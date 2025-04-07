import os
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import UNet
from dataset import DatasetForSeg, data_transform
from metrics import calculate_IOU, calculate_errors
from hParams import get_hParams
'''
(*) with torch.cuda.amp
- PyTorch에서 자동 혼합 정밀도를 지원하는 모듈
- 혼합 정밀도를 사용하면 GPU의 계산 자원을 더 효율적으로 활용할 수 있음
- 혼합 정밀도 : 부동 소수점 연산에서 16비트(half precision)와 32비트(single precision)를 적절히 혼합해 사용하는 것을 의미

(*) autocast():
- 특정 범위 내에서 혼합 정밀도를 자동으로 적용
- 이 컨텍스트 내에서는 모델의 연산이 성능을 최적화하기 위해 자동으로 FP16(half precision)과 FP32(single precision)로 전환
- FP16 연산은 FP32보다 메모리 사용량이 적고, 연산 속도가 더 빠름
- 정밀도 유지: 모든 연산을 FP16으로 수행할 경우, 정밀도가 떨어질 수 있지만, autocast()로 필요한 연산을 FP32로 유지함으로써 정밀도 보장
- forward 및 loss 계산이 자동으로 혼합 정밀도로 수행되로록 forward 계산에 이를 사용함

연산에서의 정밀도 (Precision in Computation)
- 부동소수점(Floating Point) 숫자를 얼마나 정확하게 표현하고 계산하는가?
-> 데이터 표현 방식(FP16, FP32, FP64 등)과 연산의 정확성에 관련됨.

📌 
FP16(16비트): 메모리 사용량이 적고 연산 속도가 빠르지만, 표현할 수 있는 값의 범위가 좁고 반올림 오류가 발생할 가능성이 높음
FP32(32비트): 대부분의 딥러닝 모델에서 기본적으로 사용됨
FP64(64비트): 매우 높은 정밀도를 제공하지만 연산 속도가 느리고 메모리 사용량이 큼

📌 연산 정밀도가 중요한 이유
FP16을 사용할 경우 작은 숫자가 0으로 변하는 언더플로우(Underflow) 문제가 발생할 수 있음
큰 숫자가 너무 커져서 계산할 수 없는 오버플로우(Overflow) 문제가 발생할 수 있음
학습 과정에서 오차가 누적될 가능성이 있음
'''
def train_model(batch_size, learning_rate, num_epoch, model_name):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(num_epoch)
    print('## device: ', device)
    # data
    data_dir = "./data/"
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    transform = data_transform()
    train_set = DatasetForSeg(data_dir=train_dir, transform=transform)
    test_set = DatasetForSeg(data_dir=test_dir, transform=transform)

    # DataLoader : 미니배치(batch) 단위로 데이터를 제공
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Trainer
    model = UNet(in_channel=1, out_channel=1).to(device) # grayscale
    loss_func = nn.BCEWithLogitsLoss().to(device)

    # UNet에서는 SGD, momentum=0.99였지만 메모리 절약을 위해 AdamW 사용
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    ## wandb
    wandb.init(project="unet-training", name=f"epoch{num_epoch}-batch{batch_size}", config={
        "epochs": num_epoch,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "learning_rate": learning_rate,
    })

    print('## start training ! ##')
    loss_arr = []
    for i in tqdm(range(num_epoch), total=num_epoch, desc='training...'):
        for batch, data in enumerate(train_loader):
            model.train()
            inputs = data['input'].to(device, non_blocking=True)
            input_canny = data['input_canny'].to(device, non_blocking=True)
            label = data['label'].to(device, non_blocking=True) # 데이터 로딩 속도 향상 및 CUDA 스트리밍 활용
            label = (label + 1) / 2
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(inputs, input_canny)  # forward
                loss = loss_func(output, label)

            pred_mask = (output > 0.5).float()
            iou = calculate_IOU(label, pred_mask)
            pixel_error, rand_error, warping_error = calculate_errors(label, pred_mask)
            wandb.log({"IOU": iou, "epoch": i})
            wandb.log({"Pixel Error": pixel_error, "epoch": i})
            wandb.log({"Rand Error": rand_error, "epoch": i})
            wandb.log({"Warping Error": warping_error, "epoch": i})
   
            # backward
            # AMP 스케일링 적용
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            wandb.log({"train_loss": loss.item(), "epoch": i})

            # 배치별 메모리 정리
            del inputs, label, output, pred_mask, loss
            torch.cuda.empty_cache()

        # 10 에포크마다 로그 출력
        if i % 10 == 0:
            print(f'Epoch {i} completed.')

            model.eval()
            with torch.no_grad():
                # --- Segmentation 이미지 로깅 ---
                # 단일 데이터이므로 앞에 배치 차원 추가 unsqueeze(0)
                inputs_val = test_set[0]['input'].unsqueeze(0).to(device)
                label_val = test_set[0]['label'].unsqueeze(0).to(device)
                input_canny_val = test_set[0]['input_canny'].unsqueeze(0).to(device)
                label_val = (label_val + 1) / 2
                output_val = model(inputs_val, input_canny_val)

                pred_mask = output_val.squeeze(1)  # (batch, H, W) -> 예측된 segmentation mask
                label_mask = label_val.squeeze(1)  # GT mask (batch, H, W)
                # canny_mask = input_canny_val.squeeze(1)

                pred_mask_np = pred_mask[0].cpu().detach().numpy()  # .cpu().detach()를 추가하여 GPU에서 CPU로 이동 후 numpy로 변환
                label_mask_np = label_mask[0].cpu().detach().numpy()
                # canny_mask_np = canny_mask[0].cpu().detach().numpy()
                
                wandb.log({
                    "Predicted Mask": wandb.Image(pred_mask_np, caption="Prediction"),
                    "Ground Truth": wandb.Image(label_mask_np, caption="Ground Truth"),
                })
                # 평가 후 메모리 정리
                del inputs_val, label_val, output_val, input_canny_val, pred_mask, label_mask
                torch.cuda.empty_cache()

    wandb.finish()

    os.makedirs('model/', exist_ok=True)
    torch.save(model.state_dict(), f'model/{model_name}_epoch{num_epoch}.pth') # save_model_name default : unet_vanilla

if __name__ == '__main__':
    args = get_hParams()
    train_model(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epoch=args.num_epoch,
        model_name=args.model_name)
