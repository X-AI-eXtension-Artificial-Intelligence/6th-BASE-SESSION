import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from util_evaluation_change import compute_iou, compute_pixel_accuracy
from dataset import Dataset, Normalization, ToTensor
from model import UNet

# 설정
lr = 1e-3
batch_size = 4
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 & 손실함수 & 옵티마이저 정의
net = UNet().to(device)
fn_loss = nn.CrossEntropyLoss(ignore_index=255).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 모델 weight 불러오기
def load(ckpt_dir, net, optim):
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

# 데이터셋 로딩
transform = transforms.Compose([Normalization(), ToTensor()])
dataset_test = Dataset(os.path.join(data_dir, 'test'), transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

print(f"✅ loaded testset: {len(dataset_test)} samples")

net, optim, st_epoch = load(ckpt_dir, net, optim)
print(f"✅ loaded model from {ckpt_dir} (epoch {st_epoch})")

# 유틸 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True)

# 평가 시작
with torch.no_grad():
    net.eval()
    print("✅ model set to eval mode")

    loss_arr, iou_scores, pixel_accuracies = [], [], []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device).squeeze(1).long()
        input = data['input'].to(device)

        label[label > 2] = 255  # 0,1,2 외의 라벨은 무시

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        pred = fn_class(output)

        iou = compute_iou(pred, label.unsqueeze(1))
        acc = compute_pixel_accuracy(pred, label.unsqueeze(1))
        iou_scores.append(iou.item())
        pixel_accuracies.append(acc.item())

        print(f"✅ TEST: BATCH {batch:04d} | LOSS {loss.item():.4f} | IOU {iou:.4f} | ACC {acc:.4f}")

        label_np = fn_tonumpy(label.unsqueeze(1))
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        output_np = fn_tonumpy(pred)

        for j in range(label_np.shape[0]):
            id = (batch - 1) * batch_size + j
            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output_np[j].squeeze())
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output_np[j].squeeze(), cmap='gray')

print("\n" + "="*50)
print(f"✅ AVERAGE TEST RESULT")
print(f"→ LOSS     : {np.mean(loss_arr):.4f}")
print(f"→ IoU      : {np.mean(iou_scores):.4f}")
print(f"→ Accuracy : {np.mean(pixel_accuracies):.4f}")
print("="*50)
