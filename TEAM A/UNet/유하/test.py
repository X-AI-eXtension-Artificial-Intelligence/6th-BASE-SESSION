import torch
from torch.utils.data import DataLoader
from dataset import CustomDataset
from unet import UNet
from torchmetrics import JaccardIndex

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CamVid dataset (Cambridge-driving Labeled Video Database)
# : 자동차 시점에서 촬영한 도시 도로 환경 영상을 기반으로 한 의미론적 분할 데이터셋
# -> 도로 주행 장면에서 각 픽셀을 32개의 사물/배경 클래스 중 하나로 분류하는 연구용 데이터셋

test_img_dir = './CamVid/test'
test_mask_dir = './CamVid/testannot'

test_dataset = CustomDataset(test_img_dir, test_mask_dir)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = UNet().to(device)
model.load_state_dict(torch.load('best_model.pth'))  # 학습된 모델 불러오기
model.eval()

num_classes = 32  # CamVid 클래스 수 
# mIOU를 계산하고 출력할 수 있도록 추가함
jaccard = JaccardIndex(num_classes=num_classes, task='multiclass').to(device)
correct = 0
total = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == masks).sum().item()
        total += masks.numel()

        jaccard.update(preds, masks)

print(f'Test Pixel Accuracy: {100 * correct / total:.2f}%')
print(f'Test mIoU: {jaccard.compute().item() * 100:.2f}%')

# 결과
# Test pixel Accuracy : 84% 정확하게 맞춤
# Test mIOU : 클래스별로 예측과 정답이 겹치는 비율의 평균이 51%임을 의미함