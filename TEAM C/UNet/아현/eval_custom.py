import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from dataset_custom import CityscapesCombinedDataset
from model import UNet

# 하이퍼파라미터
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 결과 저장 경로
result_dir = "results_eval"
os.makedirs(os.path.join(result_dir, "png"), exist_ok=True)
os.makedirs(os.path.join(result_dir, "numpy"), exist_ok=True)

# 데이터셋 및 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

val_dataset = CityscapesCombinedDataset(
    root_dir = "/home/work/.local/unet/datasets_city/val",
    transform=transform
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 모델 로드
model = UNet().to(device)
# model.load_state_dict(torch.load("saved_models/unet_cityscapes.pth", map_location=device))
model.load_state_dict(torch.load("/home/work/.local/unet/saved_models/unet_cityscapes.pth", map_location=device))

model.eval()

# 손실 함수
criterion = nn.BCEWithLogitsLoss()

# 평가
loss_list = []
with torch.no_grad():
    for i, (images, masks) in enumerate(val_loader):
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)
        loss_list.append(loss.item())

        # 후처리
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

        # 저장
        images_np = images.cpu().numpy().squeeze(1)
        masks_np = masks.cpu().numpy().squeeze(1)
        preds_np = preds.cpu().numpy().squeeze(1)

        for j in range(images_np.shape[0]):
            idx = i * batch_size + j
            plt.imsave(f"{result_dir}/png/input_{idx:04d}.png", images_np[j], cmap='gray')
            plt.imsave(f"{result_dir}/png/label_{idx:04d}.png", masks_np[j], cmap='gray')
            plt.imsave(f"{result_dir}/png/output_{idx:04d}.png", preds_np[j], cmap='gray')

            np.save(f"{result_dir}/numpy/input_{idx:04d}.npy", images_np[j])
            np.save(f"{result_dir}/numpy/label_{idx:04d}.npy", masks_np[j])
            np.save(f"{result_dir}/numpy/output_{idx:04d}.npy", preds_np[j])

print(f"Validation Loss: {np.mean(loss_list):.4f}")
