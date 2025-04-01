# eval.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from data_loader import get_loader  # 같은 방식의 loader 사용
from model import UNet


def compute_iou(pred, target, num_classes):
    ious = []
    for cls in range(num_classes):
        pred_inds = (pred == cls)
        target_inds = (target == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 모델 불러오기
    model = UNet().to(device)
    model.load_state_dict(torch.load("unet_custom.pth", map_location=device))
    model.eval()

    # 평가 데이터셋 로더
    val_loader = get_loader(batch_size=4, shuffle=False, num_workers=0)

    inverse_transform = transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    )

    all_ious = []

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            # 시각화 (첫 배치만)
            if batch_idx == 0:
                fig, axes = plt.subplots(images.size(0), 3, figsize=(12, 4 * images.size(0)))
                for i in range(images.size(0)):
                    img = inverse_transform(images[i]).permute(1, 2, 0).cpu().numpy()
                    mask = masks[i].cpu().numpy()
                    pred = preds[i].cpu().numpy()

                    axes[i, 0].imshow(img)
                    axes[i, 0].set_title("Input")
                    axes[i, 1].imshow(mask)
                    axes[i, 1].set_title("Ground Truth")
                    axes[i, 2].imshow(pred)
                    axes[i, 2].set_title("Prediction")

                    for j in range(3):
                        axes[i, j].axis("off")
                plt.tight_layout()
                plt.show()

            for i in range(images.size(0)):
                iou = compute_iou(preds[i], masks[i], num_classes=10)
                all_ious.append(iou)

    all_ious = np.array(all_ious)
    mean_iou_per_class = np.nanmean(all_ious, axis=0)
    mean_iou = np.nanmean(mean_iou_per_class)

    print(f"\n📊 Per-Class IoU:")
    for idx, iou in enumerate(mean_iou_per_class):
        print(f"  Class {idx}: {iou:.4f}")
    print(f"\n✅ Mean IoU: {mean_iou:.4f}")


if __name__ == '__main__':
    main()
