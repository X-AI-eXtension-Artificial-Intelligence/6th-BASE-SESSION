import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader

from tqdm import tqdm
import cv2
from PIL import Image

from model import UNet
from dataset import data_transform, DatasetForSeg
from metrics import calculate_IOU, calculate_dice_coefficient


def evaluate_model(setting_config: dict):
    device = setting_config['device']
    batch_size = setting_config['batch_size']
    model_path = setting_config['model_path']

    data_dir = "./data/"
    test_set = os.path.join(data_dir, 'test')

    transform = data_transform()
    test_set = DatasetForSeg(data_dir=test_set, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # model load
    model = UNet(in_channel=1, out_channel=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # eval
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(test_loader):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = model(inputs)  # forward

            pred_mask = (output > 0.5).float()
            label_mask = label.squeeze(1)

            dice_coefficient = calculate_dice_coefficient(label_mask, pred_mask)
            iou = calculate_IOU(label_mask, pred_mask)
            print("\n##### Dice coefficient (Dice similarity coefficient, DSC) : ", dice_coefficient)
            print("##### Intersection over Union (IOU): ", iou)

            pred_mask = output.squeeze(1)  # (batch, H, W) -> 예측된 segmentation mask
            label_mask = label.squeeze(1)  # GT mask (batch, H, W)
    
            pred_mask_np = pred_mask[0].cpu().numpy()
            label_mask_np = label_mask[0].cpu().numpy()

            pred_mask_np = Image.fromarray((pred_mask_np*255).astype(np.uint8))
            label_mask_np = Image.fromarray((label_mask_np*255).astype(np.uint8))
            os.makedirs('./result/', exist_ok=True)
            save_path = os.path.join('result', f'result_{batch}_dice_{dice_coefficient}_iou_{"{:.3f}".format(iou)}_BCELoss.png')
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(pred_mask_np, cmap='gray')
            axes[0].set_title("Predicted Mask")
            axes[1].imshow(label_mask_np, cmap='gray')
            axes[1].set_title("Ground Truth")
            plt.suptitle("Data - Pred / Ground Truth")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model_path = "./model/unet_BCELoss.pth"

    setting_config = {
        "batch_size": 1,
        "device": device,
        "model_path": model_path}

    evaluate_model(setting_config)