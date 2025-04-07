import os
import json
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
from metrics import calculate_IOU, calculate_errors
from hParams import get_hParams


def evaluate_model(batch_size, model_save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('## device:', device)

    model_path = f'model/{model_save_name}.pth'
    data_dir = "./data/"
    val_set = os.path.join(data_dir, 'val')

    transform = data_transform()
    val_set = DatasetForSeg(data_dir=val_set, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    # model load
    model = UNet(in_channel=1, out_channel=1).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    IOUs=[]
    PixelErrors=[]
    RandErrors=[]
    WarpingErrors=[]
    # eval
    model.eval()
    with torch.no_grad():
        for batch, data in enumerate(val_loader):
            label = data['label'].to(device)
            inputs = data['input'].to(device)
            output = model(inputs)  # forward

            pred_mask = (output > 0.5).float()
            label = (label + 1) / 2
            label_mask = label.squeeze(1)
            pixel_error, rand_error, warping_error = calculate_errors(label_mask, pred_mask)
            iou = calculate_IOU(label_mask, pred_mask)
            IOUs.append(iou)
            PixelErrors.append(pixel_error)
            RandErrors.append(rand_error)
            WarpingErrors.append(warping_error)

            pred_mask = output.squeeze(1)  # (batch, H, W) -> 예측된 segmentation mask
            label_mask = label.squeeze(1)  # GT mask (batch, H, W)
    
            pred_mask_np = pred_mask[0].cpu().numpy()
            label_mask_np = label_mask[0].cpu().numpy()

            pred_mask_np = Image.fromarray((pred_mask_np*255).astype(np.uint8))
            label_mask_np = Image.fromarray((label_mask_np*255).astype(np.uint8))
            os.makedirs('./eval_result/', exist_ok=True)
            save_path = os.path.join('eval_result', f'result_{batch}_iou_{"{:.3f}".format(iou)}.png')
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(pred_mask_np, cmap='gray')
            axes[0].set_title("Predicted Mask")
            axes[1].imshow(label_mask_np, cmap='gray')
            axes[1].set_title("Ground Truth")
            plt.suptitle("Data - Pred / Ground Truth")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
    result_dict = {
        "title": model_save_name,
        "Average of IOU": float(np.mean(IOUs)),
        "Average of Pixel Error": float(np.mean(PixelErrors)),
        "Average of Rand Error": float(np.mean(RandErrors)),
        "Average of Warping Error": float(np.mean(WarpingErrors))
    }
    with open('./eval_result/eval_summary.json', 'w') as json_file:
        json.dump(result_dict, json_file, ensure_ascii=False, indent=4)
    print('done.')


if __name__ == '__main__':
    args = get_hParams()
    evaluate_model(
        batch_size=args.batch_size,
        model_name=args.model_save_name)