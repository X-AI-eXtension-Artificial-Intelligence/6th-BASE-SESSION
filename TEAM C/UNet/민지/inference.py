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
from hParams import get_hParams

def inference(model_save_name):
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print('## device:', device)

    data_dir = "./data/"
    test_set = os.path.join(data_dir, 'test')
    model_path = f'model/{model_save_name}.pth'

    transform = data_transform()
    test_set = DatasetForSeg(data_dir=test_set, transform=transform)

    # model load
    model = UNet(in_channel=1, out_channel=1).to(device)
 
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.eval()
    with torch.no_grad():
        for data_cnt, data in enumerate(test_set):
            inputs = data['input'].unsqueeze(0).to(device)
            output = model(inputs)  # forward

            pred_mask = (output > 0.5).float()


            pred_mask = output.squeeze(1)  # (batch, H, W) -> 예측된 segmentation mask
    
            pred_mask_np = pred_mask[0].cpu().numpy()


            pred_mask_np = Image.fromarray((pred_mask_np*255).astype(np.uint8))
            os.makedirs('./inference_result/', exist_ok=True)
            save_path = os.path.join('inference_result', f'result_{data_cnt}.png')

            plt.imshow(pred_mask_np, cmap='gray')
            plt.title("Predicted Mask")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    args = get_hParams()
    inference(
        model_save_name=args.model_save_name
        )