import os
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class MRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 모든 마스크 파일을 기준으로 리스트 구성
        self.mask_paths = sorted(glob.glob(os.path.join(data_dir, '*_mask.tif')))
        self.img_paths = [p.replace('_mask.tif', '.tif') for p in self.mask_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]

        # 이미지 및 마스크 불러오기 (흑백)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0

        # (H, W) → (1, H, W)
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {'input': img, 'label': mask}

        # transform 적용 (선택)
        if self.transform:
            sample = self.transform(sample)

        return sample
