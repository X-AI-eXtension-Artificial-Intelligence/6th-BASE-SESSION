import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class NucleiDataset(Dataset):
    def __init__(self, root_dir, image_ids, transform=None, target_size=(256,256)):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id, 'images', f'{image_id}.png')
        mask_dir = os.path.join(self.root_dir, image_id, 'masks')

        # 이미지
        image = Image.open(img_path).convert('RGB').resize(self.target_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)  # (C, H, W)

        # 마스크 여러 장을 하나로 합치고 리사이즈
        mask = np.zeros(self.target_size, dtype=np.float32)
        if os.path.exists(mask_dir):
            for mask_file in os.listdir(mask_dir):
                mask_path = os.path.join(mask_dir, mask_file)
                mask_inst = Image.open(mask_path).convert('L').resize(self.target_size, Image.NEAREST)
                mask = np.maximum(mask, np.array(mask_inst, dtype=np.float32) / 255.0)
        mask = mask[None, ...]  # (1, H, W)

        if self.transform:
            image, mask = self.transform(image, mask)

        return {
            'input': torch.tensor(image, dtype=torch.float32),
            'label': torch.tensor(mask, dtype=torch.float32)
        }

class NucleiTestDataset(Dataset):
    def __init__(self, root_dir, image_ids, target_size=(256,256)):
        self.root_dir = root_dir
        self.image_ids = image_ids
        self.target_size = target_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.root_dir, image_id, 'images', f'{image_id}.png')
        image = Image.open(img_path).convert('RGB').resize(self.target_size, Image.BILINEAR)
        image = np.array(image, dtype=np.float32) / 255.0
        image = image.transpose(2, 0, 1)
        return {
            'input': torch.tensor(image, dtype=torch.float32),
            'image_id': image_id
        }
