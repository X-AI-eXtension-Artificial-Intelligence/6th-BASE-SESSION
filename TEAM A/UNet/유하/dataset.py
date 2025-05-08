import os
import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 리사이즈 -> 크기 통일 해줌
        image = cv2.resize(image, (480, 352))
        mask = cv2.imread(mask_path, 0)
        mask = cv2.resize(mask, (480, 352), interpolation=cv2.INTER_NEAREST)

        # 이미지를 0~1 범위로 정규화 해줌
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).long()
        return image, mask
