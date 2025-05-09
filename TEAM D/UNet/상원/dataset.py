import os
import numpy as np
from PIL import Image
import torch
import random

# ✅ 커스텀 Dataset 클래스 (npy & 이미지 데이터셋 둘 다 지원)
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None, mode='npy'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode

        # npy 모드: 기존 방식
        if mode == 'npy':
            lst_data = os.listdir(self.data_dir)
            self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
            self.lst_input = sorted([f for f in lst_data if f.startswith('input')])
        # image 모드: 폴더 기반 이미지+마스크
        elif mode == 'image':
            self.images_dir = os.path.join(self.data_dir, 'images')
            self.masks_dir = os.path.join(self.data_dir, 'masks')
            self.lst_input = sorted(os.listdir(self.images_dir))
            self.lst_label = sorted(os.listdir(self.masks_dir))
        else:
            raise ValueError("mode must be 'npy' or 'image'")

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):  
        # ✅ 이미지 & 마스크 이름 가져오기 (마스크 이름 = 이미지 이름과 동일)
        img_name = self.lst_input[index]
        mask_name = img_name  

        # 이미지 & 마스크 로드 (흑백 모드: 'L')
        input = np.array(Image.open(os.path.join(self.images_dir, img_name)).convert('L'))
        label = np.array(Image.open(os.path.join(self.masks_dir, mask_name)).convert('L'))

        # 0~1 사이로 정규화 (픽셀 스케일링)
        label = label / 255.0
        input = input / 255.0
        
        # (H, W) → (H, W, 1)로 차원 확장
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}
        
        # 데이터 증강(transform 적용)
        if self.transform:
            data = self.transform(data)

        return data

# 랜덤 회전
class RandomRotate:
    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        k = random.randint(0, 3)
        image = np.rot90(image, k)
        mask = np.rot90(mask, k)
        return {'input': image, 'label': mask}

# 가우시안 노이즈 추가
class AddNoise:
    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        noise = np.random.randn(*image.shape) * 0.05
        image = image + noise
        image = np.clip(image, 0.0, 1.0)
        return {'input': image, 'label': mask}

# (0~1) 정규화 후 평균/표준편차 기준으로 표준화
class Normalization:
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        image = (image - self.mean) / self.std
        return {'input': image, 'label': mask}

# 랜덤 수평/수직 플립
class RandomFlip:
    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)
        return {'input': image, 'label': mask}

# numpy 배열을 PyTorch Tensor로 변환 (채널 순서 변경)
class ToTensor:
    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        image = image.transpose((2, 0, 1)).astype(np.float32)
        mask = mask.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.tensor(image), 'label': torch.tensor(mask)}

# ✅ 리사이즈: 이미지 & 마스크를 고정 크기로 맞춤
class Resize:
    def __init__(self, output_size):
        self.output_size = output_size  # (H, W)

    def __call__(self, sample):
        image, mask = sample['input'], sample['label']
        image = Image.fromarray((image[:, :, 0] * 255).astype(np.uint8)).resize(self.output_size, Image.BILINEAR)
        mask = Image.fromarray((mask[:, :, 0] * 255).astype(np.uint8)).resize(self.output_size, Image.NEAREST)

        image = np.expand_dims(np.array(image) / 255.0, axis=2)
        mask = np.expand_dims(np.array(mask) / 255.0, axis=2)

        return {'input': image, 'label': mask}
