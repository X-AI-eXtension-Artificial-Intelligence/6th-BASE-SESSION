import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from skimage.transform import resize

class PersonSegDataset(TorchDataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir  = data_dir
        self.transform = transform

        # 파일 목록 정렬
        all_files = sorted(os.listdir(self.data_dir))
        self.lst_image = [f for f in all_files if f.startswith('input')]
        self.lst_label = [f for f in all_files if f.startswith('label')]

        # 입력/레이블 개수 일치 확인
        assert len(self.lst_image) == len(self.lst_label), \
            f"Number of inputs ({len(self.lst_image)}) != labels ({len(self.lst_label)})"

    def __len__(self):
        return len(self.lst_image)

    def __getitem__(self, idx):
        # .npy 로드
        image = np.load(os.path.join(self.data_dir, self.lst_image[idx])).astype(np.float32) / 255.0
        mask  = np.load(os.path.join(self.data_dir, self.lst_label[idx])).astype(np.float32)

        # 차원 추가 (H, W) -> (H, W, 1)
        if mask.ndim == 2:
            mask = mask[..., np.newaxis]
        if image.ndim == 2:
            image = image[..., np.newaxis]

        data = {'image': image, 'mask': mask}
        if self.transform:
            data = self.transform(data)

        return data

# Tensor 변환
class ToTensor(object):
    # (H, W, C) -> (C, H, W)
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        image = image.transpose(2, 0, 1).astype(np.float32)
        mask  = mask.transpose(2, 0, 1).astype(np.float32)
        return {'image': torch.from_numpy(image), 'mask': torch.from_numpy(mask)}

# 정규화
class Normalize(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std  = std

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        image = (image - self.mean) / self.std
        return {'image': image, 'mask': mask}

# 랜덤 플립 (데이터 증강)
class RandomFlip(object):
    def __call__(self, data):
        image, mask = data['image'], data['mask']
        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask  = np.fliplr(mask)
        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask  = np.flipud(mask)
        return {'image': image, 'mask': mask}

# 리사이즈 및 패딩
class ResizeWithPadding(object):
    def __init__(self, output_size):
        self.target_h, self.target_w = output_size

    def __call__(self, data):
        image, mask = data['image'], data['mask']
        h, w, c = image.shape
        scale = min(self.target_h / h, self.target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # image: bilinear(=order=1), mask: nearest(=order=0)
        img_r = resize(image, (new_h, new_w, c),
                       preserve_range=True, order=1).astype(np.float32)
        msk_r = resize(mask,  (new_h, new_w, mask.shape[2]),
                       preserve_range=True, order=0,
                       anti_aliasing=False).astype(np.float32)

        # 패딩 생성
        pad_img = np.zeros((self.target_h, self.target_w, c), dtype=np.float32)
        pad_msk = np.zeros((self.target_h, self.target_w, mask.shape[2]), dtype=np.float32)

        ph = (self.target_h - new_h) // 2
        pw = (self.target_w - new_w) // 2
        pad_img[ph:ph+new_h, pw:pw+new_w] = img_r
        pad_msk[ph:ph+new_h, pw:pw+new_w] = msk_r

        return {'image': pad_img, 'mask': pad_msk}
