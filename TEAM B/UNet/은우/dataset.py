from PIL import Image


import torchvision.transforms.functional as F
import random
import torch
from torchvision import datasets, transforms  # datasets를 임포트해야 합니다.
from torch.utils.data import DataLoader

class Dataset:
    def __init__(self, root_dir='./data', year='2012', image_set='train', batch_size=8, image_size=(256, 256), transform=None):
        self.root_dir = root_dir  # 'data_dir'을 'root_dir'로 수정
        self.year = year
        self.image_set = image_set
        self.batch_size = batch_size
        self.image_size = image_size
        self.transform = transform  # transform을 인자로 받도록 수정

        self.dataset = datasets.VOCSegmentation(root=self.root_dir, year=self.year, image_set=self.image_set, download=True)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        
        if self.transform:
            img = self.transform(img)

        return img, label

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def __len__(self):
        return len(self.dataset)


class RandomFlip(object):
    def __call__(self, img):
        # img는 PIL.Image 또는 Tensor여야 함
        if random.random() > 0.5:
            img = F.hflip(img)
        if random.random() > 0.5:
            img = F.vflip(img)
        return img

class ToTensor(object):
    def __call__(self, img):
        return F.to_tensor(img)  # PIL → Tensor


class Normalization(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
        return transforms.Normalize(mean=self.mean, std=self.std)(img)

