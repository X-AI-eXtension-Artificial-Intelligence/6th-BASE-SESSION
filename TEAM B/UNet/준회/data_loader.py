import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans

class CityscapeDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        초기화 메서드
        Args:
            image_dir (str): 이미지 파일이 저장된 디렉토리
            mask_dir (str): 마스크 파일이 저장된 디렉토리
            label_model (KMeans): 픽셀 레이블을 예측하는 데 사용할 사전 훈련된 K-means 모델
            transform (callable, optional): 이미지에 적용할 전처리 함수
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # 지원되는 확장자 목록
        supported_extensions = ('.png', '.jpg', '.jpeg')
        self.image_fns = [fn for fn in os.listdir(image_dir)
                          if fn.lower().endswith(supported_extensions)]

        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()  # 마스크는 정규화하지 않음
        ])

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        """인덱스에 해당하는 이미지와 마스크를 로드 및 전처리 후 반환"""
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        mask_fp = os.path.join(self.mask_dir, os.path.splitext(image_fn)[0] + '.png')  # 마스크 파일은 .png
        
        image = Image.open(image_fp).convert('RGB')
        mask = Image.open(mask_fp).convert('L')  # 흑백 이미지로 변환

        image = self.transform(image)
        mask = self.mask_transform(mask)
            
        mask = mask.squeeze(0)  # ToTensor 후에 추가된 첫 번째 차원을 제거

        return image, mask.long()  # 마스크 데이터를 long 타입으로 변환

def get_loader(image_dir, mask_dir, label_model, batch_size=4, shuffle=True, num_workers=0):
    """
    데이터 로더를 생성하고 반환하는 함수
    Args:
        image_dir (str): 이미지 파일이 저장된 디렉토리
        label_model (KMeans): 픽셀 레이블을 예측하는 데 사용할 K-means 모델
        batch_size (int): 배치 크기
        shuffle (bool): 데이터를 섞을지 여부
        num_workers (int): 데이터 로딩에 사용할 프로세스 수
    Returns:
        DataLoader: 생성된 데이터 로더
    """
    dataset = CityscapeDataset(image_dir, mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader