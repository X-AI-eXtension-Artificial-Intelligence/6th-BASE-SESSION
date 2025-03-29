import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.cluster import KMeans

class CityscapeDataset(Dataset):
    def __init__(self, image_dir, label_model, transform=None):
        """  
        초기화 메서드
        Args:
            image_dir (str): 이미지 파일이 저장된 디렉토리
            label_model (KMeans): 픽셀 레이블을 예측하는 데 사용할 사전 훈련된 K-means 모델
            transform (callable, optional): 이미지에 적용할 전처리 함수
        """
        self.image_dir = image_dir
        self.image_fns = [fn for fn in os.listdir(image_dir) if fn.endswith('.jpg')]
        self.label_model = label_model
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        """데이터셋 내 이미지의 총 개수 반환"""
        return len(self.image_fns)

    def __getitem__(self, index):
        """인덱스에 해당하는 이미지를 로드 및 전처리 후 반환"""
        image_fn = self.image_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        image = Image.open(image_fp)
        image = np.array(image)
        
        cityscape, label = self.split_image(image)
        
        label_class = self.label_model.predict(label.reshape(-1, 3)).reshape(label.shape[0], label.shape[1])
        label_class = torch.tensor(label_class, dtype=torch.long)
        
        cityscape = self.transform(cityscape)
        
        return cityscape, label_class

    def split_image(self, image):
        """이미지를 도시 경관과 레이블 이미지로 분리"""
        mid_point = image.shape[1] // 2
        cityscape = image[:, :mid_point, :]
        label = image[:, mid_point:, :]
        return cityscape, label

def get_loader(image_dir, label_model, batch_size=4, shuffle=True, num_workers=0):
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
    dataset = CityscapeDataset(image_dir, label_model)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
