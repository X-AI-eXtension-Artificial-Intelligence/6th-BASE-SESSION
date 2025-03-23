import torch
import torch.nn as nn
from tqdm import trange 

learning_rate = 0.001

import torch.nn as nn

def conv_2_block(in_dim, out_dim):
    """
    2개의 컨볼루션 레이어와 1개의 맥스 풀링 레이어로 구성된 블록을 생성하는 함수.
    
    Args:
        in_dim (int): 입력 채널 개수
        out_dim (int): 출력 채널 개수
    
    Returns:
        nn.Sequential: 컨볼루션 블록 (Conv2D x 2 + ReLU x 2 + MaxPool2D)
    """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 컨볼루션, 패딩 1
        nn.ReLU(),  # 활성화 함수 (ReLU)
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 또 다른 3x3 컨볼루션
        nn.ReLU(),  # 활성화 함수 (ReLU)
        nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 맥스 풀링 (다운샘플링)
    )
    return model

def conv_3_block(in_dim, out_dim):
    """
    3개의 컨볼루션 레이어와 1개의 맥스 풀링 레이어로 구성된 블록을 생성하는 함수.
    
    """
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 3x3 컨볼루션, 입력 채널 -> 출력 채널
        nn.ReLU(),  # 활성화 함수 (ReLU)
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 3x3 컨볼루션, 출력 채널 유지
        nn.ReLU(),  # 활성화 함수 (ReLU)
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 세 번째 3x3 컨볼루션, 출력 채널 유지
        nn.ReLU(),  # 활성화 함수 (ReLU)
        nn.MaxPool2d(kernel_size=2, stride=2)  # 2x2 맥스 풀링 (특성 맵 크기 절반 축소)
    )
    return model

class VGG(nn.Module):
    """
    VGG 모델을 구현한 클래스.
    
    Args:
        base_dim (int): 기본 채널 개수 (64부터 시작)
        num_classes (int): 분류할 클래스 개수 (기본값: 10, CIFAR-10)
    """
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),         # 입력(3채널) -> 64채널 (Conv2D x2 + MaxPool)
            conv_2_block(base_dim, 2*base_dim), # 64채널 -> 128채널 (Conv2D x2 + MaxPool)
            conv_3_block(2*base_dim, 4*base_dim), # 128채널 -> 256채널 (Conv2D x3 + MaxPool)
            conv_3_block(4*base_dim, 8*base_dim), # 256채널 -> 512채널 (Conv2D x3 + MaxPool)
            conv_3_block(8*base_dim, 8*base_dim), # 512채널 -> 512채널 (Conv2D x3 + MaxPool)
        )
        
        self.fc_layer = nn.Sequential(
            # CIFAR-10은 입력 이미지 크기가 32x32이므로 1x1 feature map 사용
            nn.Linear(8*base_dim*1*1, 4096),  # 512 -> 4096 Fully Connected Layer
            # IMAGENET의 경우 입력 크기가 224x224이므로 아래를 사용해야 함
            # nn.Linear(8*base_dim*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),  # 4096 -> 1000 Fully Connected Layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),  # 1000 -> num_classes Fully Connected Layer
        )

    def forward(self, x):
        """
        순전파 함수.
        
        Args:
            x (Tensor): 입력 이미지 텐서
        
        Returns:
            Tensor: 분류 결과
        """
        x = self.feature(x)  # 특징 추출
        x = x.view(x.size(0), -1)  # Fully Connected Layer에 입력을 위해 Flatten 처리
        x = self.fc_layer(x)  # 분류 진행
        return x




