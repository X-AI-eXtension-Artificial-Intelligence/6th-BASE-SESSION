import torch
import torch.nn as nn
from tqdm import trange

# 학습률 설정
learning_rate = 0.001

def conv_2_block_with_bn(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),  # 배치 정규화 추가
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_dim),  # 배치 정규화 추가
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    )
    return model

# VGG 모델을 배치 정규화 추가하여 수정
class VGG_BN(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG_BN, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block_with_bn(3, base_dim),
            conv_2_block_with_bn(base_dim, 2 * base_dim),
            conv_3_block(2 * base_dim, 4 * base_dim),
            conv_3_block(4 * base_dim, 8 * base_dim),
            conv_3_block(8 * base_dim, 8 * base_dim),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8 * base_dim * 1 * 1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
