import torch
import torch.nn as nn

# 유연한 레이어 수, 다양한 필터 크기 추가
def conv_block(in_dim, out_dim, num_layers=2, filter_size=3):
    layers = []
    for i in range(num_layers):
        layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=filter_size, padding=filter_size // 2))
        layers.append(nn.BatchNorm2d(out_dim))  # BatchNorm 추가로 학습 안정화
        layers.append(nn.ReLU())
        in_dim = out_dim
    layers.append(nn.MaxPool2d(2, 2))
    return nn.Sequential(*layers)

# VGG 아키텍처 변경!
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()

        self.feature = nn.Sequential(
            conv_block(3, base_dim, num_layers=1),           # Layer 수 감소(2->1) (얕은 구조)
            conv_block(base_dim, 2 * base_dim, num_layers=2), # 일반 VGG 구조
            conv_block(2 * base_dim, 3 * base_dim, num_layers=3), 
            conv_block(3 * base_dim, 4 * base_dim, num_layers=4, filter_size=5), # 5x5 필터 추가
            conv_block(4 * base_dim, 4 * base_dim, num_layers=2, filter_size=1)  # 1x1 필터 추가
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(4 * base_dim, 2048),   # Fully Connected Layer 노드 수 감소
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 512), #
            nn.ReLU(True),
            nn.Dropout(0.3),                # 추가 Dropout 추가
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x
