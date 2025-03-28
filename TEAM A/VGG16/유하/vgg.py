import torch
import torch.nn as nn

def conv_2_block(in_dim,out_dim): 
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), 
        # BatchNorm 추가
        ## 이전 Conv2d 레이어의 출력(feature map)을 정규화하여, 다음 레이어가 더 안정적으로 학습되도록 함
        nn.BatchNorm2d(out_dim),
        # 활성화 함수 ReLU -> LeakyReLU 변경
        ## ReLU의 문제점 : 어떤 뉴런의 경우 계속 0을 출력하면서 학습이 끊길 수 있음 (학습률이 높거나, weight 초기화가 안 좋은 경우)
        ## LeakyReLU : 0 이하의 입력값을 0이 아닌 아주 작은 음수로라도 출력시켜 학습이 유지되도록 함 / 여기서는 0.1x 값으로 출력
        nn.LeakyReLU(0.1), 
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1), 
        nn.MaxPool2d(2,2)
    )
    return model


def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1), 
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim),
        nn.LeakyReLU(0.1), 
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.LeakyReLU(0.1), 
        nn.MaxPool2d(2,2)
    )
    return model


class VGG(nn.Module):
    # base_dim 64 -> 128 변경
    ## 첫 Conv 출력 채널 값을 늘리면, 최종 feature map 채널 수도 비례하여 늘어남
    ## 채널 수가 많아지면 더 다양한 특징을 표현할 수 있음 + 더 깊은 표현 가능성이 늘어남
    def __init__(self, base_dim=128, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), # RGB 3채널 -> base_dim 128 : 기본적인 경계선 학습
            conv_2_block(base_dim,2*base_dim), # 128 -> 256 : 조금 더 복잡한 패턴 학습
            conv_3_block(2*base_dim,4*base_dim), # 256 -> 512 : 작은 객체의 특징 학습
            conv_3_block(4*base_dim,8*base_dim), # 512 -> 1024 : 큰 객체의 형태 학습
            conv_3_block(8*base_dim,8*base_dim), # 1024 -> 1024 (유지) : 최종적인 복잡한 특징 학습 / 512 : 마지막 feature map 크기
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096),
            nn.ReLU(True),
            # Dropout 비율 변경
            ## Dropout : 비율을 늘리면 과적합 방지 효과 / 비율을 낮추면 더 나은 학습을 가능하게 함
            ## 이전 기록을 보니 시간은 오래 안 걸리는데 성능이 그리 높지 않은 것 같아 Dropout 비율 0.5 -> 0.4로 낮춤 
            nn.Dropout(0.4),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(0.4),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):

        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x