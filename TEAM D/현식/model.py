import torch.nn as nn
import torch.nn.functional as F

# VGG 이전 모델들은 11x11 or 7x7 filter 사용
# VGG는 3번의 3x3 Conv를 통해 7x7 Conv와 동일한 Receptive Field를 가지면서도 연산량 감소, 비선형성 증가


## conv_2_block

# in_dim : 입력 채널 수/ out_dim : 출력 채널 수(=filter 개수)
# nn.Sequential : 모델 구성 요소를 순차적으로 실행
# nn.Conv2d : 2D Convolution
# 3x3 filter # padding 1을 통해 출력 크기가 입력 크기와 동일하게 유지
# 비선형 활성화 함수로 ReLU 적용
# 2x2 MaxPooling 수행하여 feature map의 크기를 절반으로 줄임

def conv_2_block(in_dim,out_dim): 
    model = nn.Sequential( 
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


## conv_3_block

# conv_2_block과 다르게 3개의 convolution layer로 이루어짐

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


## VGG

# nn.Module : PyTorch에서 신경망 계층을 구성하고, 학습 가능한 parameter와 연산을 관리하기 위한 기본 클래스
# base_dim : 첫 번째 convolution layer의 출력 채널수 / num_classes : 출력 class 수 
# super(VGG, self).__init__ : 부모 클래스(nn.Module)의 생성자 호출, 자식 클래스(VGG)와의 관계를 명시적으로 연결

# feature extractor에서 convolution layer를 통과할 때마다 채널이 2배 증가
# 첫 번째 convolution layer의 입력 채널 수는 3 -> RGB 이미지

# fc layer에서 최종 출력 개수는 class 수
# CIFAR10은 크기가 32x32이므로 nn.Linear(8*base_dim*1*1, 4096)
# IMAGENET이면 224x224이므로 nn.Linear(8*base_dim*7*7, 4096)

# view()를 통해 feature extractor에서 나온 feature map을 1차원 벡터로 펼침


class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), 
            conv_2_block(base_dim,2*base_dim), 
            conv_3_block(2*base_dim,4*base_dim), 
            conv_3_block(4*base_dim,8*base_dim), 
            conv_3_block(8*base_dim,8*base_dim),         
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x