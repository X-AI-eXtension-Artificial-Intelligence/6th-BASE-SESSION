import torch
import torch.nn as nn

# conv_2_block
# 2개의 Conv 레이어로 구성된 블록을 정의
def conv_2_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), # 첫 번째 Conv2d 레이어: 입력 채널(in_dim)에서 출력 채널(out_dim)로 변환, 커널 크기 3, padding 1
        nn.ReLU(), # ReLU (비선형 활성화)
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),  # 두 번째 Conv2d 레이어: 출력 채널이 다시 동일하게 out_dim
        nn.ReLU(), # ReLU (비선형 활성화)
        nn.MaxPool2d(2,2) # 2x2 크기의 풀링
    )
    return model

# conv_3_block
def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), # 첫 번째 Conv2d 레이어: 입력 채널(in_dim)에서 출력 채널(out_dim)로 변환, 커널 크기 3, padding 1
        nn.ReLU(),  # ReLU (비선형 활성화)
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1), # 두 번째 Conv2d 레이어: 출력 채널이 다시 동일하게 out_dim
        nn.ReLU(),  # ReLU (비선형 활성화)
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1), # 세세 번째 Conv2d 레이어: 출력 채널이 다시 동일하게 out_dim
        nn.ReLU(),  # ReLU (비선형 활성화)
        nn.MaxPool2d(2,2) # 2x2 크기의 풀링
    )
    return model

# Define VGG16
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10): # base_dim: 기본 채널 크기, num_classes: 출력 클래스 수 (기본 10은 CIFAR10 기준)
        super(VGG, self).__init__() # 부모 클래스 nn.Module의 생성자 호출
        self.feature = nn.Sequential(  # 특징 추출을 위한 conv 레이어들 (Sequential로 쌓음)
            conv_2_block(3,base_dim), #64 -> 첫 번째 conv_2_block: 입력 채널 3 (RGB 이미지), 출력 채널 base_dim
            conv_2_block(base_dim,2*base_dim), #128 -> 두 번째 conv_2_block: 입력 채널 base_dim, 출력 채널 2*base_dim 
            conv_3_block(2*base_dim,4*base_dim), #256 -> 세 번째 conv_3_block: 입력 채널 2*base_dim, 출력 채널 4*base_dim (예: 256)
            conv_3_block(4*base_dim,8*base_dim), #512 -> 네 번째 conv_3_block: 입력 채널 4*base_dim, 출력 채널 8*base_dim
            conv_3_block(8*base_dim,8*base_dim), #512 -> 다섯 번째 conv_3_block: 입력 채널 8*base_dim, 출력 채널 8*base_dim    
        )
        self.fc_layer = nn.Sequential(
            # CIFAR10은 크기가 32x32이므로 
            nn.Linear(8*base_dim*1*1, 4096),
            # IMAGENET이면 224x224이므로
            # nn.Linear(8*base_dim*7*7, 4096),
            nn.ReLU(True), # 활성화 함수 : ReLU
            nn.Dropout(), # dropout을 통해 과적합 방지
            nn.Linear(4096, 1000), # 선형 projecion 
            nn.ReLU(True), 
            nn.Dropout(), 
            nn.Linear(1000, num_classes), 
        )

    def forward(self, x):
        x = self.feature(x)
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc_layer(x)
        return x