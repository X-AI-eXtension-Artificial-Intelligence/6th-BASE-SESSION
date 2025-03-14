import torch.nn as nn
import torch.nn.functional as F 

# 2개의 Conv layer
def conv_2_block(in_dim, out_dim):                          # in_dim: 입력값 채널 수 #out_dim: 첫 컨볼루션 레이어의 출력 채널 수
    model = nn.Sequential(                                  # 여러 레이어를 순차적으로 쌓아서 하나의 모듈로 만들어줌
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), 
        nn.ReLU(),                                          # 활성화 함수(비선형성 증가)
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1), # 첫번째 out-dim: 이전 layer의 출력을 입력으로 받아, 동일한 채널 수 out_dim
        nn.ReLU(),
        nn.MaxPool2d(2,2),                                  # Max Pooling(사이즈 감소)
        
    )
    return model

# 3개의 Conv layer
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


# VGG16 모델 클래스스
class VGG16(nn.Module): 
    def __init__(self,base_dim,num_classes=10):
         super(VGG16,self).__init__()                 # VGG16 클래스가 nn.Module 클래스를 상속받도록 초기화화
         self.feature = nn.Sequential(
            conv_2_block(3,base_dim),                 # 입력 채널 3(RGB)
            conv_2_block(base_dim,2*base_dim),        # 채널 수를 두 배로 늘려감
            conv_3_block(2*base_dim,4*base_dim),
            conv_3_block(4*base_dim,8*base_dim),
            conv_3_block(8*base_dim,8*base_dim), 
         )
         self.fc_layer = nn.Sequential(
         nn.Linear(8*base_dim*1*1,4096),              # Fully Connected Layer
         nn.ReLU(),
         nn.Linear(4096,4096),
         nn.ReLU(True),
         nn.Dropout(),                                # 드롭아웃(과적합 방지)
         nn.Linear(4096,1000), 
         nn.ReLU(True),
         nn.Dropout(),
         nn.Linear(1000,num_classes),                 # 출력층층(CIFAR10는 10개의 클래스로 구성됨)
         )
    def forward(self,x):
        x = self.feature(x) 
        x = x.view(x.size(0),-1)                      # 1차원 벡터로 평탄화화
        x = self.fc_layer(x)
        probas = F.softmax(x,dim=1)
        return probas