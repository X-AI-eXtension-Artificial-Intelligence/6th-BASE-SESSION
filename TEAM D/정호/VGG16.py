import torch.nn as nn            # PyTorch의 신경망 모듈을 포함하는 패키지로, 다양한 신경망 레이어와 손실 함수 등을 제공
import torch.nn.functional as F  # 다양한 신경망 함수(예: 활성화 함수, 손실 함수 등)를 제공하는 모듈

# conv layer 2개
# in_dim: 입력 채널 수(예: RGB 이미지의 경우 3) # out_dim: 첫 번째 컨볼루션 레이어의 출력 채널 수(output 개수가 필터의 개수)
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(       # nn.Sequential: 여러 레이어를 순차적으로 쌓아서 하나의 모듈로 만들어주는 PyTorch 클래스           
        # Conv2d:2차원 컨볼루션 연산을 수행하는 PyTorch 클래스(입력 채널 수, 출력 채널 수, 3x3크기의 필터 사용, 패딩=1 -> 출력 데이터의 크기를 입력과 동일하게 유지)
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),  # 활성화 함수(Relu -> 비선형성 증가가)
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1), # 이전 레이어의 출력을 입력으로 받아, 동일한 채널 수 out_dim
        nn.ReLU(),
        nn.MaxPool2d(2,2), # Max pooling
        
    )
    return model

# conv layer 3개
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


# 상속을 받음으로써 PyTorch에서 샤용자 정의 신경망 모델이 파라미터 관리, GPU 지원 등의 기능을 자동으로 활용할 수 있게 함
class VGG16(nn.Module): # nn.Module: PyTorch에서 신경망 모델을 정의할 때 사용하는 기본 클래스
    # base_dim: 네트워크의 첫 번째 conv layer에서 사용할 필터 수(= 출력 채널 수)
    # num_classes: 출력층에서의 클래스 수 (CIFAR10 데이터셋은 총 10개의 레이블로 구성됨)
    def __init__(self,base_dim, num_classes=10):
         super(VGG16,self).__init__() # VGG16 클래스가 nn.Module 클래스의 모든 기능을 상속받아 사용할 수 있도록 초기화
         self.feature = nn.Sequential(
            conv_2_block(3,base_dim), # 입력 이미지의 채널 수 = 3 (RGB)
            conv_2_block(base_dim,2*base_dim), # 채널 수를 두 배로 늘려감(네트워크가 깊어질수록 더 복잡한 특징을 추출함)
            conv_3_block(2*base_dim,4*base_dim),
            conv_3_block(4*base_dim,8*base_dim),
            conv_3_block(8*base_dim,8*base_dim), 
         )

         # FC layer
         self.fc_layer = nn.Sequential(
         nn.Linear(8*base_dim*1*1,4096), # 1x1 사용 이유: 1차원 벡터로 변환한다는 의미를 담기 위함 # 4096 차원의 출력으로 변환
         nn.ReLU(),
         nn.Linear(4096,4096),
                          # 기본적으로 활성화 함수를 통과한 결과를 저장하기 위해 새로운 메모리 공간을 할당
         nn.ReLU(True),   # inplace=True : 출력 데이터를 저장하기 위해 추가적인 메모리를 할당하는 대신 입력 텐서를 직접 수정하여 결과를 저장(메모리 절약)
         nn.Dropout(),    # FC layer는 많은 파라미터를 포함하고 있어 Dropout를 통해 과적합 방지
         nn.Linear(4096,1000), 
         nn.ReLU(True),
         nn.Dropout(),
         nn.Linear(1000,num_classes),
         )
    # 순전파
    def forward(self,x):
        x = self.feature(x) 
        x = x.view(x.size(0),-1) # Pytorch의 텐서 변환 메서드 : 추출된 특징 맵 'x'를 1차원 벡터로 변환 (-1은 차원을 자동을 계산하라는 의미)
        x = self.fc_layer(x)
        probas = F.softmax(x,dim=1)
        return probas