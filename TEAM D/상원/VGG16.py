import torch.nn as nn # torch 불러오기 
import torch.nn.functional as F # 신경망 함수 불러오기

# 2개 합성곱 계층 생성
# 입력 → Conv → ReLU → Conv → ReLU → MaxPooling
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        # 커널 크기 : 3 , 패딩 : 1
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), 
        # 활성화 함수 : ReLU 사용
        nn.ReLU(), 
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        # 활성화 함수 : ReLU 사용 
        nn.ReLU(),
        # 풀링 : MaxPooling 사용 , 풀링 커널 크기: 2x2
        nn.MaxPool2d(2,2), 
        
    )
    return model
# 3개 합성곱 계층 생성
# 입력 → Conv → ReLU → Conv → ReLU → Conv → ReLU → MaxPooling
def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        # 커널 크기 : 3 , 패딩 : 1
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        # 활성화 함수 : ReLU 사용
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        # 활성화 함수 : ReLU 사용
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        # 활성화 함수 : ReLU 사용
        nn.ReLU(),
        # 풀링 : MaxPooling 사용 , 풀링 커널 크기: 2x2
        nn.MaxPool2d(2,2)
    )
    return model

# VGG16 모델 정의
class VGG16(nn.Module): 
    # 기반 채널 수 , 클래스 수 인자로 받음 
    def __init__(self,base_dim,num_classes=10): # 부모 클래스 초기화
         super(VGG16,self).__init__() 
         self.feature = nn.Sequential(
            conv_2_block(3,base_dim),           # 입력 RGB(3)
            conv_2_block(base_dim,2*base_dim),  # base_dim → 2배확장 
            conv_3_block(2*base_dim,4*base_dim),# 4배 
            conv_3_block(4*base_dim,8*base_dim),# 8배 
            conv_3_block(8*base_dim,8*base_dim),# 채널 수 유지
         )

         # Fully Connected Layer : CNN으로 뽑아낸 특징 벡터 -> 그걸 1차원 벡터로 펴서 분류 
         self.fc_layer = nn.Sequential(
         nn.Linear(8*base_dim*1*1,4096),        # 1차원으로 펴고 첫 번째 FC layer
         nn.ReLU(),                             # 활성화 함수 : ReLU 사용
         nn.Linear(4096,4096),                  # 두 번째 FC layer
         nn.ReLU(True),                         # 활성화 함수 : ReLU 사용
         nn.Dropout(),                          # 과적합 방지를 위한 Dropout
         nn.Linear(4096,1000),                  # 세 번째 FC layer
         nn.ReLU(True),                         # 활성화 함수 : ReLU 사용
         nn.Dropout(),                          # 과적합 방지를 위한 Dropout
         nn.Linear(1000,num_classes),           # 클래스 수만큼 최종 출력 
         )

    def forward(self,x):
        x = self.feature(x)                     # 합성곱 블록 통과
        x = x.view(x.size(0),-1)                # 다차원 배열을 FC Layer용 1차원 벡터로 바꾸는 과정
        x = self.fc_layer(x)                    # FC layer 통과 
        probas = F.softmax(x,dim=1)             # 소프트맥스로 클래스별 확률 출력
        return probas