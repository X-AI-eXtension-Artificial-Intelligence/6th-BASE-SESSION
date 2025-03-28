# VGG19 Architecture Summary - 3*3 Conv filter(stride = 1), MaxPooling(stride = 2), Activation(ReLU)
# VGG Input image 224*224 기준 - 포켓몬 데이터셋도 224*224

# 라이브러리 import
import torch.nn as nn #파이토치 모듈 - 레이어 구성, 정규화, 손실함수, 활성화 함수 등

# 3*3 Conv연산 2개 layer 모듈화(Max Pooling 포함)
def conv_2_block(in_dim, out_dim): #기본 인자는 들어갈때 차원이랑, 나올 때 차원
    model = nn.Sequential( # nn.Sequential 안에 연산 순차적으로 넣어서 모듈로 묶을 수 있음
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), #3*3 kernel 활용 -> MaxPooling으로 Feature map size 줄기 때문에, 급격한 feature map size 감소 방지 위해 padding=1 추가해서 원본 사이즈 유지
        nn.ReLU(), # ReLU 활성화 함수 적용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), # 동일하게 conv연산 한번더
        nn.ReLU(),
        nn.MaxPool2d(2,2) #2*2 size MaxPooling 적용
    )
    return model

# 3*3 Conv연산 4개 layer 모듈화(Max Pooling 포함)
def conv_4_block(in_dim, out_dim): # 위의 블록과 동일하게 기본 인자는 input 차원, output 차원
    model = nn.Sequential(
        # 이하 (Conv 연산 + 활성화함수 적용) 4번 반복 뒤 MaxPooling 까지는 동일
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), 
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model

# 모델 정의
class VGG19(nn.Module):
    def __init__(self, base_dim, num_classes = 150): #포켓몬 데이터셋 클래스 개수 150개
        super().__init__() #상속받은 nn.Module 부모 클래스 초기화 -> 파이토치 기능 활성화
        
        #특징 추출기(Conv layer)
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), #3개 채널 -> 64
            conv_2_block(base_dim, 2*base_dim), #64 -> 128 
            conv_4_block(2*base_dim, 4*base_dim), #128 -> 256
            conv_4_block(4*base_dim, 8*base_dim), #256 -> 512
            conv_4_block(8*base_dim, 8*base_dim) #512 -> 512
        )

        #분류기(FC layer) - 처음 Linear 224*224 max pooling 5번 -> input size 7*7
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*7*7, 2048),
            nn.ReLU(), #활성화함수
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes), #최종 256 -> 150(클래스 개수만큼)
        )

    #순전파 학습 설계(기존 설계에서 기본 dropout 삭제)
    def forward(self, x):
        x = self.feature(x) #특징 추출기 거치고
        x = x.view(x.size(0), -1) #batch 크기 제외한 크기 n*n*차원 다 펼치기
        x = self.fc_layer(x) #FC layer에 전달
        return x # 최종 class당 확률 Tensor