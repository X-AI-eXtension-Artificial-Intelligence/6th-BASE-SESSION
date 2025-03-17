import torch
import torch.nn as nn

# 2개의 convolution layer + 1개의 MaxPool2d layer
# 인자 : 입력 채널, 출력 채널
def conv_2_block(in_dim,out_dim): 
    # nn.Sequential : 여러 층을 순서대로 실행되는 하나의 블록으로 묶어줌
    model = nn.Sequential(
        # 1번째
        ## 입력 채널 수 -> 출력 채널 수로 변환하는 합성곱 연산
        ## 3*3 kernel의 합성곱 연산
        ## padding 1로 설정 -> 출력 크기 유지
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), 
        # 활성화 함수로 ReLU 사용
        nn.ReLU(),
        # 2번째
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        # 2*2 kernel의 최대 풀링 연산 -> 크기를 절반으로 줄임
        nn.MaxPool2d(2,2)
    )
    # model 반환
    return model

# 3개의 convolution layer + 1개의 MaxPool2d layer
def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        # 1번째
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        # 2번째
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        # 3번째
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        # maxpooling 연산
        nn.MaxPool2d(2,2)
    )
    # model 반환
    return model

    # conv_3_block은 conv_2_block보다 비교적 더 복잡한 패턴 학습이 가능함 (대신, 연산량이 많고 느림 / GPU 사용 권장)

## VGG 모델 정의
# Pytorch의 nn.Module을 상속받아 모델 정의 -> parameters(), state_dict(), eval(), train() 같은 기능을 사용하기 위함
class VGG(nn.Module):
    # 인자 : 첫 번째 합성곱 블록의 기본 채널 수, 분류할 클래스 수(10)
    def __init__(self, base_dim, num_classes=10):
        # 초기화
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            # 합성곱 블록 구성
            conv_2_block(3,base_dim), # RGB 3채널 -> base_dim 64 : 기본적인 경계선 학습
            conv_2_block(base_dim,2*base_dim), # 64 -> 128 : 조금 더 복잡한 패턴 학습
            conv_3_block(2*base_dim,4*base_dim), # 128 -> 256 : 작은 객체의 특징 학습
            conv_3_block(4*base_dim,8*base_dim), # 256 -> 512 : 큰 객체의 형태 학습
            conv_3_block(8*base_dim,8*base_dim), # 512 -> 512 (유지) : 최종적인 복잡한 특징 학습 / 512 : 마지막 feature map 크기
        )
        # 완전연결층 구성 - 최종 분류 담당
        self.fc_layer = nn.Sequential(
            # 1번째 완전연결층 - 512개의 입력 값이 FC Layer로 들어가서 4096개의 뉴런으로 반환됨 (1*1:feature map 크기)
            ## CIFAR10은 크기가 32x32이므로 
            nn.Linear(8*base_dim*1*1, 4096),
            ## IMAGENET이면 224x224이므로
            ## nn.Linear(8*base_dim*7*7, 4096), -> 이렇게 되는 이유 : 둘의 입력 이미지 크기가 다르기 때문에 (+CNN을 거칠수록 작아짐)
            # 활성화 함수 : ReLU (True -> inplace=True 의미)
            nn.ReLU(True),
            # 드롭아웃 (과적합 방지)
            nn.Dropout(),
            # 2번째 완전연결층
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            # 최종 출력층 (if ImageNet이면 num_classes=1000, 1000개의 클래스로 분류)
            nn.Linear(1000, num_classes),
        )

    # forward() : 입력데이터가 모델을 통과하는 경로 지정
    def forward(self, x):
        # CNN 통과
        x = self.feature(x)
        ##print(x.shape)
        # CNN의 Feature Map을 1D 벡터로 변환하는 과정
        x = x.view(x.size(0), -1)
        ##print(x.shape)
        # 완전연결층을 거쳐 최종 예측값 생성
        x = self.fc_layer(x)
        # 최종 출력 반환
        return x