import torch.nn as nn
import torch.nn.functional as F


# 합성곱 2 + Max Pooling 1
## 2개의 합성곱(Convolution) 레이어와 1개의 Max Pooling 레이어
def conv_2_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), ## 3x3 커널 -> 입력 차원을 input_dim에서 output_dim으로 변환하는 3x3 크기의 커널
        nn.ReLU(), ## 활성화 함수 ReLU -> 비선형 활성화 함수로 비선형성을 추가해 더 복잡한 함수를 학습할 수 있게 함
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1), ## 3x3 커널 -> 여기서 첫번째 output_dim은 이전 컨볼루션 레이어에서 생성된 피처 맵의 수
        nn.ReLU(), 
        nn.MaxPool2d(2,2) # 2x2 MaxPooling -> 피처 맵의 크기를 줄이고 과적합 방지
    )
    return model

# 합성곱 3 + Max Pooling 1
## 3개의 합성곱(Convolution) 레이어와 1개의 Max Pooling 레이어
def conv_3_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), ## kernel_size=3, padding=1 설정을 통해 3x3의 커널 크기를 사용하면서 입력과 동일한 공간 크기를 유지
        nn.ReLU(), ## 활성화 함수로 비선형성 증가시킴
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2) ## 2x2 MaxPooling
    )
    return model

## VGG16 모델 클래스
class VGG16(nn.Module): ## nn.Module을 상속함
    def __init__(self, base_dim, num_classes=10): 
        super(VGG16, self).__init__()
        
        ## 피처 추출 -> 입력 이미지에서 피처를 추출하기 위해 여러 컨볼루션 블록으로 구성됨
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim), ## 첫 번째 블록 (입력 채널 3) -> base_dim = 64채널
            conv_2_block(base_dim, base_dim*2), ## 두 번째 블록 (차원 증가)    
            ## conv3는 conv2보다 더 높은 수준의 피처를 추출할 수 있게 함
            conv_3_block(base_dim*2, base_dim*4), ## 세 번째 블록 (차원 증가)
            conv_3_block(base_dim*4, base_dim*8), ## 네 번째 블록 (차원 증가)
            conv_3_block(base_dim*8, base_dim*8), ## 다섯 번째 블록 (동일 차원)
        )
        ## 완전 연결 레이어로 구성됨 
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096), ## 첫 번째 FC layer
            nn.ReLU(True),
            nn.Dropout(), ## 드롭아웃 -> 과적합 방지
            nn.Linear(4096, 1000), ## 두 번째 FC layer
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes), ## 출력층 (클래스 수에 해당)
        )

    def forward(self, x):
        x = self.feature(x) ## 특성 추출
        ## self.feature: 정의된 VGG16 모델의 합성곱 레이어들
        x = x.view(x.size(0), -1) ## 배치 차원을 유지하면서 나머지 차원을 평탄화
        ## -1: 나머지 차원을 하나의 긴 벡터로 평탄화하라는 의미
        x = self.fc_layer(x) ## FC layer 통과
        return x ## 최종 출력 반환
    
# references
# https://velog.io/@euisuk-chung/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98%EB%A1%9C-CNN-%EB%AA%A8%EB%8D%B8%EC%9D%84-%EA%B5%AC%ED%98%84%ED%95%B4%EB%B3%B4%EC%9E%90-VGGNet%ED%8E%B8