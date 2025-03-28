import torch.nn as nn
import torch.nn.functional as F


# 3×3 Conv 레이어 2개 + 2×2 MaxPooling 1개
def conv_2_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), ## 3x3 커널 / 입력 차원 input_dim에서 output_dim으로 변환됨
        nn.ReLU(), ## 활성화 함수 ReLU -> 비선형 활성화 함수로 비선형성을 추가해 더 복잡한 함수를 학습할 수 있게 함
        nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1), ## 3x3 커널 / 이전 레이어의 출력 채널 수(output_dim)를 그대로 받아 다시 Conv 진행
        nn.ReLU(), 

        nn.MaxPool2d(2,2) # 2x2 MaxPooling -> 피처 맵의 크기를 줄이고 과적합 방지
    )
    return model

# 3×3 Conv 레이어 3개 + 2×2 MaxPooling 1개
def conv_3_block(input_dim, output_dim):
    model = nn.Sequential(
        nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1), ## kernel_size=3, padding=1 설정을 통해 3x3의 커널 크기를 사용하면서 입력과 동일한 공간 크기를 유지
        nn.ReLU(), # 활성화 함수, 매 합성곱 층 다음에 존재함. 비선형성 증가시키는 역할
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
            conv_2_block(3, base_dim), ## 첫 번째 블록(conv1_1, conv1_2에 해당): 3개 채널 -> base_dim개 채널 (이후, 논문에 따라, inference에서 base_dim=64로 입력)
            conv_2_block(base_dim, base_dim*2), ## 두 번째 블록(conv2_1, conv2_2) : 64 -> 128  

            ## conv3는 conv2보다 더 높은 수준의 피처를 추출할 수 있게 함-> 블록이 깊어졌으므로
            conv_3_block(base_dim*2, base_dim*4), ## 세 번째 블록 : 128 -> 256
            conv_3_block(base_dim*4, base_dim*8), ## 네 번째 블록 : 256 -> 512
            conv_3_block(base_dim*8, base_dim*8), ## 다섯 번째 블록 : 512 -> 512 (동일 차원)
        ) #[Bㅐ치사이즈, 512, 1, 1] ← 마지막 블록 출력


        ## 완전 연결 레이어로 구성됨 
        # nn.Linear(백터 길이, 뉴런 개수)
        self.fc_layer = nn.Sequential(
            # nn.Linear(8*base_dim*1*1, 4096), ## 첫 번째 FC layer
            nn.Linear(8*base_dim*1*1, 512),
            nn.ReLU(True),
            nn.Dropout(), ## 드롭아웃 -> 과적합 방지

            nn.Linear(512, num_classes), ## 출력층 (클래스 수에 해당)
        )

    def forward(self, x):
        x = self.feature(x) ## 특성 추출
        ## self.feature: 정의된 VGG16 모델의 합성곱 레이어들
        x = x.view(x.size(0), -1) ## 배치 차원을 유지하면서 나머지 차원을 평탄화
        ## -1: 나머지 차원을 하나의 긴 벡터로 평탄화하라는 의미
        x = self.fc_layer(x) ## FC layer 통과
        return x ## 최종 출력 반환

# 📌 변경 부분
# 1. 학습 속도 향상	: FC 레이어가 파라미터 대부분 차지 → 경량화 효과 큼
# 2. 실험 타당성 향상 : 데이터셋 특성에 맞게 아키텍처 조정 가능함을 검증