# torch 라이브러리를 임포트하여 텐서 연산 및 신경망 구성에 사용
import torch  
# torch.nn 모듈을 임포트하여 신경망 구성 요소(레이어, 활성화 함수 등)를 사용
import torch.nn as nn  

# 2회의 3x3 convolution, Batch Normalization, ReLU, 그리고 2x2 MaxPooling을 적용하는 블록을 정의
def conv_2_block(in_dim, out_dim):
    # nn.Sequential을 사용하여 레이어들을 순차적으로 연결
    model = nn.Sequential(
        # 입력 채널 수 in_dim에서 출력 채널 수 out_dim으로 3x3 커널을 사용하는 convolution을 수행
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # padding=1로 입력과 출력의 공간 크기를 유지
        # 출력에 대해 Batch Normalization을 적용하여 학습을 안정화
        nn.BatchNorm2d(out_dim),
        # ReLU 활성화 함수를 적용하여 비선형성을 추가
        nn.ReLU(),
        # 두번째 3x3 convolution 레이어: 채널 수 out_dim을 유지하면서 특징을 추출
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        # Batch Normalization 적용
        nn.BatchNorm2d(out_dim),
        # ReLU 활성화 함수 적용
        nn.ReLU(),
        # 2x2 MaxPooling을 적용하여 특징 맵의 공간 크기를 절반으로 줄임임
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # 구성된 sequential 모델을 반환
    return model

# 3회의 3x3 convolution, Batch Normalization, ReLU, 그리고 2x2 MaxPooling을 적용하는 블록을 정의
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        # 첫번째 3x3 convolution 레이어: 입력 채널에서 출력 채널로 변환
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        # Batch Normalization 적용
        nn.BatchNorm2d(out_dim),
        # ReLU 활성화 함수 적용
        nn.ReLU(),
        # 두번째 3x3 convolution 레이어: 채널 수를 유지하며 추가적인 특징을 추출
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        # Batch Normalization 적용
        nn.BatchNorm2d(out_dim),
        # ReLU 활성화 함수 적용
        nn.ReLU(),
        # 세번째 3x3 convolution 레이어: 더 깊은 특징 추출을 위하여 사용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        # Batch Normalization 적용
        nn.BatchNorm2d(out_dim),
        # ReLU 활성화 함수 적용
        nn.ReLU(),
        # 2x2 MaxPooling을 적용하여 출력의 공간 크기를 축소
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # 구성된 sequential 모델을 반환
    return model

# VGG 네트워크 클래스 정의: 여러 convolution 블록을 쌓아 특징을 추출한 후 FC 레이어를 통해 분류를 수행
class VGG(nn.Module):
    # 생성자 함수로, base_dim과 num_classes를 인자로 받아 초기화
    def __init__(self, base_dim, num_classes=10):
        """
        VGG 네트워크 초기화 함수이다.
        
        인자:
          - base_dim: 모델의 기본 채널 수 (예: 64)
          - num_classes: 분류할 클래스의 수 (예: CIFAR10의 경우 10)
        """
        # nn.Module의 초기화를 수행
        super(VGG, self).__init__()
        # convolution 블록들을 sequential로 연결하여 특징 추출 부분을 구성
        self.feature = nn.Sequential(
            # 첫번째 블록: 입력 RGB 이미지(채널 3)를 base_dim(예: 64) 채널로 변환
            conv_2_block(in_dim=3, out_dim=base_dim),             # 출력 채널: 64
            # 두번째 블록: base_dim 채널을 2배인 2*base_dim(예: 128)으로 확장
            conv_2_block(in_dim=base_dim, out_dim=2*base_dim),      # 출력 채널: 128
            # 세번째 블록: 2*base_dim 채널을 4배인 4*base_dim(예: 256)으로 확장
            conv_3_block(in_dim=2*base_dim, out_dim=4*base_dim),    # 출력 채널: 256
            # 네번째 블록: 4*base_dim 채널을 8배인 8*base_dim(예: 512)으로 확장
            conv_3_block(in_dim=4*base_dim, out_dim=8*base_dim),    # 출력 채널: 512
            # 다섯번째 블록: 8*base_dim 채널을 그대로 사용하여 추가 특징을 추출
            conv_3_block(in_dim=8*base_dim, out_dim=8*base_dim)     # 출력 채널: 512
        )
        # Adaptive Average Pooling을 적용하여 마지막 feature map의 공간 크기를 (1, 1)로 고정
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Fully Connected (FC) Layer 구성: flatten된 벡터를 입력받아 최종 분류 결과를 출력
        self.fc_layer = nn.Sequential(
            # 첫번째 Linear 레이어: 8*base_dim 크기의 벡터를 4096 차원으로 변환
            nn.Linear(8 * base_dim, 4096),
            # ReLU 활성화 함수 적용 (inplace=True로 메모리 효율 개선)
            nn.ReLU(True),
            # Dropout 적용하여 과적합을 방지
            nn.Dropout(),
            # 두번째 Linear 레이어: 4096 차원을 1000 차원으로 변환
            nn.Linear(4096, 1000),
            # ReLU 활성화 함수 적용
            nn.ReLU(True),
            # Dropout 적용
            nn.Dropout(),
            # 마지막 Linear 레이어: 1000 차원을 분류할 클래스 수(num_classes)로 변환
            nn.Linear(1000, num_classes)
        )

    # 순전파(forward) 메소드를 정의하여 입력 데이터를 처리
    def forward(self, x):
        # self.feature를 통해 convolution 블록을 순차적으로 적용하여 특징을 추출
        x = self.feature(x)
        # Adaptive Average Pooling을 적용하여 출력 feature map의 크기를 (1, 1)로 맞춤
        x = self.avgpool(x)
        # flatten: feature map을 1차원 벡터로 변환
        x = x.view(x.size(0), -1)
        # FC 레이어를 통과하여 최종 분류 결과를 산출
        x = self.fc_layer(x)
        # 최종 결과를 반환
        return x
