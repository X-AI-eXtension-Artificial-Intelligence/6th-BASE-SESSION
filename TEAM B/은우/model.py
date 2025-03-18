import torch
import torch.nn as nn  # PyTorch에서 신경망 구성 요소를 제공하는 모듈을 import

# 2개의 합성곱 층과 1개의 MaxPooling 층을 포함하는 블록 정의
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 합성곱, 입력 채널과 출력 채널 지정
        nn.ReLU(),  # 활성화 함수로 ReLU 사용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 또 다른 3x3 합성곱
        nn.ReLU(),  # 활성화 함수
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링, 특성 맵 크기를 절반으로 축소
    )
    return model

# 3개의 합성곱 층과 1개의 MaxPooling 층을 포함하는 블록 정의
def conv_3_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 첫 번째 3x3 합성곱
        nn.ReLU(),  # 활성화 함수
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 두 번째 3x3 합성곱
        nn.ReLU(),  # 활성화 함수
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 세 번째 3x3 합성곱
        nn.ReLU(),  # 활성화 함수
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링, 특성 맵 크기 축소
    )
    return model

# VGG 모델 정의
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()  # 부모 클래스 nn.Module의 초기화 함수 호출
        self.feature = nn.Sequential(  # VGG 모델의 특징 추출 부분
            conv_2_block(3, base_dim),  # 첫 번째 블록: 입력 채널 3(이미지 RGB), 출력 채널 base_dim
            conv_2_block(base_dim, 2*base_dim),  # 두 번째 블록: 출력 채널 2배 증가
            conv_3_block(2*base_dim, 4*base_dim),  # 세 번째 블록: 출력 채널 4배 증가
            conv_3_block(4*base_dim, 8*base_dim),  # 네 번째 블록: 출력 채널 8배 증가
            conv_3_block(8*base_dim, 8*base_dim),  # 다섯 번째 블록: 출력 채널 유지
        )
        
        # Fully Connected Layer 부분 (완전 연결층)
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096),  # 8*base_dim*1*1은 마지막 특성 맵의 크기 
            nn.ReLU(True),  # 활성화 함수 ReLU
            nn.Dropout(),  # 과적합 방지를 위한 드롭아웃
            nn.Linear(4096, 1000),  # 4096차원에서 1000차원으로
            nn.ReLU(True),  # 활성화 함수
            nn.Dropout(),  # 드롭아웃
            nn.Linear(1000, num_classes),  # 최종 출력: num_classes로 분류 (예: CIFAR10은 10개 클래스)
        )

    def forward(self, x):
        x = self.feature(x)  # 입력 x를 feature extractor에 통과시킴
        # print(x.shape)  
        x = x.view(x.size(0), -1)  # 배치 차원을 제외하고 나머지 차원을 펼침 (Flatten)
        # print(x.shape) 
        x = self.fc_layer(x)  # 완전 연결층을 통과시켜 최종 예측값을 계산
        return x  # 최종 출력값 반환
