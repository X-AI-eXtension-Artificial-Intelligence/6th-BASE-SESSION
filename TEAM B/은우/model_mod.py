import torch
import torch.nn as nn

# 2개의 합성곱 층과 1개의 MaxPooling 층을 포함하는 블록 정의
def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),  # 3x3 합성곱, padding=1
        nn.ReLU(),  # 활성화 함수로 ReLU 사용
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),  # 또 다른 3x3 합성곱
        nn.ReLU(),  # 활성화 함수
        nn.MaxPool2d(2, 2)  # 2x2 최대 풀링, 특성 맵 크기 절반으로 축소
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
        super(VGG, self).__init__()
        
        # 특성 추출 부분 (convolution + pooling)
        self.feature = nn.Sequential(
            conv_2_block(3, base_dim),  # 첫 번째 블록: 3 채널 (RGB)에서 base_dim으로 변환
            conv_2_block(base_dim, base_dim),  # 두 번째 블록: base_dim에서 동일한 base_dim으로
            conv_3_block(base_dim, 2*base_dim),  # 세 번째 블록: base_dim에서 2배로 증가
            conv_3_block(2*base_dim, 4*base_dim),  # 네 번째 블록: 2배에서 4배로 증가
        )  #마지막 블록 제거 4번째 블록 까지만 진행 
        ## 블록을 더 깊게 설계해서 구현해봤는데 런타임에러로실패 리사이즈가 안돼서 그런듯 
        #이미지가 너무 작아서 오류가 남 
        #지피티가 줄여보라고해서 블록을 줄여봤는데 성능이 올랐음 왜 줄였더니 성능이 올랐는지는 의문 
        #애초에 작은 값들롲 진행해서 과적합일것 같지도 않은데 왜 줄였는데 성능이 올랐지?
        
        #오류나서 추가 
        # feature extractor의 출력 크기를 확인하기 위한 dummy 데이터
        self.dummy_input = torch.zeros(1, 3, 32, 32)  # 예시로 32x32 크기의 이미지를 사용
        
        # feature extractor를 통과시켜 실제 출력 크기 계산
        self._to_linear = self._get_conv_output(self.dummy_input)  # feature extractor 통과 후 크기 계산

        # 완전 연결층 (Fully Connected Layer)
        self.fc_layer = nn.Sequential(
            nn.Linear(self._to_linear, 2048),  # 출력 크기를 계산한 후 첫 번째 Linear 층 
            #4*base_dim*1*1로 했는데 오류나서 
            #직접 계산안하고 출력 크기에 맞는 값으로 계산
            nn.ReLU(True),  # 활성화 함수
            nn.Dropout(),  # 드롭아웃을 적용하여 과적합 방지
            nn.Linear(2048, 512),  # 두 번째 Linear 층
            nn.ReLU(True),  # 활성화 함수
            nn.Dropout(),  # 드롭아웃
            nn.Linear(512, num_classes),  # 최종 출력층 (num_classes 클래스 분류)
        )


    def _get_conv_output(self, shape):
        """입력 텐서를 feature extractor를 통해 통과시켜, 그 출력을 펼쳐서 fully connected layer에 맞는 차원으로 반환"""
        x = self.feature(shape)  # feature extractor 통과
        return int(torch.prod(torch.tensor(x.size())))  # 텐서 크기를 펼쳐서 하나의 숫자로 반환 (차원 크기 계산)
        

    def forward(self, x):
        x = self.feature(x)  # 입력 x를 feature extractor에 통과시킴
        x = x.view(x.size(0), -1)  # 배치 차원을 제외한 나머지 차원들을 펼침 (Flatten)
        x = self.fc_layer(x)  # 완전 연결층을 통과시켜 최종 예측값을 계산
        return x  # 최종 출력값 반환
