# model.py
# 이 파일은 VGG 네트워크 모델의 구조를 정의한 파일이다.
# 여기서는 두 가지 종류의 convolution 블록(2회와 3회 convolution)과 VGG 모델 클래스를 정의한다.

import torch
import torch.nn as nn

# ------------------------------------------------------------------
# 1. conv_2_block 함수: 3x3 컨볼루션을 2번 적용하고, ReLU 활성화 후 2x2 MaxPooling 적용
# ------------------------------------------------------------------
def conv_2_block(in_dim, out_dim):
    """
    2개의 3x3 convolution 레이어와 1개의 2x2 max pooling 레이어를 구성하여,
    입력 특징 맵에서 복잡한 패턴을 추출하는 작은 블록을 생성한다.
    
    인자:
      - in_dim: 입력 채널 수 (예: 이미지의 경우 RGB이면 3)
      - out_dim: 출력 채널 수 (컨볼루션 후 생성되는 특징의 수)
      
    반환값:
      - nn.Sequential 객체: 순차적으로 레이어를 적용하는 컨테이너
    """
    model = nn.Sequential(
        # 첫번째 3x3 convolution 레이어, 패딩(padding=1)을 사용하여 출력 이미지의 크기가 입력과 동일하도록 유지한다.
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        # ReLU 활성화 함수: 음수는 0으로 바꾸고 양수는 그대로 통과시켜 비선형성을 추가한다.
        nn.ReLU(),
        # 두번째 3x3 convolution 레이어, 출력 채널 수는 동일하게 out_dim을 유지한다.
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        # 2x2 MaxPooling 레이어: 인접한 2x2 영역 중 최댓값을 선택하여, 공간 해상도를 절반으로 줄인다.
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

# ------------------------------------------------------------------
# 2. conv_3_block 함수: 3x3 컨볼루션을 3번 적용하고, ReLU 활성화 후 2x2 MaxPooling 적용
# ------------------------------------------------------------------
def conv_3_block(in_dim, out_dim):
    """
    3개의 3x3 convolution 레이어와 1개의 2x2 max pooling 레이어를 구성하여,
    입력 특징 맵에서 더욱 세밀한 패턴을 추출하는 작은 블록을 생성한다.
    
    인자:
      - in_dim: 입력 채널 수
      - out_dim: 출력 채널 수
      
    반환값:
      - nn.Sequential 객체: 순차적으로 레이어를 적용하는 컨테이너
    """
    model = nn.Sequential(
        # 첫번째 3x3 convolution 레이어
        nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        # 두번째 3x3 convolution 레이어
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        # 세번째 3x3 convolution 레이어
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
        nn.ReLU(),
        # 2x2 MaxPooling 레이어를 적용하여 feature map의 크기를 줄인다.
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    return model

# ------------------------------------------------------------------
# 3. VGG 클래스: VGG 네트워크의 전체 구조를 정의한다.
# ------------------------------------------------------------------
class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        """
        VGG 네트워크를 초기화하는 생성자.
        
        인자:
          - base_dim: 모델의 초기 채널 수. 예를 들어 64로 설정하며,
                      첫 번째 convolution 레이어의 출력 채널 수로 사용된다.
          - num_classes: 분류할 클래스의 수. CIFAR10 데이터셋은 10개의 클래스를 가진다.
        """
        super(VGG, self).__init__()
        # feature 추출 부분: 여러 convolution 블록을 쌓아 이미지의 특징을 점진적으로 추출한다.
        self.feature = nn.Sequential(
            # 첫번째 블록: RGB 이미지(채널 3)를 base_dim(예: 64) 채널로 변환한다.
            conv_2_block(in_dim=3, out_dim=base_dim),
            # 두번째 블록: base_dim 채널을 2배인 2*base_dim으로 확장한다.
            conv_2_block(in_dim=base_dim, out_dim=2*base_dim),
            # 세번째 블록: 2*base_dim 채널을 4배인 4*base_dim으로 확장한다.
            conv_3_block(in_dim=2*base_dim, out_dim=4*base_dim),
            # 네번째 블록: 4*base_dim 채널을 8배인 8*base_dim으로 확장한다.
            conv_3_block(in_dim=4*base_dim, out_dim=8*base_dim),
            # 다섯번째 블록: 8*base_dim 채널을 그대로 사용하여 더욱 복잡한 특징을 추출한다.
            conv_3_block(in_dim=8*base_dim, out_dim=8*base_dim)
        )
        # fc_layer: convolution을 통해 추출된 특징을 평탄화(flatten)한 후 분류를 수행하는 완전 연결 레이어들이다.
        self.fc_layer = nn.Sequential(
            # 첫 번째 Linear 레이어는 flatten된 벡터를 4096 차원으로 변환한다.
            # 주의: 8 * base_dim * 1 * 1의 값은 마지막 convolution 블록의 출력 크기에 따라 달라질 수 있다.
            nn.Linear(8 * base_dim * 1 * 1, 4096),
            nn.ReLU(True), # inplace=True로 설정하여여 입력 텐서의 값을 직접 덮어쓰며 ReLU 연산을 수행한다. 즉, 새로운 텐서를 생성하지 않고 기존 입력값을 수정함으로써 메모리 사용량을 줄일 수 있다.
                            # nn.ReLU(True)의 경우 입력값을 덮어쓰게 되므로, 만약 후속 연산에서 원본 입력값이 필요할 경우 문제가 발생할 수 있다.
            nn.Dropout(),  # Dropout: 과적합을 방지하기 위해 일부 뉴런을 임의로 비활성화한다.
            # 두 번째 Linear 레이어는 4096 차원을 1000 차원으로 변환한다.
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            # 마지막 Linear 레이어는 1000 차원을 최종 분류할 클래스의 수(num_classes)로 변환한다.
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        """
        forward 메소드: 입력 데이터를 순전파하여 모델의 출력을 계산한다.
        
        인자:
          - x: 입력 이미지 텐서
          
        반환값:
          - 분류 결과를 나타내는 텐서
        """
        # feature 추출 단계: convolution 블록들을 통해 이미지의 특징을 추출한다.
        x = self.feature(x)
        # fc_layer에 넣기 위해 feature map을 1차원 벡터로 평탄화한다.
        x = x.view(x.size(0), -1)
        # 완전 연결 레이어를 통해 최종 분류 결과를 계산한다.
        x = self.fc_layer(x)
        return x