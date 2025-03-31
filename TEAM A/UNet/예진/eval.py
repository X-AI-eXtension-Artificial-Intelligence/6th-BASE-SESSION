## 라이브러리 추가하기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from torchvision import transforms, datasets

## 트레이닝 파라메터 설정하기
lr = 1e-3               # 학습률 (불러온 모델에서 옵티마이저 복원에 사용)
batch_size = 4          # 한 번에 처리할 이미지 수
num_epoch = 100         # 총 에폭 수 (이 코드는 학습이 아니라 사용 안 함)

data_dir = './datasets'       # 데이터셋 폴더
ckpt_dir = './checkpoint'     # 저장된 모델(.pth) 있는 폴더
log_dir = './log'             # 로그 폴더 (이 코드는 사용 X)
result_dir = './results'      # 테스트 결과 저장 폴더


if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))    # 예측 이미지 저장 폴더
    os.makedirs(os.path.join(result_dir, 'numpy'))  # 예측 npy 저장 폴더


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU 선택


## 네트워크 구축하기
class UNet(nn.Module):      # PyTorch의 기본 신경망 클래스 상속
    def __init__(self):
        super(UNet, self).__init__()

        # CBR2d: Conv(합성곱) + BN(정규화) + ReLU(활성화)를 묶어서 하나의 블록으로 만들어주는 함수
        # => 이미지를 처리해서 특징을 뽑아주는 조합
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,    # 이미지 처리 핵심 (필터)
                                 kernel_size=kernel_size, stride=stride, padding=padding,  
                                 bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]                       # 값 정규화해서 안정적으로 만듦
            layers += [nn.ReLU()]                                                       # 음수 제거, 계산 더 잘되게 만듦

            cbr = nn.Sequential(*layers)  # 3개를 순서대로 묶음

            return cbr

        # 본격적으로 모델 제작

        # 인코더 (이미지 줄이기)
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 1: 입력 이미지가 흑백(채널 1개)이라는 뜻
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64) # 64개의 특징을 뽑아냄

        self.pool1 = nn.MaxPool2d(kernel_size=2)             # 맥스풀링: 이미지를 절반 크기로 줄임

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)  # 2, 3, 4... 반복
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128) # 채널 수는 점점 증가 (128, 256...), 이미지 크기 점점 감소 

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)  # bottleneck: 가장 압축된 지점, 이미지 엄청 줄여서 특징만 남긴 마지막 구간

        # Expansive path
        # 디코더 (이미지 키우기)
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)  

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,              # 언풀링: 이미지를 2배로 늘려줌
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)        # 인코더에서 저장해뒀던 512짜리 특징을 붙여서
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)            # 총 1024 채널 → 다시 512로 줄이면서 복원
        
        # 과정 계속 반복 (채널 수 감소, 이미지 크기 증가)
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,  
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                          kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True) # 최종 채널 1개로 복원해서 출력

    # forward 함수: 입력 x를 받고 내부 거쳐서 예측 결과 반환
    def forward(self, x):               # x: 입력 이미지 (예: 1×256×256 텐서)
        enc1_1 = self.enc1_1(x)         # 여기서 이미지 특징 뽑고 줄이기 반복 (이미지 크기 줄어들고 특징 깊어짐)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)      # 가장 압축된 곳

        dec5_1 = self.dec5_1(enc5_1)     # 이미지 크기 키우기 시작

        unpool4 = self.unpool4(dec5_1)               # 언풀링 -> 이미지 크기 키우기
        cat4 = torch.cat((unpool4, enc4_2), dim=1)   # cat: 인코더에서 저장했던 결과를 같이 써서 더 정확하게 복원함
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1)
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1)
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1)
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        x = self.fc(dec1_1)   # 최종 예측 결과 반환 (채널 1개짜리 이미지)

        return x

## Dataset 구현하기
# Dataset 클래스: 폴더 안에 있는 input_001.npy, label_001.npy 같은 파일들을 짝지어서 불러오는 역할

class Dataset(torch.utils.data.Dataset):           # PyTorch의 Dataset 클래스를 상속해서 Dataset 클래스 정의
    def __init__(self, data_dir, transform=None):  # 생성자: 데이터 폴더 경로와 transform 파이프라인을 받음
        self.data_dir = data_dir                   # 데이터가 저장될 디렉토리 생성
        self.transform = transform                 # transform: 이미지 데이터 전처리
                                                   # 1. 넘파이를 텐서형태로 2. 픽셀값 정규화 3. 이미지 변환 통해 학습 다양성 증가

        lst_data = os.listdir(self.data_dir)       # 디렉토리 안에 있는 모든 파일 이름을 리스트로 가져옴

        lst_label = [f for f in lst_data if f.startswith('label')]  # 'label'로 시작하는 파일만 추출
        lst_input = [f for f in lst_data if f.startswith('input')]  # 'input'으로 시작하는 파일만 추출

        lst_label.sort()                          # 파일 이름 기준 정렬 (순서 맞추기 위해)
        lst_input.sort()

        self.lst_label = lst_label                # 이 리스트들을 클래스 내부 변수로 저장
        self.lst_input = lst_input


    def __len__(self):                      # __len__: 전체 샘플 개수를 알려주는 함수
        return len(self.lst_label)          # 보통 label 개수 = input 개수니까 label 기준으로 길이 반환


    def __getitem__(self, index):           # __getitem__: index에 해당하는 데이터 한 쌍 불러오는 함수
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))  # input_003.npy, label_003.npy 같은 걸 불러옴
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        label = label/255.0    # 255인 픽셀값을 0~1 범위로 정규화 (딥러닝 학습 잘 되게 하려고)
        input = input/255.0

        # 현재 이미지가 흑백이라 차원이 (H, W)인데, 모델에 넣을 땐 (H, W, C) 형식이 돼아하므로 채널 차원 추가해야 함
        # np.newaxis를 이용해서 (H, W, 1)로 만들어줌
        if label.ndim == 2:                          
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label} # 후에 transform에 넘기기 좋게 딕셔너리로 input/label을 묶어줌 

        if self.transform:
            data = self.transform(data) # 만약 transform이 정의되어 있다면 전처리를 여기서 적용

        return data  # 최종적으로 input/label이 담긴 딕셔너리를 반환



## Transform 구현하기: 데이터 전처리(정규화, 텐서 변환 등)하거나 증강(랜덤 뒤집기)
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input'] # 딕셔너리에서 input/label 꺼냄
        # 이미지 배열은 보통 (H, W, C)인데, PyTorch는 (C, H, W) 형태로 받아야 함 -> 순서 변경해줌
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)} # 넘파이 배열 -> PyTorch 텐서로 변환

        return data

# 정규화
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5): # 평균 0.5, 표준편차 0.5로 설정
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input'] 

        input = (input - self.mean) / self.std # input만 정규화, label은 유지 (label은 학습용 정답이니까 건드리면 안 됨)

        data = {'label': label, 'input': input}

        return data

# 데이터 증강
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 50% 확률로 좌우, 상하 반전 → 데이터 다양성 증가
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


# 테스트 데이터 셋 불러오기

## 네트워크 학습하기
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]) # 픽셀값 정규화, 넘파이 -> 파이토치 센서

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)


## 네트워크 생성하기 -> GPU로 이동
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)                     # 테스트 데이터 개수

num_batch_test = np.ceil(num_data_test / batch_size)  # 총 배치 수


## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)    # 텐서 → 넘파이 (이미지 형태로)
fn_denorm = lambda x, mean, std: (x * std) + mean                            # 정규화 해제
fn_class = lambda x: 1.0 * (x > 0.5)                                         # 0.5 이상이면 1, 이하면 0


## 네트워크 저장하기 (모델의 가중치와 옵티마이저 상태 저장)
def save(ckpt_dir, net, optim, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)   # 저장된 모델 리스트 가져오기
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # 가장 마지막 에폭 모델 선택

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))  # 최신 모델 불러오기

    net.load_state_dict(dict_model['net'])     # 모델 파라미터 로드
    optim.load_state_dict(dict_model['optim']) # 옵티마이저 상태 로드
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])  # 에폭 번호 추출

    return net, optim, epoch

## 네트워크 학습시키기 (테스트 시작)
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

# 평가 모드 설정 + 예측 반복
with torch.no_grad():  # gradient 계산 안 함 (속도 증가)
    net.eval()         # 모델을 평가 모드로 설정 (BN, Dropout 등 동작 변경)
    loss_arr = []      # 손실 누적 리스트


    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # 손실함수 계산하기
        loss = fn_loss(output, label)

        loss_arr += [loss.item()]

        print("TEST: BATCH %04d / %04d | LOSS %.4f" %
              (batch, num_batch_test, np.mean(loss_arr)))

        # Tensorboard 저장하기
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
        output = fn_tonumpy(fn_class(output))


        for j in range(label.shape[0]):  # 배치 내 이미지 수만큼 반복
            id = num_batch_test * (batch - 1) + j  # 고유 ID 만들기

            # 이미지(.png)로 저장
            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            # 넘파이(.npy)로 저장
            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())


print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))