# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from model import UNet

# 트레이닝 파라메터 설정
lr = 1e-3
batch_size = 4
num_epoch = 100

# 디렉토리 설정
data_dir = './datasets'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# 동작 방식(GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 데이터 로더 구현
# 데이터 분배 더 효율적
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir) # data_dir 내부에 있는 파일들 가지고 오기

        # 파일 내 label 및 input 파일 각각 필터링
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 리스트 정렬 
        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 데이터 픽셀 정규화(Normalization)
        # 0~255 사이 정수로 표현되어 있는 이미지 데이터 픽셀 값을 0~1 사이 실수 값으로 변환
        # 스케일링을 통해 더 빠르고 효율적인 연산 가능
        label = label/255.0
        input = input/255.0

        # 데이터 차원 확장(pytorch 특성상)
        # np.newaxis를 통해 새로운 차원 추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data

# 기존 넘파이 배열에서 PyTorch 텐서로 변환
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']
        # numpy 차원 = (Y, X, CH)
        # tensor 차원 = (CH, Y, X)
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

# Normalization을 통해 안정적인 학습 구현
# 평균 & 표준편차 기준으로 스케일링
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

# Data Augmentation을 통해 학습 데이터 다양화
# 여기서는 input 데이터와 label 데이터를 flip함
class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:   # 랜덤 값이 0.5 초과면 적용
            label = np.fliplr(label) # 좌우 반전(Flip Left-Right)
            input = np.fliplr(input) # 좌우 반전(Flip Left-Right)

        if np.random.rand() > 0.5:   # 랜덤 값이 0.5 초과면 적용
            label = np.flipud(label) # 상하 반전(Flip Up-Down)
            input = np.flipud(input) # 상하 반전(Flip Up-Down)

        data = {'label': label, 'input': input}

        return data

# 네트워크 학습
# Transform -> Dataset -> DataLoader
# transforms.Compose를 통해 Transform class를 연결
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()]) # RandomFlop() 제거

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0) # shuffle = False로 설정

# 네트워크 생성
# U-NET 정의 및 학습 가능한 상태로 초기화
# GPU 사용이 가능하다면, 모델을 GPU로 옮기기
net = UNet().to(device)

# 손실함수 정의: 모델이 얼마나 잘못 예측했는지 평가
# Binary Cross-Entropy 계산
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer 설정: 손실함수가 제공하는 정보 기반으로 손실 값 줄이기 위해 모델 가중치 조정
# Adam 사용
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 이외 여러 variable 설정
# test 데이터의 전체 데이터 수 계산
# 이를 기반으로 전체 epoch에서 몇 번의 batch가 필요한지 계산
num_data_test = len(dataset_test)
# test 데이터 처리를 위해 필요한 batch의 총 개수 계산
num_batch_test = np.ceil(num_data_test / batch_size)


# 이외 여러 function 설정
# tensor를 numpy 배열로 변환(tensor 형식 데이터 시각화/결과 저장을 위해 쓰임)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# normalization된 데이터를 원래 scale로 되돌림
# why? 모델 출력/normalization된 데이터 시각화하려면 원래 값으로 변환해야 좋음
fn_denorm = lambda x, mean, std: (x * std) + mean
# 모델 출력을 이진 class(0 or 1)로 변환하여, 출력값이 0~1 사이 -> 0.5 기준으로 이진 분류
fn_class = lambda x: 1.0 * (x > 0.5)

# 네트워크 저장
def save(ckpt_dir, net, optim, epoch):
    # 디렉토리 확인 & 생성
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    # 모델, optimizer 상태 저장
    torch.save({'net': net.state_dict(),        # 모델 파라미터(가중치, 편향) 저장
                'optim': optim.state_dict()},   # optimizer 상태(학습률, 모멘텀) 저장
               "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))            

# 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0                   # epoch = 0 으로 초기화
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]), weights_only=True) # 모델/옵티마이저 상태
    # 모델/옵티마이저 상태 복원
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # 파일 이름에서 에폭 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

# 네트워크 학습
st_epoch = 0    # 시작 epoch = 0으로 세팅
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 네트워크 로드


with torch.no_grad(): # 기울기 계산 필요 X -> 불필요한 메모리 사용 방지
    net.eval()
    loss_arr = []

    # test 데이터에서 loss 값 계산하고, 평균 loss 저장
    for batch, data in enumerate(loader_test, 1):
        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)

        # 손실함수 계산
        loss = fn_loss(output, label)

        loss_arr += [loss.item()]
        # 값 출력
        print("TEST:  BATCH %04d / %04d | LOSS %.4f" %
            (batch, num_batch_test, np.mean(loss_arr)))

        # Tensorboard에 validation 데이터 기록
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)) 
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j
            # PNG 파일로 저장
            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')
            # Numpy 버전으로 저장
            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

# 평균 손실 값 출력
print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr))) 