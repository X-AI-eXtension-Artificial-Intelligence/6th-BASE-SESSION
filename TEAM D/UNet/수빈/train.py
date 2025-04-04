# 라이브러리 추가
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets
from model import UNet
from dataset import *
from util import *

import argparse

import matplotlib.pyplot as plt

# Parser 생성(코드 사용을 좀 더 유연하게 할 수 있음)
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# 학습률/배치 사이즈/epoch 설정
parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")

# 데이터/체크 포인트/로그 디렉토리 설정
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")

# 결과 디렉토리 설정
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

# 실행 모드 설정
parser.add_argument("--mode", default="train", type=str, dest="mode")

# 이어서 학습할지 여부 설정
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")

# args에 파싱해서 저장
args = parser.parse_args()

# 트레이닝 파라메터 설정
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

# 디렉토리 설정
data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

# 동작 방식(GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 디렉토리 없으면 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

'''
# U-NET 네트워크 구축  
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Convolution/Batch Normalization/ReLU 함수를 조합해서 하나의 레이어로 만드는 과정
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            # 레이어 저장할 리스트 생성
            layers = []
            # 2D Convolution layer
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 bias=bias)]
            # Batch Normalization layer
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU layer
            layers += [nn.ReLU()]
            # layer들을 sequential하게 묶기
            cbr = nn.Sequential(*layers)

            return cbr

        # Contracting path
        # kernel_size, stride, padding, bias는 미리 설정해뒀으니 생략
        self.enc1_1 = CBR2d(in_channels=1, out_channels=64)  # 입/출력 채널 개수 설정
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64) # 입/출력 채널 개수 설정

        self.pool1 = nn.MaxPool2d(kernel_size=2)             # Max pooling 진행(2X2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

        # Expansive path
        self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)
        
        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                          kernel_size=2, stride=2, padding=0, bias=True) # up-convolution 진행(2X2)

        self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512) # Encoder 파트에서 전달된 512 채널이 입력으로 추가되기 때문
        self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

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

        self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    # 각 레이어들 연결
    def forward(self, x):
        enc1_1 = self.enc1_1(x)      # layer 전달 및 연결
        enc1_2 = self.enc1_2(enc1_1) # layer 전달 및 연결
        pool1 = self.pool1(enc1_2)   # Pooling을 통해 size 줄이기

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1) # U-NET 구조 기준 가장 아래 있는 부분

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) # Encoder 파트에서 전달된 채널과 합치는 부분
                                                   # 'dim=1'은 concat 방향을 의미(채널 방향으로 concat한다는 것이죠)
                                                   # 0: batch, 1:channel, 2: height(y방향), 3:width(x방향)라고 합니다

        dec4_2 = self.dec4_2(cat4)   # layer 전달 및 연결
        dec4_1 = self.dec4_1(dec4_2) # layer 전달 및 연결

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

        x = self.fc(dec1_1) # U-NET 구조 중 마지막 단계. 최종 출력 생성 단계. 


        return x  # logit 형태로 출력(logit: sigmoid/softmax 함수 적용 전. 후처리 과정에서 추가 연산 필요)
'''

'''
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
'''

# 네트워크 학습
# Transform -> Dataset -> DataLoader
if mode == 'train':
    # transforms.Compose를 통해 Transform class를 연결
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    # train에 필요한 데이터셋
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # validation에 필요한 데이터셋
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)

    # 이외 여러 variable 설정
    # train/validation 데이터의 전체 데이터 수 계산
    # 이를 기반으로 전체 epoch에서 몇 번의 batch가 필요한지 계산
    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)
    # train/validation 데이터 처리를 위해 필요한 batch의 총 개수 계산
    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    # 그밖에 부수적인 variables 설정하기
    num_data_test = len(dataset_test)

    num_batch_test = np.ceil(num_data_test / batch_size)

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

# 이외 여러 function 설정
# tensor를 numpy 배열로 변환(tensor 형식 데이터 시각화/결과 저장을 위해 쓰임)
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
# normalization된 데이터를 원래 scale로 되돌림
# why? 모델 출력/normalization된 데이터 시각화하려면 원래 값으로 변환해야 좋음
fn_denorm = lambda x, mean, std: (x * std) + mean
# 모델 출력을 이진 class(0 or 1)로 변환하여, 출력값이 0~1 사이 -> 0.5 기준으로 이진 분류
fn_class = lambda x: 1.0 * (x > 0.5)

# Tensorboard 사용하기 위한 SummaryWriter 설정
# Tensorboard = 딥러닝 학습 과정 시각화 도구
# SummaryWriter = Tensorboard에 로그 데이터 기록하는데 사용
# train과 validation 과정에서 발생하는 각 데이터들을 기록
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

'''
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

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1])) # 모델/옵티마이저 상태
    # 모델/옵티마이저 상태 복원
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    # 파일 이름에서 에폭 번호 추출
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch
'''

# 네트워크 학습
st_epoch = 0    # 시작 epoch = 0으로 세팅
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim) # 네트워크 로드

# Train Loop
if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []

        for batch, data in enumerate(loader_train, 1): # loader_train에서 배치 size만큼 데이터 가져옴
            # forward pass
            label = data['label'].to(device) # 정답 데이터
            input = data['input'].to(device) # 입력 데이터

            output = net(input)

            # backward pass
            optim.zero_grad()   # 이전 배치의 기울기 초기화 -> 기울기 누적되지 않도록 조정

            loss = fn_loss(output, label) # fn_loss: 손실 함수 계산
                                          # network output & label 간 차이 계산
            loss.backward()               # 손실 함수 결과 기반으로 가중치에 대한 Gradient(기울기) 계산
            optim.step()                  # 옵티마이저가 계산된 기울기 기반으로 네트워크 가중치 조정

            # 손실함수 계산
            loss_arr += [loss.item()]
            # 값 출력
            print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                  (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

            # Tensorboard에 학습 데이터 기록
            label = fn_tonumpy(label)
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
            output = fn_tonumpy(fn_class(output))

            writer_train.add_image('label', label, num_batch_train * (epoch - 1) + batch, dataformats='NHWC') # dataformats='NHWC' = 데이터 형식 명시(Numpy의 Batch, Height, Width, Channels)
            writer_train.add_image('input', input, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')
            writer_train.add_image('output', output, num_batch_train * (epoch - 1) + batch, dataformats='NHWC')

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        # Validation Loop
        with torch.no_grad(): # validation 단계에서는 기울기 계산 필요 X -> 불필요한 메모리 사용 방지
            net.eval()
            loss_arr = []

            # validation 데이터에서 loss 값 계산하고, 평균 loss 저장
            for batch, data in enumerate(loader_val, 1):
                 # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                output = net(input)

                # 손실함수 계산
                loss = fn_loss(output, label)

                loss_arr += [loss.item()]
                # 값 출력
                print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                    (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))

                # Tensorboard에 validation 데이터 기록
                label = fn_tonumpy(label)
                input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
                output = fn_tonumpy(fn_class(output))

                writer_val.add_image('label', label, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('input', input, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')
                writer_val.add_image('output', output, num_batch_val * (epoch - 1) + batch, dataformats='NHWC')

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        # epoch 50번 마다 save
        if epoch % 50 == 0:
            save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)

    writer_train.close()
    writer_val.close()

# Test
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)
 
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