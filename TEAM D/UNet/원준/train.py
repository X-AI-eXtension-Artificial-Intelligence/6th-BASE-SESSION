## 라이브러리 추가하기
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import UNet  # U-Net 모델 정의
from dataset import *  # 데이터셋 관련 유틸리티 불러오기
from util import *  # 기타 유틸리티 함수들
import matplotlib.pyplot as plt
from torchvision import transforms, datasets

## Parser 생성하기
# 실행 시 입력받을 하이퍼파라미터 및 디렉토리 설정
parser = argparse.ArgumentParser(description="Train the UNet",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")  # 학습률 설정
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")  # 배치 크기 설정
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")  # 학습 epoch 수 설정

# 데이터, 체크포인트, 로그 및 결과 저장 경로 설정
parser.add_argument("--data_dir", default="./datasets", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--log_dir", default="./log", type=str, dest="log_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--mode", default="train", type=str, dest="mode")  # 실행 모드 설정 (train/test)
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")  # 학습 재시작 여부

args = parser.parse_args()

## 트레이닝 파라미터 설정하기
lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

mode = args.mode
train_continue = args.train_continue

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU 사용 여부 확인

print("learning rate: %.4e" % lr)
print("batch size: %d" % batch_size)
print("number of epoch: %d" % num_epoch)
print("data dir: %s" % data_dir)
print("ckpt dir: %s" % ckpt_dir)
print("log dir: %s" % log_dir)
print("result dir: %s" % result_dir)
print("mode: %s" % mode)

## 결과 저장 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

## 네트워크 학습하기
if mode == 'train':
    # 데이터 변환 정의
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
    
    dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)  # 학습 데이터셋
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)
    
    dataset_val = Dataset(data_dir=os.path.join(data_dir, 'val'), transform=transform)  # 검증 데이터셋
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_train = len(dataset_train)  # 학습 데이터 개수
    num_data_val = len(dataset_val)  # 검증 데이터 개수
    
    num_batch_train = np.ceil(num_data_train / batch_size)  # 학습 배치 개수
    num_batch_val = np.ceil(num_data_val / batch_size)  # 검증 배치 개수
else:
    transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])
    
    dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)  # 테스트 데이터셋
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)
    
    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

## 네트워크 생성하기
net = UNet().to(device)  # UNet 모델 생성 및 GPU로 이동

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)  # 바이너리 크로스엔트로피 손실 함수 사용

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)  # Adam 옵티마이저 설정

## 기타 유틸리티 함수 설정
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean  # 정규화 해제
fn_class = lambda x: 1.0 * (x > 0.5)  # 임계값을 기준으로 클래스화

## Tensorboard 를 사용하기 위한 SummaryWriter 설정
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## 네트워크 학습시키기
st_epoch = 0

if mode == 'train':
    if train_continue == "on":
        net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 체크포인트 불러오기
    
    for epoch in range(st_epoch + 1, num_epoch + 1):
        net.train()
        loss_arr = []
        
        for batch, data in enumerate(loader_train, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            
            output = net(input)  # 모델 예측값
            
            optim.zero_grad()
            loss = fn_loss(output, label)  # 손실 계산
            loss.backward()
            optim.step()
            
            loss_arr.append(loss.item())
            print(f"TRAIN: EPOCH {epoch:04d} | BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")
        
        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)
        save(ckpt_dir=ckpt_dir, net=net, optim=optim, epoch=epoch)  # 체크포인트 저장

    writer_train.close()
    writer_val.close()

# TEST MODE
else:
    net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 체크포인트 불러오기
    
    with torch.no_grad():
        net.eval()
        loss_arr = []
        
        for batch, data in enumerate(loader_test, 1):
            label = data['label'].to(device)
            input = data['input'].to(device)
            output = net(input)
            loss = fn_loss(output, label)
            loss_arr.append(loss.item())
            print(f"TEST: BATCH {batch:04d} | LOSS {np.mean(loss_arr):.4f}")
    
    print(f"AVERAGE TEST LOSS: {np.mean(loss_arr):.4f}")
