# 라이브러리
import os
import numpy as np 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import AttentionUNet
from dataset import *
from util import *

import matplotlib.pyplot as plt 
from torchvision import transforms, datasets

# 기본 경로, 데이터셋 경로, 모델 체크포인트 경로 지정
base_dir = '/home/work/XAI_BASE/BASE_4주차'
data_dir = '/home/work/XAI_BASE/BASE_4주차/dataset'
ckpt_dir = os.path.join(base_dir, "checkpoint")

# 배치 사이즈 & 초기 학습률 지정
batch_size = 4
lr = 0.001

# 이미지 전처리
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

# DataLoader 구성
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

# 배치 개수
num_data_test = len(dataset_test)
num_batch_test = np.ceil(num_data_test / batch_size)

# 결과 폴더 생성
result_dir = os.path.join(base_dir, 'result')
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# 네트워크 생성
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = AttentionUNet().to(device)

# Optimizer 설정하기(Adam)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 손실함수 정의하기(Binary CE에 시그모이드 결합된 손실함수)
fn_loss = nn.BCEWithLogitsLoss().to(device)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1) #numpy형태로 변환
fn_denorm = lambda x, mean, std: (x * std) + mean #정규화 해제
fn_class = lambda x: 1.0 * (x > 0.5) #라벨 이진화

# iou 계산을 위한 변수 초기화
total_iou = 0.0
num_samples = 0

# 모델 불러오기
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
      net.eval()
      loss_arr = []

      for batch, data in enumerate(loader_test, 1):
          label = data['label'].to(device)
          input = data['input'].to(device)

          output = net(input)

          # 손실함수 계산
          loss = fn_loss(output, label)

          loss_arr += [loss.item()]

          print("TEST: BATCH %04d / %04d | LOSS %.4f" %
                (batch, num_batch_test, np.mean(loss_arr)))


          # 모두 Numpy 객체로 변환
          label = fn_tonumpy(label)
          input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))
          output = fn_tonumpy(fn_class(output))

          # output_binary(iou 계산을 위한 이진 마스크)
          output_binary = fn_class(output)

          

          # 테스트 결과 저장하기
          for j in range(label.shape[0]):
              id = num_batch_test * (batch - 1) + j

              plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
              plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
              plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

              np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
              np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
              np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

          # 배치 내 각 샘플마다 IoU 계산
          for i in range(output_binary.shape[0]):
            # 텐서를 numpy 배열로 변환
              pred_mask = output_binary[i].squeeze() #squeeze로 채널 제거 (N,1,H,W) -> (N,H,W)
              true_mask = label[i].squeeze()
            
              iou = compute_iou(pred_mask, true_mask)
              total_iou += iou
              num_samples += 1

print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
        (batch, num_batch_test, np.mean(loss_arr)))

average_iou = total_iou / num_samples
print("Average IoU: {:.4f}".format(average_iou))