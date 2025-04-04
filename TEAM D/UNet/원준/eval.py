
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from model import UNet
from dataset import Dataset, Normalization, ToTensor

# 파라미터 설정
batch_size = 128
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './results'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 디렉토리 생성
if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

# 데이터 변환
transform = transforms.Compose([
    Normalization(mean=0.5, std=0.5),
    ToTensor()
])

# 데이터셋 및 로더
dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

# 네트워크 생성
net = UNet().to(device)
fn_loss = nn.BCEWithLogitsLoss().to(device)

# Optimizer는 로딩을 위해 필요
optim = torch.optim.Adam(net.parameters(), lr=0.001)

# 텐서 변환 함수
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

# 체크포인트 불러오기 함수
def load_model(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    last_ckpt = os.path.join(ckpt_dir, ckpt_lst[-1])
    checkpoint = torch.load(last_ckpt)

    net.load_state_dict(checkpoint['net'])
    optim.load_state_dict(checkpoint['optim'])
    epoch = int(last_ckpt.split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

# 평가 루프
net, optim, st_epoch = load_model(ckpt_dir, net, optim)
net.eval()

loss_arr = []
with torch.no_grad():
    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device)
        input = data['input'].to(device)

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        print("TEST: BATCH %04d / %04d | LOSS %.4f" % (
            batch, np.ceil(len(dataset_test) / batch_size), np.mean(loss_arr)
        ))

        # 시각화 및 저장
        label = fn_tonumpy(label)
        input = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        output = fn_tonumpy(fn_class(output))

        for j in range(label.shape[0]):
            idx = (batch - 1) * batch_size + j

            plt.imsave(os.path.join(result_dir, 'png', f'label_{idx:04d}.png'), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{idx:04d}.png'), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{idx:04d}.png'), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'numpy', f'label_{idx:04d}.npy'), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{idx:04d}.npy'), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{idx:04d}.npy'), output[j].squeeze())

print("AVERAGE TEST LOSS: %.4f" % np.mean(loss_arr))






# ## 라이브러리 추가하기
# import os
# import numpy as np

# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

# import matplotlib.pyplot as plt

# from torchvision import transforms, datasets



# ## 트레이닝 파라메터 설정하기
# lr = 0.001  # 학습률 설정
# batch_size = 128  # 배치 사이즈 설정
# num_epoch = 100  # 학습 에폭 수 설정

# data_dir = './datasets'  # 데이터 경로 설정
# ckpt_dir = './checkpoint'  # 체크포인트 저장 경로 설정
# log_dir = './log'  # 로그 파일 저장 경로 설정
# result_dir = './results'  # 결과 파일 저장 경로 설정

# # 경로를 구글 드라이브로 설정할 경우 아래 코드 사용
# # data_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/datasets'
# # ckpt_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/checkpoint'
# # log_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/log'
# # result_dir = './drive/My Drive/YouTube/youtube-002-pytorch-unet/results'

# # 결과 저장 디렉토리 생성
# if not os.path.exists(result_dir):
#     os.makedirs(os.path.join(result_dir, 'png'))  # PNG 파일을 위한 폴더 생성
#     os.makedirs(os.path.join(result_dir, 'numpy'))  # Numpy 파일을 위한 폴더 생성

# # GPU가 사용 가능한지 확인하고, 그에 맞는 디바이스 설정
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ## 네트워크 구축하기
# class UNet(nn.Module):
#     def __init__(self):
#         super(UNet, self).__init__()

#         # CBR2d: Conv + BatchNorm + ReLU
#         def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
#             layers = []
#             layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
#                                  kernel_size=kernel_size, stride=stride, padding=padding,
#                                  bias=bias)]
#             layers += [nn.BatchNorm2d(num_features=out_channels)]  # 배치 정규화
#             layers += [nn.ReLU()]  # 활성화 함수 ReLU

#             cbr = nn.Sequential(*layers)  # 여러 레이어를 순차적으로 적용

#             return cbr

#         # Contracting path (인코더)
#         self.enc1_1 = CBR2d(in_channels=1, out_channels=64)
#         self.enc1_2 = CBR2d(in_channels=64, out_channels=64)

#         self.pool1 = nn.MaxPool2d(kernel_size=2)  # 풀링 레이어

#         self.enc2_1 = CBR2d(in_channels=64, out_channels=128)
#         self.enc2_2 = CBR2d(in_channels=128, out_channels=128)

#         self.pool2 = nn.MaxPool2d(kernel_size=2)

#         self.enc3_1 = CBR2d(in_channels=128, out_channels=256)
#         self.enc3_2 = CBR2d(in_channels=256, out_channels=256)

#         self.pool3 = nn.MaxPool2d(kernel_size=2)

#         self.enc4_1 = CBR2d(in_channels=256, out_channels=512)
#         self.enc4_2 = CBR2d(in_channels=512, out_channels=512)

#         self.pool4 = nn.MaxPool2d(kernel_size=2)

#         self.enc5_1 = CBR2d(in_channels=512, out_channels=1024)

#         # Expansive path (디코더)
#         self.dec5_1 = CBR2d(in_channels=1024, out_channels=512)

#         self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512,
#                                           kernel_size=2, stride=2, padding=0, bias=True)  # 업샘플링

#         self.dec4_2 = CBR2d(in_channels=2 * 512, out_channels=512)
#         self.dec4_1 = CBR2d(in_channels=512, out_channels=256)

#         self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
#         self.dec3_1 = CBR2d(in_channels=256, out_channels=128)

#         self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
#         self.dec2_1 = CBR2d(in_channels=128, out_channels=64)

#         self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64,
#                                           kernel_size=2, stride=2, padding=0, bias=True)

#         self.dec1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
#         self.dec1_1 = CBR2d(in_channels=64, out_channels=64)

#         self.fc = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

#     def forward(self, x):
#         # Contracting path (인코더)
#         enc1_1 = self.enc1_1(x)
#         enc1_2 = self.enc1_2(enc1_1)
#         pool1 = self.pool1(enc1_2)

#         enc2_1 = self.enc2_1(pool1)
#         enc2_2 = self.enc2_2(enc2_1)
#         pool2 = self.pool2(enc2_2)

#         enc3_1 = self.enc3_1(pool2)
#         enc3_2 = self.enc3_2(enc3_1)
#         pool3 = self.pool3(enc3_2)

#         enc4_1 = self.enc4_1(pool3)
#         enc4_2 = self.enc4_2(enc4_1)
#         pool4 = self.pool4(enc4_2)

#         enc5_1 = self.enc5_1(pool4)

#         # Expansive path (디코더)
#         dec5_1 = self.dec5_1(enc5_1)

#         unpool4 = self.unpool4(dec5_1)
#         cat4 = torch.cat((unpool4, enc4_2), dim=1)  # skip connection
#         dec4_2 = self.dec4_2(cat4)
#         dec4_1 = self.dec4_1(dec4_2)

#         unpool3 = self.unpool3(dec4_1)
#         cat3 = torch.cat((unpool3, enc3_2), dim=1)  # skip connection
#         dec3_2 = self.dec3_2(cat3)
#         dec3_1 = self.dec3_1(dec3_2)

#         unpool2 = self.unpool2(dec3_1)
#         cat2 = torch.cat((unpool2, enc2_2), dim=1)  # skip connection
#         dec2_2 = self.dec2_2(cat2)
#         dec2_1 = self.dec2_1(dec2_2)

#         unpool1 = self.unpool1(dec2_1)
#         cat1 = torch.cat((unpool1, enc1_2), dim=1)  # skip connection
#         dec1_2 = self.dec1_2(cat1)
#         dec1_1 = self.dec1_1(dec1_2)

#         x = self.fc(dec1_1)

#         return x

# ## 데이터 로더를 구현하기
# class Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, transform=None):
#         self.data_dir = data_dir
#         self.transform = transform

#         lst_data = os.listdir(self.data_dir)  # 데이터 디렉토리에서 파일 목록 불러오기

#         lst_label = [f for f in lst_data if f.startswith('label')]  # 'label'로 시작하는 파일 목록
#         lst_input = [f for f in lst_data if f.startswith('input')]  # 'input'으로 시작하는 파일 목록

#         lst_label.sort()  # 라벨 파일 정렬
#         lst_input.sort()  # 입력 파일 정렬

#         self.lst_label = lst_label
#         self.lst_input = lst_input

#     def __len__(self):
#         return len(self.lst_label)  # 데이터셋의 크기 반환

#     def __getitem__(self, index):
#         label = np.load(os.path.join(self.data_dir, self.lst_label[index]))  # 라벨 파일 불러오기
#         input = np.load(os.path.join(self.data_dir, self.lst_input[index]))  # 입력 파일 불러오기

#         label = label/255.0  # 라벨 값을 [0, 1]로 정규화
#         input = input/255.0  # 입력 값을 [0, 1]로 정규화

#         if label.ndim == 2:  # 차원 맞추기 (H, W -> H, W, 1)
#             label = label[:, :, np.newaxis]
#         if input.ndim == 2:  # 차원 맞추기 (H, W -> H, W, 1)
#             input = input[:, :, np.newaxis]

#         data = {'input': input, 'label': label}

#         if self.transform:
#             data = self.transform(data)  # 변환 적용

#         return data

# ## 트렌스폼 구현하기
# class ToTensor(object):
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         # 데이터를 PyTorch 텐서로 변환
#         label = label.transpose((2, 0, 1)).astype(np.float32)  # HWC -> CHW
#         input = input.transpose((2, 0, 1)).astype(np.float32)  # HWC -> CHW

#         data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

#         return data

# class Normalization(object):
#     def __init__(self, mean=0.5, std=0.5):
#         self.mean = mean  # 평균값
#         self.std = std  # 표준편차

#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         # 입력값을 정규화
#         input = (input - self.mean) / self.std

#         data = {'label': label, 'input': input}

#         return data

# class RandomFlip(object):
#     def __call__(self, data):
#         label, input = data['label'], data['input']

#         # 데이터 랜덤으로 좌우/상하 뒤집기
#         if np.random.rand() > 0.5:
#             label = np.fliplr(label)
#             input = np.fliplr(input)

#         if np.random.rand() > 0.5:
#             label = np.flipud(label)
#             input = np.flipud(input)

#         data = {'label': label, 'input': input}

#         return data


# ## 네트워크 학습하기
# transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])  # 학습용 데이터 변환

# dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)  # 테스트 데이터셋
# loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)  # 테스트 데이터로더

# ## 네트워크 생성하기
# net = UNet().to(device)  # U-Net 모델 초기화

# ## 손실함수 정의하기
# fn_loss = nn.BCEWithLogitsLoss().to(device)  # 이진 교차 엔트로피 손실 함수

# ## Optimizer 설정하기
# optim = torch.optim.Adam(net.parameters(), lr=lr)  # Adam 옵티마이저

# ## 그밖에 부수적인 variables 설정하기
# num_data_test = len(dataset_test)  # 테스트 데이터 개수

# num_batch_test = np.ceil(num_data_test / batch_size)  # 테스트 배치 수

# ## 그밖에 부수적인 functions 설정하기
# fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)  # 텐서를 넘파이 배열로 변환
# fn_denorm = lambda x, mean, std: (x * std) + mean  # 정규화 되돌리기
# fn_class = lambda x: 1.0 * (x > 0.5)  # 예측값 이진화

# ## 네트워크 저장하기
# def save(ckpt_dir, net, optim, epoch):
#     if not os.path.exists(ckpt_dir):  # 체크포인트 디렉토리 없으면 생성
#         os.makedirs(ckpt_dir)

#     torch.save({'net': net.state_dict(), 'optim': optim.state_dict()},
#                "./%s/model_epoch%d.pth" % (ckpt_dir, epoch))  # 모델 저장

# ## 네트워크 불러오기
# def load(ckpt_dir, net, optim):
#     if not os.path.exists(ckpt_dir):  # 체크포인트 디렉토리 없으면 초기화
#         epoch = 0
#         return net, optim, epoch

#     ckpt_lst = os.listdir(ckpt_dir)  # 체크포인트 리스트 불러오기
#     ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))  # 에폭 순서대로 정렬

#     dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))  # 최신 모델 불러오기

#     net.load_state_dict(dict_model['net'])  # 모델 가중치 로드
#     optim.load_state_dict(dict_model['optim'])  # 옵티마이저 상태 로드
#     # 이렇게 하면 일치하는 키만 불러오고, 없는 키는 무시합니다. 다만, 모델이 제대로 학습되지 않을 수도 있으니 주의하세요.

#     epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])  # 마지막 에폭 번호

#     return net, optim, epoch

# ## 네트워크 학습시키기
# st_epoch = 0  # 시작 에폭
# net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)  # 체크포인트 불러오기

# with torch.no_grad():  # 평가 모드로 설정
#     net.eval()  # 평가 모드로 변경
#     loss_arr = []  # 손실값 저장 리스트

#     for batch, data in enumerate(loader_test, 1):  # 테스트 데이터 로딩
#         # forward pass
#         label = data['label'].to(device)
#         input = data['input'].to(device)

#         output = net(input)  # 네트워크 출력

#         # 손실함수 계산하기
#         loss = fn_loss(output, label)

#         loss_arr += [loss.item()]  # 손실값 저장

#         print("TEST: BATCH %04d / %04d | LOSS %.4f" %
#               (batch, num_batch_test, np.mean(loss_arr)))  # 테스트 진행상황 출력

#         # Tensorboard 저장하기
#         label = fn_tonumpy(label)  # 라벨 넘파이로 변환
#         input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5))  # 입력 데이터 되돌리기
#         output = fn_tonumpy(fn_class(output))  # 출력값 이진화

#         for j in range(label.shape[0]):  # 배치 내 각 샘플에 대해 저장
#             id = num_batch_test * (batch - 1) + j

#             plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
#             plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
#             plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

#             np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
#             np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
#             np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

# print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
#       (batch, num_batch_test, np.mean(loss_arr)))  # 전체 테스트 평균 손실 출력




