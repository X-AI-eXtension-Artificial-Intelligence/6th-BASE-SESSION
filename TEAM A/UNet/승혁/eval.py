# 필수 라이브러리
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

# 실험 설정
lr = 1e-3
batch_size = 4
num_epoch = 100

# 디렉토리 설정
data_dir = 'datasets'
ckpt_dir = './checkpoint'
log_dir = './log'
result_dir = './results'

# 결과 저장 폴더 없으면 생성
os.makedirs(os.path.join(result_dir, 'png'), exist_ok=True)
os.makedirs(os.path.join(result_dir, 'numpy'), exist_ok=True)

# 디바이스 설정 (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 네트워크 구축하기
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_ch, out_ch, k=3, s=1, p=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )

        # 인코더
        self.enc1_1, self.enc1_2 = CBR2d(1,64), CBR2d(64,64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2_1, self.enc2_2 = CBR2d(64,128), CBR2d(128,128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3_1, self.enc3_2 = CBR2d(128,256), CBR2d(256,256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4_1, self.enc4_2 = CBR2d(256,512), CBR2d(512,512)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5_1 = CBR2d(512,1024)

        # 디코더
        self.dec5_1 = CBR2d(1024,512)
        self.unpool4 = nn.ConvTranspose2d(512,512,2,2)
        self.dec4_2, self.dec4_1 = CBR2d(1024,512), CBR2d(512,256)
        self.unpool3 = nn.ConvTranspose2d(256,256,2,2)
        self.dec3_2, self.dec3_1 = CBR2d(512,256), CBR2d(256,128)
        self.unpool2 = nn.ConvTranspose2d(128,128,2,2)
        self.dec2_2, self.dec2_1 = CBR2d(256,128), CBR2d(128,64)
        self.unpool1 = nn.ConvTranspose2d(64,64,2,2)
        self.dec1_2, self.dec1_1 = CBR2d(128,64), CBR2d(64,64)
        self.fc = nn.Conv2d(64,1,1)

    def forward(self, x):
        # 인코더
        e1 = self.enc1_2(self.enc1_1(x)); p1 = self.pool1(e1)
        e2 = self.enc2_2(self.enc2_1(p1)); p2 = self.pool2(e2)
        e3 = self.enc3_2(self.enc3_1(p2)); p3 = self.pool3(e3)
        e4 = self.enc4_2(self.enc4_1(p3)); p4 = self.pool4(e4)
        m = self.enc5_1(p4)

        # 디코더
        d5 = self.dec5_1(m); up4 = self.unpool4(d5)
        d4 = self.dec4_1(self.dec4_2(torch.cat((up4, e4), 1))); up3 = self.unpool3(d4)
        d3 = self.dec3_1(self.dec3_2(torch.cat((up3, e3), 1))); up2 = self.unpool2(d3)
        d2 = self.dec2_1(self.dec2_2(torch.cat((up2, e2), 1))); up1 = self.unpool1(d2)
        d1 = self.dec1_1(self.dec1_2(torch.cat((up1, e1), 1)))

        return self.fc(d1)

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
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

        enc5_1 = self.enc5_1(pool4)

        dec5_1 = self.dec5_1(enc5_1)

        unpool4 = self.unpool4(dec5_1)
        cat4 = torch.cat((unpool4, enc4_2), dim=1)
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

        x = self.fc(dec1_1)

        return x

## 데이터 로더를 구현하기
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        # 'label', 'input'으로 시작하는 파일 목록 구분
        self.lst_label = sorted([f for f in lst_data if f.startswith('label')])
        self.lst_input = sorted([f for f in lst_data if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)  # 전체 샘플 수

    def __getitem__(self, index):
        # numpy 파일 불러오기
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 정규화 (0~1)
        label = label / 255.0
        input = input / 255.0

        # (H, W) → (H, W, 1) 채널 차원 추가
        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # transform 적용 (ToTensor, Normalization 등)
        if self.transform:
            data = self.transform(data)

        return data


# numpy → torch Tensor, CHW 형태로 변환
class ToTensor(object):
    def __call__(self, data):
        input, label = data['input'], data['label']
        input = input.transpose((2, 0, 1)).astype(np.float32)  # HWC → CHW
        label = label.transpose((2, 0, 1)).astype(np.float32)
        return {'input': torch.from_numpy(input), 'label': torch.from_numpy(label)}

# 입력 이미지 정규화 (mean/std는 미리 설정된 값)
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        data['input'] = (data['input'] - self.mean) / self.std
        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data


## 네트워크 학습하기
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), ToTensor()])

dataset_test = Dataset(data_dir=os.path.join(data_dir, 'test'), transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

## 네트워크 생성하기
net = UNet().to(device)

## 손실함수 정의하기
fn_loss = nn.BCEWithLogitsLoss().to(device)

## Optimizer 설정하기
optim = torch.optim.Adam(net.parameters(), lr=lr)

## 그밖에 부수적인 variables 설정하기
num_data_test = len(dataset_test)

num_batch_test = np.ceil(num_data_test / batch_size)

## 그밖에 부수적인 functions 설정하기
fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: 1.0 * (x > 0.5)

## 네트워크 저장하기
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

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('./%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return net, optim, epoch

## 네트워크 학습시키기
st_epoch = 0
net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

with torch.no_grad():
    net.eval()
    loss_arr = []

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

        for j in range(label.shape[0]):
            id = num_batch_test * (batch - 1) + j

            plt.imsave(os.path.join(result_dir, 'png', 'label_%04d.png' % id), label[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'input_%04d.png' % id), input[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', 'output_%04d.png' % id), output[j].squeeze(), cmap='gray')

            np.save(os.path.join(result_dir, 'numpy', 'label_%04d.npy' % id), label[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'input_%04d.npy' % id), input[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', 'output_%04d.npy' % id), output[j].squeeze())

print("AVERAGE TEST: BATCH %04d / %04d | LOSS %.4f" %
      (batch, num_batch_test, np.mean(loss_arr)))























