import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import transforms
from util_evaluation_change import compute_iou, compute_pixel_accuracy

# 설정
lr = 1e-3
batch_size = 4
data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './results'

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'png'))
    os.makedirs(os.path.join(result_dir, 'numpy'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, 1, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.enc1 = nn.Sequential(CBR2d(1, 64), CBR2d(64, 64))
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(CBR2d(64, 128), CBR2d(128, 128))
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(CBR2d(128, 256), CBR2d(256, 256))
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(CBR2d(256, 512), CBR2d(512, 512))
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = CBR2d(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = nn.Sequential(CBR2d(1024, 512), CBR2d(512, 256))
        self.up3 = nn.ConvTranspose2d(256, 256, 2, 2)
        self.dec3 = nn.Sequential(CBR2d(512, 256), CBR2d(256, 128))
        self.up2 = nn.ConvTranspose2d(128, 128, 2, 2)
        self.dec2 = nn.Sequential(CBR2d(256, 128), CBR2d(128, 64))
        self.up1 = nn.ConvTranspose2d(64, 64, 2, 2)
        self.dec1 = nn.Sequential(CBR2d(128, 64), CBR2d(64, 64))

        self.final = nn.Conv2d(64, 3, 1)  # 클래스 3개로 수정

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

# Dataset 정의
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.lst_label = sorted([f for f in os.listdir(data_dir) if f.startswith('label')])
        self.lst_input = sorted([f for f in os.listdir(data_dir) if f.startswith('input')])

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        input = input / 255.0  # 정규화
        if input.ndim == 2: input = input[:, :, np.newaxis]
        if label.ndim == 2: label = label[:, :, np.newaxis]

        data = {'label': label, 'input': input}
        if self.transform:
            data = self.transform(data)
        return data

# Transform 정의
class ToTensor(object):
    def __call__(self, data):
        label = data['label'].transpose(2, 0, 1).astype(np.int64)
        input = data['input'].transpose(2, 0, 1).astype(np.float32)
        return {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        input = (data['input'] - self.mean) / self.std
        return {'label': data['label'], 'input': input}

# 데이터셋 로딩
transform = transforms.Compose([Normalization(), ToTensor()])
dataset_test = Dataset(os.path.join(data_dir, 'test'), transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)

# 모델 로딩
net = UNet().to(device)
fn_loss = nn.CrossEntropyLoss(ignore_index=255).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)

# 모델 weight 불러오기
def load(ckpt_dir, net, optim):
    ckpt_lst = sorted(os.listdir(ckpt_dir), key=lambda f: int(''.join(filter(str.isdigit, f))))
    dict_model = torch.load(os.path.join(ckpt_dir, ckpt_lst[-1]))
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    return net, optim, epoch

net, optim, st_epoch = load(ckpt_dir, net, optim)

fn_tonumpy = lambda x: x.to('cpu').detach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean
fn_class = lambda x: torch.argmax(x, dim=1, keepdim=True)

# 테스트 시작
with torch.no_grad():
    net.eval()
    loss_arr, iou_scores, pixel_accuracies = [], [], []

    for batch, data in enumerate(loader_test, 1):
        label = data['label'].to(device).squeeze(1)
        input = data['input'].to(device)

        # ignore 처리
        label[label > 2] = 255

        output = net(input)
        loss = fn_loss(output, label)
        loss_arr.append(loss.item())

        pred = fn_class(output)

        iou = compute_iou(pred, label.unsqueeze(1))
        acc = compute_pixel_accuracy(pred, label.unsqueeze(1))
        iou_scores.append(iou.item())
        pixel_accuracies.append(acc.item())

        print(f"TEST: BATCH {batch:04d} | LOSS {loss.item():.4f} | IOU {iou:.4f} | ACC {acc:.4f}")

        label_np = fn_tonumpy(label.unsqueeze(1))
        input_np = fn_tonumpy(fn_denorm(input, 0.5, 0.5))
        output_np = fn_tonumpy(pred)

        for j in range(label_np.shape[0]):
            id = (batch - 1) * batch_size + j
            np.save(os.path.join(result_dir, 'numpy', f'label_{id:04d}.npy'), label_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'input_{id:04d}.npy'), input_np[j].squeeze())
            np.save(os.path.join(result_dir, 'numpy', f'output_{id:04d}.npy'), output_np[j].squeeze())
            plt.imsave(os.path.join(result_dir, 'png', f'label_{id:04d}.png'), label_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'input_{id:04d}.png'), input_np[j].squeeze(), cmap='gray')
            plt.imsave(os.path.join(result_dir, 'png', f'output_{id:04d}.png'), output_np[j].squeeze(), cmap='gray')

print("\n" + "="*50)
print(f"AVERAGE TEST: LOSS {np.mean(loss_arr):.4f} | IoU {np.mean(iou_scores):.4f} | Accuracy {np.mean(pixel_accuracies):.4f}")
print("="*50)
