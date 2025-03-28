import torch.nn as nn
import torch.nn.functional as F


# (BN 추가) 각 층의 입력 분포를 정규화하여 데이터 분포 변화를 줄이고, 학습 안정성을 높이기 위함

def conv_2_block(in_dim,out_dim): 
    model = nn.Sequential( 
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model



# (Conv Layer 추가) Receptive Field를 더 넓히기 위함
# (BN 추가) 각 층의 입력 분포를 정규화하여 데이터 분포 변화를 줄이고, 학습 안정성을 높이기 위함

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(),
        nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), # 추가
        nn.BatchNorm2d(out_dim), # 추가
        nn.ReLU(), #  추가
        nn.MaxPool2d(2,2)
    )
    return model



class VGG(nn.Module):
    def __init__(self, base_dim, num_classes=10):
        super(VGG, self).__init__()
        self.feature = nn.Sequential(
            conv_2_block(3,base_dim), 
            conv_2_block(base_dim,2*base_dim), 
            conv_3_block(2*base_dim,4*base_dim), 
            conv_3_block(4*base_dim,8*base_dim), 
            conv_3_block(8*base_dim,8*base_dim),         
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(8*base_dim*1*1, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x