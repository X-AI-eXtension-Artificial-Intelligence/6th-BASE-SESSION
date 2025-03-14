import torch.nn as nn
import torch.nn.functional as F 


def conv_2_block(in_dim, out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1), 
        nn.ReLU(), 
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1), 
        nn.ReLU(),
        nn.MaxPool2d(2,2),
        
    )
    return model

def conv_3_block(in_dim,out_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.Conv2d(out_dim,out_dim,kernel_size=3,padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2)
    )
    return model


class VGG16(nn.Module): 

    def __init__(self,base_dim,num_classes=10):
         super(VGG16,self).__init__() 
         self.feature = nn.Sequential(
            conv_2_block(3,base_dim), 
            conv_2_block(base_dim,2*base_dim), 
            conv_3_block(2*base_dim,4*base_dim),
            conv_3_block(4*base_dim,8*base_dim),
            conv_3_block(8*base_dim,8*base_dim), 
         )
         self.fc_layer = nn.Sequential(
         nn.Linear(8*base_dim*1*1,4096), #FC layer 생성 #1x1 이유: 1차원 벡터로 변환한다는 의미를 담기 위함
         nn.ReLU(),
         nn.ReLU(True),
         nn.Dropout(), #FC layer는 많은 파라미터를 포함하고 있어 Dropout를 통해 과적합 방지
         nn.Linear(4096,1000), 
         nn.ReLU(True),
         nn.Dropout(),
         nn.Linear(1000,num_classes),#CIFAR10 데이터는 총 10개의 레이블로 구성
         )
    def forward(self,x):
        x = self.feature(x) 
        x = x.view(x.size(0),-1) #추출된 특징 맵 'x'를 1차원 벡터로 변환
        x = self.fc_layer(x)
        probas = F.softmax(x,dim=1)
        return probas