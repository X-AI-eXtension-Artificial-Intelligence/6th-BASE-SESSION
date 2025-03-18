import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16 import VGG16
import torch
import torch.nn as nn

#setting
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

batch_size = 100

# Data
transforms = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
)

cifar10_test = datasets.CIFAR10(root="./Data/", train=False, transform=transforms, target_transform=None, download=True)

test_loader = DataLoader(cifar10_test, batch_size=batch_size)

#Train
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load('./train_model')) #사전에 훈련된 모델의 파라미터를 load

# eval
correct = 0
total = 0

model.eval() #모델을 평가모드로 설정

with torch.no_grad(): #평가 시에는 모델을 업데이트하지 않으므로, gradient를 계산할 필요가 없음 (메모리 사용량 줄임)
    for i,[image,label] in enumerate(test_loader):
        x = image.to(device)
        y = label.to(device)

        output = model.forward(x)
        
        #output의 크기 = (배치 크기) x (클래스 수)
        #1:output 텐서의 두 번째 차원을 따라 각 샘플에 대해 가장 높은 값을 가진 클래스를 찾음
        #,_: 두개의 반환 값 중 인덱스만 output_index 변수에 할당
        _,output_index = torch.max(output,1) 

        total += label.size(0) #현재 배치의 크기를 반환하여 전체 샘플 수 업데이트
        correct += (output_index==y).sum().float() #정확하게 예측된 샘플의 수 업데이트

    print("Accuracy of Test DataL {}%".format(100*correct/total))