import torchvision.datasets as datasets #Pytorch의 Vision 라이브러리 데이터셋 모듈
import torchvision.transforms as transforms #이미지 전처리 및 변환을 위한 모듈
from torch.utils.data import DataLoader #데이터를 미니배치로 로딩하기 위한 DataLoader 모듈
from VGG16 import VGG16
import torch 
import torch.nn as nn #PyTorch 모듈 중 인공 신경망 모델을 설계하는데 필요한 함수를 모아둔 모듈

#setiing
batch_size = 100 #각 반복에서 모델이 학습하는 데이터 샘플 수
learning_rate = 0.0002 #각 업데이트 단에서 얼마나 많은 양의 매개변수를 조정할지 결정
num_epoch = 100 #전체 데이터셋을 100번 반복해서 학습

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device) #CUDA GPU 사용 가능 여부 확인

transforms = transforms.Compose( 
    [transforms.ToTensor(), 
     transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))] 
)

#CIFAR10 데이터셋 load
cifar10_train = datasets.CIFAR10(root='./Data/', train=True, transform=transforms, target_transform=None, download=True)
cifar10_test = datasets.CIFAR10(root='./Data/', train=False, transform=transforms, target_transform=None, download=True)

train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True) 
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

#Train
model = VGG16(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

loss_arr = [] 

for i in range(num_epoch): 
    for j,[image,label] in enumerate(train_loader): 
        x = image.to(device) 
        y = label.to(device) 

        optimizer.zero_grad()
        output = model.forward(x) 
        loss = loss_func(output,y) 
        loss.backward() 
        optimizer.step() 

    if i%10 == 0 : #10 epoch마다 한 번씩 현재 손실을 출력
        print(f'epoch {i} loss : ', loss) 
        loss_arr.append(loss.cpu().detach().numpy()) #텐서를 gradient 계산에서 분리

#모델의 학습된 가중치들을 저장
torch.save(model.state_dict(), "./train_model")