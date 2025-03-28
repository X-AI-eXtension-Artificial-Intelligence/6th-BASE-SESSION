import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import trange
from model import VGG

# batch_size : 한 번의 모델 파라미터 업데이트로 몇 개의 데이터를 동시에 적용할지 정하는 값
## 목표 : 성능 높이기 -> 따라서 64로 줄여보기 
batch_size = 64  
# 학습률 0.0002 -> 0.0001 변경
## 학습률 : 가중치를 얼마나 크게 업데이트할지 정하는 하이퍼파라미터 (너무 크면 불안정함/너무 작으면 느림)
learning_rate = 0.0001
# 에폭수 100 -> 150 변경
## 에폭수를 늘리면 -> 모델이 더 많은 데이터를 보고 학습하도록 함 (전체 데이터셋을 100->150번 반복해서 학습하도록 함)
num_epoch = 150 
save_path = "vgg_model.pth"  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VGG(base_dim=64).to(device)


loss_func = nn.CrossEntropyLoss()  
# optimizer Adam -> AdamW 변경 + weight_decay 추가
## AdamW : Adam의 개선 버전으로 L2 정규화 적용 방식이 개선됨 
## weight_decay : 모델의 가중치가 너무 커지지 않도록 막아주는 규제 기법 (너무 큰 값->모델이 충분히 학습을 못함/너무 작은 값->효과가 없음)
### Adam에도 weight_decay 옵션이 있지만, 방식이 조금 애매해서 정규화 효과가 왜곡될 수 있음
### AdamW는 weight_decay(정규화)를 제대로 분리해서 적용하는 개선된 버전임 ! 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)  


transform = transforms.Compose(
    [transforms.ToTensor(),  
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 


cifar10_train = datasets.CIFAR10(root="../Data/", train=True, transform=transform, target_transform=None, download=True)
train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle=True)


loss_arr = []  
for i in trange(num_epoch): 
    for j,[image,label] in enumerate(train_loader):  
        x = image.to(device) 
        y_= label.to(device) 
        
        optimizer.zero_grad() 
        output = model.forward(x) 
        loss = loss_func(output,y_) 
        loss.backward() 
        optimizer.step()  

    if i % 10 ==0: 
        print(loss)
        loss_arr.append(loss.cpu().detach().numpy())

torch.save(model.state_dict(), save_path)
print(f"모델 저장 완료: {save_path}")