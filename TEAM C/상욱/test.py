import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG
import torch.nn as nn
import torch

batch_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# Transform 정의
# 이미지 데이터를 텐서로 변환
# transforms.Normalize(mean, std)
# channel = 3
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# CIFAR10 TEST 데이터 정의
cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, target_transform=None, download=True)

#test_loader 정의
test_loader = DataLoader(cifar10_test, batch_size=batch_size)

model = VGG(base_dim=64).to(device)

#훈련된 모델을 load해야됨
model.load_state_dict(torch.load('vgg_model.pth'))

# 맞은 개수, 전체 개수를 저장할 변수를 지정합니다.
correct = 0
total = 0

model.eval()

# 인퍼런스 모드를 위해 no_grad
with torch.no_grad():
    # 테스트로더에서 이미지와 정답을 부름
    for image,label in test_loader:
        
        # 두 데이터 모두 장치에 이동
        x = image.to(device)
        y= label.to(device)

        # 모델에 데이터를 넣고 결과값을 도출출
        output = model.forward(x)
        _,output_index = torch.max(output,1)

        
        # 전체 개수 += 라벨의 개수
        total += label.size(0)
        correct += (output_index == y).sum().float()
    
    # 정확도 도출
    print("Accuracy of Test Data: {}%".format(100*correct/total))