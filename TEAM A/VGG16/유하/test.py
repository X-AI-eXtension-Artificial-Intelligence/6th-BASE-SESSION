import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG


batch_size = 64
learning_rate = 0.0001
save_path = "vgg_model.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = VGG(base_dim=64).to(device)

try:
    model.load_state_dict(torch.load(save_path))
    print(f"저장된 모델 {save_path} 불러오기 성공!")
except FileNotFoundError:
    print(f"저장된 모델 {save_path}을 찾을 수 없습니다. 먼저 train.py를 실행하세요.")

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

cifar10_test = datasets.CIFAR10(root="../Data/", train=False, transform=transform, target_transform=None, download=True)
test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle=False)

# test
correct = 0  
total = 0  


model.eval()

with torch.no_grad():

    for image,label in test_loader:
        
        x = image.to(device)  
        y= label.to(device) 

        output = model.forward(x)
        _,output_index = torch.max(output,1)

        total += label.size(0)  
        correct += (output_index == y).sum().float()  
    
    print("Accuracy of Test Data: {}%".format(100*correct/total))