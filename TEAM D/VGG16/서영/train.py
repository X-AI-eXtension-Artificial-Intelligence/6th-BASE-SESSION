import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import VGG_BatchNorm


batch_size = 100
learning_rate = 0.0005
num_epoch = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


cifar10_train = datasets.CIFAR10(root="./Data/", train=True, transform=transform, target_transform=None, download=True)


train_loader = DataLoader(cifar10_train, batch_size=batch_size, shuffle=True)


model = VGG_BatchNorm(base_dim=64).to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


loss_arr = []
for i in range(num_epoch):
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

torch.save(model.state_dict(), "./models/VGG16_100.pth") 
