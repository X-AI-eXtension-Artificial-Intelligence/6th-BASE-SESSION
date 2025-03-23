import torch
import torch.optim as optim
import torch.nn as nn  
from model import VGG  
from train import train_model 
from data_loader import train_loader, test_loader  
import os  


batch_size = 32  
learning_rate = 0.001 
num_epoch = 15 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# VGG 모델을 64차원으로 설정
model = VGG(base_dim=64).to(device)  


train_model(model, train_loader, test_loader, num_epoch, batch_size, learning_rate, device)
# 'train_model' 함수로 모델 학습 실행

# 모델 저장 경로 설정
save_path = "./saved_model/vgg_model.pth"  


if not os.path.exists(os.path.dirname(save_path)):
    os.makedirs(os.path.dirname(save_path))

# 모델의 state_dict(모델 가중치)를 저장
torch.save(model.state_dict(), save_path)
# 저장된 경로를 출력
print(f"Model saved to {save_path}")
