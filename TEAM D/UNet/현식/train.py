import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm

from data_read import *
from dataset import *
from KMeans import *
from model import *

batch_size = 10
epochs = 30
lr = 0.01

dataset = CityscapeDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size = batch_size)

model = UNet(num_classes = num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = lr)

step_losses = []
epoch_losses = []

for epoch in tqdm(range(epochs)) :
  epoch_loss = 0
  
  for X, Y in tqdm(data_loader, total = len(data_loader), leave = False) :
    X, Y = X.to(device), Y.to(device)
    optimizer.zero_grad()
    Y_pred = model(X)
    loss = criterion(Y_pred, Y)
    loss.backward()
    optimizer.step()

    epoch_loss += loss.item()
    step_losses.append(loss.item())

  epoch_losses.append(epoch_loss/len(data_loader))
  print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_losses[-1]:.4f}")

model_save_path = "./saved_model/unet_verson2_weights.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
torch.save(model.state_dict(), model_save_path)