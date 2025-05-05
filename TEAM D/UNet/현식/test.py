import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from data_read import *
from dataset import *
from KMeans import *
from model import *

model_save_path = "./saved_model/unet_verson2_weights.pth"

model_ = UNet(num_classes=num_classes).to(device)
model_.load_state_dict(torch.load(model_save_path))
model_.eval() 

test_batch_size = 8
dataset = CityscapeDataset(val_dir, label_model)
data_loader = DataLoader(dataset, batch_size=test_batch_size, shuffle=False)

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)

X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)

with torch.no_grad():  
    Y_pred = model_(X)
    Y_pred = torch.argmax(Y_pred, dim=1)

inverse_transform = transforms.Compose([
    transforms.Normalize((-0.485/0.229, -0.456/0.224, -0.406/0.225), (1/0.229, 1/0.224, 1/0.225))
])

fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))
iou_scores = []

for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred[i].cpu().detach().numpy()
    
    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")


save_path = os.path.join(save_dir, "all_samples_verson2.png")
plt.savefig(save_path)
plt.close(fig)

mean_iou = np.mean(iou_scores)
print(f"Mean IoU over {test_batch_size} samples: {mean_iou:.4f}")