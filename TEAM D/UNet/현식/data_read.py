import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

root_path = './cityscapes_data'
data_dir = root_path

train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")

train_fns = os.listdir(train_dir)
val_fns = os.listdir(val_dir)

print(len(train_fns), len(val_fns))