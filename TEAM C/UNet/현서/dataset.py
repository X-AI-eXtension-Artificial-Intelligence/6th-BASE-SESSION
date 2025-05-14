import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

class DatasetForSeg(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)
        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input') and 'canny' not in f]
        lst_canny = [f for f in lst_data if f.startswith('input_canny')]

        lst_label.sort()
        lst_input.sort()
        lst_canny.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input
        self.lst_canny = lst_canny

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        inputs = np.load(os.path.join(self.data_dir, self.lst_input[index]))
        input_canny = np.load(os.path.join(self.data_dir, self.lst_canny[index]))

        label = label / 255.0
        inputs = inputs / 255.0
        input_canny = input_canny / 255.0

        label = label.astype(np.float32)
        inputs = inputs.astype(np.float32)
        input_canny = input_canny.astype(np.float32)

        if label.ndim == 2:
            label = label[:, :, np.newaxis]
        if inputs.ndim == 2:
            inputs = inputs[:, :, np.newaxis]
        if input_canny.ndim == 2:
            input_canny = input_canny[:, :, np.newaxis]

        data = {'input': inputs, 'input_canny': input_canny, 'label': label}

        if self.transform:
            data = self.transform(data)

        return data


# Transform

class ToTensor(object):
    def __call__(self, data):
        label = data['label'].transpose((2, 0, 1)).astype(np.float32)
        input = data['input'].transpose((2, 0, 1)).astype(np.float32)
        input_canny = data['input_canny'].transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label),
                'input': torch.from_numpy(input),
                'input_canny': torch.from_numpy(input_canny)}

        return data


# data_transform 함수
def data_transform():
    return ToTensor()

# # Data Augmentation -0408 

# class RandomFlip:
#     def __call__(self, data):
#         input, label = data['input'], data['label']
#         if np.random.rand() > 0.5:
#             input = np.flip(input, axis=1)
#             label = np.flip(label, axis=1)
#         if np.random.rand() > 0.5:
#             input = np.flip(input, axis=0)
#             label = np.flip(label, axis=0)
#         return {'input': input.copy(), 'label': label.copy()}

# class RandomRotate:
#     def __call__(self, data):
#         k = np.random.choice([0, 1, 2, 3])
#         input = np.rot90(data['input'], k, axes=(0, 1))
#         label = np.rot90(data['label'], k, axes=(0, 1))
#         return {'input': input.copy(), 'label': label.copy()}

# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, data):
#         for t in self.transforms:
#             data = t(data)
#         return data
