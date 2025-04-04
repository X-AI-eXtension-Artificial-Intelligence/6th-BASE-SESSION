import torch
import torch.nn.functional as F
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)


        intersection = (inputs*targets).sum()
        dice = (2* intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice