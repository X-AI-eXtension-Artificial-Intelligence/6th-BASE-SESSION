# eval.py
import torch
from torch import nn

def evaluate(model, dataloader, device):
    model.eval()
    criterion = nn.BCELoss()
    total_loss = 0.0

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            masks = torch.mean(inputs, dim=1, keepdim=True)
            masks = (masks + 1) / 2  # normalize to [0,1]

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            total_loss += loss.item()

    return total_loss / len(dataloader)
