import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, train_loader, device, num_epochs=100, lr=0.0002):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for image, label in tqdm(train_loader):
            image, label = image.to(device), label.to(device)

            optimizer.zero_grad()
            output = model(image)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        scheduler.step()  # 스케줄러 적용
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
