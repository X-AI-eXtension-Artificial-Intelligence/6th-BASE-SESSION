import os
import torch
from torch.utils.data import DataLoader
from model import UNet
from dataset import NucleiDataset
from sklearn.model_selection import train_test_split

# 데이터 준비
root_dir = './datasets/stage1_train'
all_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
train_ids, val_ids = train_test_split(all_ids, test_size=0.2, random_state=42)

train_dataset = NucleiDataset(root_dir, train_ids)
val_dataset = NucleiDataset(root_dir, val_ids)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for data in train_loader:
        x = data['input'].to(device)
        y = data['label'].to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x.size(0)
    train_loss /= len(train_dataset)

    # 검증
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data in val_loader:
            x = data['input'].to(device)
            y = data['label'].to(device)
            y_hat = model(x)
            loss = criterion(y_hat, y)
            val_loss += loss.item() * x.size(0)
    val_loss /= len(val_dataset)
    print(f"[Epoch {epoch+1}/ num_epochs] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
