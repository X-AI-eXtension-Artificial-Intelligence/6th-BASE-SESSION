import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import CustomDataset
from unet import UNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_img_dir = './CamVid/train'
train_mask_dir = './CamVid/trainannot'
val_img_dir = './CamVid/val'
val_mask_dir = './CamVid/valannot'

train_dataset = CustomDataset(train_img_dir, train_mask_dir)
val_dataset = CustomDataset(val_img_dir, val_mask_dir)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

model = UNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_loss = float('inf')  # 최적 검증 손실 추적

for epoch in range(100):
    model.train()
    running_loss = 0.0
    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # 검증
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    # 최고 성능 모델 저장
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"✅ Saved new best model (Val Loss: {avg_val_loss:.4f})")
    
    print(f'Epoch [{epoch+1}/50], Train Loss: {running_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}')
