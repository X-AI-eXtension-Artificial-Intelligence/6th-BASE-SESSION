from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# 데이터 불러오기
transform = ToTensor()
dataset = Dataset(data_dir='oxford_npy', transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# 모델, 손실, 옵티마이저
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 학습 루프
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        inputs = batch['input'].to(device)
        labels = batch['label'].to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss / len(loader):.4f}")
