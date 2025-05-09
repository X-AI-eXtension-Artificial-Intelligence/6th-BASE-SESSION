import torch
import matplotlib.pyplot as plt
from dataset_custom import CityscapesCombinedDataset
from model import UNet
import torchvision.transforms as transforms

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 데이터셋 (val or train 사용 가능)
dataset = CityscapesCombinedDataset(
    # root_dir="datasets_city/train",  
    root_dir = "/home/work/.local/unet/datasets_city/train",
    transform=transform
)

# 학습된 모델 불러오기
model = UNet().to(device)
model.load_state_dict(torch.load("/home/work/.local/unet/saved_models/unet_cityscapes.pth", map_location=device))

model.eval()

# 한 장 예측
idx = 0  # 바꾸면 다른 이미지 확인 가능
image, label = dataset[idx]
input_tensor = image.unsqueeze(0).to(device)  # [1, C, H, W]

with torch.no_grad():
    output = model(input_tensor)
    output = torch.sigmoid(output)
    output = output.squeeze().cpu().numpy()

# 시각화용 텐서 -> numpy 변환
input_np = image.squeeze().cpu().numpy()
label_np = label.squeeze().cpu().numpy()

# 시각화
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(input_np, cmap='gray')
plt.title('Input')

plt.subplot(1, 3, 2)
plt.imshow(label_np, cmap='gray')
plt.title('Ground Truth')

plt.subplot(1, 3, 3)
plt.imshow(output, cmap='gray')
plt.title('Prediction')

plt.tight_layout()

plt.savefig("output_result.png")  # 결과 저장
plt.show()  # GUI 환경이면 띄우기, CLI면 무시됨
