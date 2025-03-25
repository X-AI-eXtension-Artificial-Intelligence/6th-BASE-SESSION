import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from vgg16 import VGG16
from torchvision import datasets
import matplotlib.pyplot as plt

# Device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CIFAR-10 클래스 목록
cifar10_test = datasets.CIFAR10(root="./Data/", train=False, download=True)
class_names = cifar10_test.classes

# 이미지 전처리 정의
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 모델 로드
model = VGG16(base_dim=64).to(device)
model.load_state_dict(torch.load("./train_model"))
model.eval()

# 이미지 폴더 경로
image_dir = os.path.join(os.path.dirname(__file__), 'dog')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 자동 반복 예측 및 시각화
with torch.no_grad():
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        image_pil = Image.open(image_path).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

        print(f"[{image_name}] ▶ 예측: {predicted_class} (클래스 인덱스: {predicted.item()})")

        # 시각화 출력
        plt.imshow(image_pil)
        plt.title(f"Predicted: {predicted_class}")
        plt.axis('off')
        plt.show()
