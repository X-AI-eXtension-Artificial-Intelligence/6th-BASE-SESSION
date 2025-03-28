import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from vgg16 import VGG16 as VGG16_Original
from vgg16_light import VGG16 as VGG16_Light
from torchvision import datasets
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time

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

# 모델 로드 함수
def load_model(model_class, path):
    model = model_class(base_dim=64).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

model_orig = load_model(VGG16_Original, "./train_model")
model_light = load_model(VGG16_Light, "./train_model_light")

# 이미지 폴더 경로
image_dir = os.path.join(os.path.dirname(__file__), 'dog')
image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

# 자동 반복 예측 및 시각화
with torch.no_grad():
    for image_name in image_files:
        image_path = os.path.join(image_dir, image_name)
        image_pil = Image.open(image_path).convert('RGB')
        image_tensor = transform(image_pil).unsqueeze(0).to(device)

        # 원본 VGG 예측
        start = time.time()
        output_orig = model_orig(image_tensor)
        time_orig = time.time() - start
        _, pred_orig = torch.max(output_orig, 1)
        class_orig = class_names[pred_orig.item()]
        prob_orig = F.softmax(output_orig, dim=1)
        confidence_orig = prob_orig.max().item()

        # 경량 VGG 예측
        start = time.time()
        output_light = model_light(image_tensor)
        time_light = time.time() - start
        _, pred_light = torch.max(output_light, 1)
        class_light = class_names[pred_light.item()]
        prob_light = F.softmax(output_light, dim=1)
        confidence_light = prob_light.max().item()

        print(f"[{image_name}]")
        print(f" ✅ Original VGG ▸ {class_orig} (예측 확률: {confidence_orig*100:.2f}%, 시간: {time_orig*1000:.1f}ms)")
        print(f" 🔹 Light VGG    ▸ {class_light} (예측 확률: {confidence_light*100:.2f}%, 시간: {time_light*1000:.1f}ms)")

        # 시각화 출력
        plt.imshow(image_pil)
        plt.title(f"Original: {class_orig} | Light: {class_light}")
        plt.axis('off')
        plt.show()
