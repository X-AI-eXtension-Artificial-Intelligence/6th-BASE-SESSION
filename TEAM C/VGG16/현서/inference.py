import torch
from torchvision import transforms
from PIL import Image
from model import VGG

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 로드
model = VGG(base_dim=64, num_classes=100).to(device)
model.load_state_dict(torch.load("VGG.pth", map_location=device))
model.eval()

# 예측 함수
def predict(image_path):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # CIFAR100에 맞게 사이즈 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()

# 예시 사용
if __name__ == "__main__":
    image_path = "/Users/hyunseonam/Lyle/Study/Coding/X-AI/Basic/Images/IMG_0382.JPG"   # 예측할 이미지 경로
    class_idx = predict(image_path)
    print(f"Predicted class index: {class_idx}")
