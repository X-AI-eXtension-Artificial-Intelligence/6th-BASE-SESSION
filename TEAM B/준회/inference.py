## 추론하는 코드

import torch
import torchvision.transforms as transforms
from PIL import Image
from VGG16 import VGG16


## 학습시킨 모델 불러오는 함수
def load_model(model_path, device):
    model = VGG16(base_dim=64)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


## 이미지 파일 전처리 함수
def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  ## CIFAR-10 크기에 맞게 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    image = Image.open(image_path).convert('RGB')  ## 이미지 열기 및 RGB로 변환
    return transform(image).unsqueeze(0)  ## 배치 차원을 추가

## 이미지 예측 함수
def predict_image(model, image_tensor, device, classes):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)  ## 가장 높은 확률의 클래스 선택
    return classes[predicted.item()]

if __name__ == '__main__':
    ## 장치 설정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    ## CIFAR-10 클래스 정의
    classes = ('cane', 'cavallo', 'elefante', 'farfalla', 'gallina', 
               'gatto', 'mucca', 'pecora', 'ragno', 'scoiattolo')

    ## 모델 로드
    model_path = "./VGG16/vgg16_animals_10.pth"  ## 저장된 모델 경로
    model = load_model(model_path, device)

    ## 이미지 전처리
    image_path = "./VGG16/코끼리사진.jpg"  ## 테스트할 이미지 경로
    image_tensor = preprocess_image(image_path)

    ## 이미지 예측
    predicted_class = predict_image(model, image_tensor, device, classes)

    ## 결과 출력
    print(f"The image '{image_path}' is predicted as: {predicted_class}")
