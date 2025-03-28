import torch
from model import VGG 
from data_loader import test_loader  
import os 

# 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model_path = './saved_model/vgg_model.pth'  

# 모델 로드
model = VGG(64).to(device)  # VGG 모델 인스턴스화 
model.load_state_dict(torch.load(model_path))  # 저장된 모델 가중치 로드
model.eval()  # 평가 모드로 설정
# 테스트
correct = 0 
total = 0  


with torch.no_grad():
    for images, labels in test_loader:  
        images = images.to(device) 
        labels = labels.to(device)  
        
        outputs = model(images)  # 모델을 통해 예측값 계산
        _, predicted = torch.max(outputs.data, 1)  # 예측값에서 가장 높은 확률의 클래스를 선택
        
        total += labels.size(0)  # 전체 샘플 수 업데이트
        correct += (predicted == labels).sum().item()  # 정확하게 예측한 샘플의 수 업데이트

# 정확도 계산
accuracy = 100 * correct / total  # 정확도 = (정확한 예측 수 / 전체 예측 수) * 100
print(f'Test Accuracy: {accuracy}%')  # 정확도 출력
