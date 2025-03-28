import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from dataset import dataset_loader 
from model import VGG19            

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# test split DataLoader 준비 
test_loader = dataset_loader("test", batch_size= 32, shuffle=False)

# 모델 정의 및 가중치 로드
model = VGG19(base_dim=64).to(device)

# 모델 가중치 불러오기
model.load_state_dict(torch.load("VGG19_model.pth"))
model.eval()  # 평가 모드 전환

# 손실 함수 정의
loss_func = nn.CrossEntropyLoss()

correct = 0
total = 0

# 테스트
with torch.no_grad():  # gradient 계산 안함
    for batch in test_loader:
        # 학습 시와 동일하게 데이터는 딕셔너리 형태로 반환됨
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        output = model(images)
        _, output_index = torch.max(output, 1)

        total += labels.size(0)
        correct += (output_index == labels).sum().item()

print("Total samples: ", total)
print("Correct predictions: ", correct)
print("Accuracy of Test Data: {}%".format(100 * correct / total))
