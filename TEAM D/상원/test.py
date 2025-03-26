import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from VGG16_Improved import VGG16_Improved  # 개선된 VGG 모델 사용

# ---------------- 설정 ----------------
batch_size = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print("Device:", device)

# ---------------- 전처리 ----------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# ---------------- 테스트 데이터 로드 ----------------
test_dataset = datasets.CIFAR10(root='./Data/', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ---------------- 모델 로드 ----------------
model = VGG16_Improved(base_dim=64, num_classes=10).to(device)
model.load_state_dict(torch.load("./trained_vgg16_improved.pth"))

# ---------------- 평가 함수 ----------------
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            x = images.to(device)
            y = labels.to(device)

            output = model(x)
            _, predicted = torch.max(output, 1)

            total += y.size(0)
            correct += (predicted == y).sum().item()

    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# ---------------- 평가 수행 ----------------
evaluate(model, test_loader)
