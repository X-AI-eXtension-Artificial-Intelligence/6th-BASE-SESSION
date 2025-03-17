#라이브러리 import
import torchvision
import torchvision.transforms as transforms #이미지 전처리
import torchvision.datasets as datasets #torch 내장 데이터셋
from torch.utils.data import DataLoader #train/test 세트 준비
import matplotlib.pyplot as plt
import numpy as np

#Transform 정의
def data_transform(train = True):
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])])

    #torchvision 내장 CIFAR10 Dataset 활용(target_transform - 레이블은 변환 없음)
    fashionmnist_dataset = datasets.FashionMNIST(root = "../Data/", train = train, transform=transform, target_transform=None, download = True)
    return fashionmnist_dataset

#클래스 정의
class_names = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 데이터셋 이미지 시각화
def imshow(img):
    img = img / 2 + 0.5  # 정규화 해제
    npimg = img.numpy()

    if npimg.shape[0] == 1:  # FashionMNIST는 흑백 이미지 (1, H, W)
        npimg = npimg.squeeze(0)  # (1, H, W) -> (H, W)

    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')  # 흑백 이미지 시각화
    plt.savefig('FashionMNIST_Image.png')  # 이미지 저장
    plt.show()
    plt.close()

# 데이터 로더에서 무작위로 이미지 가져와서 격자 형태로 시각화
def random_visualize(data_loader):
    # 학습용 이미지를 무작위로 가져오기
    dataiter = iter(data_loader)
    images, labels = next(dataiter)

    # make_grid로 여러 이미지 grid 형태로 묶어서 출력
    imshow(torchvision.utils.make_grid(images, nrow=10, padding=2, normalize=True))

    # 배치 크기만큼 이미지 클래스 라벨 출력
    print(' '.join('%5s' % class_names[labels[j].item()] for j in range(len(labels))))
