import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def data_transform(train=True):
    # Transform 정의
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    os.makedirs('./dataset/', exist_ok=True)
    data_set = datasets.CIFAR100(root="./dataset/", train=train, transform=transform, download=True)
    return data_set

# 이미지 확인용
def denormalize_image(image, mean, std):
    # 이미지를 정규화에서 복원. 위에서 학습을 위해 정규화를 수행했으므로!
    mean = np.array(mean)
    std = np.array(std)
    image = image.numpy().transpose((1, 2, 0))  # (C, H, W) -> (H, W, C)
    image = image * std + mean  # 역정규화
    return image

def show_images(dataset):
    print("### Number of classes:", len(dataset.classes))
    print("### Number of samples:", len(dataset))
    print("### Shape of each image:", dataset[0][0].shape)

    random_index = np.random.randint(0, len(dataset))
    image, label = dataset[random_index]

    image = denormalize_image(image, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # 이미지 시각화
    plt.imshow(image)
    plt.title(f"CIFAR-100 - Class: {label}")
    plt.show()


if __name__ == '__main__':
    train_set = data_transform()
    show_images(train_set)