import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloader(batch_size=100):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 데이터 증강
        transforms.RandomCrop(32, padding=4),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_set = datasets.CIFAR10(root="./Data/", train=True, transform=transform, download=True)
    test_set = datasets.CIFAR10(root="./Data/", train=False, transform=transform, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
