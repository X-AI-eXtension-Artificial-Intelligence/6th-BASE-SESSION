import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.transforms.functional import to_tensor

class SegmentationTransform:
    def __init__(self, size=(256, 256)):
        self.size = size
        self.img_transform = transforms.Compose([
            transforms.Resize(self.size),
            transforms.ToTensor()
        ])

        self.mask_transform = transforms.Resize(self.size, interpolation=transforms.InterpolationMode.NEAREST)

    def __call__(self, image, target):
        image = self.img_transform(image)

        # 마스크는 PIL Image → numpy → long tensor
        target = self.mask_transform(target)
        target = np.array(target, dtype=np.int64)
        target = torch.from_numpy(target).long()

        return image, target

def get_loader(batch_size=4, shuffle=True, num_workers=2):
    transform = SegmentationTransform()

    train_dataset = OxfordIIITPet(
        root='data',
        split='trainval',
        target_types='segmentation',
        download=True,
        transforms=transform
    )

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader
