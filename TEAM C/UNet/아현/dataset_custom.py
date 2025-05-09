from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms as transforms


# 출처 : https://velog.io/@jarvis_geun/U-Net-%EC%8B%A4%EC%8A%B5
# https://www.kaggle.com/datasets/dansbecker/cityscapes-image-pairs

class CityscapesCombinedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.filenames = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.filenames[idx])
        combined = Image.open(img_path).convert("L")

        w, h = combined.size
        left = combined.crop((0, 0, w // 2, h))       # input image
        right = combined.crop((w // 2, 0, w, h))       # label mask

        if self.transform:
            left = self.transform(left)
            right = self.transform(right)

        return left, right
