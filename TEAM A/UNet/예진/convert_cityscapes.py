import os
import numpy as np
from PIL import Image
from tqdm import tqdm

color_map = {
    (128, 64, 128): 0,
    (244, 35, 232): 1,
    (70, 70, 70): 2,
    (102, 102, 156): 3,
    (153, 153, 153): 4,
    (250, 170, 30): 5,
    (220, 220, 0): 6,
    (0, 0, 142): 7,
    (0, 0, 70): 8,
    (0, 60, 100): 9
}

def rgb_to_class(mask_rgb):
    h, w, _ = mask_rgb.shape
    mask_class = np.zeros((h, w), dtype=np.uint8)
    for rgb, cls in color_map.items():
        match = np.all(mask_rgb == rgb, axis=-1)
        mask_class[match] = cls
    return mask_class

def convert_images(img_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    img_list = sorted(os.listdir(img_dir))[:1200]  # 시간문제로 개수 제한

    for i, fname in enumerate(tqdm(img_list)):
        img = np.array(Image.open(os.path.join(img_dir, fname)))
        input_img = img[:, :256, :]
        label_rgb = img[:, 256:, :]
        label_class = rgb_to_class(label_rgb)

        np.save(os.path.join(save_dir, f'input_{i:03d}.npy'), input_img)
        np.save(os.path.join(save_dir, f'label_{i:03d}.npy'), label_class)
        
if __name__ == "__main__":
    convert_images("./cityscapes/cityscapes_data/train", "./datasets/train")
    convert_images("./cityscapes/cityscapes_data/val", "./datasets/val")
    convert_images("./cityscapes/cityscapes_data/val", "./datasets/test")  # 테스트용 임시 재사용


