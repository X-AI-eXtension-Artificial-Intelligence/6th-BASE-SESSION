import os
import numpy as np
from PIL import Image
from tqdm import tqdm

img_dir = "oxford_pets/images"
mask_dir = "oxford_pets/annotations/trimaps"
save_dir = "oxford_npy"
os.makedirs(save_dir, exist_ok=True)

image_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

for i, fname in enumerate(tqdm(image_files)):
    img_path = os.path.join(img_dir, fname)
    mask_path = os.path.join(mask_dir, fname.replace('.jpg', '.png'))

    if not os.path.exists(mask_path):
        continue

    image = Image.open(img_path).convert("L").resize((128, 128))
    mask = Image.open(mask_path).convert("L").resize((128, 128))

    image = np.array(image)
    mask = np.array(mask)

    # 동물만 1로, 나머지는 0
    mask = (mask == 2).astype(np.uint8)

    np.save(os.path.join(save_dir, f"input_{i:04d}.npy"), image)
    np.save(os.path.join(save_dir, f"label_{i:04d}.npy"), mask)
