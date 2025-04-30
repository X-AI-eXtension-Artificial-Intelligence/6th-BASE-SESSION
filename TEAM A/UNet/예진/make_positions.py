from PIL import Image
import numpy as np
import os

# 저장할 경로
save_dir = './datasets/train'
os.makedirs(save_dir, exist_ok=True)

# 라벨 tif 열기
img = Image.open('./datasets/train-labels.tif')
for i in range(img.n_frames):
    img.seek(i)
    label = np.array(img)

    # 위치 기반 클래스 변환
    h, w = label.shape
    pos_label = np.zeros_like(label, dtype=np.uint8)
    pos_label[:h//3, :] = 0     # 상단 1/3 → 클래스 0
    pos_label[h//3:2*h//3, :] = 1  # 중간 1/3 → 클래스 1
    pos_label[2*h//3:, :] = 2   # 하단 1/3 → 클래스 2

    # 저장
    np.save(os.path.join(save_dir, f'label_{i:03d}.npy'), pos_label)
