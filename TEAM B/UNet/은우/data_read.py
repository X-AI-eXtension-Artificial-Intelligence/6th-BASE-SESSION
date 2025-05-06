from datasets import load_dataset
import os
import numpy as np
from PIL import Image

# 저장 경로
data_dir = './datasets'
dir_train = os.path.join(data_dir, 'train')
dir_val = os.path.join(data_dir, 'val')
dir_test = os.path.join(data_dir, 'test')

# 디렉토리 생성
for d in [dir_train, dir_val, dir_test]:
    os.makedirs(d, exist_ok=True)

# 데이터셋 로드
dataset = load_dataset("beans")

# 저장
for split_name, save_dir in zip(['train', 'validation', 'test'], [dir_train, dir_val, dir_test]):
    split = dataset[split_name]
    for i, item in enumerate(split):
        img = item['image']  # 이미 PIL.Image 타입
        img_np = np.array(img)  # numpy 배열로 변환

        label = item['labels']  # 클래스 번호 (0, 1, 2)

        # 이미지와 레이블을 각각 .npy 형식으로 저장
        np.save(os.path.join(save_dir, f"input_{i:03d}.npy"), img_np)
        np.save(os.path.join(save_dir, f"label_{i:03d}.npy"), label)
