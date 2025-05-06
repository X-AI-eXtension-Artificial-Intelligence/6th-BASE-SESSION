"""
https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset
Human Segmentation Dataset

train set : 2133개
val set : 267개
test set : 267개
"""

import opendatasets as od
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

# 1. 루트 경로
root_path = '/home/work/XAI_BASE/BASE_5주차'

# 2. 데이터셋 다운로드 (이미 있으면 생략)
od.download("https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset", data_dir=root_path)

# 3. 정확한 데이터셋 경로 (2번 중첩된 폴더 반영)
dataset_path = os.path.join(
    root_path,
    'supervisely-filtered-segmentation-person-dataset',
    'supervisely_person_clean_2667_img',
    'supervisely_person_clean_2667_img'
)

# 4. 이미지/마스크 폴더 지정
image_dir = os.path.join(dataset_path, 'images')
mask_dir = os.path.join(dataset_path, 'masks')

# 5. 파일 리스트 확인 및 정렬
image_list = sorted(os.listdir(image_dir))
mask_list = sorted(os.listdir(mask_dir))

# 6. train/val/test 분할
X_train, X_temp, y_train, y_temp = train_test_split(image_list, mask_list, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 7. 저장 경로 생성
save_root = os.path.join(root_path, 'npy_split')
os.makedirs(save_root, exist_ok=True)
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(save_root, split), exist_ok=True)

# 8. .npy 저장 함수
def save_npy_split(x_list, y_list, split_name):
    save_dir = os.path.join(save_root, split_name)
    for i, (x_name, y_name) in enumerate(zip(x_list, y_list)):
        # RGB 입력 이미지
        img = np.asarray(Image.open(os.path.join(image_dir, x_name)).convert('RGB'))  

        # 마스크: 흑백으로 로드 → 정규화(0~1) → 1채널로 reshape
        mask = Image.open(os.path.join(mask_dir, y_name)).convert('L')                 
        mask = np.array(mask).astype(np.float32) / 255.0                              
        mask = mask[:, :, np.newaxis]                                            

        # 저장
        np.save(os.path.join(save_dir, f'input_{i:04d}.npy'), img)
        np.save(os.path.join(save_dir, f'label_{i:04d}.npy'), mask)

    print(f"{split_name} 세트 저장 완료: {len(x_list)}개")

# 9. 저장 실행
save_npy_split(X_train, y_train, 'train')
save_npy_split(X_val, y_val, 'val')
save_npy_split(X_test, y_test, 'test')