# 📁 Step 1: data_read.py 🥸
# 이미지 데이터를 불러와 train/val/test로 나누고 .npy로 저장

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 디렉토리 설정
dir_data = './datasets'

# 파일 이름 설정
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# tif 파일 열기 (다중 프레임 이미지)
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size         # 이미지 크기
nframe = img_label.n_frames     # 프레임 개수 (3D 이미지)

# 학습/검증/테스트 분할 개수 설정
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 저장 폴더 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 폴더가 없으면 생성
for dir_path in [dir_save_train, dir_save_val, dir_save_test]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 프레임 인덱스 무작위 셔플
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# 데이터 저장 (학습용)
offset_nframe = 0
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])     # 셔플된 인덱스로 프레임 이동
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)                  # 프레임 → numpy 배열로 변환
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, f'label_{i:03d}.npy'), label_)  # 저장
    np.save(os.path.join(dir_save_train, f'input_{i:03d}.npy'), input_)

# 검증용
offset_nframe = nframe_train
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_val, f'input_{i:03d}.npy'), input_)

# 테스트용
offset_nframe = nframe_train + nframe_val
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, f'label_{i:03d}.npy'), label_)
    np.save(os.path.join(dir_save_test, f'input_{i:03d}.npy'), input_)

# 마지막 프레임 시각화 (테스트용)
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')
plt.show()