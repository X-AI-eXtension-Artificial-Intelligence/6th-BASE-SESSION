## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = '/content/drive/MyDrive/XAI 코드 구현/unet/datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# 라벨과 입력 이미지 파일 load
img_label = Image.open(os.path.join(dir_data, name_label)) 
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size # image 너비(nx), 높이(ny)
nframe = img_label.n_frames # image 총 frame의 수 (tif는 멀티프레임)

##
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 학습, 검증, 테스트 데이터 저장할 디렉토리 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 존재하지 않으면 만들어줌
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## frame 섞기 (shuffle)
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# 학습 데이터 저장
offset_nframe = 0 # 시작 인덱스

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) 
    img_input.seek(id_frame[i + offset_nframe])
    # numpy 배열로 변환
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# 검증 데이터 저장
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])
    # numpy 배열로 변환
    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)









