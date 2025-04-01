## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = '/content/drive/MyDrive/XAI 코드 구현/unet/datasets'   # 데이터셋이 저장된 디렉토리 경로

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# 라벨 및 입력 이미지 파일 불러오기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))
# 이미지 크기 및 프레임 개수 확인
ny, nx = img_label.size # 이미지의 너비(nx)와 높이(ny) 저장
nframe = img_label.n_frames # 이미지의 총 프레임 개수 저장 (멀티프레임 TIF 파일)

# 데이터셋 분할을 위한 프레임 개수 설정
nframe_train = 24
nframe_val = 3
nframe_test = 3

# 학습, 검증, 테스트 데이터를 저장할 디렉토리 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##
# 프레임을 무작위로 섞기
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# 학습 데이터 저장
offset_nframe = 0 # 시작 인덱스 설정

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) # 랜덤하게 선택된 프레임으로 이동
    img_input.seek(id_frame[i + offset_nframe]) # 입력 이미지도 동일한 프레임으로 이동

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)
    # NumPy 배열을 검증 데이터 디렉토리에 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## 검증 데이터 저장
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)









