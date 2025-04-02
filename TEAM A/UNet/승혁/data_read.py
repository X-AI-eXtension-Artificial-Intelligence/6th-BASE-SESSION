import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'   # 정답(label) 이미지 파일 이름
name_input = 'train-volume.tif'   # 입력(input) 이미지 파일 이름

# tif 이미지 열기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size       # 이미지 한 프레임의 크기 (width, height)
nframe = img_label.n_frames   # 전체 프레임 수

## train, val, test 이미지 데이터 개수 정하기
nframe_train = 24             
nframe_val = 3                
nframe_test = 3               

# 저장할 디렉토리 생성
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 존재하지 않으면 폴더 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 512x512x30 이미지에서 frame 순서를 랜덤하게 섞음
id_frame = np.arange(nframe)       # [0, 1, 2, ..., 29]
np.random.shuffle(id_frame)        # 무작위로 섞음

## 학습용 데이터 24개 저장
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])  # 랜덤 순서로 프레임 선택
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)   # label 이미지 배열로 변환
    input_ = np.asarray(img_input)   # input 이미지 배열로 변환

    # numpy 배열로 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

a = label_.shape
a

## 검증용 데이터 3개 저장
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 테스트용 데이터 3개 저장
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 추출한 label/input 이미지 시각화
plt.subplot(121)
plt.imshow(label_, cmap='gray')   # label 이미지 출력
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')   # input 이미지 출력
plt.title('input')

plt.show()









