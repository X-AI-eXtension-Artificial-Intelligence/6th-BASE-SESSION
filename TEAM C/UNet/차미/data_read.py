## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

## 데이터셋을 나눠서 저장
nframe_train = 24
nframe_val = 3
nframe_test = 3

## 데이터가 저장될 디렉토리 지정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

## 디렉토리 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 실제로 저장
## 데이터셋을 랜덤하게 저장하기 위해 셔플
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## 데이터셋 저장
offset_nframe = 0

for i in range(nframe_train):
    ## i(현재 인덱스) / offset_nframe(이동하고자 하는 프레임 수)
    ## 현재 위치에서 offset_nframe만큼 떨어진 프레임으로 이동하기 위해 id_frame에서 해당 프레임의 식별자를 찾고 그 프레임을 img_label 객체를 통해 찾아가도록 함
    img_label.seek(id_frame[i + offset_nframe])  
    img_input.seek(id_frame[i + offset_nframe])
    ## seek.(): 특정 위치를 찾아가기 위해 사용되는 함수

    ## np.asarray(): 입력 데이터를 넘파이 배열 (ndarray)로 변환해줌
    label_ = np.asarray(img_label)  
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## 
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

##
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 생성된 데이터셋을 맷플랏라이브러리를 통해 출력
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()
## 왼쪽은 세그맨테이션 이미지 (검정 부분은 1, 하얀 부분은 0)
## 오른쪽은 입력 이미지