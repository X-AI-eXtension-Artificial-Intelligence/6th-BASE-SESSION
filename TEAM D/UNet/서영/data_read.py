## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = 'Unet_3주차/datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames

## train, val, test 이미지 데이터 개수 정하기
## 총 이미지 데이터 수가 30개임임
nframe_train = 24
nframe_val = 3
nframe_test = 3

dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 512x512x30에서 선별할 frame shuffle해서 추출출
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)
'slice형태로 이미지가 묶여서 나오기 때문에 별개의 이미지로 분리하는 작업 필요'
'3차원의 별도의 이미지로 제공되면 분리 코드는 필요 없음'
'RGB면 상관 없는데, Gray Scale이면 분리 코드 필요 CT나 MRI 쪽 의료영상 데이터는 대부분 Gray Scale'

## 24개의 train 데이터 추출
offset_nframe = 0

for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

'왜 numpy 형태로 저장할까?'
## -> 이미지 shape이 (512,512)
a = label_.shape
a
'(H,W) =(이미지 높이, 이미지 너비)'
'-> (H,W,C) =(이미지 높이, 이미지 너비, 이미지 채널)'
'numpy 형식 이용하면 (H,W)에서 축 하나 쉽게 늘릴 수 있음음'

## 3개의 val 데이터 추출
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 3개의 test 데이터 추출
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

##
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()









