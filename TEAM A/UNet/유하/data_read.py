## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'

name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

### .tif 파일을 열고 PIL 이미지 객체로 저장 
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size # 가로 너비, 세로 높이 형식인데, 이름 반대로 되어 있음 주의
nframe = img_label.n_frames # tif 파일은 여러 장의 이미지가 하나로 묶인 형식 -> .n>frame : 몇 장의 이미지가 들어있는지 알 수 있음

## 데이터 분할 설정
nframe_train = 24
nframe_val = 3
nframe_test = 3

## 저장할 폴더 경로 설정
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

## 폴더가 없을 시, 새로 만들기
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 프레임 인덱스 랜덤으로 변환 -> 데이터를 랜덤하게 섞어서 train, val, test 데이터로 나누기 위해 
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

## 훈련 데이터 저장
offset_nframe = 0

for i in range(nframe_train): # 24장만큼 반복 
    img_label.seek(id_frame[i + offset_nframe]) # 랜덤으로 섞인 순서로 접근하도록 함 
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) # 넘파이 배열로 변환
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## 검증 데이터 저장
offset_nframe = nframe_train

for i in range(nframe_val): # 24번부터 시작
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test): # 27번부터 시작
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 시각화 
plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()







