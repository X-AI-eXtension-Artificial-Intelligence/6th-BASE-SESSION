## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image   # 파이썬에서 이미지 처리를 위한 라이브러리 
                        # Image.open 로 이미지 열기. 리사이즈, 회전, 자르기 등의 작업 가능 
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'  # train 24장, val 3장, test 3장이 있음. label과 

name_label = 'train-labels.tif'  # 이미지 label 지정 
name_input = 'train-volume.tif'  # input 이미지 이름 지정 

img_label = Image.open(os.path.join(dir_data, name_label))  # 경로와 파일명을 결합해 이미지 열기 
img_input = Image.open(os.path.join(dir_data, name_input)) 

ny, nx = img_label.size  # 가로 세로 크기 받아오기 
nframe = img_label.n_frames  # 프레임 수 받아오기. -> 추후 프레임을 하나씩 처리할 수 있게 준비 

## .tif 이미지 안에는 여러 장의 이미지(프레임)가 들어 있으니까
nframe_train = 24  # 훈련 프레임 24 
nframe_val = 3  # 검증 프레임 3 
nframe_test = 3  # 실험 프레임 3  로 사용하겠다! 

dir_save_train = os.path.join(dir_data, 'train')  # 저장하기 위한 경로 설정 
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

if not os.path.exists(dir_save_train):  # 폴더가 없으면 생성하기 
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##
id_frame = np.arange(nframe)  # 프레임에 인덱스 부여 
np.random.shuffle(id_frame)  # 프레임 인덱스 셔플 

##
offset_nframe = 0

for i in range(nframe_train):  # train데이터 저장. 24였음 
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)  # numpy 바이너리 파일로 변환 
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)  # .npy로 저장 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
offset_nframe = nframe_train  # 훈련 프레임 다음 프레임을 쓰려고 변수 할당 

for i in range(nframe_val):  # 3장의 검증용 프레임 저장 
    img_label.seek(id_frame[i + offset_nframe])  # 훈련 프레임 다음 프레임 
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

##
offset_nframe = nframe_train + nframe_val  # 평가 프레임 저장 

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

##
plt.subplot(121)  # 1행 2열 중 1번째 그림 
plt.imshow(label_, cmap='gray')  # label_ 이미지를 흑백(cmap='gray')으로 시각화.
plt.title('label')

plt.subplot(122)
plt.imshow(input_, cmap='gray')
plt.title('input')

plt.show()








