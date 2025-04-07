"""
논문에서 언급되었던 ISBI 2012 EM Segmentation Membrane Dataset

train-volumne.tif 파일 -> 훈련 이미지(512*512 grayscale 세포 Image) 30장
train-labels.tif 파일 -> 정답에 해당하는 Segmentation Map(배경과 세포를 구분하며, 배경은 1이고 세포는 255)
test-volumne.tif 파일 -> 테스트 이미지
"""

# 라이브러리 
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 데이터 불러오기
dir_data = './dataset' #경로 설정
name_label = 'train-labels.tif' #label 파일
name_input = 'train-volume.tif' #train 이미지 파일

img_label = Image.open(os.path.join(dir_data, name_label)) 
img_input = Image.open(os.path.join(dir_data, name_input))

ny, nx = img_label.size
nframe = img_label.n_frames #이미지 개수 확인

# 이미지(frame)개수 설정
nframe_train = 24 
nframe_val = 3
nframe_test = 3

# 경로명 설정(기존 dataset폴더에 하위 폴더로 train, val, test)
# 이미지 나누어서 넣을 각 train,test,val 폴더 생성
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 만약 기존에 따로 설정되지 않았다면 폴더 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

# 전체 이미지 프레임 번호, 인덱스 배열 만들고 [0,1,2...30] -> 랜덤으로 순서를 shuffle
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# tif의 멀티 프레임 이미지를 이미지 개별 파일로 분할해서 저장
# 개별 train 이미지를 npy 파일로 저장(grayscale)

# offset 0으로 초기 설정
offset_nframe = 0

# train 이미지 저장 - 지정 횟수만큼 반복
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe]) # 섞인 랜덤 배열에서 i번째 프레임 이미지 선택(.seek -> 포인터만 해당 파일에 놓는 역할)
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) #해당 인덱스 번호 label numpy array 형태로 변환
    input_ = np.asarray(img_input) #해당 인덱스 번호 train image numpy array 형태로 변환

    # 지정경로에 npy 형태로 각각 이미지, 라벨 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_) 
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

# valid도 동일하게 진행(offset을 nframe_train로 설정하면, train 개수 그 이후부터)
offset_nframe = nframe_train

for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

# test 추출도 동일하게 진행(offset을 nframe_train + nframe_val로 설정하면, 나머지 3개 추출 가능)
offset_nframe = nframe_train + nframe_val

for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)