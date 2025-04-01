## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
dir_data = './datasets'  # 데이터가 저장된 디렉토리 경로 지정

name_label = 'train-labels.tif'  # 레이블 이미지 파일명
name_input = 'train-volume.tif'  # 입력 이미지 파일명

# 이미지 파일을 PIL 라이브러리로 열기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

# 레이블 이미지의 크기 (ny: 세로 크기, nx: 가로 크기)와 프레임 수(nframe)를 얻기
ny, nx = img_label.size
nframe = img_label.n_frames  # 이미지의 프레임 수 (3D 데이터의 각 슬라이스)



# 훈련, 검증, 테스트 데이터의 프레임 수 정의
nframe_train = 24  # 훈련 데이터 프레임 수
nframe_val = 3  # 검증 데이터 프레임 수
nframe_test = 3  # 테스트 데이터 프레임 수

# 훈련, 검증, 테스트 데이터 저장 경로 정의
dir_save_train = os.path.join(dir_data, 'train')
dir_save_val = os.path.join(dir_data, 'val')
dir_save_test = os.path.join(dir_data, 'test')

# 각 데이터 디렉토리가 존재하지 않으면 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

##
# 이미지 프레임에 대한 인덱스를 생성하여 섞기
id_frame = np.arange(nframe)  # 프레임 인덱스 배열 생성
np.random.shuffle(id_frame)  # 배열 섞기 (무작위로 샘플링)

##
# 훈련 데이터 저장
offset_nframe = 0  # 훈련 데이터의 시작 위치

# nframe_train만큼 반복하면서 훈련 데이터를 저장
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동
    img_input.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동

    label_ = np.asarray(img_label)  # 레이블 이미지를 배열로 변환
    input_ = np.asarray(img_input)  # 입력 이미지를 배열로 변환

    # 훈련 데이터로 저장 (파일명은 'label_000.npy'와 'input_000.npy' 형식)
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

##
# 검증 데이터 저장
offset_nframe = nframe_train  # 검증 데이터의 시작 위치 (훈련 데이터가 끝난 후부터 시작)

# nframe_val만큼 반복하면서 검증 데이터를 저장
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동
    img_input.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동

    label_ = np.asarray(img_label)  # 레이블 이미지를 배열로 변환
    input_ = np.asarray(img_input)  # 입력 이미지를 배열로 변환

    # 검증 데이터로 저장 (파일명은 'label_000.npy'와 'input_000.npy' 형식)
    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

##
# 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val  # 테스트 데이터의 시작 위치 (훈련+검증 데이터가 끝난 후부터 시작)

# nframe_test만큼 반복하면서 테스트 데이터를 저장
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동
    img_input.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동

    label_ = np.asarray(img_label)  # 레이블 이미지를 배열로 변환
    input_ = np.asarray(img_input)  # 입력 이미지를 배열로 변환

    # 테스트 데이터로 저장 (파일명은 'label_000.npy'와 'input_000.npy' 형식)
    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

##
# 마지막으로 하나의 입력 이미지와 레이블 이미지를 출력
plt.subplot(121)  # 첫 번째 서브플롯
plt.imshow(label_, cmap='gray')  # 레이블 이미지를 회색조로 출력
plt.title('label')  # 서브플롯 제목: 'label'

plt.subplot(122)  # 두 번째 서브플롯
plt.imshow(input_, cmap='gray')  # 입력 이미지를 회색조로 출력
plt.title('input')  # 서브플롯 제목: 'input'

plt.show()  # 플롯을 화면에 표시
