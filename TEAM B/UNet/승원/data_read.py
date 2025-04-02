## 필요한 패키지 등록
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

## 데이터 불러오기
# 데이터셋 디렉토리 설정
dir_data = './datasets'

# 라벨 및 입력 이미지 파일명 지정
name_label = 'train-labels.tif'
name_input = 'train-volume.tif'

# 이미지 파일 열기
img_label = Image.open(os.path.join(dir_data, name_label))
img_input = Image.open(os.path.join(dir_data, name_input))

# 이미지 크기 및 프레임 수 확인
ny, nx = img_label.size  # 이미지 크기 (너비, 높이)
nframe = img_label.n_frames  # 전체 프레임 수

## 데이터 분할 개수 설정
nframe_train = 24  # 학습 데이터 개수
nframe_val = 3     # 검증 데이터 개수
nframe_test = 3    # 테스트 데이터 개수

# 학습, 검증, 테스트 데이터를 저장할 디렉토리 경로 설정
dir_save_train = os.path.join(dir_data, 'train')  # 학습 데이터 저장 디렉토리
dir_save_val = os.path.join(dir_data, 'val')      # 검증 데이터 저장 디렉토리
dir_save_test = os.path.join(dir_data, 'test')    # 테스트 데이터 저장 디렉토리

# 디렉토리가 존재하지 않으면 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)
if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)
if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)

## 데이터 프레임 순서 섞기
id_frame = np.arange(nframe)  # 프레임 인덱스 배열 생성
np.random.shuffle(id_frame)   # 프레임 순서를 랜덤하게 섞음

## 학습 데이터 저장
offset_nframe = 0  # 시작 프레임 오프셋 설정
for i in range(nframe_train):
    img_label.seek(id_frame[i + offset_nframe])  # 해당 프레임으로 이동
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)  # 이미지 데이터를 NumPy 배열로 변환
    input_ = np.asarray(img_input)

    # NumPy 배열을 파일로 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)

## 검증 데이터 저장
offset_nframe = nframe_train  # 학습 데이터 이후 프레임부터 시작
for i in range(nframe_val):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)

## 테스트 데이터 저장
offset_nframe = nframe_train + nframe_val  # 검증 데이터 이후 프레임부터 시작
for i in range(nframe_test):
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label) # 이미지 데이터를 NumPy 배열로 변환
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)

## 저장된 데이터 시각화
plt.subplot(121)  # 첫 번째 서브플롯
plt.imshow(label_, cmap='gray')  # 라벨 이미지 시각화
plt.title('label')  # 라벨 이미지 제목 설정

plt.subplot(122)  # 두 번째 서브플롯
plt.imshow(input_, cmap='gray')  # 입력 이미지 시각화
plt.title('input')  # 입력 이미지 제목 설정

plt.show()  # 시각화된 이미지를 화면에 표시