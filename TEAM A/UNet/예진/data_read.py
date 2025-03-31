# 이미지 데이터를 학습, 검증, 테스트용으로 나눠서 저장하는 코드

import os                        
import numpy as np              
from PIL import Image           
import matplotlib.pyplot as plt 


dir_data = './datasets'           # 이미지 파일 저장된 폴더

# tif 파일: 여러 장의 이미지가 들어 있는 멀티프레임 이미지
name_label = 'train-labels.tif'   # 정답 이미지 파일
name_input = 'train-volume.tif'   # 입력 이미지 파일


img_label = Image.open(os.path.join(dir_data, name_label))  # label 이미지 열기
img_input = Image.open(os.path.join(dir_data, name_input))  # input 이미지 열기


ny, nx = img_label.size        # 각 이미지의 크기: 너비(nx), 높이(ny)
nframe = img_label.n_frames    # 전체 프레임 수 

# ex. 256x256 크기의 이미지가 30장 있다면 -> ny=256, nx=256, nframe=30


nframe_train = 24  # 학습용 프레임 장 수
nframe_val = 3     # 검증용 프레임 장 수 
nframe_test = 3    # 테스트용 프레임 장 수


# 데이터 저장 위치 지정
dir_save_train = os.path.join(dir_data, 'train')  # 학습 데이터 저장 경로
dir_save_val = os.path.join(dir_data, 'val')      # 검증 데이터 저장 경로
dir_save_test = os.path.join(dir_data, 'test')    # 테스트 데이터 저장 경로

# 각 경로 존재하지 않으면 새로 생성
if not os.path.exists(dir_save_train):
    os.makedirs(dir_save_train)

if not os.path.exists(dir_save_val):
    os.makedirs(dir_save_val)

if not os.path.exists(dir_save_test):
    os.makedirs(dir_save_test)


# 프레임 인덱스를 무작위 섞기
# 왜? 모델이 특정 순서에 의존하지 않도록 하기 위해서
id_frame = np.arange(nframe)    # 0부터 nframe-1번까지 번호 만들고 [0, 1, 2, ..., 29] 순서 shuffle
np.random.shuffle(id_frame)     



# 학습용 데이터 저장

offset_nframe = 0  # 시작 위치

for i in range(nframe_train):  # 프레임 선택하고 배열로 바꿔서 train 폴더에 저장

    # seek(n): 멀티프레임 이미지에서 n번째 이미지를 선택
    img_label.seek(id_frame[i + offset_nframe])  # i번째 label 프레임 선택
    img_input.seek(id_frame[i + offset_nframe])  # i번째 input 프레임 선택

    label_ = np.asarray(img_label)  # 이미지 -> NumPy 배열 변환
    input_ = np.asarray(img_input)

    # 'label_000.npy' 같은 형식으로 저장
    np.save(os.path.join(dir_save_train, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_train, 'input_%03d.npy' % i), input_)


# 검증용 데이터 저장

offset_nframe = nframe_train  # 학습용 다음부터 시작

for i in range(nframe_val):  # 프레임 선택하고 배열로 바꿔서 val 폴더에 저장
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_val, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_val, 'input_%03d.npy' % i), input_)


# 테스트용 데이터 저장

offset_nframe = nframe_train + nframe_val  # 검증용 다음부터 시작

for i in range(nframe_test):  # 프레임 선택하고 배열로 바꿔서 test 폴더에 저장
    img_label.seek(id_frame[i + offset_nframe])
    img_input.seek(id_frame[i + offset_nframe])

    label_ = np.asarray(img_label)
    input_ = np.asarray(img_input)

    np.save(os.path.join(dir_save_test, 'label_%03d.npy' % i), label_)
    np.save(os.path.join(dir_save_test, 'input_%03d.npy' % i), input_)


# 왜 tif 말고 npy로 저장? 모델 학습에 바로 쓰기 좋게 최적화된 배열 형식 -> 속도 빠르고 코딩 쉬움

# ------------------------------
# 시각화

plt.subplot(121)                      # 첫 번째 이미지 영역
plt.imshow(label_, cmap='gray')       # label 이미지 출력 (회색)
plt.title('label')                    

plt.subplot(122)                      # 두 번째 이미지 영역
plt.imshow(input_, cmap='gray')       # input 이미지 출력 (회색)
plt.title('input')                    

plt.show()                            
