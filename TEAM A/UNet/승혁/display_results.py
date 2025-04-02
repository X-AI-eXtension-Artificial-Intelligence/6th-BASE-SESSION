import os
import numpy as np
import matplotlib.pyplot as plt

## 결과 디렉토리 경로 설정 (모델 추론 결과들이 저장된 경로)
result_dir = '/result/numpy'

# 디렉토리 안에 있는 파일 목록 불러오기
lst_data = os.listdir(result_dir)

# 파일 이름이 'label'로 시작하는 것들만 추출 (정답 이미지)
lst_label = [f for f in lst_data if f.startswith('label')]

# 파일 이름이 'input'으로 시작하는 것들만 추출 (입력 이미지)
lst_input = [f for f in lst_data if f.startswith('input')]

# 파일 이름이 'output'으로 시작하는 것들만 추출 (모델 예측 결과)
lst_output = [f for f in lst_data if f.startswith('output')]

# 파일 리스트를 이름 기준으로 정렬 (ex: label_000.npy, label_001.npy, ...)
lst_label.sort()
lst_input.sort()
lst_output.sort()

## 시각화할 이미지의 인덱스 선택 (여기선 첫 번째 이미지 선택)
id = 0

# numpy 파일 불러오기
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## 시각화
plt.subplot(131)                      # 1행 3열 중 첫 번째 subplot
plt.imshow(input, cmap='gray')       # 입력 이미지 (입력 채널 1개니까 회색조)
plt.title('input')                   # 제목

plt.subplot(132)                      # 두 번째 subplot
plt.imshow(label, cmap='gray')       # 정답 이미지
plt.title('label')

plt.subplot(133)                      # 세 번째 subplot
plt.imshow(output, cmap='gray')      # 예측 결과 이미지
plt.title('output')








