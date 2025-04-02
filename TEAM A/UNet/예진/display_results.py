# 모델 예측 결과 시각화해주는 코드

import os                         
import numpy as np                
import matplotlib.pyplot as plt   

# 결과 파일들이 저장된 경로 설정
result_dir = './results/numpy'    # 예측 결과(.npy)들이 저장된 폴더

# 폴더 안의 모든 파일 이름 가져오기
lst_data = os.listdir(result_dir)

# 파일 이름별로 분류
lst_label = [f for f in lst_data if f.startswith('label')]   # 정답 마스크 파일 목록
lst_input = [f for f in lst_data if f.startswith('input')]   # 입력 이미지 파일 목록
lst_output = [f for f in lst_data if f.startswith('output')] # 모델 예측 결과 파일 목록

# 이름 기준으로 정렬 (순서 맞추기 위해)
lst_label.sort()
lst_input.sort()
lst_output.sort()

# 시각화할 이미지 인덱스 선택 (예: 0번째)
id = 0

# 각 결과 파일 불러오기
label = np.load(os.path.join(result_dir, lst_label[id]))    # 정답 마스크 불러오기
input = np.load(os.path.join(result_dir, lst_input[id]))    # 입력 이미지 불러오기
output = np.load(os.path.join(result_dir, lst_output[id]))  # 예측 결과 불러오기

# 이미지 시각화 (1행 3열: 입력 / 정답 / 예측)
plt.figure(figsize=(12, 4))  # 전체 그림 크기 지정

plt.subplot(131)                  # 첫 번째 그림 (입력)
plt.imshow(input, cmap='gray')    # 흑백 이미지로 출력
plt.title('Input')                
plt.axis('off')                   # 축 제거

plt.subplot(132)                  # 두 번째 그림 (정답 라벨)
plt.imshow(label, cmap='gray')
plt.title('Label')
plt.axis('off')

plt.subplot(133)                  # 세 번째 그림 (모델 예측 결과)
plt.imshow(output, cmap='gray')
plt.title('Output')
plt.axis('off')

plt.tight_layout()
plt.show()
