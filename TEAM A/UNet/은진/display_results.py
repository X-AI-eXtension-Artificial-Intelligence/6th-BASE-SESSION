# 📁 Step 6: display_results.py 👹
# 테스트 결과로 저장된 input/label/output 이미지들을 시각화해서 비교

import os
import numpy as np
import matplotlib.pyplot as plt

# 결과 파일이 저장된 디렉토리 설정
result_dir = './results/numpy'

# 결과 디렉토리 안의 모든 파일 리스트
datalist = os.listdir(result_dir)

# label, input, output 파일 분류
lst_label = [f for f in datalist if f.startswith('label')]
lst_input = [f for f in datalist if f.startswith('input')]
lst_output = [f for f in datalist if f.startswith('output')]

# 정렬 (파일 이름 순서대로 맞춰야 함)
lst_label.sort()
lst_input.sort()
lst_output.sort()

# 시각화할 인덱스 선택 (여기선 첫 번째 이미지)
id = 0

# 각각 numpy 배열로 불러오기
label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

# 세 개의 이미지를 나란히 출력
plt.subplot(131)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')

plt.show()
plt.savefig('comparison.png') 