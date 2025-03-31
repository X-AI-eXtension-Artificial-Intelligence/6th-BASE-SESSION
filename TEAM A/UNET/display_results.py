import os
import numpy as np
import matplotlib.pyplot as plt


##
result_dir = './results/numpy'  # 경로 설정 

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

##
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

# 한 눈에 보기 
plt.subplot(131)
plt.imshow(input, cmap='gray')  # input 이미지 
plt.title('input')

plt.subplot(132)
plt.imshow(label, cmap='gray')  # label 이미지 
plt.title('label')

plt.subplot(133)
plt.imshow(output, cmap='gray')  # 분류이미지 
plt.title('output')

# plt.show()  # 벡엔드 설정상 그림이 안뜰 수 있음 -> 바로 저장하는게 좋을 선택 
plt.savefig(f"{result_dir}/fig1.png", dpi=300)







