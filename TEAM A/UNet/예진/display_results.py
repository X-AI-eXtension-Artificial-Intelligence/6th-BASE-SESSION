# display_results.py
import os
import numpy as np
import matplotlib.pyplot as plt

result_dir = './results/numpy'
lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if f.startswith('label')]
lst_input = [f for f in lst_data if f.startswith('input')]
lst_output = [f for f in lst_data if f.startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.imshow(input.squeeze(), cmap='gray')
plt.title('Input')
plt.axis('off')

plt.subplot(132)
plt.imshow(label, cmap='tab10')  # ▼ 변경: 다중 클래스용 컬러맵 적용
plt.title('Label')
plt.axis('off')

plt.subplot(133)
plt.imshow(output, cmap='tab10')  # ▼ 변경: 다중 클래스용 컬러맵 적용
plt.title('Output')
plt.axis('off')

plt.tight_layout()
plt.show()
