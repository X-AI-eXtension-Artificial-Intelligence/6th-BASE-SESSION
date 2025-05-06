from PIL import Image
import numpy as np
import os

# 입력 이미지 경로
input_image_path = 'Han_river.jpg'  # 변환할 한강 이미지 파일명
output_npy_path = 'hangang.npy'     # 저장할 .npy 파일명

# 이미지 불러오기 (흑백으로 변환)
img = Image.open(input_image_path).convert('L')  # 'L'은 흑백

# 크기 조정 (512x512로 변경)
img = img.resize((512, 512))

# 넘파이 배열로 변환
img_array = np.array(img)  # shape = (512, 512)

# 저장
np.save(output_npy_path, img_array)

print(f"변환 완료: {output_npy_path}, shape = {img_array.shape}")