import os
import numpy as np
from PIL import Image

### 위암 병변(병이 있는 부위)을 이미지 속에서 찾아서 영역을 표시하는 작업 ###

# 1️⃣ 세그멘테이션 모델을 불러오고 (Attention U-Net),

# 2️⃣ 데이터셋을 로드해서 (npy or 이미지/마스크 쌍),

# 3️⃣ 모델에 이미지를 넣고 → 병변이 어디 있는지 마스크를 예측하게 학습하고,

# 4️⃣ 예측한 마스크와 정답 마스크를 비교해서 손실(loss)을 계산하고,

# 5️⃣ 이걸 계속 반복해서 모델의 성능을 점점 높이는 구조.

# 경로 설정
images_dir = './datasets/Kvasir-SEG/images'
masks_dir = './datasets/Kvasir-SEG/masks'

# 저장할 경로 (train 폴더로 예시)
save_dir = './datasets/train'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 이미지/마스크 리스트 불러오기
# ✅ 이미지 : 우리가 모델에 입력하는 실제 이미지
# ✅ 마스크 : 이미지에 대해 레이블(정답)을 표시한 이진/다중 클래스 이미지
image_list = sorted(os.listdir(images_dir))
mask_list = sorted(os.listdir(masks_dir))

# 변환 loop -> ✅ 이미지 & 마스크를 한 쌍씩 읽어서 반복 처리
for i, (img_name, mask_name) in enumerate(zip(image_list, mask_list)):
    # ✅ 이미지 로드 & 변환
    img = Image.open(os.path.join(images_dir, img_name)).convert('L')  # 흑백으로 변환
    mask = Image.open(os.path.join(masks_dir, mask_name)).convert('L')

    # numpy 배열로 변환
    img_np = np.array(img)
    mask_np = np.array(mask)

    # 저장
    np.save(os.path.join(save_dir, f'input_{i:03d}.npy'), img_np)
    np.save(os.path.join(save_dir, f'label_{i:03d}.npy'), mask_np)

    print(f'Saved: input_{i:03d}.npy & label_{i:03d}.npy')
