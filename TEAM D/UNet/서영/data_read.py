import numpy as np
import os
import matplotlib.pyplot as plt
import deeplake

# Deep Lake 데이터셋 로드
ds = deeplake.load("hub://activeloop/drive-train")
print("Available tensors:", list(ds.tensors.keys()))

# 전체 프레임 수
nframe = len(ds["rgb_images"])

# train/val/test 자동 분할
nframe_train = int(nframe * 0.7)
nframe_val = int(nframe * 0.15)
nframe_test = nframe - nframe_train - nframe_val

print(f"총 {nframe}개 중 → train: {nframe_train}, val: {nframe_val}, test: {nframe_test}")

# 저장 경로
dir_save_train = "./data/train"
dir_save_val = "./data/val"
dir_save_test = "./data/test"

os.makedirs(dir_save_train, exist_ok=True)
os.makedirs(dir_save_val, exist_ok=True)
os.makedirs(dir_save_test, exist_ok=True)

# 인덱스 셔플
id_frame = np.arange(nframe)
np.random.shuffle(id_frame)

# --- 저장 함수 ---
def save_split(start_idx, count, save_dir):
    last_input, last_label = None, None
    for i in range(count):
        idx = int(id_frame[start_idx + i])
        input_ = ds["rgb_images"][idx].numpy()
        label_ = ds["manual_masks/mask"][idx].numpy()

        np.save(os.path.join(save_dir, f"input_{i:03d}.npy"), input_)
        np.save(os.path.join(save_dir, f"label_{i:03d}.npy"), label_)

        last_input, last_label = input_, label_  # 마지막 이미지 저장
    return last_input, last_label

# 데이터 저장 + 마지막 이미지 리턴
_, _ = save_split(0, nframe_train, dir_save_train)
_, _ = save_split(nframe_train, nframe_val, dir_save_val)
input_, label_ = save_split(nframe_train + nframe_val, nframe_test, dir_save_test)

# 시각화
# 시각화용 마지막 이미지 전처리
if label_.ndim == 3 and label_.shape[2] == 2:
    label_ = label_[:, :, 0]

if input_.ndim == 3 and input_.shape[2] == 2:
    input_ = input_[:, :, 0]

plt.subplot(121)
plt.imshow(label_, cmap='gray')
plt.title('label')

plt.subplot(122)
plt.imshow(input_)
plt.title('input')

plt.show()
