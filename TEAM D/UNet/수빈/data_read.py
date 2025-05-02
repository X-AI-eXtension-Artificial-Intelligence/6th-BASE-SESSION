import deeplake
import numpy as np
import matplotlib.pyplot as plt

# Deep Lake에서 데이터 로드
train_ds = deeplake.load('hub://activeloop/drive-train')
test_ds = deeplake.load('hub://activeloop/drive-test')

print("train_ds tensors:", train_ds.tensors.keys())
print("test_ds tensors:", test_ds.tensors.keys())



# train 세트에서 샘플 하나 가져오기
train_sample = train_ds[0]
train_image = train_sample['rgb_images'].numpy() / 255.0
train_mask = train_sample['manual_masks/mask'].numpy() / 255.0

# test 세트에서 샘플 하나 가져오기
test_sample = test_ds[0]
test_image = test_sample['rgb_images'].numpy() / 255.0
test_mask = test_sample['masks'].numpy() / 255.0

# train 시각화
plt.figure(figsize=(10, 5))
plt.subplot(2, 2, 1)
plt.imshow(train_image)
plt.title('Train Image')

plt.subplot(2, 2, 2)
plt.imshow(train_mask[:, :, 0], cmap='gray')
plt.title('Train Mask')

# test 시각화
plt.subplot(2, 2, 3)
plt.imshow(test_image)
plt.title('Test Image')

plt.subplot(2, 2, 4)
plt.imshow(test_mask[:, :, 0], cmap='gray') 
plt.title('Test Mask')

plt.tight_layout()
plt.show()
