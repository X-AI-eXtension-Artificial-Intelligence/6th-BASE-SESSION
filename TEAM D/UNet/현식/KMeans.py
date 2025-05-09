import numpy as np
from sklearn.cluster import KMeans

# RGB 색상 값(0~255)에서 무작위로 1000개의 색상 샘플을 만든 것
num_items = 1000
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)

num_classes = 10
label_model = KMeans(n_clusters = num_classes)
label_model.fit(color_array)