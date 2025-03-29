from sklearn.cluster import KMeans
import numpy as np
import joblib  # 모델 저장 및 로드를 위한 라이브러리

# 적당한 데이터를 생성하거나 로드
data = np.random.rand(1000, 3) * 255
data = data.astype(int)

# KMeans 모델 생성 및 훈련
kmeans = KMeans(n_clusters=10, random_state=42)
kmeans.fit(data)
  
# 모델 저장
joblib.dump(kmeans, 'kmeans_model.pkl')
