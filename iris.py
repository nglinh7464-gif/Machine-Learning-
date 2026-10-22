import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

class MyKMeans:
    def __init__(self, K=3, max_iters=100, tol=1e-4):
        self.K = K
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # 1. Khởi tạo ngẫu nhiên K tâm cụm từ các điểm dữ liệu
        random_indices = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = X[random_indices]

        for _ in range(self.max_iters):
            # 2. Gán nhãn cho từng điểm dựa vào khoảng cách gần nhất tới tâm cụm
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            self.labels = np.argmin(distances, axis=1)

            # 3. Tính lại tâm cụm mới
            new_centroids = np.array([X[self.labels == k].mean(axis=0) if np.sum(self.labels == k) > 0 else self.centroids[k] for k in range(self.K)])

            # 4. Kiểm tra điều kiện hội tụ (nếu tâm không đổi)
            if np.all(np.abs(new_centroids - self.centroids) < self.tol):
                break
                
            self.centroids = new_centroids

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

# 1. Tải tập dữ liệu Iris và chỉ lấy 2 cột đầu tiên (sepal length, sepal width)[cite: 7]
iris = load_iris()
X = iris.data[:, :2] 
y_true = iris.target

# 2. Huấn luyện MyKMeans với K=3
kmeans = MyKMeans(K=3, max_iters=100)
kmeans.fit(X)
y_pred = kmeans.labels

# 3. Tính chỉ số Rand Index (Adjusted Rand Index)
ari = adjusted_rand_score(y_true, y_pred)
print(f"Chỉ số Adjusted Rand Index (ARI) so với nhãn thật: {ari:.4f}")

# 4. Vẽ biểu đồ phân tán so sánh phân cụm và nhãn thật[cite: 7]
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Biểu đồ kết quả K-Means thủ công
axes[0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=50, edgecolor='k')
axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Tâm cụm')
axes[0].set_title('Kết quả phân cụm bằng MyKMeans (K=3)')
axes[0].set_xlabel('Sepal length')
axes[0].set_ylabel('Sepal width')
axes[0].legend()
axes[0].grid(True)

# Biểu đồ nhãn thực tế của Iris
axes[1].scatter(X[:, 0], X[:, 1], c=y_true, cmap='Set1', s=50, edgecolor='k')
axes[1].set_title('Nhãn thực tế (Ground Truth)')
axes[1].set_xlabel('Sepal length')
axes[1].set_ylabel('Sepal width')
axes[1].grid(True)

plt.tight_layout()
plt.show()