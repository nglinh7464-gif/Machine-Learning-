import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# 1. Tải và chuẩn bị dữ liệu
data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- PHẦN a: SVM Tuyến tính với các giá trị C khác nhau ---
print("--- PHẦN a: SVM TUYẾN TÍNH ---")
C_values = [0.1, 1, 10]
for c in C_values:
    svm_lin = SVC(kernel='linear', C=c, random_state=42)
    svm_lin.fit(X_train_scaled, y_train)
    y_pred = svm_lin.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    n_sv = svm_lin.n_support_
    print(f"C = {c} -> Độ chính xác: {acc:.4f} | Số lượng support vectors (mỗi lớp): {n_sv} | Tổng SV: {sum(n_sv)}")

# --- PHẦN b: SVM Kernel RBF, thay đổi C và gamma, trực quan hóa bằng PCA ---
print("\n--- PHẦN b: SVM KERNEL RBF & TRỰC QUAN HÓA ---")
# Giảm chiều dữ liệu xuống 2 chiều bằng PCA để vẽ đồ thị đường biên
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)

# Thử nghiệm các bộ tham số C và gamma
configs = [
    {'C': 1, 'gamma': 0.01, 'title': 'RBF (C=1, gamma=0.01)'},
    {'C': 1, 'gamma': 1.0, 'title': 'RBF (C=1, gamma=1.0)'},
    {'C': 100, 'gamma': 10.0, 'title': 'RBF (C=100, gamma=10.0 - Overfitting)'}
]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Tạo lưới để vẽ đường biên
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

for idx, config in enumerate(configs):
    # Huấn luyện trên dữ liệu 2D PCA
    clf = SVC(kernel='rbf', C=config['C'], gamma=config['gamma'])
    clf.fit(X_train_pca, y_train)
    
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = axes[idx]
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_title(config['title'])
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

plt.tight_layout()
plt.show()