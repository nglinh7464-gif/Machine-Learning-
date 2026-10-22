import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# 1. Tải dữ liệu Iris
iris = load_iris()
# Chỉ lấy 2 thuộc tính đầu tiên và 2 lớp đầu tiên (Setosa & Versicolor)
X = iris.data[:100, :2]
y = iris.target[:100]

# Chuẩn hóa dữ liệu để Gradient Descent hội tụ ổn định
X_scaled = (X - X.mean(axis=0)) / X.std(axis=0)

class LogisticRegressionGD:
    def __init__(self, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.losses = []

        for _ in range(self.epochs):
            linear_model = np.dot(X, self.w) + self.b
            y_predicted = self._sigmoid(linear_model)

            # Hàm mất mát Binary Cross-Entropy
            eps = 1e-15
            loss = -np.mean(y * np.log(y_predicted + eps) + (1 - y) * np.log(1 - y_predicted + eps))
            self.losses.append(loss)

            # Gradient Descent
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.w -= self.lr * dw
            self.b -= self.lr * db

# 2. Thử nghiệm với 3 Learning Rate khác nhau
learning_rates = [0.01, 0.1, 1.0]
models = {}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Vẽ đồ thị Loss theo Epoch[cite: 5]
for lr in learning_rates:
    model = LogisticRegressionGD(lr=lr, epochs=200)
    model.fit(X_scaled, y)
    models[lr] = model
    axes[0].plot(range(200), model.losses, label=f'lr = {lr}')

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Đồ thị Loss theo Epoch')
axes[0].legend()
axes[0].grid(True)

# Biểu diễn đường biên phân lớp trên mặt phẳng 2D[cite: 5]
colors = ['red', 'blue']
for i, color in zip([0, 1], colors):
    axes[1].scatter(X_scaled[y == i, 0], X_scaled[y == i, 1], label=iris.target_names[i], color=color)

x1_min, x1_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
x1_vals = np.linspace(x1_min, x1_max, 100)

for lr in learning_rates:
    model = models[lr]
    # Phương trình đường biên: w1*x1 + w2*x2 + b = 0 => x2 = -(w1*x1 + b) / w2
    x2_vals = -(model.w[0] * x1_vals + model.b) / model.w[1]
    axes[1].plot(x1_vals, x2_vals, label=f'Ranh giới (lr={lr})')

axes[1].set_xlabel('Sepal length (scaled)')
axes[1].set_ylabel('Sepal width (scaled)')
axes[1].set_title('Đường biên phân lớp trên mặt phẳng 2D')
axes[1].set_ylim(X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5)
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()