import numpy as np

class Perceptron:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Đảm bảo nhãn là 0 và 1 hoặc -1 và 1
        y_encoded = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(linear_output)
                
                # Cập nhật trọng số nếu dự đoán sai
                if y_encoded[idx] * y_predicted <= 0:
                    update = self.lr * y_encoded[idx]
                    self.weights += update * x_i
                    self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return np.where(linear_output >= 0, 1, 0)

# Ví dụ kiểm thử
if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # Cổng AND

    model = Perceptron(lr=0.1, n_iters=10)
    model.fit(X, y)
    print("Dự đoán cổng AND:", model.predict(X))