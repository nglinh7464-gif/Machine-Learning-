import numpy as np

class NeuralNetwork:
    def __init__(self, layers, task='classification', lr=0.01):
        """
        layers: Danh sách số lượng nơ-ron từng tầng, ví dụ [input_dim, hidden1, hidden2, output_dim]
        task: 'classification' hoặc 'regression'
        """
        self.layers = layers
        self.task = task
        self.lr = lr
        self.weights = []
        self.biases = []

        # Khởi tạo trọng số và bias
        for i in range(len(layers) - 1):
            w = np.random.randn(layers[i], layers[i+1]) * 0.01
            b = np.zeros((1, layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def forward(self, X):
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._sigmoid(z)
            activations.append(a)

        # Tầng đầu ra
        z_out = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        if self.task == 'classification':
            a_out = self._sigmoid(z_out)
        else:
            a_out = z_out # Tuyến tính cho hồi quy
        activations.append(a_out)
        
        return activations

    def fit(self, X, y, epochs=1000):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        for _ in range(epochs):
            # Forward pass
            activations = self.forward(X)
            
            # Backpropagation
            deltas = [None] * len(self.weights)
            
            # Đạo hàm tầng đầu ra
            error = activations[-1] - y
            if self.task == 'classification':
                deltas[-1] = error * self._sigmoid_derivative(activations[-1])
            else:
                deltas[-1] = error

            # Lan truyền ngược cho các tầng ẩn
            for i in reversed(range(len(self.weights) - 1)):
                error_hidden = np.dot(deltas[i+1], self.weights[i+1].T)
                deltas[i] = error_hidden * self._sigmoid_derivative(activations[i+1])

            # Cập nhật weights và biases
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * np.dot(activations[i].T, deltas[i]) / len(X)
                self.biases[i] -= self.lr * np.sum(deltas[i], axis=0, keepdims=True) / len(X)

    def predict(self, X):
        activations = self.forward(X)
        output = activations[-1]
        if self.task == 'classification':
            return (output >= 0.5).astype(int)
        return output

# Ví dụ kiểm thử
if __name__ == "__main__":
    X = np.array([[0,0], [0,1], [1,0], [1,1]])
    y = np.array([[0], [1], [1], [0]]) # Cổng XOR

    # Mạng có 1 tầng ẩn gồm 4 nơ-ron
    nn = NeuralNetwork(layers=[2, 4, 1], task='classification', lr=0.5)
    nn.fit(X, y, epochs=5000)
    print("Dự đoán bài toán XOR:")
    print(nn.predict(X))