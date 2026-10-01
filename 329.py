import numpy as np


class Perceptron:

    # Ham khoi tao
    def __init__(self, learning_rate=0.1, epochs=10):

        self.learning_rate = learning_rate
        self.epochs = epochs

        # Trong so
        self.w = None


    # Ham fit de huan luyen
    def fit(self, X, y):

        # Khoi tao trong so bang 0
        self.w = np.zeros(X.shape[1])

        # Lap qua cac epoch
        for epoch in range(self.epochs):

            # Duyet tung mau du lieu
            for i in range(len(X)):

                # Tinh w^T * x
                z = np.dot(self.w, X[i])

                # Du doan
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Neu du doan sai
                if y_pred != y[i]:

                    # Cap nhat trong so
                    self.w = (
                        self.w
                        + self.learning_rate
                        * y[i]
                        * X[i]
                    )

        return self


    # Ham predict de du doan
    def predict(self, X):

        # Tinh X * w
        z = np.dot(X, self.w)

        # Neu z >= 0 -> 1
        # Neu z < 0 -> -1
        y_pred = np.where(z >= 0, 1, -1)

        return y_pred


# =================================================
# CHUONG TRINH CHINH
# =================================================

print("BAI 3.29 - CLASS PERCEPTRON")
print("---------------------------")


# Du lieu huan luyen
X = np.array([
    [1, 2],
    [2, 3],
    [-1, -2],
    [-2, -3]
])


# Nhan
y = np.array([
    1,
    1,
    -1,
    -1
])


# Tao model
model = Perceptron(
    learning_rate=0.1,
    epochs=10
)


# Huan luyen
model.fit(X, y)


print("Trong so sau khi huan luyen:")
print(model.w)


# Du lieu moi
X_new = np.array([
    [3, 4],
    [-3, -4]
])


# Du doan
y_pred = model.predict(X_new)


print("\nDu lieu moi:")
print(X_new)

print("\nNhan du doan:")
print(y_pred)