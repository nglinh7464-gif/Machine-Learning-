# Bai 3.30
# Phan lop nhi phan bang Perceptron


import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


# =================================================
# 1. XAY DUNG CLASS PERCEPTRON
# =================================================

class Perceptron:

    def __init__(self, learning_rate=0.01, epochs=100):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.w = None


    # Ham huan luyen
    def fit(self, X, y):

        # Khoi tao trong so
        self.w = np.zeros(X.shape[1])

        # Lap qua cac epoch
        for epoch in range(self.epochs):

            # Duyet tung mau
            for i in range(len(X)):

                # Tinh w^T * x
                z = np.dot(self.w, X[i])

                # Du doan
                if z >= 0:
                    y_pred = 1
                else:
                    y_pred = -1

                # Neu phan lop sai
                if y_pred != y[i]:

                    # Cap nhat trong so
                    self.w = (
                        self.w
                        + self.learning_rate
                        * y[i]
                        * X[i]
                    )

        return self


    # Ham du doan
    def predict(self, X):

        # Tinh w^T * x
        z = np.dot(X, self.w)

        # Chuyen thanh nhan
        y_pred = np.where(z >= 0, 1, -1)

        return y_pred


# =================================================
# 2. DOC DU LIEU IRIS
# =================================================

print("BAI 3.30 - PHAN LOP NHI PHAN")
print("----------------------------")


iris = load_iris()


# X la cac dac trung
X = iris.data


# y la nhan
y = iris.target


print("So luong du lieu ban dau:", len(X))


# =================================================
# 3. CHI LAY 2 LOP
# =================================================

# Chi lay lop 0 va lop 1
mask = y < 2

X = X[mask]
y = y[mask]


# Chuyen:
# 0 -> -1
# 1 -> +1

y = np.where(y == 0, -1, 1)


print("So luong du lieu sau khi lay 2 lop:", len(X))


# =================================================
# 4. CHIA TRAIN VA TEST
# =================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("So mau train:", len(X_train))
print("So mau test :", len(X_test))


# =================================================
# 5. CHUAN HOA DU LIEU
# =================================================

scaler = StandardScaler()


# Hoc cach chuan hoa tu tap train
X_train = scaler.fit_transform(X_train)


# Chuan hoa tap test theo train
X_test = scaler.transform(X_test)


# =================================================
# 6. TAO MO HINH
# =================================================

model = Perceptron(
    learning_rate=0.01,
    epochs=100
)


# =================================================
# 7. HUAN LUYEN
# =================================================

model.fit(X_train, y_train)


print("\nTrong so sau khi huan luyen:")
print(model.w)


# =================================================
# 8. DU DOAN
# =================================================

y_pred = model.predict(X_test)


# =================================================
# 9. TINH CAC DO DO
# =================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


precision = precision_score(
    y_test,
    y_pred,
    pos_label=1
)


recall = recall_score(
    y_test,
    y_pred,
    pos_label=1
)


f1 = f1_score(
    y_test,
    y_pred,
    pos_label=1
)


# =================================================
# 10. IN KET QUA
# =================================================

print("\nKET QUA DANH GIA")
print("----------------")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1-score :", f1)