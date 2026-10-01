import numpy as np


# Vector w
w = np.array([1, 2, -10])

# Vector x
x = np.array([3, 4, 1])

# Nhan thuc te
y = -1


print("BAI 3.27 - PERCEPTRON")
print("---------------------")

# Tinh w^T * x
wx = np.dot(w, x)

print("w =", w)
print("x =", x)

print("\nw^T x =", wx)


# Xac dinh nhan du doan
if wx >= 0:
    y_pred = 1
else:
    y_pred = -1


print("Nhan du doan =", y_pred)
print("Nhan thuc te =", y)


# Kiem tra phan lop sai
if y_pred != y:
    print("=> Diem du lieu bi phan lop sai")
else:
    print("=> Diem du lieu duoc phan lop dung")