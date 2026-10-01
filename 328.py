import numpy as np

# Vector w ban dau
w = np.array([-2, 1, 0])

# Vector x
x = np.array([2, 3, 1])

# Nhan thuc te
y = 1

# Learning rate
eta = 1


print("BAI 3.28 - PERCEPTRON")
print("---------------------")

print("w ban dau =", w)
print("x =", x)
print("y =", y)


# ------------------------------------------------
# 1. Kiem tra mau co bi phan lop sai hay khong
# ------------------------------------------------

wx = np.dot(w, x)

print("\nTruoc khi cap nhat:")
print("w^T x =", wx)


# Du doan
if wx >= 0:
    y_pred = 1
else:
    y_pred = -1


print("Nhan du doan =", y_pred)


# ------------------------------------------------
# 2. Neu sai thi cap nhat
# ------------------------------------------------

if y_pred != y:

    print("Mau bi phan lop sai")

    # Cong thuc:
    # w = w + eta * y * x
    w = w + eta * y * x

    print("w sau cap nhat =", w)

else:

    print("Mau duoc phan lop dung")
    print("Khong can cap nhat")


# ------------------------------------------------
# 3. Tinh lai w^T x
# ------------------------------------------------

wx_new = np.dot(w, x)

print("\nSau khi cap nhat:")
print("w^T x =", wx_new)


# Du doan lai
if wx_new >= 0:
    y_pred_new = 1
else:
    y_pred_new = -1


print("Nhan du doan moi =", y_pred_new)