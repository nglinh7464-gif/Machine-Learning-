# Ham f(x)
def f(x):
    return x**2 - 4*x + 5


# Dao ham f'(x)
def df(x):
    return 2*x - 4


# Diem khoi tao
x = 5

# Learning rate
eta = 0.2

print("BAI 3.26 - GRADIENT DESCENT")
print("---------------------------")

print("Dao ham: f'(x) = 2x - 4")
print("Learning rate =", eta)

# Gia tri ban dau
print("\nBuoc 0:")
print("x =", x)
print("f(x) =", f(x))

# Thuc hien 4 buoc
for i in range(1, 5):

    # Tinh dao ham
    gradient = df(x)

    # Cap nhat x
    x = x - eta * gradient

    print("\nBuoc", i)
    print("x =", x)
    print("f(x) =", f(x))


print("\nKet luan:")
print("Gradient Descent dang hoi tu ve x = 2")
print("Gia tri nho nhat la f(2) = 1")