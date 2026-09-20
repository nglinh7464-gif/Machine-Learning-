import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error


# ============================================================
# 1. TẠO DỮ LIỆU GIẢ LẬP (1000 CĂN NHÀ)
# ============================================================

np.random.seed(42)
n_samples = 1000

# x1 = Dien tich
x1 = np.random.uniform(30, 200, n_samples)

# x2 = So phong ngu
x2 = np.random.randint(1, 6, n_samples)

# x3 = Khoang cach den trung tam
x3 = np.random.uniform(1, 30, n_samples)

# Tao nhieu
noise = np.random.normal(0, 200, n_samples)

# Cong thuc gia nha
y = 50 * x1 + 100 * x2 - 30 * x3 + 500 + noise

# Tao DataFrame
X = pd.DataFrame({
    'x1': x1,
    'x2': x2,
    'x3': x3
})


# ============================================================
# 2. CHIA TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 3. TAO MO HINH POLYNOMIAL BAC 3
# ============================================================

poly = PolynomialFeatures(degree=3)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

overfit_model = LinearRegression()

overfit_model.fit(X_train_poly, y_train)


# ============================================================
# 4. TINH RMSE CHO MO HINH OVERFIT
# ============================================================

train_pred = overfit_model.predict(X_train_poly)
test_pred = overfit_model.predict(X_test_poly)

train_rmse = root_mean_squared_error(y_train, train_pred)
test_rmse = root_mean_squared_error(y_test, test_pred)

print("--- DAU HIEU OVERFITTING ---")
print(f"Train RMSE: {train_rmse:.2f}")
print(f"Test RMSE : {test_rmse:.2f}")
print()


# ============================================================
# 5. K-FOLD CROSS VALIDATION
# ============================================================

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ------------------------------------------------------------
# 5a. MO HINH OVERFIT
# ------------------------------------------------------------

cv_scores_overfit = cross_val_score(
    overfit_model,
    X_train_poly,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=kf
)

overfit_cv_rmse = -cv_scores_overfit.mean()

print(
    f"K-Fold CV RMSE (Mo hinh Overfit): "
    f"{overfit_cv_rmse:.2f}"
)


# ------------------------------------------------------------
# 5b. MO HINH TUYEN TINH DON GIAN
# ------------------------------------------------------------

simple_model = LinearRegression()

cv_scores_simple = cross_val_score(
    simple_model,
    X_train,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=kf
)

simple_cv_rmse = -cv_scores_simple.mean()

print(
    f"K-Fold CV RMSE (Mo hinh Gian don): "
    f"{simple_cv_rmse:.2f}"
)


# ------------------------------------------------------------
# 5c. RIDGE REGRESSION
# ------------------------------------------------------------

ridge_model = Ridge(alpha=100.0)

cv_scores_ridge = cross_val_score(
    ridge_model,
    X_train_poly,
    y_train,
    scoring='neg_root_mean_squared_error',
    cv=kf
)

ridge_cv_rmse = -cv_scores_ridge.mean()

print(
    f"K-Fold CV RMSE (Mo hinh Ridge): "
    f"{ridge_cv_rmse:.2f}"
)


# ============================================================
# 6. BIEU DO 1:
#    GIA THUC TE VS GIA DU DOAN
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    test_pred,
    alpha=0.5
)

# Duong y = x
min_value = min(y_test.min(), test_pred.min())
max_value = max(y_test.max(), test_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle='--'
)

plt.xlabel("Gia thuc te")
plt.ylabel("Gia du doan")
plt.title("Gia thuc te vs Gia du doan - Polynomial bac 3")

plt.grid(True)
plt.show()


# ============================================================
# 7. BIEU DO 2:
#    SO SANH RMSE CUA 3 MO HINH
# ============================================================

models = [
    "Polynomial\nOverfit",
    "Linear\nDon gian",
    "Ridge"
]

rmse_values = [
    overfit_cv_rmse,
    simple_cv_rmse,
    ridge_cv_rmse
]

plt.figure(figsize=(8, 6))

plt.bar(
    models,
    rmse_values
)

plt.ylabel("RMSE")
plt.xlabel("Mo hinh")
plt.title("So sanh RMSE cua 3 mo hinh")

plt.grid(
    axis='y',
    linestyle='--',
    alpha=0.5
)

plt.show()