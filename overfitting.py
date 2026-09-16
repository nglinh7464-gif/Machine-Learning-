import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error

# 1. TẠO DỮ LIỆU GIẢ LẬP (1000 căn nhà)
np.random.seed(42)
n_samples = 1000

x1 = np.random.uniform(30, 200, n_samples)   # Diện tích (m2)
x2 = np.random.randint(1, 6, n_samples)       # Số phòng ngủ
x3 = np.random.uniform(1, 30, n_samples)     # Khoảng cách trung tâm (km)

# Giá nhà = 50*x1 + 100*x2 - 30*x3 + nhiễu
noise = np.random.normal(0, 200, n_samples)
y = 50 * x1 + 100 * x2 - 30 * x3 + 500 + noise

X = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3})

# Chia tập Train/Test cơ bản
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. CỐ TÌNH TẠO OVERFITTING (Dùng degree=3 để chạy siêu nhanh)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

overfit_model = LinearRegression()
overfit_model.fit(X_train_poly, y_train)

# Tính RMSE bằng hàm mới root_mean_squared_error
train_rmse = root_mean_squared_error(y_train, overfit_model.predict(X_train_poly))
test_rmse = root_mean_squared_error(y_test, overfit_model.predict(X_test_poly))

print("--- DẤU HIỆU OVERFITTING ---")
print(f"Train RMSE: {train_rmse:.2f} (Thấp trên tập huấn luyện)")
print(f"Test RMSE : {test_rmse:.2f} (Cao trên tập kiểm thử)\n")

# 3. ÁP DỤNG K-FOLD CROSS-VALIDATION ĐỂ ĐÁNH GIÁ VÀ KHẮC PHỤC
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 3a. Dùng K-Fold đánh giá mô hình Overfit
cv_scores_overfit = cross_val_score(overfit_model, X_train_poly, y_train, 
                                    scoring='neg_root_mean_squared_error', cv=kf)
print(f"K-Fold CV RMSE (Mô hình Overfit) : {-cv_scores_overfit.mean():.2f}")

# 3b. Khắc phục 1: Dùng mô hình đơn giản hơn (Tuyến tính gốc bậc 1)
simple_model = LinearRegression()
cv_scores_simple = cross_val_score(simple_model, X_train, y_train, 
                                   scoring='neg_root_mean_squared_error', cv=kf)
print(f"K-Fold CV RMSE (Mô hình Giản đơn): {-cv_scores_simple.mean():.2f}")

# 3c. Khắc phục 2: Dùng Regularization (Ridge Regression) trên đa thức
ridge_model = Ridge(alpha=100.0)
cv_scores_ridge = cross_val_score(ridge_model, X_train_poly, y_train, 
                                  scoring='neg_root_mean_squared_error', cv=kf)
print(f"K-Fold CV RMSE (Mô hình Ridge)    : {-cv_scores_ridge.mean():.2f}")