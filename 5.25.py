import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

# 1. Tạo file CSV chứa thông tin khách hàng giả lập[cite: 4]
np.random.seed(42)
df_cust = pd.DataFrame({
    'Tuoi': np.random.randint(18, 65, 100),
    'ThuNhap_Trieu': np.random.randint(5, 50, 100),
    'DiemTinDung': np.random.randint(300, 850, 100),
    'NhanVay': np.random.choice([0, 1], 100),            # Biến mục tiêu phân lớp
    'ChiTieu_Trieu': np.random.uniform(1.0, 20.0, 100) # Biến mục tiêu số (hồi quy)
})
df_cust.to_csv('khach_hang.csv', index=False)

# 2. Đọc dữ liệu từ file CSV[cite: 4]
data = pd.read_csv('khach_hang.csv')
X = data[['Tuoi', 'ThuNhap_Trieu', 'DiemTinDung']]
y_cls = data['NhanVay']
y_reg = data['ChiTieu_Trieu']

# 3. Phân lớp + Cắt tỉa + Tầm quan trọng + Trực quan hóa[cite: 4]
clf = DecisionTreeClassifier(min_samples_leaf=5, random_state=42) # Cắt tỉa bằng min_samples_leaf[cite: 4]
clf.fit(X, y_cls)

print("--- BÀI TOÁN PHÂN LỚP ---")
for col, imp in zip(X.columns, clf.feature_importances_):
    print(f"Tầm quan trọng đặc trưng '{col}': {imp:.4f}")

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=X.columns, class_names=['KhongVay', 'Vay'], filled=True)
plt.title("Cây Phân Lớp Khách Hàng")
plt.savefig("cay_phan_lop.png") # Lưu thành file ảnh[cite: 4]
plt.close()

# 4. Hồi quy + Cắt tỉa + Tầm quan trọng + Trực quan hóa[cite: 4]
reg = DecisionTreeRegressor(min_samples_leaf=5, random_state=42)
reg.fit(X, y_reg)

print("\n--- BÀI TOÁN HỒI QUY ---")
for col, imp in zip(X.columns, reg.feature_importances_):
    print(f"Tầm quan trọng đặc trưng '{col}': {imp:.4f}")

plt.figure(figsize=(12, 8))
plot_tree(reg, feature_names=X.columns, filled=True)
plt.title("Cây Hồi Quy Khách Hàng")
plt.savefig("cay_hoi_quy.png") # Lưu thành file ảnh[cite: 4]
plt.close()

print("\nĐã xuất đồ thị cây ra hai file ảnh: 'cay_phan_lop.png' và 'cay_hoi_quy.png'.")