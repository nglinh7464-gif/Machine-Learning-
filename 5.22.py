import math
import pandas as pd
import numpy as np
# Tập dữ liệu Mua máy tính[cite: 3]
data = {
    'Tuoi': ['Tre', 'Tre', 'Trung nien', 'Gia', 'Gia', 'Gia', 'Trung nien', 'Tre', 'Tre', 'Gia', 'Tre', 'Trung nien', 'Trung nien', 'Gia'],
    'Thu nhap': ['Cao', 'Cao', 'Cao', 'Trung binh', 'Thap', 'Thap', 'Thap', 'Trung binh', 'Thap', 'Trung binh', 'Trung binh', 'Trung binh', 'Cao', 'Trung binh'],
    'Sinh vien': ['Khong', 'Khong', 'Khong', 'Khong', 'Co', 'Co', 'Co', 'Khong', 'Co', 'Co', 'Co', 'Khong', 'Co', 'Khong'],
    'Danh gia tin dung': ['Tot', 'Xuat sac', 'Tot', 'Tot', 'Tot', 'Xuat sac', 'Xuat sac', 'Tot', 'Tot', 'Tot', 'Xuat sac', 'Xuat sac', 'Tot', 'Xuat sac'],
    'Mua may tinh': ['Khong', 'Khong', 'Co', 'Co', 'Co', 'Khong', 'Co', 'Khong', 'Co', 'Co', 'Co', 'Co', 'Co', 'Khong']
}

df = pd.DataFrame(data)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    ent = 0
    for i in range(len(elements)):
        p = counts[i] / np.sum(counts)
        ent -= p * math.log2(p)
    return ent

def info_gain(data, split_attribute_name, target_name="Mua may tinh"):
    total_entropy = entropy(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = 0
    for i in range(len(vals)):
        sub_data = data[data[split_attribute_name] == vals[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(sub_data[target_name])
    return total_entropy - weighted_entropy

def build_tree(data, original_data, features, target_attribute_name="Mua may tinh"):
    # Nếu tất cả các giá trị mục tiêu đồng nhất
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    # Nếu tập dữ liệu rỗng
    elif len(data) == 0:
        return np.unique(original_data[target_attribute_name])[np.argmax(np.unique(original_data[target_attribute_name], return_counts=True)[1])]
    
    # Nếu không còn đặc trưng nào để phân chia
    elif len(features) == 0:
        return np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
    
    else:
        # Chọn đặc trưng có Information Gain cao nhất
        item_values = [info_gain(data, feature, target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        
        for value in np.unique(data[best_feature]):
            sub_data = data[data[best_feature] == value]
            subtree = build_tree(sub_data, data, features, target_attribute_name)
            tree[best_feature][value] = subtree
            
        return tree

def predict(tree, sample):
    for key in sample.keys():
        if key in tree:
            try:
                result = tree[key][sample[key]]
            except KeyError:
                return None
            if isinstance(result, dict):
                return predict(result, sample)
            else:
                return result

# Xây dựng và kiểm thử[cite: 3]
features = list(df.columns[:-1])
tree = build_tree(df, df, features)

print("Cây quyết định dạng Dictionary:")
print(tree)

# Kiểm tra thử nghiệm[cite: 3]
test_sample_1 = {'Tuoi': 'Tre', 'Thu nhap': 'Cao', 'Sinh vien': 'Khong', 'Danh gia tin dung': 'Tot'}
test_sample_2 = {'Tuoi': 'Trung nien', 'Thu nhap': 'Thap', 'Sinh vien': 'Co', 'Danh gia tin dung': 'Xuat sac'}

print("Dự đoán mẫu 1:", predict(tree, test_sample_1)) # Kỳ vọng: Khong
print("Dự đoán mẫu 2:", predict(tree, test_sample_2)) # Kỳ vọng: Co