
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt


# ============================================================
# 1. TẠO DỮ LIỆU
# ============================================================

data = {
    "Outlook": [
        "Sunny", "Sunny", "Overcast", "Rain",
        "Rain", "Rain", "Overcast", "Sunny",
        "Sunny", "Rain", "Sunny", "Overcast",
        "Overcast", "Rain"
    ],

    "Temperature": [
        "Hot", "Hot", "Hot", "Mild",
        "Cool", "Cool", "Cool", "Mild",
        "Cool", "Mild", "Mild", "Mild",
        "Hot", "Mild"
    ],

    "Humidity": [
        "High", "High", "High", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "Normal", "Normal", "High",
        "Normal", "High"
    ],

    "Wind": [
        "Weak", "Strong", "Weak", "Weak",
        "Weak", "Strong", "Strong", "Weak",
        "Weak", "Weak", "Strong", "Strong",
        "Weak", "Strong"
    ],

    "Play": [
        "No", "No", "Yes", "Yes",
        "Yes", "No", "Yes", "No",
        "Yes", "Yes", "Yes", "Yes",
        "Yes", "No"
    ]
}

df = pd.DataFrame(data)

print("=" * 60)
print("DỮ LIỆU")
print("=" * 60)
print(df)


# ============================================================
# 2. HÀM TÍNH ENTROPY
# ============================================================

def entropy(data):

    values, counts = np.unique(
        data,
        return_counts=True
    )

    total = len(data)

    result = 0

    for count in counts:

        p = count / total

        result -= p * math.log2(p)

    return result


# ============================================================
# 3. HÀM TÍNH INFORMATION GAIN
# ============================================================

def information_gain(df, feature, target):

    # Entropy ban đầu
    parent_entropy = entropy(
        df[target].values
    )

    weighted_entropy = 0

    # Các giá trị của thuộc tính
    values = df[feature].unique()

    for value in values:

        subset = df[
            df[feature] == value
        ]

        probability = (
            len(subset) / len(df)
        )

        subset_entropy = entropy(
            subset[target].values
        )

        weighted_entropy += (
            probability
            * subset_entropy
        )

    return (
        parent_entropy
        - weighted_entropy
    )


# ============================================================
# 4. TÍNH INFORMATION GAIN
# ============================================================

features = [
    "Outlook",
    "Temperature",
    "Humidity",
    "Wind"
]

print()
print("=" * 60)
print("INFORMATION GAIN")
print("=" * 60)

for feature in features:

    gain = information_gain(
        df,
feature,
        "Play"
    )

    print(
        feature,
        "=",
        round(gain, 4)
    )


# ============================================================
# 5. CÂY ID3
# ============================================================

def id3(data, features, target):

    # --------------------------------------------------------
    # Trường hợp 1:
    # Tất cả các mẫu có cùng nhãn
    # --------------------------------------------------------

    if len(data[target].unique()) == 1:

        return {
            "type": "leaf",
            "label": data[target].iloc[0]
        }


    # --------------------------------------------------------
    # Trường hợp 2:
    # Không còn thuộc tính
    # --------------------------------------------------------

    if len(features) == 0:

        majority = data[target].mode()[0]

        return {
            "type": "leaf",
            "label": majority
        }


    # --------------------------------------------------------
    # Chọn thuộc tính có Information Gain lớn nhất
    # --------------------------------------------------------

    gains = {}

    for feature in features:

        gains[feature] = information_gain(
            data,
            feature,
            target
        )

    best_feature = max(
        gains,
        key=gains.get
    )


    print()
    print(
        "Chọn thuộc tính:",
        best_feature,
        "| Gain =",
        round(gains[best_feature], 4)
    )


    # --------------------------------------------------------
    # Tạo nút
    # --------------------------------------------------------

    tree = {
        "type": "node",
        "feature": best_feature,
        "branches": {}
    }


    # --------------------------------------------------------
    # Tạo các nhánh
    # --------------------------------------------------------

    values = data[best_feature].unique()

    remaining_features = [
        feature
        for feature in features
        if feature != best_feature
    ]


    for value in values:

        subset = data[
            data[best_feature] == value
        ]

        # Nếu nhánh không có dữ liệu
        if len(subset) == 0:
            continue

        # Xây cây con
        subtree = id3(
            subset,
            remaining_features,
            target
        )

        tree["branches"][value] = subtree


    return tree


# ============================================================
# 6. XÂY DỰNG CÂY ID3
# ============================================================

print()
print("=" * 60)
print("XÂY DỰNG CÂY ID3")
print("=" * 60)

tree = id3(
    df,
    features,
    "Play"
)


# ============================================================
# 7. IN CÂY RA MÀN HÌNH
# ============================================================

def print_tree(tree, level=0):

    space = "    " * level

    # Nếu là lá
    if tree["type"] == "leaf":

        print(space + "→ " + tree["label"])
    return


    # Nếu là node
    print(space + "[" + tree["feature"] + "]")


    for value, subtree in tree["branches"].items():

        print(
            space
            + "├── "
            + str(value)
        )

        print_tree(
            subtree,
            level + 1
        )


print()
print("=" * 60)
print("CÂY ID3")
print("=" * 60)

print_tree(tree)


# ============================================================
# 8. VẼ CÂY THEO NHÁNH
# ============================================================

fig, ax = plt.subplots(
    figsize=(18, 10)
)

ax.axis("off")


# ------------------------------------------------------------
# Hàm đếm số lá
# ------------------------------------------------------------

def count_leaves(tree):

    if tree["type"] == "leaf":
        return 1

    total = 0

    for subtree in tree["branches"].values():
        total += count_leaves(subtree)

    return total


# ------------------------------------------------------------
# Hàm vẽ cây
# ------------------------------------------------------------

def draw_tree(
    tree,
    x,
    y,
    dx,
    parent_x=None,
    parent_y=None,
    branch_label=""
):

    # --------------------------------------------------------
    # Vẽ đường nối với node cha
    # --------------------------------------------------------

    if parent_x is not None:

        ax.plot(
            [parent_x, x],
            [parent_y, y],
            linewidth=1
        )

        # Tên nhánh
        mid_x = (parent_x + x) / 2
        mid_y = (parent_y + y) / 2

        ax.text(
            mid_x,
            mid_y,
            str(branch_label),
            fontsize=10,
            ha="center",
            va="center"
        )


    # --------------------------------------------------------
    # Nếu là node lá
    # --------------------------------------------------------

    if tree["type"] == "leaf":

        ax.text(
            x,
            y,
            tree["label"],
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                edgecolor="black"
            )
        )

        return


    # --------------------------------------------------------
    # Nếu là node quyết định
    # --------------------------------------------------------

    ax.text(
        x,
        y,
        tree["feature"],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="white",
            edgecolor="black"
        )
    )


    branches = list(
        tree["branches"].items()
    )

    n = len(branches)


    # Khoảng cách giữa các nhánh
    if n > 1:
        step = dx / (n - 1)
    else:
        step = 0


    start_x = x - dx / 2


    for i, (value, subtree) in enumerate(branches):

        child_x = start_x + i * step

        child_y = y - 1.5

        draw_tree(
            subtree,
            child_x,
            child_y,
            dx / max(n, 1),
            x,
            y,
            value
        )


# ------------------------------------------------------------
# Vẽ
# ------------------------------------------------------------

draw_tree(
    tree,
    x=0.5,
    y=1,
    dx=0.9
)

plt.title(
    "Cây quyết định ID3",
    fontsize=16
)

plt.tight_layout()

plt.show()


# ============================================================
# 9. HÀM DỰ ĐOÁN
# ============================================================

def predict_one(tree, sample):

    # Nếu là node lá
    if tree["type"] == "leaf":

        return tree["label"]


    feature = tree["feature"]

    value = sample[feature]


    # Nếu giá trị tồn tại trong cây
    if value in tree["branches"]:

        return predict_one(
            tree["branches"][value],
            sample
        )


    # Nếu gặp giá trị chưa từng xuất hiện
    # thì trả về nhãn phổ biến nhất
    return
# ============================================================

# 10. DỰ ĐOÁN TOÀN BỘ DỮ LIỆU

# ============================================================

predictions = []

for _, row in df.iterrows():

    prediction = predict_one(

        tree,

        row

    )

    predictions.append(

        prediction

    )

print()

print("=" * 60)

print("KẾT QUẢ DỰ ĐOÁN")

print("=" * 60)

result = df.copy()

result["Prediction"] = predictions

print(result)

# ============================================================

# 11. TÍNH ACCURACY

# ============================================================

accuracy = sum(

    result["Play"] == result["Prediction"]

) / len(result)

print()

print("=" * 60)

print("ĐÁNH GIÁ MÔ HÌNH")

print("=" * 60)

print(

    "Accuracy =",

    round(accuracy, 4)

)

# ============================================================

# 12. TÍNH PRECISION, RECALL, F1

# ============================================================

# Chuyển Yes/No thành 1/0

y_true = result["Play"].map({

    "No": 0,

    "Yes": 1

})

y_pred = result["Prediction"].map({

    "No": 0,

    "Yes": 1

})

# Loại bỏ trường hợp Unknown

mask = y_pred.notna()

y_true = y_true[mask]

y_pred = y_pred[mask]

TP = sum(

    (y_true == 1) &

    (y_pred == 1)

)

TN = sum(

    (y_true == 0) &

    (y_pred == 0)

)

FP = sum(

    (y_true == 0) &

    (y_pred == 1)

)

FN = sum(

    (y_true == 1) &

    (y_pred == 0)

)

# Precision

if TP + FP != 0:

    precision = TP / (TP + FP)

else:

    precision = 0

# Recall

if TP + FN != 0:

    recall = TP / (TP + FN)

else:

    recall = 0

# F1-score

if precision + recall != 0:

    f1 = (

        2

        * precision

        * recall

        / (precision + recall)

    )

else:

    f1 = 0

print(

    "Precision =",

    round(precision, 4)

)

print(

    "Recall    =",

    round(recall, 4)

)

print(

    "F1-score  =",

    round(f1, 4)

)