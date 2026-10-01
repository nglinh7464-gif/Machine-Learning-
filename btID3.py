import math
import pandas as pd

import matplotlib.pyplot as plt
# =========================
# 1. TẠO DỮ LIỆU
# =========================

data = {
    "age": [
        "<=30", "<=30", "31...40", ">40", ">40",
        ">40", "31...40", "<=30", "<=30", ">40",
        "<=30", "31...40", "31...40", ">40"
    ],

    "income": [
        "high", "high", "high", "medium", "low",
        "low", "low", "medium", "low", "medium",
        "medium", "medium", "high", "medium"
    ],

    "student": [
        "no", "no", "no", "no", "yes",
        "yes", "yes", "no", "yes", "yes",
        "yes", "no", "yes", "no"
    ],

    "credit_rating": [
        "fair", "excellent", "fair", "fair", "fair",
        "excellent", "excellent", "fair", "fair", "fair",
        "excellent", "excellent", "fair", "excellent"
    ],

    "buys_computer": [
        "no", "no", "yes", "yes", "yes",
        "no", "yes", "no", "yes", "yes",
        "yes", "yes", "yes", "no"
    ]
}

df = pd.DataFrame(data)

print("DỮ LIỆU:")
print(df)


# =========================
# 2. TÍNH ENTROPY
# =========================

def entropy(data):
    """
    Tính Entropy của tập dữ liệu
    """

    total = len(data)

    # Đếm số lượng yes/no
    counts = data.value_counts()

    result = 0

    for count in counts:
        p = count / total

        result -= p * math.log2(p)

    return result


# =========================
# 3. TÍNH INFORMATION GAIN
# =========================

def information_gain(df, attribute, target):
    """
    Gain = Entropy(S)
           - tổng trọng số Entropy(Sv)
    """

    # Entropy ban đầu
    total_entropy = entropy(df[target])

    weighted_entropy = 0

    # Chia dữ liệu theo từng giá trị của thuộc tính
    for value in df[attribute].unique():

        subset = df[df[attribute] == value]

        weight = len(subset) / len(df)

        subset_entropy = entropy(subset[target])

        weighted_entropy += weight * subset_entropy

    gain = total_entropy - weighted_entropy

    return gain


# =========================
# 4. TÍNH GAIN CHO TẤT CẢ
# =========================

attributes = [
    "age",
    "income",
    "student",
    "credit_rating"
]

print("\n==============================")
print("ENTROPY BAN DAU")
print("==============================")

print("Entropy =", round(entropy(df["buys_computer"]), 4))


print("\n==============================")
print("INFORMATION GAIN")
print("==============================")

for attribute in attributes:

    gain = information_gain(
        df,
        attribute,
        "buys_computer"
    )

    print(
        attribute,
        "=",
        round(gain, 4)
    )


# =========================
# 5. TÌM THUỘC TÍNH TỐT NHẤT
# =========================

best_attribute = max(
    attributes,
    key=lambda x: information_gain(
        df,
        x,
        "buys_computer"
    )
)

print("\nThuoc tinh duoc chon lam NUT GOC:")
print(best_attribute)


# =========================
# 6. HÀM ID3
# =========================

def id3(df, attributes, target):

    # Nếu tất cả các mẫu đều YES
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Nếu không còn thuộc tính để chia
    if len(attributes) == 0:
        return df[target].mode()[0]

    # Chọn thuộc tính có Gain lớn nhất
    best_attribute = max(
        attributes,
        key=lambda x: information_gain(
            df,
            x,
            target
        )
    )

    tree = {
        best_attribute: {}
    }

    # Các thuộc tính còn lại
    remaining_attributes = [
        x for x in attributes
        if x != best_attribute
    ]

    # Chia dữ liệu theo thuộc tính tốt nhất
    for value in df[best_attribute].unique():

        subset = df[
            df[best_attribute] == value
        ]

        # Nếu subset rỗng
        if len(subset) == 0:

            tree[best_attribute][value] = \
                df[target].mode()[0]

        else:

            tree[best_attribute][value] = id3(
                subset,
                remaining_attributes,
                target
            )

    return tree


# =========================
# 7. XÂY DỰNG CÂY ID3
# =========================

tree = id3(
    df,
    attributes,
    "buys_computer"
)


print("\n==============================")
print("CAY QUYET DINH ID3")
print("==============================")

print(tree)


# =========================
# 8. VẼ CÂY ID3
# =========================

def draw_tree(tree, parent=None, edge_text="", pos=None,
              level=0, x=0.5, width=1.0):

    if pos is None:
        pos = {}

    # Lấy tên nút hiện tại
    node = list(tree.keys())[0]

    # Nếu là nút gốc
    if parent is None:
        pos[node] = (x, 1 - level * 0.2)

    else:
        pos[node] = (x, 1 - level * 0.2)

        # Vẽ đường nối
        plt.plot(
            [parent[0], x],
            [parent[1], 1 - level * 0.2],
            'k-'
        )

        # Ghi giá trị nhánh
        plt.text(
            (parent[0] + x) / 2,
            (parent[1] + 1 - level * 0.2) / 2,
            edge_text,
            ha="center",
            va="center"
        )

    children = tree[node]

    # Nếu là lá
    if not isinstance(children, dict):
        return

    child_items = list(children.items())

    n = len(child_items)

    # Khoảng cách giữa các nhánh
    if n > 1:
        step = width / (n - 1)
    else:
        step = width

    start_x = x - width / 2

    for i, (edge, child) in enumerate(child_items):

        child_x = start_x + i * step

        child_y = 1 - (level + 1) * 0.2

        # Nếu child là YES / NO
        if not isinstance(child, dict):

            plt.text(
                child_x,
                child_y,
                child.upper(),
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black"
                )
            )

            # Vẽ đường từ node cha
            plt.plot(
                [x, child_x],
                [1 - level * 0.2, child_y],
                'k-'
            )

            # Ghi tên nhánh
            plt.text(
                (x + child_x) / 2,
                (1 - level * 0.2 + child_y) / 2,
                edge,
                ha="center"
            )

        else:

            # Vẽ nút con
            plt.text(
                child_x,
                child_y,
                list(child.keys())[0],
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black"
                )
            )

            # Đường nối
            plt.plot(
                [x, child_x],
                [1 - level * 0.2, child_y],
                'k-'
            )

            # Tên nhánh
            plt.text(
                (x + child_x) / 2,
                (1 - level * 0.2 + child_y) / 2,
                edge,
                ha="center"
            )

            # Vẽ tiếp cây con
            draw_subtree(
                child,
                child_x,
                child_y,
                level + 1,
                width / 2
            )


def draw_subtree(tree, x, y, level, width):

    node = list(tree.keys())[0]

    # Vẽ tên nút
    plt.text(
        x,
        y,
        node,
        ha="center",
        va="center",
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black"
        )
    )

    children = tree[node]

    if not isinstance(children, dict):
        return

    child_items = list(children.items())

    n = len(child_items)

    if n == 1:
        positions = [x]
    else:
        step = width / (n - 1)

        positions = [
            x - width / 2 + i * step
            for i in range(n)
        ]

    child_y = y - 0.2

    for i, (edge, child) in enumerate(child_items):

        child_x = positions[i]

        # Vẽ đường nối
        plt.plot(
            [x, child_x],
            [y - 0.03, child_y + 0.03],
            'k-'
        )

        # Tên nhánh
        plt.text(
            (x + child_x) / 2,
            (y + child_y) / 2,
            edge,
            ha="center"
        )

        # Nếu là lá
        if not isinstance(child, dict):

            plt.text(
                child_x,
                child_y,
                child.upper(),
                ha="center",
                va="center",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor="black"
                )
            )

        else:

            draw_subtree(
                child,
                child_x,
                child_y,
                level + 1,
                width / 2
            )


# =========================
# 9. HIỂN THỊ CÂY
# =========================

plt.figure(figsize=(12, 7))

root = list(tree.keys())[0]

plt.text(
    0.5,
    0.9,
    root,
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)

draw_subtree(
    tree,
    0.5,
    0.9,
    0,
    0.8
)

plt.axis("off")

plt.title("Cay quyet dinh ID3")

plt.show()