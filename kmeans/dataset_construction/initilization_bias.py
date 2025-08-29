import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# -----------------------------
# 类 1：左侧两个上下排列的子簇（共 600 点）
# -----------------------------
mean1_upper = [-1.5,  2]   # 左上
mean1_lower = [-1.5, -2]   # 左下
cov1 = 0.8 * np.eye(2)     # 协方差矩阵

class1_upper = rng.multivariate_normal(mean1_upper, cov1, 300)
class1_lower = rng.multivariate_normal(mean1_lower, cov1, 300)
class1 = np.vstack([class1_upper, class1_lower])  # 600 点

# -----------------------------
# 类 2：右侧两个上下排列的子簇（共 600 点）
# -----------------------------
mean2_upper = [1.5,  2]    # 右上
mean2_lower = [1.5, -2]    # 右下
cov2 = 0.8 * np.eye(2)

class2_upper = rng.multivariate_normal(mean2_upper, cov2, 300)
class2_lower = rng.multivariate_normal(mean2_lower, cov2, 300)
class2 = np.vstack([class2_upper, class2_lower])  # 600 点

# -----------------------------
# 合并数据和标签
# -----------------------------
data = np.vstack([class1, class2])          # 1200 x 2
labels = np.concatenate([
    np.ones(600, dtype=int),                # 类 1（左）
    np.full(600, 2, dtype=int)              # 类 2（右）
])

# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_init_bias.mat", {"data": data, "class": labels})
