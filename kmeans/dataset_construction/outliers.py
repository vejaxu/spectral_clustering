import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# -----------------------------
# 类 1 和 类 2：两个靠近但可分的主类
# -----------------------------
mean1 = [0, 0]
mean2 = [3, 0]              # 水平距离为 3，靠近但不重叠
cov = 0.4 * np.eye(2)       # 类内扩散适中

class1 = rng.multivariate_normal(mean1, cov, 600)
class2 = rng.multivariate_normal(mean2, cov, 600)

# -----------------------------
# 异常点：少量（6 个），集中，构成第 3 个类（但很小）
# -----------------------------
n_outliers = 6
mean_outlier = [10, 0]      # 远离主群
cov_outlier = 0.1 * np.eye(2)  # 非常集中
outliers = rng.multivariate_normal(mean_outlier, cov_outlier, n_outliers)

# -----------------------------
# 合并数据（全部用于聚类输入）
# -----------------------------
data = np.vstack([class1, class2, outliers])  # 总共 1206 个点

# -----------------------------
# 真实标签（包含三个类）
# -----------------------------
labels = np.concatenate([
    np.ones(600, dtype=int),                    # 类 1 → label=1
    np.full(600, 2, dtype=int),                 # 类 2 → label=2
    np.full(n_outliers, 3, dtype=int)           # 异常点 → label=3（真实类别）
])
# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_outliers.mat", {"data": data, "class": labels})
