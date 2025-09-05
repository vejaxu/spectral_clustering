import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# -----------------------------
# 三个类
# -----------------------------
# 类 1：左主类
mean1 = [-1.0, 0]
cov = 0.4 * np.eye(2)
class1 = rng.multivariate_normal(mean1, cov, 600)

# 类 2：右主类
mean2 = [1.0, 0]
class2 = rng.multivariate_normal(mean2, cov, 600)

# 类 3：异常点（作为第3类）
n_outliers = 6
mean_outlier = [0, 5]
cov_outlier = 0.1 * np.eye(2)
outliers = rng.multivariate_normal(mean_outlier, cov_outlier, n_outliers)

# -----------------------------
# 合并数据
# -----------------------------
data = np.vstack([class1, class2, outliers])  # 1206 x 2

# -----------------------------
# 真实标签：三个类（1,2,3），仅用于构造理解
# 但在后续聚类中，我们只关心“左 vs 右”是真实结构
# -----------------------------
labels_true = np.concatenate([
    np.ones(600, dtype=int),              # 类 1 → label=1
    np.full(600, 2, dtype=int),           # 类 2 → label=2
    np.full(n_outliers, 3, dtype=int)     # 类 3 → label=3（异常点）
])

# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_outliers.mat", {"data": data, "class": labels_true})
