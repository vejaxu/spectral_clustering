import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# -----------------------------
# 类 1 和 类 2：两个靠近但可分的主类（位于中心）
# -----------------------------
mean1 = [0, 0]           # 类 1 中心
mean2 = [3, 0]           # 类 2 中心
cov = 0.4 * np.eye(2)    # 类内扩散适中

class1 = rng.multivariate_normal(mean1, cov, 600)
class2 = rng.multivariate_normal(mean2, cov, 600)

# -----------------------------
# 异常点：少量（6 个），远离两个类，且到两者距离相近
# -----------------------------
n_outliers = 6
# 放在两个类的正上方中点位置
mean_outlier = [1.5, 8]  # x 在 0 和 3 中间，y=8 → 到两个类的距离 ≈ sqrt(1.5^2 + 8^2) ≈ 8.14
cov_outlier = 0.1 * np.eye(2)  # 非常集中
outliers = rng.multivariate_normal(mean_outlier, cov_outlier, n_outliers)

# -----------------------------
# 合并数据
# -----------------------------
data = np.vstack([class1, class2, outliers])  # 1206 x 2

# -----------------------------
# 真实标签：异常点属于 类2（label=2）
# -----------------------------
labels = np.concatenate([
    np.ones(600, dtype=int),                    # 类 1 → label=1
    np.full(600, 2, dtype=int),                 # 类 2 → label=2
    np.full(n_outliers, 2, dtype=int)           # 异常点 → 属于类2（离群样本）
])

# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_outliers.mat", {"data": data, "class": labels})
