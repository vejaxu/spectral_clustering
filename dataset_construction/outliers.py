import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# 类 1 & 类 2：两个比较接近的小方差高斯分布
mean1 = [0, 0]
mean2 = [2, 2]  # 均值靠近
cov_small = 0.2 * np.eye(2)  # 小方差

class1 = rng.multivariate_normal(mean1, cov_small, 600)
class2 = rng.multivariate_normal(mean2, cov_small, 600)

# 类 3：远离的大方差高斯分布（异常点）
mean3 = [10, -10]   # 尽量远离
cov_large = 5 * np.eye(2)  # 大方差
class3 = rng.multivariate_normal(mean3, cov_large, 10)

# 合并数据
data = np.vstack([class1, class2, class3])
labels = np.concatenate([
    np.ones(600, dtype=int),
    np.full(600, 2, dtype=int),
    np.full(10, 3, dtype=int)
])

# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_outliers.mat", {"data": data, "class": labels})
