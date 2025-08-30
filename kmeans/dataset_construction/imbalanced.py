import numpy as np
from scipy.io import savemat

# 设置随机种子
rng = np.random.default_rng(42)

# ----------------------------
# 生成三类球状数据
# ----------------------------

# 类1：大球，中心在原点，点数多，方差适中（球状扩散）
data1 = rng.multivariate_normal([0, 0], 1.0 * np.eye(2), 1000)

# 类2：小球，中心在 [4, 0]，方差小（紧凑）
data2 = rng.multivariate_normal([4, 0], 0.2 * np.eye(2), 200)

# 类3：小球，中心在 [5, 0]，与类2 靠近但分离
data3 = rng.multivariate_normal([5, 0], 0.2 * np.eye(2), 200)

# 合并数据和标签
data = np.vstack([data1, data2, data3])
labels = np.concatenate([
    np.ones(1000, dtype=int),
    np.full(200, 2, dtype=int),
    np.full(200, 3, dtype=int)
])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_imbalanced.mat", {"data": data, "class": labels})
