import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# 四个类的均值，排成一个 2x2 格子
means = [
    [-5, -5],  # 左下
    [ 5, -5],  # 右下
    [-5,  5],  # 左上
    [ 5,  5]   # 右上
]

cov = np.eye(2)  # 各类相同方差
n_per_class = 300

# 生成四个类
classes = [rng.multivariate_normal(mean, cov, n_per_class) for mean in means]

# 合并数据
data = np.vstack(classes)
labels = np.concatenate([np.full(n_per_class, i+1, dtype=int) for i in range(4)])

# 保存
savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_init_bias.mat", {"data": data, "class": labels})
