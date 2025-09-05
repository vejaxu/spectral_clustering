import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# 使用各向异性协方差：纵向更扩散，增强“上下”感知
cov = np.array([[0.3, 0.0],
                [0.0, 0.8]])  # y 方向方差更大

# -----------------------------
# 类 1：左侧类，上下两个子簇
# -----------------------------
mean1_upper = [-2.5,  1.5]
mean1_lower = [-2.5, -1.5]

class1_upper = rng.multivariate_normal(mean1_upper, cov, 300)
class1_lower = rng.multivariate_normal(mean1_lower, cov, 300)
class1 = np.vstack([class1_upper, class1_lower])

# -----------------------------
# 类 2：右侧类，上下两个子簇
# -----------------------------
mean2_upper = [2.5,  1.5]
mean2_lower = [2.5, -1.5]

class2_upper = rng.multivariate_normal(mean2_upper, cov, 300)
class2_lower = rng.multivariate_normal(mean2_lower, cov, 300)
class2 = np.vstack([class2_upper, class2_lower])

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
