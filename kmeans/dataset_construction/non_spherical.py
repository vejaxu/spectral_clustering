import numpy as np
from scipy.io import savemat
from sklearn.datasets import make_moons

# rng = np.random.default_rng(42)

# mean1, cov1 = [0, 0], [[10, 0], [0, 0.1]]
# data1_a = rng.multivariate_normal(mean1, cov1, 600)

# mean2, cov2 = [5, 5], [[0.1, 0], [0, 10]]
# data1_b = rng.multivariate_normal(mean2, cov2, 600)

# data = np.vstack([data1_a, data1_b])
# labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])

# rng = np.random.default_rng(42)
# mean1 = [0, 2]
# cov1 = [[10, 0], [0, 0.1]]
# data1 = rng.multivariate_normal(mean1, cov1, 600)

# mean2 = [0, -2]
# cov2 = [[10, 0], [0, 0.1]]
# data2 = rng.multivariate_normal(mean2, cov2, 600)

# data = np.vstack([data1, data2])
# labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])


# 设置随机种子生成器（和你风格一致）
rng = np.random.default_rng(42)

# 生成标准 Two-Moons 数据（600+600样本，噪声控制清晰度）
X, y = make_moons(n_samples=1200, noise=0.08, random_state=42)

# 🔁 旋转 90 度：将左右月牙 → 变成上下月牙
# 旋转矩阵：90度顺时针 = [[0, 1], [-1, 0]]
def rotate_90(X):
    return np.dot(X, np.array([[0, 1], [-1, 0]]))

X_rot = rotate_90(X)

# 📦 合并数据（直接使用旋转后的数据）
data = X_rot

# 🏷️ 标签保持不变（0 和 1）
labels = y.astype(int) + 1  # 将 0→1, 1→2，符合你习惯（class 1, class 2）

# 💾 保存为 .mat 文件（路径和键名完全一致）
savemat(
    "/home/xwj/aaa/clustering/data/kmeans/dataset_non_spherical_3.mat",
    {"data": data, "class": labels}
)