import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

data2_a = rng.multivariate_normal([0, 0], np.eye(2), 1100)
data2_b = rng.multivariate_normal([5, 0], np.eye(2), 100)

data = np.vstack([data2_a, data2_b])
labels = np.concatenate([np.ones(1100, dtype=int), np.full(100, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_imbalanced.mat", {"data": data, "class": labels})
