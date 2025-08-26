import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

data3_a = rng.multivariate_normal([0, 0], np.eye(2), 600)
cov3_b = [[4, 0], [0, 0.2]]
data3_b = rng.multivariate_normal([2, 0], cov3_b, 600)

data = np.vstack([data3_a, data3_b])
labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_overlapping.mat", {"data": data, "class": labels})
