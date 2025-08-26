import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

data5_a = rng.multivariate_normal([-5, 0], np.eye(2), 600)
data5_b = rng.multivariate_normal([5, 0], np.eye(2), 600)

data = np.vstack([data5_a, data5_b])
labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_initilization_bias.mat", {"data": data, "class": labels})
