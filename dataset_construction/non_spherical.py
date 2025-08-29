import numpy as np
from scipy.io import savemat

# rng = np.random.default_rng(42)

# mean1, cov1 = [0, 0], [[10, 0], [0, 0.1]]
# data1_a = rng.multivariate_normal(mean1, cov1, 600)

# mean2, cov2 = [5, 5], [[0.1, 0], [0, 10]]
# data1_b = rng.multivariate_normal(mean2, cov2, 600)

# data = np.vstack([data1_a, data1_b])
# labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])

rng = np.random.default_rng(42)
mean1 = [0, 2]
cov1 = [[10, 0], [0, 0.1]]
data1 = rng.multivariate_normal(mean1, cov1, 600)

mean2 = [0, -2]
cov2 = [[10, 0], [0, 0.1]]
data2 = rng.multivariate_normal(mean2, cov2, 600)

data = np.vstack([data1, data2])
labels = np.concatenate([np.ones(600, dtype=int), np.full(600, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_non_spherical_2.mat", {"data": data, "class": labels})
