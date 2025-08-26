import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

data_core = rng.multivariate_normal([0, 0], np.eye(2), 1190)
outliers = 20 + rng.uniform(-1, 1, size=(10, 2))

data = np.vstack([data_core, outliers])
labels = np.concatenate([np.ones(1190, dtype=int), np.full(10, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_outliers2.mat", {"data": data, "class": labels})
