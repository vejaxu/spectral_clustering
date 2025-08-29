import numpy as np
from scipy.io import savemat

rng = np.random.default_rng(42)

# Ring
n_ring = 1000
angles = rng.uniform(0, 2*np.pi, n_ring)
radii = 5 + rng.normal(0, 0.5, n_ring)
ring = np.column_stack([radii*np.cos(angles), radii*np.sin(angles)])

# Core
core = rng.multivariate_normal([0, 0], np.eye(2), 200)

data = np.vstack([ring, core])
labels = np.concatenate([np.ones(n_ring, dtype=int), np.full(200, 2, dtype=int)])

savemat("/home/xwj/aaa/clustering/data/kmeans/dataset_metric_mismatch.mat", {"data": data, "class": labels})
