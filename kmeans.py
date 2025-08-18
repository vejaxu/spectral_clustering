import numpy as np
from scipy.io import loadmat
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import time
import os


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def align_labels(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    row_ind, col_ind = linear_sum_assignment(-cm)
    label_mapping = {col: row for row, col in zip(row_ind, col_ind)}
    aligned_pred = np.array([label_mapping[label] for label in pred_labels])
    return aligned_pred


def load_data_from_mat(filename, x_key='X', y_key='y'):
    data = loadmat(filename)
    print("加载变量名:", [k for k in data.keys() if not k.startswith('__')])
    X = data[x_key]
    y = data[y_key].ravel()
    return X, y


def run_kmeans(X, n_clusters, n_runs, y_true):
    nmis = []
    aris = []
    f1s = []
    times = []

    for i in range(n_runs):
        clustering = KMeans(
            n_clusters=n_clusters, 
            random_state=42, 
            n_init=10
        )
        start_time = time.time()
        labels = clustering.fit_predict(X)
        end_time = time.time()
        labels = align_labels(y_true, labels)
        nmi = normalized_mutual_info_score(y_true, labels)
        ari = adjusted_rand_score(y_true, labels)
        f1 = f1_score(y_true, labels, average='macro')
        nmis.append(nmi)
        aris.append(ari)
        f1s.append(f1)
        times.append(end_time - start_time)

    return np.mean(nmis), np.mean(aris), np.mean(f1s), np.mean(times), labels, nmis, aris, f1s


if __name__ == '__main__':
    # === 修改为你的数据文件和变量名 ===
    key = "4C"
    mat_file = f'../data/{key}.mat'

    # spiral 4C AC RinG complex9 spam
    x_key = 'data'
    y_key = 'class'

    # landsat waveform3
    """ x_key = 'data'
    y_key = 'label' """

    # pendigits
    """ x_key = 'X'
    y_key = 'gtlabels' """

    n_runs = 10

    X, y_true = load_data_from_mat(mat_file, x_key, y_key)
    n_clusters = np.unique(y_true).size
    print(f"{x_key} clusters: {n_clusters}")
    print(f"dataset points: {X.shape[0]}")
    print(f"dataset features: {X.shape[1]}")

    mean_nmi, mean_ari, mean_f1, mean_times, labels, nmis, aris, f1s = run_kmeans(X, n_clusters, n_runs, y_true)
    print(f"mean NMI = {mean_nmi:.3f}")
    print(f"mean ari = {mean_ari:.3f}")
    print(f"mean f1 = {mean_f1:.3f}")
    print(f"mean time: {np.mean(mean_times):.3f}")

    if X.shape[1] > 2:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        X_embedded = X

    def plot_tsne(X_2d, labels, title):
        # plt.subplot(1, 2, subplot_idx)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar(scatter, label='Class Labels')
        plt.grid(True)
        plt.axis('equal')

    # plt.figure(figsize=(12, 6))
    # plot_tsne(X_embedded, y_true, "True Labels", 1)
    # # plot_tsne(X_embedded, best_overall_labels, "Spectral Clustering", 2)
    # plot_tsne(X_embedded, labels, "Kmeans", 2)
    # plt.tight_layout()
    # plt.savefig(f"fig_kmeans/{key}.jpg", dpi=300)

    plt.figure(figsize=(6, 6))
    plot_tsne(X_embedded, labels, "KMeans")
    plt.tight_layout()
    os.makedirs("fig_kmeans", exist_ok=True)
    plt.savefig(f"fig_kmeans/{key}.jpg", dpi=300)