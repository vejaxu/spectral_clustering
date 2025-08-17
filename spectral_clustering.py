import numpy as np
from scipy.io import loadmat
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
import time


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


def run_spectral_clustering(X, n_clusters, gamma, n_runs, y_true):
    nmis = []
    aris = []
    f1s = []
    times = []

    for i in range(n_runs):
        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='rbf',
            gamma=gamma,
            assign_labels='kmeans',
            random_state=42
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
    key = "sparse_8_dense_1_dense_1"
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

    # gamma_values = [2 ** i for i in range(-5, 6)]
    # best_overall_nmi = -1
    # best_overall_ari = -1
    # best_overall_f1 = -1
    # time_lst = []
    # best_gamma = None
    # best_overall_labels = None
    # for gamma in tqdm(gamma_values, desc="Grid Search over gamma"):
    #     mean_nmi, mean_ari, mean_f1, mean_times, labels, nmis, aris, f1s = run_spectral_clustering(X, n_clusters, gamma, n_runs, y_true)
    #     time_lst.append(mean_times)
    #     if mean_nmi > best_overall_nmi:
    #         best_overall_nmi = mean_nmi
    #         best_overall_labels = labels
    #         best_gamma = gamma
    #     if mean_ari > best_overall_ari:
    #         best_overall_ari = mean_ari
    #     if mean_f1 > best_overall_f1:
    #         best_overall_f1 = mean_f1
    #     print()
    #     print(f"gamma value: {gamma}")
    #     print(f"nmis: {nmis}")
    #     print(mean_nmi)
    #     print(f"aris: {aris}")
    #     print(mean_ari)
    #     print(f"f1s: {f1s}")
    #     print(mean_f1)
    #     print("*" * 500)
    # print(f"Best gamma = {best_gamma:.5f}")
    # print(f"Best mean NMI = {best_overall_nmi:.3f}")
    # print(f"Best mean ari = {best_overall_ari:.3f}")
    # print(f"Best mean f1 = {best_overall_f1:.3f}")
    # print(f"mean time: {np.mean(time_lst):.3f}")

    mean_nmi, mean_ari, mean_f1, mean_times, labels, nmis, aris, f1s = run_kmeans(X, n_clusters, n_runs, y_true)
    print(f"mean NMI = {mean_nmi:.3f}")
    print(f"mean ari = {mean_ari:.3f}")
    print(f"mean f1 = {mean_f1:.3f}")
    print(f"mean time: {np.mean(mean_times):.3f}")

    if X.shape[1] > 2:
        X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        X_embedded = X
    # 可视化函数，风格统一
    def plot_tsne(X_2d, labels, title, subplot_idx):
        plt.subplot(1, 2, subplot_idx)
        scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
        plt.title(title)
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.colorbar(scatter, label='Class Labels')
        plt.grid(True)
        plt.axis('equal')
    # 可视化整体布局
    plt.figure(figsize=(12, 6))
    plot_tsne(X_embedded, y_true, "True Labels", 1)
    # plot_tsne(X_embedded, best_overall_labels, "Spectral Clustering", 2)
    plot_tsne(X_embedded, labels, "Kmeans", 2)
    plt.tight_layout()
    plt.savefig(f"fig_kmeans/{key}.jpg", dpi=300)
    # plt.show()