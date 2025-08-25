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
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
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
            random_state=i,  # 使用不同的随机种子
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


def plot_decision_boundaries(X, y, model, centers, resolution=0.02, title="Decision Boundaries"):
    """绘制决策边界"""
    # 设置颜色
    unique_labels = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
    color_list = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
                  'lightcyan', 'wheat', 'lavender', 'lightgray', 'lightsteelblue']
    
    # 绘制决策面
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # 创建自定义颜色映射
    if len(unique_labels) <= len(color_list):
        cmap = ListedColormap(color_list[:max(1, len(unique_labels))])
    else:
        cmap = plt.cm.Set1
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # 绘制数据点
    for idx, cl in enumerate(unique_labels):
        mask = (y == cl)
        if np.sum(mask) > 0:  # 确保有点要绘制
            plt.scatter(X[mask, 0], X[mask, 1],
                       alpha=0.8, c=[colors[idx % len(colors)]], 
                       label=f'Class {cl}', edgecolors='black', linewidth=0.5, s=50)
    
    # 绘制聚类中心
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', marker='x', s=200, linewidths=3, label='Centers')


def plot_voronoi_with_boundaries(X_2d, labels, centers, title="Voronoi Diagram with Boundaries"):
    """绘制维诺图与决策边界的综合图"""
    # 检查是否有足够的点来创建维诺图
    if len(centers) < 3:
        print(f"警告: 只有 {len(centers)} 个中心点，无法创建维诺图（至少需要3个点）")
        # 绘制简单的决策边界
        plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=50)
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3)
        plt.title(f"{title} (Insufficient points for Voronoi - n={len(centers)})")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True, alpha=0.3)
        return
    
    # 创建颜色映射
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
    color_list = ['lightcoral', 'lightblue', 'lightgreen', 'lightyellow', 'lightpink', 
                  'lightcyan', 'wheat', 'lavender', 'lightgray', 'lightsteelblue']
    
    # 创建网格用于决策边界
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 计算每个网格点到中心的距离（维诺图分配）
    distances = np.sqrt(((grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
    voronoi_labels = np.argmin(distances, axis=1)
    
    # 绘制维诺图区域背景
    plt.scatter(grid_points[:, 0], grid_points[:, 1], 
               c=[colors[label % len(colors)] for label in voronoi_labels], 
               alpha=0.1, s=1)
    
    # 绘制原始数据点
    for i, label in enumerate(unique_labels):
        cluster_points = X_2d[labels == label]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i % len(colors)]], alpha=0.8, s=60, 
                       edgecolors='black', linewidth=0.5, label=f'Class {label}')
    
    # 绘制中心点
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', marker='x', s=300, linewidths=4, label='Centers')
    
    # 绘制决策边界（等高线）
    Z = voronoi_labels.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=np.arange(len(centers))-0.5, 
                colors='black', linewidths=2, alpha=0.8)
    
    # 绘制维诺图边界
    try:
        vor = Voronoi(centers)
        voronoi_plot_2d(vor, show_vertices=False, line_colors='darkred', 
                        line_width=2, line_style='--', point_size=0, ax=plt.gca())
    except Exception as e:
        print(f"创建维诺图时出错: {e}")
        print("继续绘制其他元素...")
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlim([x_min, x_max])
    plt.ylim([y_min, y_max])
    plt.grid(True, alpha=0.3)


def plot_simple_boundaries(X_2d, labels, centers, title="Simple Boundaries"):
    """简单的边界可视化（适用于少于3个聚类的情况）"""
    plt.figure(figsize=(10, 8))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.Set1(np.linspace(0, 1, max(3, len(unique_labels))))
    
    # 对于2个聚类，可以绘制分隔线
    if len(centers) == 2:
        # 创建网格
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                             np.linspace(y_min, y_max, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        
        # 计算距离
        distances = np.sqrt(((grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
        voronoi_labels = np.argmin(distances, axis=1)
        
        # 绘制背景
        plt.scatter(grid_points[:, 0], grid_points[:, 1], 
                   c=[colors[label % len(colors)] for label in voronoi_labels], 
                   alpha=0.1, s=1)
        
        # 绘制分隔线
        mid_point = (centers[0] + centers[1]) / 2
        direction = centers[1] - centers[0]
        perpendicular = np.array([-direction[1], direction[0]])
        
        # 绘制分隔线
        line_length = 10
        line_points_x = [mid_point[0] - line_length * perpendicular[0], 
                        mid_point[0] + line_length * perpendicular[0]]
        line_points_y = [mid_point[1] - line_length * perpendicular[1], 
                        mid_point[1] + line_length * perpendicular[1]]
        plt.plot(line_points_x, line_points_y, 'r--', linewidth=2, label='Decision Boundary')
    
    # 绘制数据点
    for i, label in enumerate(unique_labels):
        cluster_points = X_2d[labels == label]
        if len(cluster_points) > 0:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       c=[colors[i % len(colors)]], alpha=0.8, s=60, 
                       edgecolors='black', linewidth=0.5, label=f'Class {label}')
    
    # 绘制中心点
    plt.scatter(centers[:, 0], centers[:, 1], 
               c='black', marker='x', s=300, linewidths=4, label='Centers')
    
    plt.title(title, fontsize=14)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)


def save_visualization_for_key(X_embedded, y_true, labels, centers, key, kmeans_model):
    """为特定key保存所有相关图片"""
    # 创建特定key的目录
    key_dir = f"fig_kmeans/{key}"
    os.makedirs(key_dir, exist_ok=True)

    # 0. 数据集可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_true, cmap='viridis', alpha=0.7, s=15)
    plt.title(f"Label - {key}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label='Class Labels')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{key_dir}/{key}_dataset.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 1. 基础t-SNE可视化
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
    plt.title(f"KMeans Clustering - {key}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.colorbar(scatter, label='Class Labels')
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{key_dir}/{key}_clustering_result.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 决策边界可视化
    plt.figure(figsize=(10, 8))
    try:
        plot_decision_boundaries(X_embedded, labels, kmeans_model, centers, resolution=0.1, 
                               title=f"Decision Boundaries - {key}")
        plt.tight_layout()
        plt.savefig(f"{key_dir}/{key}_decision_boundaries.jpg", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"绘制决策边界时出错: {e}")
        # 绘制简单版本
        plt.figure(figsize=(10, 8))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='viridis', alpha=0.7, s=15)
        plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3)
        plt.title(f"KMeans Clustering - {key} (Simple View)")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.savefig(f"{key_dir}/{key}_simple_view.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 高分辨率决策边界
    plt.figure(figsize=(12, 10))
    try:
        plot_decision_boundaries(X_embedded, labels, kmeans_model, centers, resolution=0.02,
                               title=f"High Resolution Decision Boundaries - {key}")
        plt.tight_layout()
        plt.savefig(f"{key_dir}/{key}_decision_boundaries_high_res.jpg", dpi=300, bbox_inches='tight')
    except Exception as e:
        print(f"绘制高分辨率决策边界时出错: {e}")
    plt.close()
    
    # 4. 根据聚类数量选择合适的边界可视化
    plt.figure(figsize=(10, 8))
    if len(centers) >= 3:
        # 使用维诺图可视化
        plot_voronoi_with_boundaries(X_embedded, labels, centers, f"Voronoi Diagram - {key}")
    else:
        # 使用简单边界可视化
        plot_simple_boundaries(X_embedded, labels, centers, f"Simple Boundaries - {key}")
    
    plt.tight_layout()
    plt.savefig(f"{key_dir}/{key}_boundaries.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 简化版边界对比图
    plt.figure(figsize=(12, 5))
    
    # 左侧：基础聚类
    plt.subplot(1, 2, 1)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=30)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3)
    plt.title(f"Clustering Result - {key}")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, alpha=0.3)
    
    # 右侧：带边界
    plt.subplot(1, 2, 2)
    if len(centers) >= 3:
        try:
            plot_voronoi_with_boundaries(X_embedded, labels, centers, f"Boundaries - {key}")
        except:
            plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap=plt.cm.Set1, alpha=0.7, s=30)
            plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidths=3)
            plt.title(f"Boundaries - {key} (Fallback)")
    else:
        plot_simple_boundaries(X_embedded, labels, centers, f"Simple Boundaries - {key}")
        plt.title(f"Simple Boundaries - {key}")
    
    plt.tight_layout()
    plt.savefig(f"{key_dir}/{key}_comparison.jpg", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已为 {key} 保存所有可视化结果到 {key_dir}/")


def process_single_dataset(key, mat_file, x_key, y_key, n_runs=10):
    """处理单个数据集"""
    print(f"\n{'='*50}")
    print(f"处理数据集: {key}")
    print(f"{'='*50}")
    
    try:
        X, y_true = load_data_from_mat(mat_file, x_key, y_key)
        n_clusters = np.unique(y_true).size
        print(f"{x_key} clusters: {n_clusters}")
        print(f"dataset points: {X.shape[0]}")
        print(f"dataset features: {X.shape[1]}")

        mean_nmi, mean_ari, mean_f1, mean_times, labels, nmis, aris, f1s = run_kmeans(X, n_clusters, n_runs, y_true)
        print(f"mean NMI = {mean_nmi:.3f}")
        print(f"mean ARI = {mean_ari:.3f}")
        print(f"mean F1 = {mean_f1:.3f}")
        print(f"mean time: {np.mean(mean_times):.3f}")

        # 获取聚类中心
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_model.fit(X)
        centers = kmeans_model.cluster_centers_

        if X.shape[1] > 2:
            X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X)
        else:
            X_embedded = X

        # 为当前key保存可视化结果
        save_visualization_for_key(X_embedded, y_true, labels, centers, key, kmeans_model)
        
        return {
            'key': key,
            'nmi': mean_nmi,
            'ari': mean_ari,
            'f1': mean_f1,
            'time': np.mean(mean_times),
            'n_clusters': n_clusters,
            'n_points': X.shape[0],
            'n_features': X.shape[1]
        }
        
    except Exception as e:
        print(f"处理 {key} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # === 修改为你的数据文件和变量名 ===
    key = "outliers2"
    mat_file = f'../data/kmeans/dataset_{key}.mat'

    # landsat waveform3
    """ x_key = 'data'
    y_key = 'label' """

    # pendigits
    """ x_key = 'X'
    y_key = 'gtlabels' """

    # spiral 4C AC RinG complex9 spam
    x_key = 'data'
    y_key = 'class'

    n_runs = 10

    # 处理单个数据集
    result = process_single_dataset(key, mat_file, x_key, y_key, n_runs)
    
    if result:
        print(f"\n{'='*60}")
        print("处理完成")
        print(f"{'='*60}")
        print(f"Dataset: {result['key']}")
        print(f"Clusters: {result['n_clusters']}")
        print(f"Points: {result['n_points']}")
        print(f"Features: {result['n_features']}")
        print(f"NMI: {result['nmi']:.3f}")
        print(f"ARI: {result['ari']:.3f}")
        print(f"F1: {result['f1']:.3f}")
        print(f"Time: {result['time']:.3f}s")
        print(f"可视化结果已保存到 fig_kmeans/{key}/")
