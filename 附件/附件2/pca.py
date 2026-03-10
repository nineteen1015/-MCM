import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # ================== 样式设置 ==================
# sns.set(style="whitegrid", palette="bright", font_scale=1.2)


# ================== 数据准备 ==================
def load_data():
    """加载并预处理数据"""
    data = pd.read_excel('Data.xlsx')
    X = data.iloc[:, :-1].values  # 特征数据
    labels = data.iloc[:, -1].values if data.shape[1] > 1 else None  # 标签数据
    return X, labels


# ================== PCA核心分析 ==================
def perform_pca_analysis(X):
    """执行完整PCA分析"""
    # 数据标准化
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 完整PCA分析
    pca_full = PCA()
    pca_full.fit(X_std)

    # 降维后的数据（取前两个主成分）
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X_std)

    return X_std, X_pca, pca_full


# ================== 可视化模块 ==================
def plot_contribution(pca_obj):
    """绘制贡献率图表"""
    explained_variance = pca_obj.explained_variance_
    explained_ratio = pca_obj.explained_variance_ratio_
    cumulative_ratio = np.cumsum(explained_ratio)

    plt.figure(figsize=(14, 6))

    # 主成分贡献率柱状图
    ax1 = plt.subplot(121)
    bars = ax1.bar(range(1, len(explained_ratio) + 1),
                   explained_ratio,
                   color='#1f77b4',
                   alpha=0.7,
                   edgecolor='black')

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.1%}',
                 ha='center', va='bottom')

    # 累积贡献率折线图
    ax2 = ax1.twinx()
    ax2.plot(range(1, len(cumulative_ratio) + 1),
             cumulative_ratio,
             color='#ff7f0e',
             marker='o',
             linewidth=2)

    # 图表装饰
    ax1.set_title("主成分方差贡献分析", fontsize=14)
    ax1.set_xlabel("主成分序号", fontsize=12)
    ax1.set_ylabel("方差贡献率", fontsize=12)
    ax2.set_ylabel("累积方差贡献率", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(False)

    # 数据表格
    ax3 = plt.subplot(122)
    ax3.axis('off')

    table_data = []
    for i, (ev, ratio, cum) in enumerate(zip(explained_variance,
                                             explained_ratio,
                                             cumulative_ratio)):
        table_data.append([
            f"PC{i + 1}",
            f"{ev:.4f}",
            f"{ratio * 100:.2f}%",
            f"{cum * 100:.2f}%"
        ])

    table = ax3.table(cellText=table_data[:15],
                      colLabels=["主成分", "特征值", "贡献率", "累积贡献率"],
                      loc='center',
                      cellLoc='center',
                      colColours=['#d3d3d3'] * 4)

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    plt.tight_layout()
    plt.show()


def plot_comparison(X_std, X_pca, labels):
    """绘制对比图表"""
    plt.figure(figsize=(16, 6))

    # 原始数据分布
    plt.subplot(1, 2, 1)
    if labels is not None:
        sns.scatterplot(x=X_std[:, 0], y=X_std[:, 1], hue=labels,
                        palette="viridis", s=100, edgecolor="k")
    else:
        plt.scatter(X_std[:, 0], X_std[:, 1], c='royalblue', s=100, edgecolor="k")
    plt.title("原始数据分布（前两个特征）", fontsize=14)
    plt.xlabel("特征 1", fontsize=12)
    plt.ylabel("特征 2", fontsize=12)

    # 降维结果
    plt.subplot(1, 2, 2)
    if labels is not None:
        sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels,
                        palette="plasma", s=100, edgecolor="k")
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c='crimson', s=100, edgecolor="k")
    plt.title("PCA降维结果（二维）", fontsize=14)
    plt.xlabel("主成分 1", fontsize=12)
    plt.ylabel("主成分 2", fontsize=12)

    plt.tight_layout()
    plt.show()


def plot_flowchart():
    """绘制算法流程图"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    steps = [
        ("1. 数据标准化", (0.1, 0.5)),
        ("2. 计算协方差矩阵", (0.3, 0.5)),
        ("3. 特征值分解", (0.5, 0.5)),
        ("4. 选择主成分", (0.7, 0.5)),
        ("5. 数据投影", (0.9, 0.5))
    ]

    for i, (step, (x, y)) in enumerate(steps):
        ax.text(x, y, step, ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.5', fc='#90EE90', ec='black', lw=2),
                fontsize=12)
        if i < len(steps) - 1:
            ax.annotate('', xy=(x + 0.18, y), xytext=(x - 0.18, y),
                        arrowprops=dict(arrowstyle='->', lw=3, color='#000080'))

    plt.title("PCA算法流程图", fontsize=14, pad=20)
    plt.show()


# ================== 主程序 ==================
if __name__ == "__main__":
    # 数据加载
    X, labels = load_data()

    # 执行PCA分析
    X_std, X_pca, pca_full = perform_pca_analysis(X)

    # 绘制贡献率图表
    plot_contribution(pca_full)

    # 绘制对比图表
    plot_comparison(X_std, X_pca, labels)

    # 绘制流程图
    plot_flowchart()