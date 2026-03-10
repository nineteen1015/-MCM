# ======================= 初始化设置 =======================
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import VarianceThreshold  # 关键修复
from scipy.optimize import minimize
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 全局随机种子
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

# ======================= 数据准备 =======================
def load_data():
    """加载并预处理数据"""
    data_a = pd.read_excel('5-X.xlsx', header=None).values.astype(np.float32)
    data_b = pd.read_excel('5-Y.xlsx', header=None).values.ravel().astype(np.float32)

    # 特征筛选（去除低方差特征）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_a)
    selector = VarianceThreshold(threshold=0.5)
    X_filtered = selector.fit_transform(X_scaled)

    print(f"原始特征数: {data_a.shape[1]} -> 筛选后: {X_filtered.shape[1]}")
    return X_filtered, data_b


# ======================= 模型定义 =======================
class ModelWrapper:
    """模型结果容器"""

    def __init__(self):
        self.results = {}

    def add_result(self, name, y_pred, metrics):
        self.results[name] = {'predictions': y_pred, 'metrics': metrics}

    def show_results(self):
        """格式化输出结果"""
        results_df = pd.DataFrame(
            {k: v['metrics'] for k, v in self.results.items()}
        ).T.sort_values(by='R^2', ascending=False)

        # 控制台输出
        print("\n======== 模型性能对比 ========")
        print(results_df[['R^2', 'MSE', 'MAE']].to_string(
            float_format=lambda x: f"{x:.4f}",
            justify='center',
            col_space=12
        ))

        # 可视化输出
        plt.figure(figsize=(10, 6))
        sns.heatmap(results_df[['R^2']].T, annot=True, fmt=".2f", cmap="YlGnBu",
                    cbar=False, linewidths=1, linecolor='gray')
        plt.title('模型R^2值对比热力图', pad=20)
        plt.xticks(rotation=30)
        plt.show()


# ======================= 模型实现 =======================
def run_linear(X_train, X_test, y_train, y_test):
    """模型1: 线性回归（带交叉验证）"""
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        LinearRegression()
    )
    # 交叉验证防止数据泄漏
    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"线性回归交叉验证R^2: {scores.mean():.4f} (±{scores.std():.4f})")

    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)


def run_polynomial(X_train, X_test, y_train, y_test):
    """模型2: 改进多项式回归"""
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=10),
        PolynomialFeatures(degree=2, interaction_only=True),
        RidgeCV(alphas=np.logspace(1, 3, 20))
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)


def run_ridge(X_train, X_test, y_train, y_test):
    """模型3: 岭回归（带交叉验证）"""
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=0.95),
        RidgeCV(alphas=np.logspace(3, 5, 20), cv=5)
    )
    pipeline.fit(X_train, y_train)
    return pipeline.predict(X_test)


def run_nn(X_train, X_test, y_train, y_test):
    """模型4: 神经网络"""
    # 标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

    # 网络结构
    class Net(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            return self.layers(x)

    # 训练配置
    model = Net(X_train_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.MSELoss()

    # 训练循环
    X_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)

    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    # 预测
    with torch.no_grad():
        y_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    return scaler_y.inverse_transform(y_pred).ravel()


def run_optimization(X_train, X_test, y_train, y_test):
    """模型5: 改进优化算法（正则化多项式）"""
    # 特征工程
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[:, :15])  # 取前15个重要特征
    X_test_poly = poly.transform(X_test[:, :15])

    # 正则化回归
    model = RidgeCV(alphas=np.logspace(1, 4, 50), cv=5)
    model.fit(X_train_poly, y_train)
    return model.predict(X_test_poly)


# ======================= 可视化模块 =======================
def plot_comparisons(wrapper, y_test):
    """综合可视化"""
    colors = sns.color_palette("husl", len(wrapper.results))
    n_models = len(wrapper.results)

    # 动态布局
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    # 残差分布
    plt.figure(figsize=(5 * n_cols, 4 * n_rows))
    for i, (name, res) in enumerate(wrapper.results.items(), 1):
        plt.subplot(n_rows, n_cols, i)
        residuals = y_test - res['predictions']
        sns.histplot(residuals, kde=True, color=colors[i - 1], bins=30)
        plt.title(f'{name}残差分布\n偏度: {pd.Series(residuals).skew():.2f}')
        plt.xlabel('残差值')
    plt.tight_layout()

    # 误差分布
    plt.figure(figsize=(12, 6))
    error_data = {name: np.abs(y_test - res['predictions'])
                  for name, res in wrapper.results.items()}
    sns.boxplot(data=pd.DataFrame(error_data), palette=colors)
    plt.title('绝对误差分布箱线图')
    plt.ylabel('绝对误差')
    plt.xticks(rotation=20)
    plt.show()


# ======================= 主程序 =======================
if __name__ == "__main__":
    # 数据加载
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 模型初始化
    wrapper = ModelWrapper()
    models = {
        "岭回归": run_ridge,
        "神经网络": run_nn,
        "线性回归": run_linear,
        "多项式回归": run_polynomial,
        "优化算法": run_optimization
    }

    # 模型训练与评估
    for name, model_func in models.items():
        y_pred = model_func(X_train, X_test, y_train, y_test)
        metrics = {
            'R^2': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }
        wrapper.add_result(name, y_pred, metrics)

    # 结果展示
    wrapper.show_results()
    plot_comparisons(wrapper, y_test)
# # 综合模型对比分析
# import matplotlib
# matplotlib.use('TkAgg')
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.decomposition import PCA
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.optimize import minimize
# import torch
# import torch.nn as nn
# import torch.optim as optim
#
# # 设置可视化样式
# sns.set_theme(style="whitegrid")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # ------------------------- 数据准备 -------------------------
# # 读取数据
# data_a = pd.read_excel('3-X.xlsx', header=None).values
# data_b = pd.read_excel('3-Y.xlsx', header=None).values.ravel()
#
# # 全局随机种子
# RANDOM_STATE = 42
# np.random.seed(RANDOM_STATE)
# torch.manual_seed(RANDOM_STATE)
#
#
# # ------------------------- 模型定义 -------------------------
# class ModelWrapper:
#     def __init__(self):
#         self.results = {}
#
#     def add_model(self, name, y_pred, metrics):
#         self.results[name] = {
#             'predictions': y_pred,
#             'metrics': metrics
#         }
#
#
# # ------------------------- 模型训练函数 -------------------------
# def run_linear_regression(X_train, X_test, y_train, y_test):
#     """模型1: 线性回归（PCA+标准化）"""
#     # 预处理
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X_train)
#     pca = PCA(n_components=0.95)
#     X_pca = pca.fit_transform(X_scaled)
#     # 训练
#     model = LinearRegression()
#     model.fit(X_pca, y_train)
#     # 测试预处理
#     X_test_scaled = scaler.transform(X_test)
#     X_test_pca = pca.transform(X_test_scaled)
#     y_pred = model.predict(X_test_pca)
#     return y_pred
#
#
# def run_polynomial_regression(X_train, X_test, y_train, y_test):
#     """模型2: 多项式回归（均值特征）"""
#     # 特征转换
#     X_train_mean = np.mean(X_train, axis=1).reshape(-1, 1)
#     X_test_mean = np.mean(X_test, axis=1).reshape(-1, 1)
#     # 多项式特征
#     poly = PolynomialFeatures(degree=2)
#     X_train_poly = poly.fit_transform(X_train_mean)
#     X_test_poly = poly.transform(X_test_mean)
#     # 训练
#     model = LinearRegression()
#     model.fit(X_train_poly, y_train)
#     y_pred = model.predict(X_test_poly)
#     return y_pred
#
#
# def run_ridge_regression(X_train, X_test, y_train, y_test):
#     """模型3: 岭回归（多列特征）"""
#     # 标准化
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     # 添加偏置项
#     X_train_ = np.hstack([X_train_scaled, np.ones((len(X_train_scaled), 1))])
#     X_test_ = np.hstack([X_test_scaled, np.ones((len(X_test_scaled), 1))])
#     # 训练
#     model = Ridge(alpha=1.0)
#     model.fit(X_train_, y_train)
#     y_pred = model.predict(X_test_)
#     return y_pred
#
#
# def run_neural_network(X_train, X_test, y_train, y_test):
#     """模型4: 神经网络"""
#     # 标准化
#     scaler_X = StandardScaler()
#     scaler_y = StandardScaler()
#     X_train_scaled = scaler_X.fit_transform(X_train)
#     X_test_scaled = scaler_X.transform(X_test)
#     y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
#     # 转换为张量
#     X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).reshape(-1, 1)
#     X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
#
#     # 定义模型
#     class Net(nn.Module):
#         def __init__(self, input_dim):
#             super().__init__()
#             self.layers = nn.Sequential(
#                 nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128),
#                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
#
#         def forward(self, x):
#             return self.layers(x)
#
#     model = Net(X_train_scaled.shape[1])
#     # 训练配置
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     # 早停法训练
#     best_loss, patience, no_improve = float('inf'), 20, 0
#     for epoch in range(1000):
#         model.train()
#         optimizer.zero_grad()
#         outputs = model(X_train_tensor)
#         loss = criterion(outputs, y_train_tensor)
#         loss.backward()
#         optimizer.step()
#         # 验证
#         model.eval()
#         with torch.no_grad():
#             val_pred = model(X_test_tensor)
#             val_loss = criterion(val_pred, torch.tensor(scaler_y.transform(y_test.reshape(-1, 1)), dtype=torch.float32))
#         if val_loss < best_loss:
#             best_loss, no_improve = val_loss, 0
#         else:
#             no_improve += 1
#         if no_improve >= patience: break
#     # 预测
#     with torch.no_grad():
#         y_pred_scaled = model(X_test_tensor).numpy()
#     y_pred = scaler_y.inverse_transform(y_pred_scaled).ravel()
#     return y_pred
#
#
# def run_optimization(X_train, X_test, y_train, y_test):
#     """模型5: 优化算法（均值特征）"""
#     # 特征转换
#     X_single = np.mean(X_train, axis=1)
#
#     # 定义目标函数
#     def objective(params):
#         a, b = params
#         return np.mean((a * X_single + b - y_train) ** 2)
#
#     # 优化
#     result = minimize(objective, [1.0, 0.0], method='L-BFGS-B')
#     a_opt, b_opt = result.x
#     # 测试预测
#     X_test_single = np.mean(X_test, axis=1)
#     y_pred = a_opt * X_test_single + b_opt
#     return y_pred
#
# # ------------------------- 新增数据探索可视化 -------------------------
# # 1. 特征与B的散点图（前5个特征示例）
# plt.figure(figsize=(15, 8))
# for i in range(5):  # 展示前5个特征
#     plt.subplot(2, 3, i+1)
#     sns.scatterplot(x=data_a[:, i], y=data_b, alpha=0.6, edgecolor='none')
#     plt.title(f'特征 {i+1} vs B', fontsize=10)
#     plt.xlabel(f'特征 {i+1}', fontsize=8)
#     plt.ylabel('B', fontsize=8)
#     plt.grid(linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.suptitle('特征与目标变量B的散点图（前5个特征示例）', y=1.02, fontsize=12)
# plt.show()
#
# # 2. 相关系数热力图（含B）
# # 合并数据并计算相关系数
# df_combined = pd.DataFrame(data_a).iloc[:, :10]  # 取前10个特征避免过载
# df_combined['B'] = data_b
# corr_matrix = df_combined.corr()
#
# # 绘制热力图
# plt.figure(figsize=(12, 8))
# mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 隐藏上半三角
# sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=True, fmt=".2f",
#             cbar_kws={'label': 'Pearson相关系数'},
#             annot_kws={'size': 8}, vmin=-1, vmax=1)
# plt.title('特征与目标变量B的相关性热力图（前10个特征）', fontsize=12)
# plt.xticks(rotation=45, ha='right', fontsize=8)
# plt.yticks(rotation=0, fontsize=8)
# plt.tight_layout()
# plt.show()
#
# # ------------------------- 主程序 -------------------------
# if __name__ == "__main__":
#     # 统一数据划分
#     X_train, X_test, y_train, y_test = train_test_split(
#         data_a, data_b, test_size=0.2, random_state=RANDOM_STATE
#     )
#
#     # 初始化结果容器
#     wrapper = ModelWrapper()
#
#     # 运行所有模型
#     models = {
#         "线性回归": run_linear_regression,
#         "多项式回归": run_polynomial_regression,
#         "岭回归": run_ridge_regression,
#         "神经网络": run_neural_network,
#         "优化算法": run_optimization
#     }
#
#     for name, func in models.items():
#         y_pred = func(X_train, X_test, y_train, y_test)
#         metrics = {
#             'MSE': mean_squared_error(y_test, y_pred),
#             'R2': r2_score(y_test, y_pred),
#             'MAE': np.mean(np.abs(y_test - y_pred))
#         }
#         wrapper.add_model(name, y_pred, metrics)
#     # ------------------------- 可视化增强 -------------------------
#     # 设置统一颜色主题
#     model_colors = sns.color_palette("husl", n_colors=len(models))
#
#
#     # 扩展1: 残差分布图（带KDE）
#     plt.figure(figsize=(15, 10))
#     for i, (name, result) in enumerate(wrapper.results.items(), 1):
#         residuals = y_test - result['predictions']
#         plt.subplot(2, 3, i)
#         sns.histplot(residuals, kde=True, color=model_colors[i - 1],
#                      bins=30, edgecolor='w', linewidth=0.5)
#         plt.axvline(0, color='r', linestyle='--', linewidth=1)
#         plt.title(f'{name}残差分布\n偏度: {pd.Series(residuals).skew():.2f}')
#         plt.xlabel('残差值')
#     plt.tight_layout()
#     plt.suptitle('各模型残差分布对比', y=1.02)
#     plt.show()
#
#     # 扩展2: 误差箱线图对比
#     error_data = {
#         name: np.abs(y_test - result['predictions'])
#         for name, result in wrapper.results.items()
#     }
#     plt.figure(figsize=(12, 6))
#     sns.boxplot(data=pd.DataFrame(error_data), palette=model_colors,
#                 showmeans=True, meanprops={"marker": "o", "markerfacecolor": "white"})
#     plt.title('各模型绝对误差分布箱线图', fontsize=14)
#     plt.ylabel('绝对误差', fontsize=12)
#     plt.xticks(rotation=20, fontsize=10)
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()
#
#     # 扩展3: 累积分布函数图（CDF）
#     plt.figure(figsize=(10, 6))
#     for (name, result), color in zip(wrapper.results.items(), model_colors):
#         abs_errors = np.abs(y_test - result['predictions'])
#         sorted_errors = np.sort(abs_errors)
#         cdf = np.linspace(0, 1, len(sorted_errors))
#         plt.plot(sorted_errors, cdf, label=name, color=color, linewidth=2)
#
#     plt.axvline(x=0.1, color='grey', linestyle='--', alpha=0.7)
#     plt.text(0.11, 0.5, '误差阈值', rotation=90, va='center')
#     plt.title('累积误差分布函数（CDF）', fontsize=14)
#     plt.xlabel('绝对误差', fontsize=12)
#     plt.ylabel('累积概率', fontsize=12)
#     plt.legend(loc='lower right')
#     plt.grid(linestyle='--', alpha=0.5)
#     plt.show()
#
#     # 扩展4: 特征重要性图（以岭回归为例）
#     ridge_model = Ridge(alpha=1.0).fit(
#         np.hstack([StandardScaler().fit_transform(data_a), np.ones((len(data_a), 1))]),
#         data_b
#     )
#     coef = ridge_model.coef_[:-1]  # 排除偏置项
#
#     plt.figure(figsize=(12, 6))
#     sns.barplot(x=np.arange(len(coef)), y=np.abs(coef),
#                 palette=sns.color_palette("Blues_d", n_colors=len(coef)))
#     plt.title('岭回归特征重要性（系数绝对值）', fontsize=14)
#     plt.xlabel('特征编号', fontsize=12)
#     plt.ylabel('重要性', fontsize=12)
#     plt.xticks(rotation=45)
#     plt.grid(axis='y', linestyle='--', alpha=0.5)
#     plt.show()
#
#     # 扩展5: 预测趋势对比图
#     sorted_idx = np.argsort(y_test)
#     plt.figure(figsize=(15, 8))
#     plt.plot(y_test[sorted_idx], label='真实值', color='black',
#              linewidth=3, alpha=0.8, marker='o', markersize=5)
#
#     for (name, result), color in zip(wrapper.results.items(), model_colors):
#         plt.plot(result['predictions'][sorted_idx],
#                  label=name, color=color, linewidth=2, alpha=0.7)
#
#     plt.title('模型预测趋势对比', fontsize=14)
#     plt.xlabel('样本序号（按真实值排序）', fontsize=12)
#     plt.ylabel('数值', fontsize=12)
#     plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
#     plt.grid(linestyle='--', alpha=0.5)
#     plt.tight_layout()
#     plt.show()
#
