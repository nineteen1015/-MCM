#优化算法框架（SciPy 最小化）
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def convert_to_single_column(data, method='mean'):
    """将多列数据转换为单列"""
    if method == 'mean':
        return np.mean(data, axis=1).reshape(-1, 1)
    elif method == 'sum':
        return np.sum(data, axis=1).reshape(-1, 1)
    elif method == 'pca':
        pca = PCA(n_components=1)
        return pca.fit_transform(data)
    else:
        raise ValueError("可用方法：'mean', 'sum', 'pca'")

# 读取数据
data_a = pd.read_excel('A.xlsx').values
data_b = pd.read_excel('B.xlsx').values.flatten()

# 数据校验
if len(data_a) != len(data_b):
    raise ValueError("数据集A和B的样本数量不一致")

# 选择转换方法（可修改为'sum'或'pca'）
method = 'mean'
X_single = convert_to_single_column(data_a, method=method).flatten()

# 定义优化目标函数
def objective(params):
    a, b = params
    y_pred = a * X_single + b
    return np.mean((y_pred - data_b) ** 2)  # MSE作为损失函数

# 参数优化
initial_guess = [1.0, 0.0]
result = minimize(objective, initial_guess, method='L-BFGS-B')

# 获取优化参数
a_opt, b_opt = result.x
optimized_values = a_opt * X_single + b_opt

# 计算评估指标
r2 = r2_score(data_b, optimized_values)
mae = np.mean(np.abs(optimized_values - data_b))

# 计算准确率（误差在10%以内视为正确）
threshold = 0.10  # 可调整阈值
relative_error = np.abs((optimized_values - data_b) / data_b)
accuracy = np.mean(relative_error <= threshold) * 100

# 生成对比图表
plt.figure(figsize=(12, 6))

# 真实值 vs 预测值散点图
plt.subplot(1, 2, 1)
plt.scatter(data_b, optimized_values, alpha=0.6)
plt.plot([min(data_b), max(data_b)], [min(data_b), max(data_b)], 'r--')
plt.xlabel('真实值(B)')
plt.ylabel('优化后预测值')
plt.title(f'真实值 vs 预测值\nR²={r2:.2f}'.replace('²', '^2'))

# 误差分布直方图
plt.subplot(1, 2, 2)
plt.hist(relative_error, bins=30, edgecolor='k', alpha=0.7)
plt.axvline(threshold, color='r', linestyle='--')
plt.xlabel('相对误差')
plt.ylabel('频率')
plt.title('误差分布')

plt.tight_layout()
plt.show()

# 输出结果
print(f"优化参数：a = {a_opt:.4f}, b = {b_opt:.4f}")
print(f"R²分数：{r2:.4f}")
print(f"平均绝对误差：{mae:.4f}")
print(f"准确率（误差≤{threshold*100}%）：{accuracy:.2f}%")
# # 优化算法（优化版：非线性模型 + 动态阈值）
# import matplotlib
# matplotlib.use('TkAgg')
# import pandas as pd
# import numpy as np
# from scipy.optimize import minimize
# from sklearn.preprocessing import StandardScaler, PolynomialFeatures  # 补充导入
# from sklearn.decomposition import PCA                                # 补充导入
# from sklearn.metrics import r2_score
# import matplotlib.pyplot as plt
# import seaborn as sns
#
# # 设置可视化样式
# sns.set_theme(style="whitegrid")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 读取数据
# data_a = pd.read_excel('A.xlsx').values
# data_b = pd.read_excel('B.xlsx').values.flatten()
#
# # 数据预处理（多列PCA）
# scaler = StandardScaler()
# data_a_scaled = scaler.fit_transform(data_a)
# pca = PCA(n_components=1)
# X_single = pca.fit_transform(data_a_scaled).flatten()
#
# # 定义非线性目标函数（二次多项式）
# def objective(params):
#     a, b, c = params
#     y_pred = a * X_single**2 + b * X_single + c
#     return np.mean((y_pred - data_b) ** 2)
#
# # 参数优化
# initial_guess = [0.5, 1.0, 0.0]  # 初始猜测包含二次项系数
# result = minimize(objective, initial_guess, method='L-BFGS-B')
# a_opt, b_opt, c_opt = result.x
# optimized_values = a_opt * X_single**2 + b_opt * X_single + c_opt
#
# # 动态阈值设定（基于数据分布）
# threshold = np.percentile(np.abs(data_b), 90) * 0.1  # 前90%数据的10%
# accuracy = np.mean(np.abs(optimized_values - data_b) <= threshold) * 100
#
# # 评估指标
# r2 = r2_score(data_b, optimized_values)
# mae = np.mean(np.abs(optimized_values - data_b))
# print(f"优化参数: a={a_opt:.4f}, b={b_opt:.4f}, c={c_opt:.4f}")
# print(f"R²: {r2:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"准确率（阈值={threshold:.2f}）: {accuracy:.2f}%")
#
# # 可视化
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.scatter(data_b, optimized_values, alpha=0.6)
# plt.plot([data_b.min(), data_b.max()], [data_b.min(), data_b.max()], 'r--')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# plt.title('非线性优化预测效果')
#
# plt.subplot(1, 2, 2)
# plt.hist(np.abs(optimized_values - data_b), bins=30, edgecolor='k')
# plt.axvline(threshold, color='r', linestyle='--')
# plt.xlabel('绝对误差')
# plt.ylabel('频数')
# plt.title('误差分布')
# plt.tight_layout()
# plt.show()

