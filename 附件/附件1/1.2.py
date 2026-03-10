#非线性回归
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
def convert_to_single_column(data, method='pca'):
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
data_a = pd.read_excel('A.xlsx', header=None)
data_b = pd.read_excel('B.xlsx', header=None)

# 转换为单列数据（示例使用mean方法）
X = convert_to_single_column(data_a.values, method='mean')
y = data_b.values.ravel()  # 转换为1D数组

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用多项式回归作为非线性模型示例
degree = 2
poly_features = PolynomialFeatures(degree=degree)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_pred = model.predict(X_test_poly)

# 计算评估指标
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"R²分数: {r2:.4f}")
print(f"均方误差: {mse:.4f}")

# 生成预测结果图表
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, label='预测值')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='理想拟合线')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'真实值 vs 预测值 (R²={r2:.2f}, MSE={mse:.2f})'.replace('²', '^2'))
plt.legend()
plt.show()
