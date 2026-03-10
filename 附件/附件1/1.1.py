#线性回归分析
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# 读取数据
A = pd.read_excel('A.xlsx', header=None).values
B = pd.read_excel('B.xlsx', header=None).values.ravel()

# 数据预处理（PCA + 标准化）
scaler = StandardScaler()
A_scaled = scaler.fit_transform(A)  # 标准化多列数据
pca = PCA(n_components=0.95)        # 保留95%方差的主成分
A_pca = pca.fit_transform(A_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(A_pca, B, test_size=0.2, random_state=42)

# 模型训练与评估
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"测试集 MSE: {mse:.4f}")
print(f"测试集 R²: {r2:.4f}".replace('²', '^2'))

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('线性回归预测效果')
plt.show()
