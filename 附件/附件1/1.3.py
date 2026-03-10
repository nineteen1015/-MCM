# 矩阵分解（优化版：多列特征 + 正则化）
import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data_a = pd.read_excel('A.xlsx').values
data_b = pd.read_excel('B.xlsx').values

# 数据标准化（保留多列特征）
scaler = StandardScaler()
X = scaler.fit_transform(data_a)
y = data_b.ravel()

# 添加偏置项
X = np.hstack([X, np.ones((X.shape[0], 1))])

# 使用岭回归解决共线性问题
model = Ridge(alpha=1.0)  # 正则化参数
model.fit(X, y)
y_pred = model.predict(X)

# 评估指标
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")

# 可视化系数重要性
coef = model.coef_[:-1]  # 排除偏置项
plt.figure(figsize=(10, 6))
plt.bar(range(len(coef)), np.abs(coef))
plt.xlabel('特征编号')
plt.ylabel('系数绝对值')
plt.title('岭回归特征重要性')
plt.show()