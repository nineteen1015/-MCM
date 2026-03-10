import matplotlib
matplotlib.use('TkAgg')
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 读取数据
X = pd.read_excel('5-X.xlsx', header=None)
Y = pd.read_excel('5-Y.xlsx', header=None).values.ravel()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维（保留95%方差）
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_scaled)
print(f"保留的主成分数量: {pca.n_components_}")

# 重构数据
X_reconstructed = pca.inverse_transform(X_reduced)
X_reconstructed = scaler.inverse_transform(X_reconstructed)

# 计算重构误差
recon_mse = np.mean((X - X_reconstructed) ** 2)
print(f"重构均方误差: {recon_mse:.4f}")

# 建立回归模型
X_train, X_test, Y_train, Y_test = train_test_split(X_reconstructed, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

# 评估模型
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print(f"测试集MSE: {mse:.4f}")
print(f"测试集R²: {r2:.4f}")

# 交叉验证评估泛化性
cv_r2 = cross_val_score(model, X_reconstructed, Y, cv=5, scoring='r2')
print(f"交叉验证R²: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f}")

# 分析不同主成分数的影响
max_components = X_scaled.shape[1]
components = np.arange(1, max_components + 1)
recon_errors, cv_scores = [], []

for n in components:
    pca = PCA(n_components=n)
    X_red = pca.fit_transform(X_scaled)
    X_recon = scaler.inverse_transform(pca.inverse_transform(X_red))
    recon_mse = np.mean((X - X_recon) ** 2)
    recon_errors.append(recon_mse)

    model = LinearRegression()
    cv_r2 = cross_val_score(model, X_recon, Y, cv=5, scoring='r2').mean()
    cv_scores.append(cv_r2)

# 绘制结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(components, recon_errors, 'b-o')
plt.xlabel('主成分数量')
plt.ylabel('重构MSE')
plt.title('主成分数量与重构误差')

plt.subplot(1, 2, 2)
plt.plot(components, cv_scores, 'r-o')
plt.xlabel('主成分数量')
plt.ylabel('交叉验证R^2')
plt.title('主成分数量与模型性能')
plt.tight_layout()
plt.show()

# 复杂度分析
print(f"PCA时间复杂度: O(n_samples * n_features^2 + n_features^3)")
print(f"线性回归复杂度: O(n_samples * n_features^2)")