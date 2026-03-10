import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

# 读取数据
X = pd.read_excel('4-X.xlsx').values
y = pd.read_excel('4-Y.xlsx').values.ravel()

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 创建管道：多项式特征 + 岭回归
pipeline = Pipeline([
    ('poly', PolynomialFeatures(include_bias=False)),
    ('ridge', Ridge())
])

# 参数网格：减少参数组合数量
param_grid = {
    'poly__degree': [1, 2],  # 移除3次多项式以减少特征数量
    'ridge__alpha': np.logspace(-3, 3, 20)  # 减少alpha数量
}

# 调整并行进程数，避免资源耗尽
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='r2',
    cv=5,
    n_jobs=2  # 减少并行进程数，根据系统资源调整
)

grid_search.fit(X_train, y_train)

# 输出最优参数及评估结果
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print(f"最优参数: {best_params}")

y_pred = best_model.predict(X_test)
test_r2 = r2_score(y_test, y_pred)
test_mse = mean_squared_error(y_test, y_pred)
print(f"测试集R²: {test_r2:.4f}")
print(f"测试集MSE: {test_mse:.4f}")

# 参数与拟合优度的相关性分析
cv_results = grid_search.cv_results_
params = cv_results['params']
degrees = [p['poly__degree'] for p in params]
alphas = [p['ridge__alpha'] for p in params]
scores = cv_results['mean_test_score']

corr_degree, _ = spearmanr(degrees, scores)
corr_alpha, _ = spearmanr(alphas, scores)
print(f"Degree与R²的Spearman相关系数: {corr_degree:.4f}")
print(f"Alpha与R²的Spearman相关系数: {corr_alpha:.4f}")

# 模型稳定性评估（交叉验证标准差）
cv_std = np.mean(cv_results['std_test_score'])
print(f"交叉验证R²标准差均值: {cv_std:.4f}")

# 适用性评估（训练集与测试集R²对比）
train_pred = best_model.predict(X_train)
train_r2 = r2_score(y_train, train_pred)
print(f"训练集R²: {train_r2:.4f}, 测试集R²: {test_r2:.4f}")