import matplotlib
matplotlib.use('TkAgg')
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import normaltest
from joblib import Parallel, delayed

# ===== 1. 中文显示解决方案 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ===== 2. 数据预处理优化 =====
# 读取数据
df_x = pd.read_excel('3-X.xlsx', header=None)
df_y = pd.read_excel('3-Y.xlsx', header=None)

# 数据校验
assert df_x.shape[0] == df_y.shape[0], "样本数不匹配"
X = df_x.values.astype(np.float32)
Y = df_y.values.flatten().astype(np.float32)
print(f"原始数据维度: {X.shape}, Y方差: {np.var(Y):.4f}")

# 改进的并行去噪
def denoise_feature(col_data):
    try:
        coeffs = pywt.wavedec(col_data, 'db4', level=5, mode='periodization')
        sigma = np.median(np.abs(coeffs[-1]))/0.6745
        uthresh = sigma * np.sqrt(2*np.log(len(col_data)))
        coeffs = [pywt.threshold(c, uthresh, 'soft') for c in coeffs]
        return pywt.waverec(coeffs, 'db4', mode='periodization')[:len(col_data)]
    except:
        return col_data

print("并行去噪中...")
X_denoised = np.column_stack(
    Parallel(n_jobs=-1)(delayed(denoise_feature)(X[:,i]) for i in range(X.shape[1]))
)

# ===== 3. 特征工程优化 =====
scaler = StandardScaler()
X_processed = scaler.fit_transform(X_denoised)

# 特征筛选（保留方差>1e-8的特征）
nonzero_var_mask = np.var(X_processed, axis=0) > 1e-8
X_processed = X_processed[:, nonzero_var_mask]
print(f"有效特征数: {X_processed.shape[1]}")

# ===== 关键修复：调整PCA参数 =====
# 计算可用的最大主成分数
max_components = min(X_processed.shape[0], X_processed.shape[1])  # 10000和有效特征数中取小值
print(f"可用最大主成分数: {max_components}")

# 设置保留85%方差（自动计算实际成分数）
pca = PCA(n_components=0.85, whiten=True, svd_solver='auto')  # 修改求解器为auto
X_pca = pca.fit_transform(X_processed)
print(f"PCA实际保留主成分数: {pca.n_components_}")

# ===== 4. 多项式特征生成 =====
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_pca)
print(f"多项式特征维度: {X_poly.shape}")

# ===== 5. 建模优化 =====
# ===== 5. 建模优化 =====
model = RidgeCV(
    alphas=np.logspace(3, 5, 10),
    cv=5
)
model.fit(X_poly, Y)
Y_pred = model.predict(X_poly)

# ===== 6. 可视化优化 =====
def plot_separate_figures():
    # 去噪对比图（前500个样本）
    plt.figure(figsize=(10, 4))
    plt.plot(X[:500, 0], alpha=0.5, label='原始特征')
    plt.plot(X_denoised[:500, 0], label='去噪特征')
    plt.title('特征去噪效果对比（前500样本）')
    plt.legend()
    plt.show()

    # 主成分分布图
    plt.figure(figsize=(10, 4))
    plt.hist(X_pca[:, 0], bins=50)
    plt.title('第一主成分分布')
    plt.show()

    # 回归拟合图（随机1000样本）
    plt.figure(figsize=(10, 4))
    sample_idx = np.random.choice(len(Y), 1000, replace=False)
    plt.scatter(X_pca[sample_idx, 0], Y[sample_idx], alpha=0.3, label='实际值')
    plt.scatter(X_pca[sample_idx, 0], Y_pred[sample_idx], alpha=0.3, color='r', label='预测值')
    plt.title('回归拟合效果（随机1000样本）')
    plt.legend()
    plt.show()

    # 残差分析图
    plt.figure(figsize=(10, 4))
    plt.scatter(Y_pred, Y - Y_pred, alpha=0.3)
    plt.axhline(0, color='r', linestyle='--')
    plt.title('残差分布')
    plt.show()

plot_separate_figures()

# ===== 7. 模型评估 =====
residuals = Y - Y_pred
results = {
    "R^2": r2_score(Y, Y_pred),
    "BP检验p值": het_breuschpagan(residuals, sm.add_constant(X_poly))[1],
    "正态性检验p值": normaltest(residuals).pvalue,
    "最佳alpha": model.alpha_,
    "MSE": np.mean(residuals**2)
}

print("\n======== 模型评估结果 ========")
for k, v in results.items():
    print(f"{k}: {v:.4f}")