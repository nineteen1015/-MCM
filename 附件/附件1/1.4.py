import matplotlib
matplotlib.use('TkAgg')
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

# 设置可视化样式
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ----------------------
# 数据预处理（修复数据泄漏）
# ----------------------
# 读取原始数据（不标准化）
data_a = pd.read_excel('A.xlsx', header=None).values
data_b = pd.read_excel('B.xlsx', header=None).values

# 先划分数据集再标准化
X_train, X_test, y_train, y_test = train_test_split(
    data_a, data_b, test_size=0.2, random_state=42
)

# 分别对输入输出做标准化
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)

X_train = scaler_X.transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.transform(y_train)
y_test = scaler_y.transform(y_test)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)


# ----------------------
# 增强的神经网络模型
# ----------------------
class EnhancedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


model = EnhancedModel(input_dim=X_train.shape[1])

# ----------------------
# 优化器与学习率调度
# ----------------------
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

# ----------------------
# 早停法训练
# ----------------------
best_loss = float('inf')
patience = 30
no_improve = 0
train_losses = []
val_losses = []

for epoch in range(1000):
    # 训练阶段
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
    optimizer.step()
    train_losses.append(loss.item())

    # 验证阶段
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test)
        val_loss = criterion(val_outputs, y_test)
        val_losses.append(val_loss.item())

    # 学习率调整
    scheduler.step(val_loss)

    # 早停逻辑
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        no_improve += 1

    if no_improve >= patience:
        print(f"早停触发于第 {epoch + 1} 轮，最佳验证损失: {best_loss:.4f}")
        break

# ----------------------
# 加载最佳模型评估
# ----------------------
model.load_state_dict(torch.load('best_model.pth'))

# 反标准化预测结果
with torch.no_grad():
    y_pred = model(X_test).numpy()
    y_pred = scaler_y.inverse_transform(y_pred)
    y_true = scaler_y.inverse_transform(y_test.numpy())

# 计算误差指标
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y_true - y_pred))
r2 = r2_score(y_true, y_pred)

print(f"\n评估结果:")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R^2: {r2:.4f}")

# ----------------------
# 可视化分析
# ----------------------
# 损失曲线
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='训练损失')
plt.plot(val_losses, label='验证损失')
plt.title('训练过程损失曲线')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()

# 预测值分布对比
plt.figure(figsize=(8, 6))
sns.histplot(y_true.flatten(), color='blue', label='真实值', kde=True, alpha=0.5)
sns.histplot(y_pred.flatten(), color='red', label='预测值', kde=True, alpha=0.5)
plt.title('预测值与真实值分布对比')
plt.legend()
plt.show()

# 残差分析
residuals = y_true - y_pred
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_pred.flatten(), y=residuals.flatten(), alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差图')
plt.xlabel('预测值')
plt.ylabel('残差')
plt.grid(True)
plt.show()
# # 神经网络（优化版：网络结构优化 + 早停法）
# import matplotlib
# matplotlib.use('TkAgg')
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import r2_score, mean_squared_error
# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np
# import seaborn as sns
#
# # 设置可视化样式
# sns.set_theme(style="whitegrid")
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# # 读取数据
# data_a = pd.read_excel('A.xlsx', header=None).values
# data_b = pd.read_excel('B.xlsx', header=None).values
#
# # 数据预处理（多列标准化）
# scaler_X = StandardScaler()
# scaler_y = StandardScaler()
# X = scaler_X.fit_transform(data_a)
# y = scaler_y.fit_transform(data_b)
#
# # 转换为张量并划分数据集
# X_tensor = torch.tensor(X, dtype=torch.float32)
# y_tensor = torch.tensor(y, dtype=torch.float32)
# X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)
#
#
# # 定义神经网络模型
# class RegressionModel(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, 128),
#             nn.ReLU(),
#             nn.BatchNorm1d(128),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# model = RegressionModel(input_dim=X.shape[1])
#
# # 定义损失函数和优化器
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # 训练循环（早停法）
# best_loss = float('inf')
# patience = 20
# no_improve = 0
# train_losses = []
# val_losses = []
#
# for epoch in range(1000):
#     model.train()
#     optimizer.zero_grad()
#     outputs = model(X_train)
#     loss = criterion(outputs, y_train)
#     loss.backward()
#     optimizer.step()
#     train_losses.append(loss.item())
#
#     # 验证损失
#     model.eval()
#     with torch.no_grad():
#         val_outputs = model(X_test)
#         val_loss = criterion(val_outputs, y_test)
#         val_losses.append(val_loss.item())
#
#     # 早停逻辑
#     if val_loss < best_loss:
#         best_loss = val_loss
#         no_improve = 0
#         torch.save(model.state_dict(), 'best_model.pth')
#     else:
#         no_improve += 1
#     if no_improve >= patience:
#         print(f"早停触发于第 {epoch + 1} 轮")
#         break
#
# # 加载最佳模型
# model.load_state_dict(torch.load('best_model.pth', weights_only=True))  # 修复警告
#
# # 预测与评估
# # 预测与评估
# model.eval()
# with torch.no_grad():
#     y_pred = model(X_test)
# y_pred = scaler_y.inverse_transform(y_pred.numpy())
# y_true = scaler_y.inverse_transform(y_test.numpy())
#
# mse = mean_squared_error(y_true, y_pred)
# r2 = r2_score(y_true, y_pred)
# print(f"测试集 MSE: {mse:.4f}")
# print(f"测试集 R²: {r2:.4f}")
#
# # 新增真实值 vs 预测值散点图
# plt.figure(figsize=(8, 6))
# plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='预测值')
# plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='理想拟合线')
# plt.xlabel('真实值')
# plt.ylabel('预测值')
# plt.title('真实值 vs 预测值')
# plt.legend()
# plt.grid(True)
# plt.show()
#
# # 可视化训练过程
# plt.plot(train_losses, label='训练损失')
# plt.plot(val_losses, label='验证损失')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('训练与验证损失曲线')
# plt.show()