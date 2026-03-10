import matplotlib
matplotlib.use('TkAgg')
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings

# 中文字体显示设置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class GeoDataProcessor:
    """地理空间数据处理模块（支持多行多列输入）"""

    def __init__(self, scale=1e6, segment_length=2500, sample_points=100):
        self.scale = scale  # 比例尺参数
        self.segment_length = segment_length  # 分段长度（米）
        self.sample_points = sample_points  # 采样点数

    def _parse_row(self, row):
        """解析单行数据为坐标序列"""
        try:
            # 过滤无效值并转换为二维数组
            coords = row[~pd.isnull(row)].astype(float).reshape(-1, 2)
            if len(coords) < 2:
                return None
            return coords
        except Exception as e:
            print(f"解析行数据失败：{str(e)}")
            return None

    def _resample_curve(self, points):
        """基于香农采样定理的重采样"""
        # 计算曲线总长度
        cum_length = np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))
        if len(cum_length) == 0:
            return None
        total_length = cum_length[-1]

        # 生成等间距采样点
        t = np.linspace(0, total_length, self.sample_points)
        return np.array([
            np.interp(t, cum_length, points[1:, 0]),
            np.interp(t, cum_length, points[1:, 1])
        ]).T

    def _segment_curve(self, points):
        """智能曲线分段算法（完整实现）"""
        if len(points) < 2:
            return []

        # 计算累计长度
        diffs = np.diff(points, axis=0)
        lengths = np.linalg.norm(diffs, axis=1)
        cum_lengths = np.cumsum(lengths)
        total_length = cum_lengths[-1]

        # 计算理论分段数
        num_segments = max(1, int(total_length // self.segment_length))
        if num_segments == 0:
            return [points]

        # 确定分割点
        split_indices = []
        target_length = total_length / num_segments
        current_target = target_length

        for i, cl in enumerate(cum_lengths):
            if cl >= current_target:
                split_indices.append(i + 1)  # 分割点在两点之间
                current_target += target_length

        # 执行分段
        segments = []
        prev_idx = 0
        for idx in split_indices:
            if idx > prev_idx and idx <= len(points):
                segments.append(points[prev_idx:idx + 1])
                prev_idx = idx

        # 处理剩余部分
        if prev_idx < len(points) - 1:
            remaining = points[prev_idx:]
            remaining_length = total_length - cum_lengths[prev_idx - 1]

            if remaining_length > self.segment_length * (2 / 3):
                segments.append(remaining)
            elif remaining_length > self.segment_length * (1 / 3):
                # 反向延拓处理
                extended = np.vstack([remaining, remaining[-2::-1]])
                segments.append(extended[:int(self.sample_points * 1.5)])

        return segments

    def process(self, raw_df):
        """完整预处理流程"""
        processed = []
        valid_count = 0

        for _, row in raw_df.iterrows():
            points = self._parse_row(row.values)
            if points is None:
                continue

            segments = self._segment_curve(points)
            for seg in segments:
                # 重采样
                sampled = self._resample_curve(seg)
                if sampled is None or len(sampled) < 2:
                    continue

                # 计算坐标增量并处理闭合差
                deltas = np.diff(sampled, axis=0)
                if len(deltas) == 0:
                    continue

                # 闭合差分配
                error = np.sum(deltas, axis=0)
                correction = error / len(deltas)
                corrected = deltas - correction

                # 归一化处理
                normalized = corrected / self.segment_length
                processed.append(normalized.flatten())
                valid_count += 1

        print(f"有效处理{valid_count}条数据片段")
        return np.array(processed)


class EnhancedConvAutoencoder(nn.Module):
    """改进的卷积自编码器"""

    def __init__(self, input_dim, compression_ratio=0.25):
        super().__init__()
        self.input_dim = input_dim
        seq_len = input_dim // 2

        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv1d(2, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, int(32 * compression_ratio), 5, padding=2)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(int(32 * compression_ratio), 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 64, 5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(64, 2, 5, padding=2)
        )

    def forward(self, x):
        x = x.view(-1, 2, self.input_dim // 2)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(-1, self.input_dim)


# 示例使用流程
if __name__ == "__main__":
    # 1. 数据加载
    try:
        df = pd.read_excel('Data.xlsx', header=None)
        print(f"原始数据维度：{df.shape}")
        print("前5行样例：\n", df.head())
    except Exception as e:
        print(f"数据加载失败：{str(e)}")
        exit()

    # 2. 数据预处理
    processor = GeoDataProcessor(scale=1e6, segment_length=2500)
    processed_data = processor.process(df)

    if len(processed_data) == 0:
        print("错误：预处理后数据为空，请检查：")
        print("1. 输入数据是否包含有效坐标对")
        print("2. 分段长度参数是否合适")
        exit()

    print(f"\n预处理后数据维度：{processed_data.shape}")

    # 3. 划分数据集
    X_train, X_val = train_test_split(processed_data, test_size=0.2, random_state=42)
    print(f"\n训练集样本数：{len(X_train)}，验证集样本数：{len(X_val)}")

    # 4. 创建数据加载器
    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train)),
        batch_size=64,
        shuffle=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val)),
        batch_size=64,
        pin_memory=True
    )

    # 5. 模型训练
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedConvAutoencoder(input_dim=X_train.shape[1], compression_ratio=0.25).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_loss = float('inf')
    for epoch in range(100):
        model.train()
        train_loss = []
        for batch in train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = nn.MSELoss()(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        # 验证
        model.eval()
        val_loss = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0].to(device)
                outputs = model(inputs)
                val_loss.append(nn.MSELoss()(outputs, inputs).item())

        avg_train = np.mean(train_loss)
        avg_val = np.mean(val_loss)
        scheduler.step(avg_val)

        # 保存最佳模型
        if avg_val < best_loss:
            best_loss = avg_val
            torch.save(model.state_dict(), 'best_model.pth')

        print(f'Epoch {epoch + 1:03d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}')

    # 6. 结果评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_sample = torch.FloatTensor(X_val[:10]).to(device)
    with torch.no_grad():
        reconstructed = model(test_sample).cpu().numpy()

    # 计算压缩指标
    original_size = X_train.nbytes / 1024 ** 2  # MB
    compressed_size = os.path.getsize('best_model.pth') / 1024 ** 2

    print(f"\n压缩效率报告：")
    print(f"- 原始数据尺寸：{original_size:.2f} MB")
    print(f"- 压缩后尺寸：{compressed_size:.2f} MB")
    print(f"- 压缩率：{compressed_size / original_size:.1%}")
    print(f"- 存储节省率：{1 - compressed_size / original_size:.1%}")

    # 可视化对比
    plt.figure(figsize=(10, 6))
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        orig = X_val[i].reshape(-1, 2)
        recon = reconstructed[i].reshape(-1, 2)
        plt.plot(orig[:, 0], orig[:, 1], 'b-', label='原始')
        plt.plot(recon[:, 0], recon[:, 1], 'r--', label='重建')
        plt.title(f'样本{i + 1}对比')
    plt.tight_layout()
    plt.show()