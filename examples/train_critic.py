"""训练 Critic 模型

使用产业链共振特征训练 Critic，学习打分函数
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class KlineWindow:
    """K线窗口数据结构"""
    def __init__(self, features):
        self.open = features['open']
        self.high = features['high']
        self.low = features['low']
        self.close = features['close']
        self.volume = features['volume']


class CriticDataset(Dataset):
    """Critic 训练数据集"""
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 主标的窗口
        main_window = self._normalize_window(sample['main_window'])

        # 关联标的窗口（固定数量，不足则填充零）
        max_related = 8
        related_windows = []
        for i, (code, window) in enumerate(sample['related_windows'].items()):
            if i >= max_related:
                break
            related_windows.append(self._normalize_window(window))

        # 填充到固定长度
        while len(related_windows) < max_related:
            related_windows.append(np.zeros_like(main_window))

        related_windows = np.stack(related_windows, axis=0)  # [8, 20, 5]

        # 标签（未来收益率）
        label = sample['future_return']

        return (
            torch.FloatTensor(main_window),
            torch.FloatTensor(related_windows),
            torch.FloatTensor([label])
        )

    def _normalize_window(self, window):
        """归一化K线窗口"""
        # 提取 OHLCV
        open_price = window['open']
        high = window['high']
        low = window['low']
        close = window['close']
        volume = window['volume']

        # 价格归一化（相对于首日收盘价）
        base_price = close[0]
        norm_open = (open_price - base_price) / base_price
        norm_high = (high - base_price) / base_price
        norm_low = (low - base_price) / base_price
        norm_close = (close - base_price) / base_price

        # 成交量归一化（log1p + 标准化）
        norm_volume = np.log1p(volume)
        norm_volume = (norm_volume - norm_volume.mean()) / (norm_volume.std() + 1e-8)

        # 组合特征 [window_size, 5]
        features = np.stack([norm_open, norm_high, norm_low, norm_close, norm_volume], axis=1)
        return features


class CriticNetwork(nn.Module):
    """Critic 神经网络"""
    def __init__(self, window_size=20, max_related=8):
        super().__init__()

        # 主标的编码器
        self.main_encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 关联标的编码器（共享权重）
        self.related_encoder = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )

        # 产业链共振特征提取
        self.resonance_net = nn.Sequential(
            nn.Linear(64 * max_related, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 融合主标的和共振特征
        self.fusion_net = nn.Sequential(
            nn.Linear(64 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, main_window, related_windows):
        """
        Args:
            main_window: [batch, 20, 5]
            related_windows: [batch, 8, 20, 5]
        Returns:
            scores: [batch, 1]
        """
        batch_size = main_window.size(0)

        # 编码主标的
        main_feat = self.main_encoder(main_window.transpose(1, 2))  # [batch, 64, 1]
        main_feat = main_feat.squeeze(-1)  # [batch, 64]

        # 编码关联标的
        related_windows = related_windows.view(batch_size * 8, 20, 5)
        related_feat = self.related_encoder(related_windows.transpose(1, 2))  # [batch*8, 64, 1]
        related_feat = related_feat.squeeze(-1)  # [batch*8, 64]
        related_feat = related_feat.view(batch_size, 8 * 64)  # [batch, 512]

        # 提取共振特征
        resonance_feat = self.resonance_net(related_feat)  # [batch, 64]

        # 融合特征
        combined = torch.cat([main_feat, resonance_feat], dim=1)  # [batch, 128]
        score = self.fusion_net(combined)  # [batch, 1]

        return score


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0

    for main_window, related_windows, labels in tqdm(loader, desc="Training"):
        main_window = main_window.to(device)
        related_windows = related_windows.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(main_window, related_windows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for main_window, related_windows, labels in loader:
            main_window = main_window.to(device)
            related_windows = related_windows.to(device)

            outputs = model(main_window, related_windows)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(labels.cpu().numpy())

    predictions = np.array(predictions).flatten()
    targets = np.array(targets).flatten()

    # 计算排序相关性
    from scipy.stats import spearmanr
    corr, _ = spearmanr(predictions, targets)

    return corr, predictions, targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='训练样本文件')
    parser.add_argument('--output', type=str, required=True, help='模型输出路径')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    print("=" * 60)
    print("训练 Critic 模型")
    print("=" * 60)

    # 加载数据
    print(f"\n加载训练数据: {args.data}")
    with open(args.data, 'rb') as f:
        samples = pickle.load(f)
    print(f"  总样本数: {len(samples)}")

    # 按时间排序划分训练集和验证集（避免数据泄露）
    samples_sorted = sorted(samples, key=lambda x: x['date'])
    split_idx = int(len(samples_sorted) * 0.8)
    train_samples = samples_sorted[:split_idx]
    val_samples = samples_sorted[split_idx:]

    print(f"  训练集: {len(train_samples)} (时间范围: {train_samples[0]['date']} ~ {train_samples[-1]['date']})")
    print(f"  验证集: {len(val_samples)} (时间范围: {val_samples[0]['date']} ~ {val_samples[-1]['date']})")

    # 创建数据加载器
    train_dataset = CriticDataset(train_samples)
    val_dataset = CriticDataset(val_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 创建模型
    model = CriticNetwork().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # 训练
    print(f"\n开始训练 (epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr})")
    best_corr = -1
    best_epoch = 0
    patience = 10  # Early stopping patience
    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_corr, _, _ = evaluate(model, train_loader, device)
        val_corr, _, _ = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {train_loss:.4f}, Train Corr: {train_corr:.4f}, Val Corr: {val_corr:.4f}")

        # 过拟合预警
        if train_corr - val_corr > 0.15:
            print(f"  ⚠️  过拟合预警：训练集相关性 ({train_corr:.4f}) 显著高于验证集 ({val_corr:.4f})")

        if val_corr > best_corr:
            best_corr = val_corr
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), args.output)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n验证集相关性连续 {patience} 个 epoch 未提升，提前停止训练")
                break

    print(f"\n训练完成！")
    print(f"  最佳 epoch: {best_epoch}")
    print(f"  最佳验证相关性: {best_corr:.4f}")
    print(f"  模型已保存: {args.output}")


if __name__ == '__main__':
    main()
