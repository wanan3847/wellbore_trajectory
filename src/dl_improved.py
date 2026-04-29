#!/usr/bin/env python3
"""
改进版深度学习模型（窗口化序列处理 v3）
核心改进：用全局注意力池化替代中心点提取 + FocalLoss 处理类别不均衡。
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2 import Config, macro_f1_with_tolerance


# ==================== 窗口化数据集 ====================

class DrillingWindowDataset(Dataset):
    """滑窗数据集 — 为每个测点提供其所在井段的连续窗口。"""
    def __init__(self, X, y=None, well_ids=None, window_size=51):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64) if y is not None else None
        self.window_size = window_size
        self.half_win = window_size // 2

        if well_ids is not None and len(well_ids) > 0:
            well_ids = np.asarray(well_ids)
            sort_idx = np.argsort(well_ids, kind='stable')
            self.X = X[sort_idx]
            self.y = y[sort_idx] if y is not None else None
            self._well_ids = well_ids[sort_idx]

            self.well_ranges = []
            current = self._well_ids[0]
            start = 0
            for i in range(1, len(self._well_ids)):
                if self._well_ids[i] != current:
                    self.well_ranges.append((current, start, i))
                    current = self._well_ids[i]
                    start = i
            self.well_ranges.append((current, start, len(self._well_ids)))
        else:
            self.X = X
            self.y = y
            self._well_ids = None
            self.well_ranges = []

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        for wid, start, end in self.well_ranges:
            if start <= idx < end:
                break

        half = self.half_win
        lo = max(start, idx - half)
        hi = min(end, idx + half + 1)
        window = self.X[lo:hi]

        before = idx - start
        after  = end - 1 - idx
        pad_before = max(0, half - before)
        pad_after  = max(0, half - after)

        # 使用反射填充(较重复填充减少边沿伪影)
        if pad_before > 0:
            reflect_len = min(pad_before, hi - lo)
            if reflect_len > 0:
                reflect = self.X[lo:lo+reflect_len][::-1]
                if reflect.shape[0] < pad_before:
                    reflect = np.concatenate([reflect] * (pad_before // reflect.shape[0] + 1), axis=0)
                window = np.concatenate([reflect[:pad_before], window], axis=0)
            else:
                prefix = self.X[lo:lo+1].repeat(pad_before, axis=0)
                window = np.concatenate([prefix, window], axis=0)
        if pad_after > 0:
            reflect_len = min(pad_after, hi - lo)
            if reflect_len > 0:
                reflect = self.X[hi-reflect_len:hi][::-1]
                if reflect.shape[0] < pad_after:
                    reflect = np.concatenate([reflect] * (pad_after // reflect.shape[0] + 1), axis=0)
                window = np.concatenate([window, reflect[:pad_after]], axis=0)
            else:
                suffix = self.X[hi-1:hi].repeat(pad_after, axis=0)
                window = np.concatenate([window, suffix], axis=0)

        x = torch.FloatTensor(window).permute(1, 0)

        if self.y is not None:
            return x, torch.LongTensor([self.y[idx]])[0]
        return x


# ==================== 辅助模块 ====================

class FocalLoss(nn.Module):
    """Focal Loss — 缓解类别不均衡"""
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.scale = dim ** -0.5

    def forward(self, x):
        scores = torch.matmul(self.query, x.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, x).squeeze(1)


# ==================== V3 模型系列 ====================

class HybridV3(nn.Module):
    """
    Conv1d → BiLSTM(特征预处理) → TransformerEncoder(核心注意力) → 全局注意力池化 → FC
    顺序架构（无残差冲突），LSTM 作为 Transformer 的前置特征处理器。
    """
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.pos_encoder = PositionalEncoding(hidden_dim * 2, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        self.pool = AttentionPooling(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)
        x_pos = self.pos_encoder(lstm_out)
        tf_out = self.transformer(x_pos)

        pooled = self.pool(tf_out)
        return self.fc(pooled)


class LSTMOnlyV3(nn.Module):
    """Conv1d → BiLSTM → 全局注意力池化 → FC"""
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(
            64, hidden_dim, num_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.pool = AttentionPooling(hidden_dim * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        pooled = self.pool(lstm_out)
        return self.fc(pooled)


class TransformerOnlyV3(nn.Module):
    """Conv1d → TransformerEncoder → 全局注意力池化 → FC"""
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 num_layers=2, num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(hidden_dim)

        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers)

        self.pool = AttentionPooling(hidden_dim)

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.LayerNorm(64),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)
        tf_out = self.transformer(x)
        pooled = self.pool(tf_out)
        return self.fc(pooled)


# ==================== 模型工厂 ====================

MODEL_REGISTRY = {
    'hybrid_v3': HybridV3,
    'lstm_only': LSTMOnlyV3,
    'transformer_only': TransformerOnlyV3,
}


def build_model(model_type, n_features, window_size=51, hidden_dim=128,
                num_classes=4, dropout=0.3, **kwargs):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"未知模型类型: {model_type}，可选: {list(MODEL_REGISTRY.keys())}")
    cls = MODEL_REGISTRY[model_type]
    extra = {'num_layers': 2}
    extra.update(kwargs)
    return cls(n_features, window_size, hidden_dim, num_classes=num_classes,
               dropout=dropout, **extra)


# ==================== 训练函数 ====================

def train_enhanced_model(X_train, y_train, well_ids_train,
                          X_val, y_val, well_ids_val,
                          model_type='hybrid_v3',
                          n_features=None, window_size=51,
                          hidden_dim=128, lr=0.001, batch_size=256,
                          epochs=150, dropout=0.3, weight_decay=1e-4,
                          patience=30, verbose=True,
                          class_weights=None):
    """训练增强版深度学习模型（CE Loss + ReduceLROnPlateau）"""
    device = Config.DEVICE
    if n_features is None:
        n_features = X_train.shape[1]
    n_classes = len(np.unique(np.concatenate([y_train, y_val])))

    model = build_model(model_type, n_features, window_size, hidden_dim,
                        num_classes=n_classes, dropout=dropout).to(device)

    if class_weights is None:
        cw = torch.FloatTensor([1.0, 100.0, 100.0, 100.0]).to(device)
    else:
        cw = torch.FloatTensor(class_weights).to(device)
    criterion = FocalLoss(alpha=cw, gamma=2.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-5
    )

    train_ds = DrillingWindowDataset(X_train, y_train, well_ids_train, window_size)
    val_ds   = DrillingWindowDataset(X_val, y_val, well_ids_val, window_size)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=batch_size)

    # 使用 dataset 内部已排序的 well_ids 和 labels（用于容错 F1 评估）
    sorted_well_ids_val = val_ds._well_ids
    sorted_y_val = val_ds.y

    best_f1 = 0.0
    stale = 0

    for ep in range(epochs):
        model.train()
        loss_sum = 0.0
        for bx, by in train_ld:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += loss.item()

        model.eval()
        preds = []
        with torch.no_grad():
            for bx, _ in val_ld:
                out = model(bx.to(device))
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())

        # 使用容错 F1 作为验证指标（与最终评分标准一致）
        f1 = macro_f1_with_tolerance(sorted_y_val, np.array(preds), sorted_well_ids_val)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_{model_type}.pth")
            stale = 0
        else:
            stale += 1

        if verbose and (ep + 1) % 15 == 0:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"    [{model_type}] Epoch {ep+1:3d}: Loss {loss_sum/len(train_ld):.4f}  "
                  f"ValF1={f1:.4f}  Best={best_f1:.4f}  LR={lr_now:.2e}")
        if stale >= patience:
            if verbose:
                print(f"    [{model_type}] 提前停止 (Epoch {ep+1})")
            break

    model.load_state_dict(torch.load(f"best_{model_type}.pth"))
    if verbose:
        print(f"    [{model_type}] 训练完成，最佳验证 F1={best_f1:.4f}")
    return model


def predict_enhanced_proba(model, X_test, well_ids_test, window_size=51):
    """窗口化预测，返回概率矩阵 [n, num_classes]"""
    model.eval()
    device = Config.DEVICE
    ds = DrillingWindowDataset(X_test, y=None, well_ids=well_ids_test, window_size=window_size)
    ld = DataLoader(ds, batch_size=256)
    probas = []
    with torch.no_grad():
        for bx in ld:
            data = bx[0] if isinstance(bx, list) else bx
            probas.append(F.softmax(model(data.to(device)), dim=1).cpu().numpy())
    return np.vstack(probas)
