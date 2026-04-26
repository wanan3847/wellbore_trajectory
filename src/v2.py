# === 1. 基础与数据处理 ===
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

# === 2. 数据库与科学计算 ===
from sqlalchemy import create_engine
from scipy import signal
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# === 3. 机器学习核心 (Sklearn & Tree Models) ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# === 4. 深度学习框架 (PyTorch) ===
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

print("=" * 80)
print("基于钻井轨迹数据的造斜识别")
print("=" * 80)

# ==================== 配置参数 ====================
class Config:
    RANDOM_SEED = 42 #保证实验的可重复性
    TEST_SIZE = 0.15
    BATCH_SIZE = 256 #每次迭代喂给模型的数据量
    STAGE1_EPOCHS = 80  # 关键点检测 哪儿可能是点，哪儿肯定不是
    STAGE2_EPOCHS = 100  # 关键点分类 细分到底是“增斜点”、“稳斜点”还是“降斜点”
    LEARNING_RATE = 0.001 #决定了模型在寻找最优解时步子跨的大小
    PATIENCE = 15 #防止过拟合

    # 容错参数
    TOLERANCE_FIRST = 1  # 第一个增斜点容错
    TOLERANCE_REST = 2   # 其余关键点容错

    # 类别权重
    CLASS_WEIGHTS_DETECTION = {0: 1, 1: 150}  # 阶段1：检测 关键点的权重是普通点的 150 倍
    CLASS_WEIGHTS_CLASSIFY = {0: 1, 1: 200, 2: 200, 3: 200}  # 阶段2：分类

    # 负样本采样
    HARD_NEGATIVE_WINDOW = 150  # 关键点前后150个点作为困难负样本
    NEGATIVE_SAMPLE_RATIO = 0.8  # 保留30%的负样本

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#有显卡则用显卡加速

np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)

# =================== 5. 增强型深度学习模型 (CNN + RNN/LSTM + Transformer/Attention) ====================

class DrillingDataset(Dataset): #将Pandas表格数据转化为神经网络
    # （CNN/Transformer）能够理解的张量（Tensor）格式的关键桥梁
    def __init__(self, X, y=None):#初始化“仓库”

        self.X = torch.FloatTensor(X.copy())#将特征数据（如井斜、深度、方位等）转换为 32位浮点型张量
        self.y = torch.LongTensor(y) if y is not None else None#将标签数据（分类结果 0, 1, 2, 3）转换为 长整型张量

    def __len__(self):#告诉程序这个数据集总共有多少行数据
        return len(self.X)

    def __getitem__(self, idx):#核心“取货”逻辑
        # 增加一个维度以适配CNN (Batch, Channel, Length)
        x = self.X[idx].unsqueeze(0)#在数据的最前面增加一个维度
        # CNN 才能把这些特征当成一根“线”来提取局部特征
        if self.y is not None:#如果是训练阶段（有 y），它会成对返回特征和对应的关键点标签。
                              #如果是预测阶段（y 为 None），它只返回特征。
            return x, self.y[idx]
        return x

class HybridAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=4):
        
        super(HybridAttentionModel, self).__init__()

        # 1. 卷积层：提取局部特征变化（一维卷积）
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)

        # 2. 双向LSTM：捕获钻井前后趋势
        # 输入维度需要考虑卷积后的特征
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=0.3)

        # 3. Multi-Head Attention: 模仿 Transformer 的核心机制
        # embed_dim 为 BiLSTM 的输出维度: hidden_dim * 2
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2,
                                               num_heads=8,
                                               dropout=0.1,
                                               batch_first=True)

        # 4. 全连接分类层
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, 1, Features)

        # LSTM 期望输入: (Batch, SeqLen, Features)
        lstm_out, _ = self.lstm(x)

        # Transformer Attention 层
        # lstm_out shape: (Batch, 1, hidden*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)

        # 残差连接与池化 (由于当前处理是逐点采样，SeqLen=1)
        out = attn_out[:, -1, :]
        logits = self.fc(out)
        return logits

# ==================== 6. 深度学习训练与预测集成函数 ====================

def train_dl_model(X_train, y_train, X_val, y_val, input_dim):
    device = Config.DEVICE
    model = HybridAttentionModel(input_dim).to(device)

    # 类别权重处理 (解决钻井关键点极其稀疏的问题)
    weights = torch.FloatTensor([1.0, 50.0, 50.0, 50.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)

    train_loader = DataLoader(DrillingDataset(X_train, y_train), batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(DrillingDataset(X_val, y_val), batch_size=Config.BATCH_SIZE)

    best_f1 = 0
    patience_counter = 0

    print("  开始训练深度学习模型 (CNN-BiLSTM-Attention)...")
    for epoch in range(Config.STAGE2_EPOCHS):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证环节
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_x, _ in val_loader:
                outputs = model(batch_x.to(device))
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())

        # 使用你定义的容错 F1 分数进行评估
        current_f1 = f1_score(y_val, val_preds, average='macro', zero_division=0)

        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(model.state_dict(), "best_hybrid_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch + 1}: Loss {train_loss / len(train_loader):.4f}, Val F1 {current_f1:.4f}")

        if patience_counter >= Config.PATIENCE:
            print(f"    提前停止于 Epoch {epoch + 1}")
            break

    model.load_state_dict(torch.load("best_hybrid_model.pth"))
    return model

def predict_dl_proba(model, X_test):
    model.eval()
    device = Config.DEVICE
    loader = DataLoader(DrillingDataset(X_test), batch_size=Config.BATCH_SIZE)
    probas = []
    with torch.no_grad():
        for batch_x in loader:
            # 处理无标签的 Dataset 返回值
            data = batch_x[0] if isinstance(batch_x, list) else batch_x
            outputs = model(data.to(device))
            proba = F.softmax(outputs, dim=1)
            probas.append(proba.cpu().numpy())
    return np.vstack(probas)

# ==================== 1. 带容错的评估指标 ====================
def macro_f1_with_tolerance(y_true, y_pred, well_ids, tolerance_first=1, tolerance_rest=2):
    """
    实现带容错机制的Macro F1评估
    - 第一个增斜点：容错±1
    - 其余关键点：容错±2
    """
    y_true_adjusted = y_true.copy()
    y_pred_adjusted = y_pred.copy()

    for well_id in np.unique(well_ids):
        mask = well_ids == well_id
        indices = np.where(mask)[0]

        true_labels = y_true[mask]
        pred_labels = y_pred[mask]

        # 找出真实的关键点位置
        true_keypoints = {}
        for kp in [1, 2, 3]:
            kp_indices = indices[true_labels == kp]
            if len(kp_indices) > 0:
                true_keypoints[kp] = kp_indices[0]  # 假设每类只有一个

        # 找出预测的关键点位置
        pred_keypoints = {}
        for kp in [1, 2, 3]:
            kp_indices = indices[pred_labels == kp]
            if len(kp_indices) > 0:
                pred_keypoints[kp] = kp_indices[0]

        # 应用容错机制
        for kp in [1, 2, 3]:
            if kp in true_keypoints and kp in pred_keypoints:
                true_pos = true_keypoints[kp]
                pred_pos = pred_keypoints[kp]

                # 确定容错范围
                tolerance = tolerance_first if kp == 1 else tolerance_rest

                if abs(true_pos - pred_pos) <= tolerance:
                    # 在容错范围内，将预测位置调整为真实位置
                    local_pred_idx = np.where(indices == pred_pos)[0][0]
                    local_true_idx = np.where(indices == true_pos)[0][0]
                    y_pred_adjusted[pred_pos] = 0  # 清除预测位置
                    y_pred_adjusted[true_pos] = kp  # 标记为真实位置

    # 计算调整后的Macro F1
    return f1_score(y_true_adjusted, y_pred_adjusted, average='macro', zero_division=0)


# ==================== 2. 从设计轨迹提取先验关键点 ====================
def extract_keypoints_from_design(design_data):
    """
    从设计轨迹中提取理论关键点位置
    基于井斜角变化率的分析
    """
    if design_data is None or len(design_data) == 0:
        return {}

    jx_design = design_data['JX_design'].values
    xjs = design_data['XJS'].values

    # 计算井斜角变化率
    jx_diff = np.diff(jx_design, prepend=jx_design[0])
    jx_diff_smooth = signal.savgol_filter(jx_diff, window_length=min(11, len(jx_diff)), polyorder=2)

    keypoints = {}

    # 寻找增斜点：变化率从接近0变为显著正值
    for i in range(10, len(jx_diff_smooth) - 10):
        if jx_diff_smooth[i-5:i].mean() < 0.05 and jx_diff_smooth[i:i+5].mean() > 0.15:
            keypoints[1] = i
            break

    # 寻找稳斜点：变化率从正值变为接近0
    if 1 in keypoints:
        for i in range(keypoints[1] + 20, len(jx_diff_smooth) - 10):
            if jx_diff_smooth[i-5:i].mean() > 0.1 and abs(jx_diff_smooth[i:i+10].mean()) < 0.05:
                keypoints[2] = i
                break

    # 寻找降斜点：变化率从接近0变为显著负值
    if 2 in keypoints:
        for i in range(keypoints[2] + 20, len(jx_diff_smooth) - 10):
            if abs(jx_diff_smooth[i-5:i].mean()) < 0.05 and jx_diff_smooth[i:i+5].mean() < -0.1:
                keypoints[3] = i
                break

    return keypoints


def align_design_to_actual(design_data, actual_data):
    """
    使用DTW对齐设计轨迹和实际轨迹
    返回对齐的索引映射
    """
    if design_data is None or len(design_data) == 0:
        return None

    # 提取关键特征用于对齐
    design_features = design_data[['JX_design', 'LJCZJS_design']].values
    actual_features = actual_data[['JX', 'LJCZJS']].values

    # 标准化
    design_features = (design_features - design_features.mean(axis=0)) / (design_features.std(axis=0) + 1e-6)
    actual_features = (actual_features - actual_features.mean(axis=0)) / (actual_features.std(axis=0) + 1e-6)

    # DTW对齐
    try:
        distance, path = fastdtw(design_features, actual_features, dist=euclidean)

        # 构建映射：design_idx -> actual_idx
        alignment_map = {}
        for design_idx, actual_idx in path:
            if design_idx not in alignment_map:
                alignment_map[design_idx] = []
            alignment_map[design_idx].append(actual_idx)

        # 取平均作为最佳对齐位置
        alignment_map = {k: int(np.mean(v)) for k, v in alignment_map.items()}

        return alignment_map
    except:
        return None


# ==================== 3. 增强的特征工程 ====================
def create_advanced_features_v2(df, is_train=True):
    """
    重构版特征工程：解决 PerformanceWarning 和 FutureWarning
    """
    print(f"\n[2/9] 增强特征工程v2 {'(训练集)' if is_train else '(测试集)'}...")
    df = df.copy()

    # 处理数值转换
    for col in ['LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    all_well_frames = []

    for well_id, group in tqdm(df.groupby('转换后JH'), desc="  处理井号"):
        group = group.sort_values('XJS').reset_index(drop=True)
        n = len(group)

        # 使用字典存储当前井的所有列，最后一次性转成 DataFrame
        f_dict = {}
        f_dict['id'] = group['id'].values
        f_dict['well_id'] = group['转换后JH'].values
        f_dict['XJS'] = group['XJS'].values
        f_dict['JX'] = group['JX'].values
        f_dict['FW'] = group['FW'].values
        f_dict['LJCZJS'] = group['LJCZJS'].values

        # 井内相对位置
        f_dict['depth_pct'] = np.arange(n) / max(n - 1, 1)

        # ========== 核心物理特征 ==========
        jx_series = group['JX']
        jx_diff1 = jx_series.diff().fillna(0)
        jx_diff2 = jx_diff1.diff().fillna(0)
        jx_diff3 = jx_diff2.diff().fillna(0)

        f_dict['JX_diff_1'] = jx_diff1.values
        f_dict['JX_diff_2'] = jx_diff2.values
        f_dict['JX_diff_3'] = jx_diff3.values

        # ========== 单调性和转折点特征 ==========
        for w in [5, 10, 15, 20]:
            inc_list, dec_list = [], []
            for i in range(n):
                window = jx_series.iloc[i: min(n, i + w)].values
                if len(window) > 1:
                    diffs = np.diff(window)
                    inc_list.append(int(np.all(diffs >= -0.01)))
                    dec_list.append(int(np.all(diffs <= 0.01)))
                else:
                    inc_list.append(0)
                    dec_list.append(0)
            f_dict[f'monotonic_inc_{w}'] = np.array(inc_list)
            f_dict[f'monotonic_dec_{w}'] = np.array(dec_list)
            # 计算变化（这里直接对数组操作）
            f_dict[f'monotonic_change_{w}'] = (
                    np.abs(np.diff(inc_list, prepend=inc_list[0])) +
                    np.abs(np.diff(dec_list, prepend=dec_list[0]))
            )

        # ========== 变化点检测特征 ==========
        for w in [10, 20, 30]:
            var_before = jx_diff1.rolling(w, min_periods=1).std().shift(1).fillna(0)
            var_after = jx_diff1.rolling(w, min_periods=1).std().shift(-1).fillna(0)
            f_dict[f'variance_ratio_{w}'] = (var_after / (var_before + 1e-6)).values

            mean_before = jx_diff1.rolling(w, min_periods=1).mean().shift(1).fillna(0)
            mean_after = jx_diff1.rolling(w, min_periods=1).mean().shift(-1).fillna(0)
            f_dict[f'mean_change_{w}'] = (mean_after - mean_before).values
            f_dict[f'sign_change_{w}'] = (np.sign(mean_before) != np.sign(mean_after)).astype(int).values

        # ========== 滑窗统计特征 ==========
        for w in [3, 5, 10, 15, 20, 30, 50]:
            rolling = jx_series.rolling(w, min_periods=1, center=True)
            f_dict[f'JX_mean_{w}'] = rolling.mean().values
            f_dict[f'JX_std_{w}'] = rolling.std().fillna(0).values
            f_dict[f'JX_max_{w}'] = rolling.max().values
            f_dict[f'JX_min_{w}'] = rolling.min().values
            f_dict[f'JX_range_{w}'] = f_dict[f'JX_max_{w}'] - f_dict[f'JX_min_{w}']

            diff_rolling = jx_diff1.rolling(w, min_periods=1, center=True)
            f_dict[f'JX_diff_mean_{w}'] = diff_rolling.mean().values
            f_dict[f'JX_diff_std_{w}'] = diff_rolling.std().fillna(0).values

        # ========== 局部极值特征 ==========
        for w in [5, 10, 15]:
            f_dict[f'is_local_max_{w}'] = (jx_series == jx_series.rolling(w, center=True, min_periods=1).max()).astype(
                int).values
            f_dict[f'is_local_min_{w}'] = (jx_series == jx_series.rolling(w, center=True, min_periods=1).min()).astype(
                int).values

        # ========== 设计轨迹特征 (解决 FutureWarning) ==========
        has_design = not group['JX_design'].isna().all()
        if has_design:
            # 使用新版 ffill().bfill() 避免 FutureWarning
            jx_design = group['JX_design'].ffill().bfill().fillna(0)
            fw_design = group['FW_design'].ffill().bfill().fillna(0)
            lj_design = group['LJCZJS_design'].ffill().bfill().fillna(0)

            f_dict['JX_design'] = jx_design.values
            f_dict['FW_design'] = fw_design.values
            f_dict['LJCZJS_design'] = lj_design.values

            f_dict['JX_deviation'] = (group['JX'] - jx_design).values
            f_dict['JX_deviation_abs'] = np.abs(f_dict['JX_deviation'])
            f_dict['FW_deviation'] = (group['FW'] - fw_design).values
            f_dict['FW_deviation_abs'] = np.abs(f_dict['FW_deviation'])

            design_data = group[['XJS', 'JX_design', 'LJCZJS_design']].copy()
            design_data = design_data[~design_data['JX_design'].isna()]

            if len(design_data) > 20:
                prior_keypoints = extract_keypoints_from_design(design_data)
                alignment_map = align_design_to_actual(design_data, group[['JX', 'LJCZJS']])
                for kp_type in [1, 2, 3]:
                    if alignment_map and kp_type in prior_keypoints and prior_keypoints[kp_type] in alignment_map:
                        actual_idx = alignment_map[prior_keypoints[kp_type]]
                        dist_to_prior = np.abs(np.arange(n) - actual_idx)
                        f_dict[f'dist_to_prior_kp{kp_type}'] = dist_to_prior
                        f_dict[f'near_prior_kp{kp_type}'] = (dist_to_prior < 15).astype(int)
                    else:
                        f_dict[f'dist_to_prior_kp{kp_type}'] = np.full(n, 999)
                        f_dict[f'near_prior_kp{kp_type}'] = np.zeros(n)
            else:
                for kp_type in [1, 2, 3]:
                    f_dict[f'dist_to_prior_kp{kp_type}'] = np.full(n, 999)
                    f_dict[f'near_prior_kp{kp_type}'] = np.zeros(n)
        else:
            # 无设计数据填充默认值
            for feat in ['JX_design', 'FW_design', 'LJCZJS_design', 'JX_deviation',
                         'JX_deviation_abs', 'FW_deviation', 'FW_deviation_abs']:
                f_dict[feat] = np.zeros(n)
            for kp_type in [1, 2, 3]:
                f_dict[f'dist_to_prior_kp{kp_type}'] = np.full(n, 999)
                f_dict[f'near_prior_kp{kp_type}'] = np.zeros(n)

        # ========== 井级别全局特征 (使用 np.full 保持长度一致) ==========
        f_dict['well_max_jx'] = np.full(n, jx_series.max())
        f_dict['well_min_jx'] = np.full(n, jx_series.min())
        f_dict['well_mean_jx'] = np.full(n, jx_series.mean())
        f_dict['well_total_depth'] = np.full(n, group['XJS'].max())
        f_dict['well_n_points'] = np.full(n, n)
        f_dict['jx_percentile'] = jx_series.rank(pct=True).values

        if is_train and '关键点' in group.columns:
            f_dict['label'] = group['关键点'].values

        # 将当前井的字典一次性转为 DataFrame
        all_well_frames.append(pd.DataFrame(f_dict))

    # 一次性合并所有井
    result_df = pd.concat(all_well_frames, ignore_index=True)

    # 最终数据清洗
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"  完成！共生成 {len(result_df.columns)} 列特征")
    return result_df


# ==================== 4. 困难负样本采样 ==========
def sample_hard_negatives(df, window=50, ratio=0.3):
    """
    采样困难负样本：保留关键点附近的负样本 + 随机采样部分远离关键点的负样本
    """
    print("\n[3/9] 困难负样本采样...")

    sampled_list = []

    for well_id in df['well_id'].unique():
        group = df[df['well_id'] == well_id].copy()

        # 找出所有关键点
        keypoint_indices = group[group['label'] != 0].index.tolist()

        if len(keypoint_indices) == 0:
            # 没有关键点的井，随机采样负样本
            neg_indices = group[group['label'] == 0].index
            if len(neg_indices) > 0:
                n_sample = max(1, int(len(neg_indices) * ratio))
                sampled_neg = np.random.choice(neg_indices, size=n_sample, replace=False)
                sampled_list.append(group.loc[sampled_neg])
        else:
            # 保留所有关键点
            sampled_list.append(group[group['label'] != 0])

            # 困难负样本：关键点前后window范围内
            hard_neg_indices = set()
            for kp_idx in keypoint_indices:
                kp_pos = group.index.get_loc(kp_idx)
                start = max(0, kp_pos - window)
                end = min(len(group), kp_pos + window + 1)
                for pos in range(start, end):
                    idx = group.index[pos]
                    if group.loc[idx, 'label'] == 0:
                        hard_neg_indices.add(idx)

            if hard_neg_indices:
                sampled_list.append(group.loc[list(hard_neg_indices)])

            # 简单负样本：随机采样
            all_neg_indices = set(group[group['label'] == 0].index.tolist())
            easy_neg_indices = all_neg_indices - hard_neg_indices

            if easy_neg_indices:
                n_easy_sample = max(1, int(len(easy_neg_indices) * ratio))
                sampled_easy = np.random.choice(list(easy_neg_indices),
                                              size=min(n_easy_sample, len(easy_neg_indices)),
                                              replace=False)
                sampled_list.append(group.loc[sampled_easy])

    result = pd.concat(sampled_list, ignore_index=False).sort_index()

    print(f"  原始样本数: {len(df)}")
    print(f"  采样后样本数: {len(result)}")
    print(f"  压缩比: {len(result)/len(df):.2%}")

    return result


# ==================== 5. 两阶段模型 ==========
class KeypointDetector(nn.Module):
    """阶段1：关键点检测（二分类）"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)  # 二分类：0=非关键点, 1=任何关键点
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        return self.fc(out)


class KeypointClassifier(nn.Module):
    """阶段2：关键点分类（四分类，但主要关注1/2/3）"""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True,
                           bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=4, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        return self.fc(out)


# ==================== 6. 动态规划后处理 ==========
def dp_post_process(well_data, model_proba, prior_info=None):
    """
    使用动态规划搜索最优关键点组合
    考虑：
    1. 物理约束（顺序、间距）
    2. 模型置信度
    3. 先验信息
    4. 容错扩展
    """
    n = len(well_data)
    jx = well_data['JX'].values
    jx_diff = np.diff(jx, prepend=jx[0])

    # 为每种关键点找候选位置（top-K个高置信度）
    candidates = {1: [], 2: [], 3: []}
    K = 30  # 每类保留top-K个候选

    for kp in [1, 2, 3]:
        # 基础候选：模型高置信度位置
        kp_proba = model_proba[:, kp]
        top_k_indices = np.argsort(kp_proba)[-K:]

        for idx in top_k_indices:
            if kp_proba[idx] > 0.05:  # 最低置信度阈值
                score = kp_proba[idx]

                # 物理特征检查（连续评分，越符合物理规律得分越高）
                phys_score = 0
                if kp == 1 and idx < n - 10:  # 增斜点：井斜角增大趋势
                    future_trend = jx[idx+1:idx+11].mean() - jx[idx]
                    if future_trend > 0:
                        phys_score = 0.3 * min(future_trend / 1.0, 1.0)
                elif kp == 2 and idx >= 10 and idx < n - 10:  # 稳斜点：井斜角变化小
                    local_std = jx_diff[idx-10:idx+10].std()
                    phys_score = 0.3 * max(0, 1 - local_std / 0.3)
                elif kp == 3 and idx < n - 10:  # 降斜点：井斜角减小趋势
                    future_trend = jx[idx+1:idx+11].mean() - jx[idx]
                    if future_trend < 0:
                        phys_score = 0.3 * min(-future_trend / 0.6, 1.0)

                # 先验信息加权
                prior_score = 0
                if prior_info and kp in prior_info:
                    dist = abs(idx - prior_info[kp])
                    if dist < 15:
                        prior_score = 0.2 * (1 - dist / 15)

                total_score = score + phys_score + prior_score
                candidates[kp].append((idx, total_score))

        # 按分数排序
        candidates[kp].sort(key=lambda x: x[1], reverse=True)

    # 动态规划找最优组合
    best_combo = None
    best_score = -1

    # 遍历所有可能的组合
    for kp1_idx, kp1_score in candidates[1][:10]:  # 增斜点候选
        for kp2_idx, kp2_score in candidates[2][:10]:  # 稳斜点候选
            # 约束：稳斜点必须在增斜点之后至少10个点
            if kp2_idx < kp1_idx + 10:
                continue

            for kp3_idx, kp3_score in candidates[3][:10] + [(None, 0)]:  # 降斜点候选（可选）
                # 约束：降斜点必须在稳斜点之后至少10个点
                if kp3_idx is not None and kp3_idx < kp2_idx + 10:
                    continue

                # 计算组合得分
                combo_score = kp1_score + kp2_score + (kp3_score if kp3_idx is not None else 0)

                if combo_score > best_score:
                    best_score = combo_score
                    if kp3_idx is not None:
                        best_combo = {1: kp1_idx, 2: kp2_idx, 3: kp3_idx}
                    else:
                        best_combo = {1: kp1_idx, 2: kp2_idx}

    # 如果没有找到有效组合，使用最高置信度的单个点
    if best_combo is None:
        best_combo = {}
        for kp in [1, 2, 3]:
            if candidates[kp]:
                best_combo[kp] = candidates[kp][0][0]

    return best_combo


def advanced_post_process(test_features, tree_preds, dl_preds, weights):
    """
    增强版后处理流程
    """
    print("\n[9/9] 增强后处理...")

    # 加权融合预测概率
    final_proba = np.zeros((len(test_features), 4))
    for name, proba in tree_preds.items():
        final_proba += proba * weights[name]
    for name, proba in dl_preds.items():
        final_proba += proba * weights[name]

    final_predictions = np.zeros(len(test_features), dtype=int)

    # 逐井处理
    for well_id in tqdm(test_features['well_id'].unique(), desc="  逐井后处理"):
        mask = test_features['well_id'] == well_id
        group = test_features[mask].reset_index(drop=True)
        group_indices = np.where(mask)[0]
        group_proba = final_proba[group_indices]

        # 提取先验信息
        prior_keypoints = {}
        for kp in [1, 2, 3]:
            col_name = f'dist_to_prior_kp{kp}'
            if col_name in group.columns:
                dists = group[col_name].values
                if dists.min() < 900:  # 有有效先验
                    prior_keypoints[kp] = int(dists.argmin())

        # 动态规划搜索
        best_combo = dp_post_process(group, group_proba, prior_keypoints)

        # 标记关键点
        for kp, idx in best_combo.items():
            if 0 <= idx < len(group):
                global_idx = group_indices[idx]
                final_predictions[global_idx] = kp

    return final_predictions


# ==================== 数据库交互模块 ====================

def load_data_from_db(engine):
    """
    从数据库加载原始数据，并自动处理字段名映射
    """

    train_sql = """
    SELECT
        id,
        well_id        AS 转换后JH,
        depth          AS XJS,
        inclination    AS JX,
        azimuth        AS FW,
        tvd            AS LJCZJS,
        design_incl    AS JX_design,
        design_azim    AS FW_design,
        design_tvd     AS LJCZJS_design,
        label          AS 关键点
    FROM well_trajectory_train
    """

    test_sql = """
    SELECT
        id,
        well_id        AS 转换后JH,
        depth          AS XJS,
        inclination    AS JX,
        azimuth        AS FW,
        tvd            AS LJCZJS,
        design_incl    AS JX_design,
        design_azim    AS FW_design,
        design_tvd     AS LJCZJS_design
    FROM well_trajectory_val
    """

    train_df = pd.read_sql(train_sql, engine)
    test_df = pd.read_sql(test_sql, engine)

    train_df = train_df.replace('N/A', np.nan)
    test_df = test_df.replace('N/A', np.nan)

    return train_df, test_df

# ==================== 7. 主流程 ====================
def main():
    '''
    db_uri = 'mysql+pymysql://root:12345678@127.0.0.1:3306/wellbore_trajectory'
    engine = create_engine(db_uri)

    # [1/9] 数据加载
    print("\n[1/9] 正在从数据库提取原始钻井轨迹数据...")
    try:
        train_df, test_df = load_data_from_db(engine)
        print(f"  数据库连接成功！读取训练集 {len(train_df)} 行，测试集 {len(test_df)} 行。")
    except Exception as e:
        print(f"  数据库读取失败: {e}");
        return
    '''
    print("\n[1/9] 正在从本地目录提取原始钻井轨迹数据...")
    try:
        
        import pandas as pd
        import pandas as pd

        base_path = os.path.dirname(os.path.abspath(__file__))

# 拼接路径：从 src 往上一级到项目根目录，再进 data
        train_path = os.path.join(base_path, '../data/train.csv')
        test_path = os.path.join(base_path, '../data/validation_without_label.csv')

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        print(f"  文件读取成功！读取训练集 {len(train_df)} 行，测试集 {len(test_df)} 行。")
    except Exception as e:
        print(f"  文件读取失败: {e}")
        return
    # 预处理
    for df in [train_df, test_df]:
        df.replace('N/A', np.nan, inplace=True)
        numeric_cols = ['XJS', 'JX', 'FW', 'LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    # 过滤测试集 (仅保留无设计数据的部分进行预测)
    mask = (test_df['JX_design'].isna() & test_df['FW_design'].isna() & test_df['LJCZJS_design'].isna())
    test_df = test_df[mask].copy()

    # [2/9] 特征工程
    train_features = create_advanced_features_v2(train_df, is_train=True)
    test_features = create_advanced_features_v2(test_df, is_train=False)

    # [3/9] 困难负样本采样
    train_features_sampled = sample_hard_negatives(
        train_features,
        window=Config.HARD_NEGATIVE_WINDOW,
        ratio=Config.NEGATIVE_SAMPLE_RATIO
    )

    # [4/9] 准备特征
    print("\n[4/9] 准备特征...")
    exclude_cols = ['id', 'well_id', 'label']
    feature_cols = [col for col in train_features_sampled.columns if col not in exclude_cols]

    for col in feature_cols:
        if col not in test_features.columns: test_features[col] = 0

    X = np.nan_to_num(train_features_sampled[feature_cols].values, nan=0.0)
    y = train_features_sampled['label'].values
    well_ids = train_features_sampled['well_id'].values

    # 方差过滤
    selector = VarianceThreshold(threshold=0.001)
    X = selector.fit_transform(X)
    feature_cols_filtered = [f for f, m in zip(feature_cols, selector.get_support()) if m]

    # 划分验证集
    unique_wells = np.unique(well_ids)
    train_wells, val_wells = train_test_split(unique_wells, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED)
    train_mask, val_mask = np.isin(well_ids, train_wells), np.isin(well_ids, val_wells)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_mask])
    X_val_scaled = scaler.transform(X[val_mask])
    y_train, y_val = y[train_mask], y[val_mask]
    well_ids_val = well_ids[val_mask]

    # [5/9] 训练树模型 (验证阶段)
    print("\n[5/9] 训练树模型 (验证阶段)...")
    tree_scores = {}
    sample_weights_val = np.array([Config.CLASS_WEIGHTS_CLASSIFY[int(label)] for label in y_train])

    # XGBoost 验证
    xgb_model = xgb.XGBClassifier(n_estimators=1000, max_depth=16, learning_rate=0.05, random_state=Config.RANDOM_SEED,
                                  n_jobs=-1, eval_metric='mlogloss', early_stopping_rounds=50)
    xgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)], sample_weight=sample_weights_val,
                  verbose=False)
    tree_scores['xgb'] = macro_f1_with_tolerance(y_val, xgb_model.predict(X_val_scaled), well_ids_val)

    # LightGBM 验证
    lgb_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=16, learning_rate=0.05, random_state=Config.RANDOM_SEED,
                                   n_jobs=-1, class_weight=Config.CLASS_WEIGHTS_CLASSIFY, verbose=-1)
    lgb_model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    tree_scores['lgb'] = macro_f1_with_tolerance(y_val, lgb_model.predict(X_val_scaled), well_ids_val)

    # CatBoost 验证
    cat_model = cb.CatBoostClassifier(iterations=1000, depth=12, learning_rate=0.05, random_state=Config.RANDOM_SEED,
                                      verbose=False, class_weights=Config.CLASS_WEIGHTS_CLASSIFY,
                                      early_stopping_rounds=50)
    cat_model.fit(X_train_scaled, y_train, eval_set=(X_val_scaled, y_val), use_best_model=True)
    tree_scores['cat'] = macro_f1_with_tolerance(y_val, cat_model.predict(X_val_scaled).flatten(), well_ids_val)

    # [6/9] 训练深度学习 Transformer 模型
    print("\n[6/9] 训练深度学习 Transformer 模型...")
    input_dim = X_train_scaled.shape[1]
    dl_model = train_dl_model(X_train_scaled, y_train, X_val_scaled, y_val, input_dim)

    dl_val_proba = predict_dl_proba(dl_model, X_val_scaled)
    dl_val_pred = np.argmax(dl_val_proba, axis=1)
    tree_scores['transformer'] = macro_f1_with_tolerance(y_val, dl_val_pred, well_ids_val)
    print(f"    Transformer 容错F1: {tree_scores['transformer']:.4f}")

    # [7/9] 全量训练集成模型
    print("\n[7/9] 全量训练集成模型...")
    X_full = scaler.fit_transform(X)
    sample_weights_full = np.array([Config.CLASS_WEIGHTS_CLASSIFY[int(label)] for label in y])

    # 重新定义并训练全量模型
    print("  训练 XGBoost 全量...")
    xgb_full = xgb.XGBClassifier(n_estimators=xgb_model.best_iteration + 50, max_depth=16, learning_rate=0.05,
                                 random_state=Config.RANDOM_SEED, n_jobs=-1)
    xgb_full.fit(X_full, y, sample_weight=sample_weights_full, verbose=False)

    print("  训练 LightGBM 全量...")
    lgb_full = lgb.LGBMClassifier(n_estimators=lgb_model.best_iteration_ + 50, max_depth=16, learning_rate=0.05,
                                  random_state=Config.RANDOM_SEED, n_jobs=-1,
                                  class_weight=Config.CLASS_WEIGHTS_CLASSIFY, verbose=-1)
    lgb_full.fit(X_full, y)

    print("  训练 CatBoost 全量...")
    cat_full = cb.CatBoostClassifier(iterations=cat_model.best_iteration_ + 50, depth=12, learning_rate=0.05,
                                     random_state=Config.RANDOM_SEED, verbose=False,
                                     class_weights=Config.CLASS_WEIGHTS_CLASSIFY)
    cat_full.fit(X_full, y)

    # [8/9] 生成融合预测
    print("\n[8/9] 生成融合预测...")
    X_test_scaled = scaler.transform(np.nan_to_num(test_features[feature_cols_filtered].values, nan=0.0))

    tree_test_preds = {
        'xgb': xgb_full.predict_proba(X_test_scaled),
        'lgb': lgb_full.predict_proba(X_test_scaled),
        'cat': cat_full.predict_proba(X_test_scaled),
        'transformer': predict_dl_proba(dl_model, X_test_scaled)  # 使用验证集表现最佳的DL模型权重
    }

    # 计算融合权重
    weights_val = {name: score ** 1.5 for name, score in tree_scores.items()}
    total_w = sum(weights_val.values())
    final_weights = {name: w / total_w for name, w in weights_val.items()}

    for name, w in final_weights.items():
        print(f"  模型: {name:12s} | 权重: {w:.4f} | 验证F1: {tree_scores[name]:.4f}")

    # [9/9] 增强后处理
    final_predictions = advanced_post_process(test_features, tree_test_preds, {}, final_weights)

    # 生成提交并保存
    submission = pd.DataFrame({'id': test_features['id'], '关键点': final_predictions})
    submission.to_csv('submission_optimized_v3.csv', index=False)

    model_assets = {
        "tree_models": {"xgb": xgb_full, "lgb": lgb_full, "cat": cat_full},
        "dl_model_state": dl_model.state_dict(),
        "scaler": scaler,
        "selector": selector,
        "feature_cols": feature_cols_filtered,
        "final_weights": final_weights
    }
    joblib.dump(model_assets, "drilling_model_full_v3.pkl")
    print("\n✅ 所有任务完成！预测结果已保存至 CSV 和数据库。")
    return submission
if __name__ == '__main__':
    submission = main()
