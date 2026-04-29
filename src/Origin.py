import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report
from sklearn.feature_selection import VarianceThreshold

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import gc
from scipy import signal
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

print("=" * 80)
print("定向钻井关键点识别 - 深度优化版 v2.0")
print("=" * 80)

# ==================== 配置参数 ====================
class Config:
    RANDOM_SEED = 42
    TEST_SIZE = 0.15
    BATCH_SIZE = 256
    STAGE1_EPOCHS = 80  # 关键点检测
    STAGE2_EPOCHS = 100  # 关键点分类
    LEARNING_RATE = 0.001
    PATIENCE = 15

    # 容错参数
    TOLERANCE_FIRST = 1  # 第一个增斜点容错
    TOLERANCE_REST = 2   # 其余关键点容错

    # 类别权重
    CLASS_WEIGHTS_DETECTION = {0: 1, 1: 150}  # 阶段1：检测
    CLASS_WEIGHTS_CLASSIFY = {0: 1, 1: 100, 2: 100, 3: 100}  # 阶段2：分类

    # 负样本采样
    HARD_NEGATIVE_WINDOW = 150  # 关键点前后50个点作为困难负样本
    NEGATIVE_SAMPLE_RATIO = 0.8  # 保留30%的负样本

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(Config.RANDOM_SEED)
torch.manual_seed(Config.RANDOM_SEED)


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
    创建增强版特征，重点添加物理特征和设计轨迹先验
    """
    print(f"\n[2/9] 增强特征工程v2 {'(训练集)' if is_train else '(测试集)'}...")
    df = df.copy()

    # 处理N/A
    for col in ['LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    feature_list = []

    for well_id, group in tqdm(df.groupby('转换后JH'), desc="  处理井号"):
        group = group.sort_values('XJS').reset_index(drop=True)
        n = len(group)

        features = pd.DataFrame()
        features['id'] = group['id']
        features['well_id'] = group['转换后JH']

        # ========== 基础特征 ==========
        features['XJS'] = group['XJS']
        features['JX'] = group['JX']
        features['FW'] = group['FW']
        features['LJCZJS'] = group['LJCZJS']

        # 井内相对位置
        features['depth_pct'] = np.arange(n) / max(n-1, 1)

        # ========== 核心物理特征：多阶差分和导数 ==========
        # 一阶差分
        jx_diff1 = group['JX'].diff().fillna(0)
        features['JX_diff_1'] = jx_diff1

        # 二阶差分（加速度）- 关键特征！
        jx_diff2 = jx_diff1.diff().fillna(0)
        features['JX_diff_2'] = jx_diff2

        # 三阶差分（急动度）
        features['JX_diff_3'] = jx_diff2.diff().fillna(0)

        # ========== 单调性和转折点特征 ==========
        for w in [5, 10, 15, 20]:
            # 前向单调性
            monotonic_inc = []
            monotonic_dec = []
            for i in range(n):
                end = min(n, i + w)
                window = group['JX'].iloc[i:end].values
                if len(window) > 1:
                    diffs = np.diff(window)
                    monotonic_inc.append(int(np.all(diffs >= -0.01)))  # 单调递增或平稳
                    monotonic_dec.append(int(np.all(diffs <= 0.01)))   # 单调递减或平稳
                else:
                    monotonic_inc.append(0)
                    monotonic_dec.append(0)
            features[f'monotonic_inc_{w}'] = monotonic_inc
            features[f'monotonic_dec_{w}'] = monotonic_dec

            # 单调性转折
            features[f'monotonic_change_{w}'] = (
                features[f'monotonic_inc_{w}'].diff().fillna(0).abs() +
                features[f'monotonic_dec_{w}'].diff().fillna(0).abs()
            )

        # ========== 变化点检测特征 ==========
        for w in [10, 20, 30]:
            # 前后方差比
            var_before = jx_diff1.rolling(w, min_periods=1).std().shift(1).fillna(0)
            var_after = jx_diff1.rolling(w, min_periods=1).std().shift(-1).fillna(0)
            features[f'variance_ratio_{w}'] = var_after / (var_before + 1e-6)

            # 前后均值变化
            mean_before = jx_diff1.rolling(w, min_periods=1).mean().shift(1).fillna(0)
            mean_after = jx_diff1.rolling(w, min_periods=1).mean().shift(-1).fillna(0)
            features[f'mean_change_{w}'] = mean_after - mean_before

            # 前后符号变化
            sign_before = np.sign(mean_before)
            sign_after = np.sign(mean_after)
            features[f'sign_change_{w}'] = (sign_before != sign_after).astype(int)

        # ========== 滑窗统计特征 ==========
        windows = [3, 5, 10, 15, 20, 30, 50]
        for w in windows:
            rolling = group['JX'].rolling(w, min_periods=1, center=True)
            features[f'JX_mean_{w}'] = rolling.mean()
            features[f'JX_std_{w}'] = rolling.std().fillna(0)
            features[f'JX_max_{w}'] = rolling.max()
            features[f'JX_min_{w}'] = rolling.min()
            features[f'JX_range_{w}'] = features[f'JX_max_{w}'] - features[f'JX_min_{w}']

            # 差分的滑窗
            diff_rolling = jx_diff1.rolling(w, min_periods=1, center=True)
            features[f'JX_diff_mean_{w}'] = diff_rolling.mean()
            features[f'JX_diff_std_{w}'] = diff_rolling.std().fillna(0)

        # ========== 局部极值特征 ==========
        for w in [5, 10, 15]:
            features[f'is_local_max_{w}'] = (
                group['JX'] == group['JX'].rolling(w, center=True, min_periods=1).max()
            ).astype(int)
            features[f'is_local_min_{w}'] = (
                group['JX'] == group['JX'].rolling(w, center=True, min_periods=1).min()
            ).astype(int)

        # ========== 设计轨迹特征和先验信息 ==========
        has_design = not group['JX_design'].isna().all()

        if has_design:
            features['JX_design'] = group['JX_design'].fillna(method='ffill').fillna(method='bfill').fillna(0)
            features['FW_design'] = group['FW_design'].fillna(method='ffill').fillna(method='bfill').fillna(0)
            features['LJCZJS_design'] = group['LJCZJS_design'].fillna(method='ffill').fillna(method='bfill').fillna(0)

            # 偏差特征
            features['JX_deviation'] = group['JX'] - features['JX_design']
            features['JX_deviation_abs'] = features['JX_deviation'].abs()
            features['FW_deviation'] = group['FW'] - features['FW_design']
            features['FW_deviation_abs'] = features['FW_deviation'].abs()

            # 从设计轨迹提取先验关键点
            design_data = group[['XJS', 'JX_design', 'LJCZJS_design']].copy()
            design_data = design_data[~design_data['JX_design'].isna()]

            if len(design_data) > 20:
                prior_keypoints = extract_keypoints_from_design(design_data)

                # DTW对齐
                alignment_map = align_design_to_actual(design_data, group[['JX', 'LJCZJS']])

                # 距离先验关键点的距离
                for kp_type, design_idx in prior_keypoints.items():
                    if alignment_map and design_idx in alignment_map:
                        actual_idx = alignment_map[design_idx]
                        dist_to_prior = np.abs(np.arange(n) - actual_idx)
                        features[f'dist_to_prior_kp{kp_type}'] = dist_to_prior
                        features[f'near_prior_kp{kp_type}'] = (dist_to_prior < 15).astype(int)
                    else:
                        features[f'dist_to_prior_kp{kp_type}'] = 999
                        features[f'near_prior_kp{kp_type}'] = 0
            else:
                for kp_type in [1, 2, 3]:
                    features[f'dist_to_prior_kp{kp_type}'] = 999
                    features[f'near_prior_kp{kp_type}'] = 0
        else:
            for feat in ['JX_design', 'FW_design', 'LJCZJS_design',
                        'JX_deviation', 'JX_deviation_abs', 
                        'FW_deviation', 'FW_deviation_abs']:
                features[feat] = 0
            for kp_type in [1, 2, 3]:
                features[f'dist_to_prior_kp{kp_type}'] = 999
                features[f'near_prior_kp{kp_type}'] = 0

        # ========== 井级别全局特征 ==========
        features['well_max_jx'] = group['JX'].max()
        features['well_min_jx'] = group['JX'].min()
        features['well_mean_jx'] = group['JX'].mean()
        features['well_total_depth'] = group['XJS'].max()
        features['well_n_points'] = n

        # 当前JX在全井中的百分位
        features['jx_percentile'] = group['JX'].rank(pct=True)

        # 标签
        if is_train and '关键点' in group.columns:
            features['label'] = group['关键点'].values

        feature_list.append(features)

    result_df = pd.concat(feature_list, ignore_index=True)

    # 数据清洗
    numeric_cols = result_df.select_dtypes(include=[np.number]).columns
    result_df[numeric_cols] = result_df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    result_df[numeric_cols] = result_df[numeric_cols].fillna(0)

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
    K = 10  # 每类保留top-K个候选

    for kp in [1, 2, 3]:
        # 基础候选：模型高置信度位置
        kp_proba = model_proba[:, kp]
        top_k_indices = np.argsort(kp_proba)[-K:]

        for idx in top_k_indices:
            if kp_proba[idx] > 0.1:  # 最低置信度阈值
                score = kp_proba[idx]

                # 物理特征检查
                phys_score = 0
                if kp == 1:  # 增斜点：要求后续井斜角增大
                    if idx < n - 10:
                        future_trend = jx[idx+1:idx+11].mean() - jx[idx]
                        if future_trend > 0.5:
                            phys_score = 0.3
                elif kp == 2:  # 稳斜点：要求井斜角变化小
                    if idx >= 10 and idx < n - 10:
                        local_std = jx_diff[idx-10:idx+10].std()
                        if local_std < 0.2:
                            phys_score = 0.3
                elif kp == 3:  # 降斜点：要求后续井斜角减小
                    if idx < n - 10:
                        future_trend = jx[idx+1:idx+11].mean() - jx[idx]
                        if future_trend < -0.3:
                            phys_score = 0.3

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
    for kp1_idx, kp1_score in candidates[1][:5]:  # 增斜点候选
        for kp2_idx, kp2_score in candidates[2][:5]:  # 稳斜点候选
            # 约束：稳斜点必须在增斜点之后至少20个点
            if kp2_idx < kp1_idx + 20:
                continue

            for kp3_idx, kp3_score in candidates[3][:5] + [(None, 0)]:  # 降斜点候选（可选）
                # 约束：降斜点必须在稳斜点之后至少20个点
                if kp3_idx is not None and kp3_idx < kp2_idx + 20:
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


# ==================== 7. 主流程 ====================
def main():
    # 1. 加载数据
    print("\n[1/9] 加载数据...")
    train_df = pd.read_csv('SINOPEC-02/train.csv')
    test_df = pd.read_csv('SINOPEC-02/validation_without_label.csv')

    # 过滤测试集
    for col in ['JX_design', 'FW_design', 'LJCZJS_design']:
        if col in test_df.columns:
            test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
    mask = (test_df['JX_design'].isna() & test_df['FW_design'].isna() & test_df['LJCZJS_design'].isna())
    test_df = test_df[mask].copy()

    print(f"  训练集: {train_df.shape}, 测试集: {test_df.shape}")

    # 2. 增强特征工程
    train_features = create_advanced_features_v2(train_df, is_train=True)
    test_features = create_advanced_features_v2(test_df, is_train=False)

    # 3. 困难负样本采样
    train_features_sampled = sample_hard_negatives(
        train_features,
        window=Config.HARD_NEGATIVE_WINDOW,
        ratio=Config.NEGATIVE_SAMPLE_RATIO
    )

    # 4. 准备特征
    print("\n[4/9] 准备特征...")
    exclude_cols = ['id', 'well_id', 'label']
    feature_cols = [col for col in train_features_sampled.columns if col not in exclude_cols]

    # 确保测试集特征一致
    for col in feature_cols:
        if col not in test_features.columns:
            test_features[col] = 0

    X = train_features_sampled[feature_cols].values
    y = train_features_sampled['label'].values
    well_ids = train_features_sampled['well_id'].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 方差过滤
    selector = VarianceThreshold(threshold=0.001)
    X = selector.fit_transform(X)
    feature_cols_filtered = [f for f, m in zip(feature_cols, selector.get_support()) if m]
    print(f"  特征数: {len(feature_cols_filtered)}")

    # 划分训练/验证集
    unique_wells = np.unique(well_ids)
    train_wells, val_wells = train_test_split(unique_wells, test_size=Config.TEST_SIZE, 
                                               random_state=Config.RANDOM_SEED)

    train_mask = np.isin(well_ids, train_wells)
    val_mask = np.isin(well_ids, val_wells)

    X_train, X_val = X[train_mask], X[val_mask]
    y_train, y_val = y[train_mask], y[val_mask]
    well_ids_val = well_ids[val_mask]

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 测试集
    X_test = test_features[feature_cols_filtered].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练树模型
    print("\n[5/9] 训练树模型...")
    tree_models = {}
    tree_scores = {}

    sample_weights = np.array([Config.CLASS_WEIGHTS_CLASSIFY[int(label)] for label in y_train])

    # XGBoost
    print("  XGBoost...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, max_depth=16, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED, n_jobs=-1,
        eval_metric='mlogloss', early_stopping_rounds=50
    )
    xgb_model.fit(X_train_scaled, y_train, 
                  eval_set=[(X_val_scaled, y_val)],
                  sample_weight=sample_weights, verbose=False)

    val_pred_xgb = xgb_model.predict(X_val_scaled)
    xgb_f1_strict = f1_score(y_val, val_pred_xgb, average='macro')
    xgb_f1_tolerant = macro_f1_with_tolerance(y_val, val_pred_xgb, well_ids_val)
    tree_models['xgb'] = xgb_model
    tree_scores['xgb'] = xgb_f1_tolerant
    print(f"    严格F1: {xgb_f1_strict:.4f}, 容错F1: {xgb_f1_tolerant:.4f}")

    # LightGBM
    print("  LightGBM...")
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, max_depth=16, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED, n_jobs=-1,
        class_weight=Config.CLASS_WEIGHTS_CLASSIFY, verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_val_scaled, y_val)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])

    val_pred_lgb = lgb_model.predict(X_val_scaled)
    lgb_f1_tolerant = macro_f1_with_tolerance(y_val, val_pred_lgb, well_ids_val)
    tree_models['lgb'] = lgb_model
    tree_scores['lgb'] = lgb_f1_tolerant
    print(f"    容错F1: {lgb_f1_tolerant:.4f}")

    # CatBoost
    print("  CatBoost...")
    cat_model = cb.CatBoostClassifier(
        iterations=1000, depth=12, learning_rate=0.05,
        random_state=Config.RANDOM_SEED, verbose=False,
        class_weights=Config.CLASS_WEIGHTS_CLASSIFY,
        early_stopping_rounds=50
    )
    cat_model.fit(X_train_scaled, y_train,
                  eval_set=(X_val_scaled, y_val),
                  use_best_model=True)

    val_pred_cat = cat_model.predict(X_val_scaled).flatten()
    cat_f1_tolerant = macro_f1_with_tolerance(y_val, val_pred_cat, well_ids_val)
    tree_models['cat'] = cat_model
    tree_scores['cat'] = cat_f1_tolerant
    print(f"    容错F1: {cat_f1_tolerant:.4f}")

    # 6-8. 简化：只用树模型集成（深度学习部分省略以控制代码长度）

    # 9. 全量训练
    print("\n[6/9] 全量训练...")
    X_full = scaler.fit_transform(X)
    X_test_full = scaler.transform(X_test)

    sample_weights_full = np.array([Config.CLASS_WEIGHTS_CLASSIFY[int(label)] for label in y])

    print("  XGBoost全量...")
    xgb_full = xgb.XGBClassifier(
        n_estimators=xgb_model.best_iteration + 50,
        max_depth=16, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED, n_jobs=-1
    )
    xgb_full.fit(X_full, y, sample_weight=sample_weights_full, verbose=False)

    print("  LightGBM全量...")
    lgb_full = lgb.LGBMClassifier(
        n_estimators=lgb_model.best_iteration_ + 50,
        max_depth=16, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=Config.RANDOM_SEED, n_jobs=-1,
        class_weight=Config.CLASS_WEIGHTS_CLASSIFY, verbose=-1
    )
    lgb_full.fit(X_full, y)

    print("  CatBoost全量...")
    cat_full = cb.CatBoostClassifier(
        iterations=cat_model.best_iteration_ + 50,
        depth=12, learning_rate=0.05,
        random_state=Config.RANDOM_SEED, verbose=False,
        class_weights=Config.CLASS_WEIGHTS_CLASSIFY
    )
    cat_full.fit(X_full, y)

    # 10. 预测
    print("\n[7/9] 生成预测...")
    tree_test_preds = {
        'xgb': xgb_full.predict_proba(X_test_full),
        'lgb': lgb_full.predict_proba(X_test_full),
        'cat': cat_full.predict_proba(X_test_full)
    }

    # 11. 计算权重
    print("\n[8/9] 计算融合权重...")
    weights = {name: score**2 for name, score in tree_scores.items()}
    total_weight = sum(weights.values())
    weights = {name: w / total_weight for name, w in weights.items()}

    for name, weight in weights.items():
        print(f"  {name}: {weight:.4f} (F1: {tree_scores[name]:.4f})")

    # 12. 增强后处理
    final_predictions = advanced_post_process(
        test_features, tree_test_preds, {}, weights
    )

    # 13. 生成提交
    submission = pd.DataFrame({
        'id': test_features['id'],
        '关键点': final_predictions
    })
    submission.to_csv('submission_optimized_v2.csv', index=False)

    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"\n提交文件: submission_optimized_v2.csv")
    print(f"\n预测分布:")
    for label in [0, 1, 2, 3]:
        count = (final_predictions == label).sum()
        pct = count / len(final_predictions) * 100
        print(f"  类别{label}: {count:>5} ({pct:>5.2f}%)")

    return submission


if __name__ == '__main__':
    submission = main()