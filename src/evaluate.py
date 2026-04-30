#!/usr/bin/env python3
"""
================================================================================
  钻井轨迹造斜关键点识别 - 完整评估脚本 V3
  Evaluate & Score your Super Hybrid Model
================================================================================

  功能:
    1. 从 train.csv 按井号智能分层划分训练集/测试集
    2. 完整运行 6 个独立模型 + 3 种混合集成
    3. 表1：各模型容错测试集评分 ⭐
    4. 表2：显著性检验（配对 t 检验）
    5. 表3：物理容错偏移量分析
    6. 输出 MD 报告

  用法:
    python src/evaluate.py                          # 默认 15% 测试井
    python src/evaluate.py --test_ratio 0.2         # 自定义比例
    python src/evaluate.py --quick                  # 快速模式（少轮数）
    python src/evaluate.py --output ./eval_results  # 指定输出目录
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from scipy import stats as scipy_stats

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2 import (
    Config, create_advanced_features_v2, sample_hard_negatives,
    macro_f1_with_tolerance, advanced_post_process,
    advanced_post_process_v2, dp_post_process,
    optimize_dp_params
)
from dl_improved import (
    DrillingWindowDataset,
    build_model, train_enhanced_model, predict_enhanced_proba,
    MODEL_REGISTRY
)
from v3 import (
    TwoStagePipeline, kfold_tree_ensemble, kfold_predict_proba,
    augment_kp3, v4_predict_on_test
)

# =====================================================================
#  1. 智能井号划分
# =====================================================================

def stratified_well_split(df, test_ratio=0.15, random_seed=42):
    """按井号分层划分训练/测试井"""
    well_info = {}
    for well_id, group in df.groupby('转换后JH'):
        labels = set(group['关键点'].unique())
        well_info[well_id] = {
            'has_1': int(1 in labels),
            'has_2': int(2 in labels),
            'has_3': int(3 in labels),
            'n_points': len(group),
        }

    info_df = pd.DataFrame.from_dict(well_info, orient='index')
    strat_labels = info_df[['has_1', 'has_2', 'has_3']].apply(
        lambda r: f"{r['has_1']}{r['has_2']}{r['has_3']}", axis=1
    )

    all_wells = info_df.index.tolist()
    unique_strat = strat_labels.value_counts()
    if (unique_strat > 1).all():
        train_wells, test_wells = train_test_split(
            all_wells, test_size=test_ratio, random_state=random_seed,
            stratify=strat_labels
        )
    else:
        print("  ⚠  某些关键点组合只有 1 口井，改用普通随机划分")
        train_wells, test_wells = train_test_split(
            all_wells, test_size=test_ratio, random_state=random_seed
        )

    print(f"\n{'='*60}")
    print(f"  训练井数: {len(train_wells)}  |  测试井数: {len(test_wells)}")
    print(f"  训练井: {', '.join(train_wells)}")
    print(f"  测试井: {', '.join(test_wells)}")
    print(f"{'='*60}\n")
    return train_wells, test_wells


# =====================================================================
#  2. 核心训练 Pipeline
# =====================================================================

def train_pipeline(train_df, hyperparams=None, dl_model_types=None,
                  v4_mode=False):
    """
    在 train_df 上完整训练混合模型（树模型 + 多个 DL 变体）。
    v4_mode: 额外训练 K 折树集成 + 两阶段 DL，用于生成更高精度的预测。
    返回: (模型资产 dict, 验证集得分 dict)
    """
    if hyperparams is None:
        hyperparams = {}
    if dl_model_types is None:
        dl_model_types = ['hybrid_v3', 'lstm_only', 'transformer_only']

    # 关键修复：NaN标签必须填充为0，否则被当成关键点导致训练数据污染
    if '关键点' in train_df.columns:
        n_nan = train_df['关键点'].isna().sum()
        if n_nan > 0:
            train_df['关键点'] = train_df['关键点'].fillna(0).astype(int)
            print(f"  [修复] 填充了 {n_nan} 个 NaN 标签为 0")

    hp = {**{
        'hard_neg_window': 50,
        'neg_sample_ratio': 0.638,
        'tree_n_estimators': 600,
        'tree_max_depth': 12,
        'tree_learning_rate': 0.05,
        'xgb_subsample': 0.789,
        'xgb_colsample_bytree': 0.648,
        'xgb_min_child_weight': 8,
        'xgb_gamma': 3.8,
        'xgb_reg_lambda': 1.33,
        'xgb_reg_alpha': 3.86,
        'lgb_num_leaves': 31,
        'lgb_subsample': 0.809,
        'lgb_colsample_bytree': 0.771,
        'lgb_min_child_samples': 15,
        'lgb_reg_lambda': 1.08,
        'lgb_reg_alpha': 0.314,
        'cat_depth': 8,
        'cat_l2_leaf_reg': 3.0,
        'cat_subsample': 0.803,
        'cat_border_count': 235,
        'dl_hidden_dim': 128,
        'dl_lr': 0.001,
        'dl_batch_size': 256,
        'dl_dropout': 0.3,
        'dl_weight_decay': 1e-5,
        'dl_epochs': 150,
        'dl_window_size': 51,
        'class_weight_cls_positive': 50,
    }, **hyperparams}

    # --- 特征工程 ---
    print("[1] 特征工程 ...")
    train_features = create_advanced_features_v2(train_df, is_train=True)

    # --- 可选: KP3 数据增强 (仅 v4_mode) ---
    if v4_mode and 'label' in train_features.columns:
        print("\n[V4] KP3 数据增强 ...")
        kp3_before = int((train_features['label'] == 3).sum())
        exclude = ['id', 'well_id', 'label']
        feat_cols_aug = [c for c in train_features.columns if c not in exclude]
        X_aug, y_aug, wid_aug = augment_kp3(
            train_features[feat_cols_aug].values,
            train_features['label'].values,
            train_features['well_id'].values,
            n_shift=3, noise_std=0.01
        )
        kp3_after = int((y_aug == 3).sum())
        print(f"    KP3 样本: {kp3_before} → {kp3_after} (+{kp3_after - kp3_before})")
        aug_df = pd.DataFrame(X_aug, columns=feat_cols_aug)
        aug_df['label'] = y_aug
        aug_df['well_id'] = wid_aug
        if 'id' in train_features.columns:
            n_orig = len(train_features)
            aug_df['id'] = np.concatenate([
                train_features['id'].values,
                np.full(len(aug_df) - n_orig, -1, dtype=np.int64)
            ])
        train_features = aug_df

    # --- 负采样 ---
    print("[2] 困难负样本采样 ...")
    train_sampled = sample_hard_negatives(
        train_features,
        window=hp['hard_neg_window'],
        ratio=hp['neg_sample_ratio']
    )

    # --- 特征矩阵 ---
    exclude_cols = ['id', 'well_id', 'label']
    feature_cols = [c for c in train_sampled.columns if c not in exclude_cols]
    X = np.nan_to_num(train_sampled[feature_cols].values, nan=0.0)
    y = train_sampled['label'].values.astype(int)
    well_ids = train_sampled['well_id'].values

    selector = VarianceThreshold(threshold=0.001)
    X = selector.fit_transform(X)
    X_raw = X.copy()  # 保存缩放前的原始特征（K折用）
    feature_cols_filtered = [f for f, m in zip(feature_cols, selector.get_support()) if m]

    # --- 内部验证划分 ---
    unique_wells = np.unique(well_ids)
    train_wells_inner, val_wells_inner = train_test_split(
        unique_wells, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
    )
    train_mask = np.isin(well_ids, train_wells_inner)
    val_mask   = np.isin(well_ids, val_wells_inner)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X[train_mask])
    X_val_scaled   = scaler.transform(X[val_mask])
    y_train, y_val = y[train_mask], y[val_mask]
    well_ids_val   = well_ids[val_mask]

    cw_dict = {0: 1, 1: hp['class_weight_cls_positive'], 2: int(hp['class_weight_cls_positive'] * 1.5), 3: int(hp['class_weight_cls_positive'] * 4)}
    sample_weights_train = np.array([cw_dict[int(l)] for l in y_train])

    tree_models = {}
    val_scores  = {}

    # --- 3. 训练树模型 ---
    print("[3] 训练树模型 ...")

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=hp['tree_n_estimators'],
        max_depth=hp['tree_max_depth'],
        learning_rate=hp['tree_learning_rate'],
        subsample=hp['xgb_subsample'],
        colsample_bytree=hp['xgb_colsample_bytree'],
        min_child_weight=hp['xgb_min_child_weight'],
        gamma=hp['xgb_gamma'],
        reg_lambda=hp['xgb_reg_lambda'],
        reg_alpha=hp['xgb_reg_alpha'],
        random_state=Config.RANDOM_SEED,
        n_jobs=1, eval_metric='mlogloss',
        early_stopping_rounds=50,
        verbosity=0,
    )
    xgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        sample_weight=sample_weights_train,
        verbose=False,
    )
    val_scores['xgb'] = macro_f1_with_tolerance(y_val, xgb_model.predict(X_val_scaled), well_ids_val)
    tree_models['xgb'] = xgb_model
    print(f"    XGBoost 验证 F1: {val_scores['xgb']:.4f}")

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=hp['tree_n_estimators'],
        max_depth=hp['tree_max_depth'],
        learning_rate=hp['tree_learning_rate'],
        num_leaves=hp['lgb_num_leaves'],
        subsample=hp['lgb_subsample'],
        colsample_bytree=hp['lgb_colsample_bytree'],
        min_child_samples=hp['lgb_min_child_samples'],
        reg_lambda=hp['lgb_reg_lambda'],
        reg_alpha=hp['lgb_reg_alpha'],
        class_weight=cw_dict,
        random_state=Config.RANDOM_SEED,
        n_jobs=1, verbose=-1,
    )
    lgb_model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    val_scores['lgb'] = macro_f1_with_tolerance(y_val, lgb_model.predict(X_val_scaled), well_ids_val)
    tree_models['lgb'] = lgb_model
    print(f"    LightGBM 验证 F1: {val_scores['lgb']:.4f}")

    # CatBoost
    cat_model = cb.CatBoostClassifier(
        iterations=hp['tree_n_estimators'],
        depth=hp['cat_depth'],
        learning_rate=hp['tree_learning_rate'],
        l2_leaf_reg=hp['cat_l2_leaf_reg'],
        subsample=hp['cat_subsample'],
        bootstrap_type='Bernoulli',
        border_count=hp['cat_border_count'],
        class_weights=cw_dict,
        random_state=Config.RANDOM_SEED,
        verbose=False,
        early_stopping_rounds=50,
    )
    cat_model.fit(
        X_train_scaled, y_train,
        eval_set=(X_val_scaled, y_val),
        use_best_model=True,
    )
    val_scores['cat'] = macro_f1_with_tolerance(y_val, cat_model.predict(X_val_scaled).flatten(), well_ids_val)
    tree_models['cat'] = cat_model
    print(f"    CatBoost 验证 F1: {val_scores['cat']:.4f}")

    # --- 4. 训练深度学习 V3 系列模型 ---
    input_dim = X_train_scaled.shape[1]
    ws = hp['dl_window_size']
    dl_cw = [1.0, float(cw_dict[1]), float(cw_dict[2]), float(cw_dict[3])]

    dl_models = {}
    trained_types = []
    for mtype in dl_model_types:
        print(f"\n[4] 训练深度学习 {mtype} 模型 ...")
        dl_model = train_enhanced_model(
            X_train_scaled, y_train, well_ids[train_mask],
            X_val_scaled, y_val, well_ids[val_mask],
            model_type=mtype, n_features=input_dim, window_size=ws,
            hidden_dim=hp['dl_hidden_dim'],
            lr=hp['dl_lr'], batch_size=hp['dl_batch_size'],
            epochs=hp['dl_epochs'], dropout=hp['dl_dropout'],
            weight_decay=hp['dl_weight_decay'],
            class_weights=dl_cw,
        )
        dl_val_proba = predict_enhanced_proba(dl_model, X_val_scaled, well_ids[val_mask], ws)
        dl_val_pred = np.argmax(dl_val_proba, axis=1)
        val_scores[mtype] = macro_f1_with_tolerance(y_val, dl_val_pred, well_ids_val)
        dl_models[mtype] = dl_model
        trained_types.append(mtype)
        print(f"    {mtype:>16s} 验证 F1: {val_scores[mtype]:.4f}")

    # --- 收集验证集预测用于权重优化 ---
    print("\n[权重优化] 收集验证集预测...")
    val_probas = {}
    min_f1_threshold = 0.5
    for name in ['xgb', 'lgb', 'cat']:
        if name in val_scores and val_scores[name] >= min_f1_threshold:
            val_probas[name] = tree_models[name].predict_proba(X_val_scaled)
            print(f"    收集 {name}: 验证F1={val_scores[name]:.4f}")
    for mtype in trained_types:
        if mtype in val_scores and val_scores[mtype] >= min_f1_threshold:
            val_probas[mtype] = predict_enhanced_proba(dl_models[mtype], X_val_scaled, well_ids[val_mask], ws)
            print(f"    收集 {mtype}: 验证F1={val_scores[mtype]:.4f}")

    def eval_val_ensemble(weights_dict):
        pred = np.zeros((len(X_val_scaled), 4))
        for name, w in weights_dict.items():
            if name in val_probas:
                pred += val_probas[name] * w
        return macro_f1_with_tolerance(y_val, np.argmax(pred, axis=1), well_ids_val)

    best_ensemble_f1 = 0
    best_final_weights = None
    sorted_models = sorted(val_probas.keys(), key=lambda n: val_scores[n], reverse=True)
    print(f"\n    [权重搜索] 候选模型: {sorted_models}")

    # 策略1: 单模型
    w1 = {sorted_models[0]: 1.0}
    f1_1 = eval_val_ensemble(w1)
    best_ensemble_f1 = f1_1
    best_final_weights = w1
    print(f"    [单模型] {sorted_models[0]}: F1={f1_1:.4f} ✓")

    # 策略2: 等权重 top-k
    for k in range(2, len(sorted_models) + 1):
        top_k = sorted_models[:k]
        w = {n: 1.0 / k for n in top_k}
        f1 = eval_val_ensemble(w)
        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_final_weights = w
            print(f"    [等权重top{k}]: F1={f1:.4f} ✓ (优于 {best_ensemble_f1:.4f})")

    # 策略3: score² 加权 top-k
    for k in range(2, len(sorted_models) + 1):
        top_k = sorted_models[:k]
        scores = {n: val_scores[n] for n in top_k}
        total = sum(v ** 2 for v in scores.values())
        w = {n: (val_scores[n] ** 2) / total for n in top_k}
        f1 = eval_val_ensemble(w)
        if f1 > best_ensemble_f1:
            best_ensemble_f1 = f1
            best_final_weights = w
            print(f"    [平方加权top{k}]: F1={f1:.4f} ✓ (优于 {best_ensemble_f1:.4f})")

    final_weights = best_final_weights
    print(f"\n    → 最优策略: 验证F1={best_ensemble_f1:.4f}")
    for n, w in final_weights.items():
        print(f"       {n:16s}: 权重={w:.4f}")

    # --- 5. 全量训练集成模型 ---
    print("\n[5] 全量训练集成模型 ...")
    X_full = scaler.fit_transform(X)
    sample_weights_full = np.array([cw_dict[int(l)] for l in y])

    best_iter = {}
    best_iter['xgb'] = getattr(xgb_model, 'best_iteration', None) or max(50, int(hp['tree_n_estimators'] * 0.6))
    best_iter['lgb'] = getattr(lgb_model, 'best_iteration_', None) or max(50, int(hp['tree_n_estimators'] * 0.6))
    best_iter['cat'] = getattr(cat_model, 'best_iteration_', None) or max(50, int(hp['tree_n_estimators'] * 0.6))

    xgb_full = xgb.XGBClassifier(
        n_estimators=best_iter['xgb'] + 50,
        max_depth=hp['tree_max_depth'],
        learning_rate=hp['tree_learning_rate'],
        subsample=hp['xgb_subsample'],
        colsample_bytree=hp['xgb_colsample_bytree'],
        min_child_weight=hp['xgb_min_child_weight'],
        gamma=hp['xgb_gamma'],
        reg_lambda=hp['xgb_reg_lambda'],
        reg_alpha=hp['xgb_reg_alpha'],
        random_state=Config.RANDOM_SEED,
        n_jobs=1, verbosity=0,
    )
    xgb_full.fit(X_full, y, sample_weight=sample_weights_full, verbose=False)

    lgb_full = lgb.LGBMClassifier(
        n_estimators=best_iter['lgb'] + 50,
        max_depth=hp['tree_max_depth'],
        learning_rate=hp['tree_learning_rate'],
        num_leaves=hp['lgb_num_leaves'],
        subsample=hp['lgb_subsample'],
        colsample_bytree=hp['lgb_colsample_bytree'],
        min_child_samples=hp['lgb_min_child_samples'],
        reg_lambda=hp['lgb_reg_lambda'],
        reg_alpha=hp['lgb_reg_alpha'],
        class_weight=cw_dict,
        random_state=Config.RANDOM_SEED,
        n_jobs=1, verbose=-1,
    )
    lgb_full.fit(X_full, y)

    cat_full = cb.CatBoostClassifier(
        iterations=best_iter['cat'] + 50,
        depth=hp['cat_depth'],
        learning_rate=hp['tree_learning_rate'],
        l2_leaf_reg=hp['cat_l2_leaf_reg'],
        subsample=hp['cat_subsample'],
        bootstrap_type='Bernoulli',
        border_count=hp['cat_border_count'],
        class_weights=cw_dict,
        random_state=Config.RANDOM_SEED,
        verbose=False,
    )
    cat_full.fit(X_full, y)

    print("\n  最终模型融合权重:")
    for name, w in final_weights.items():
        print(f"    {name:16s}  权重: {w:.4f}  |  验证 F1: {val_scores[name]:.4f}")

    # ---- K折树集成 (始终训练, 作为主模型) ----
    kfold_models = None
    kfold_weights = None
    kfold_val_f1 = None
    kfold_scaler = None
    two_stage_pipeline = None
    two_stage_val_f1 = None

    print("\n[4] 训练 K折树集成 ...")
    kfold_models, oof_preds, kfold_weights, kfold_val_f1, kfold_scaler = kfold_tree_ensemble(
        X_raw, y, well_ids, verbose=True
    )
    print(f"    K折平均 ValF1={np.mean(kfold_val_f1):.4f} ± {np.std(kfold_val_f1):.4f}")

    # ---- DP 参数网格搜索 (仅 v4_mode) ----
    best_dp_params = None
    dp_grid_score = None
    if v4_mode:
        print("\n[V4] DP 参数网格搜索...")
        val_features_df = train_sampled[val_mask].copy().reset_index(drop=True)
        val_tree_preds = {}
        for name in ['xgb', 'lgb', 'cat']:
            if name in val_scores and val_scores[name] > 0.1:
                val_tree_preds[name] = tree_models[name].predict_proba(X_val_scaled)
        best_dp_params, dp_grid_score, _ = optimize_dp_params(
            val_features_df, val_tree_preds,
            y_val, well_ids_val,
            verbose=True
        )
        print(f"  → 最佳 DP 参数: {best_dp_params}, 验证 F1={dp_grid_score:.4f}")

    # ---- 可选: 两阶段 DL (仅 v4_mode) ----
    if v4_mode:
        print("\n[V4] 训练两阶段 DL ...")
        n_features = X_raw.shape[1]
        two_stage = TwoStagePipeline(
            n_features=n_features, window_size=hp['dl_window_size'],
            hidden_dim=hp['dl_hidden_dim'], lr=hp['dl_lr'],
            batch_size=hp['dl_batch_size'], dropout=hp['dl_dropout']
        )
        two_stage_val_f1 = two_stage.train(
            X[train_mask], y[train_mask], well_ids[train_mask],
            X[val_mask], y[val_mask], well_ids[val_mask],
            epochs=hp['dl_epochs'], patience=30, verbose=True
        )
        two_stage_pipeline = two_stage
        print(f"    两阶段 DL 验证 F1={two_stage_val_f1:.4f}")

    assets = {
        'tree_models': {'xgb': xgb_full, 'lgb': lgb_full, 'cat': cat_full},
        'dl_models': dl_models,
        'scaler': scaler,
        'selector': selector,
        'feature_cols': feature_cols_filtered,
        'final_weights': final_weights,
        'val_scores': val_scores,
        'hyperparams': hp,
        'dl_window_size': hp['dl_window_size'],
        'dl_model_types': trained_types,
        'kfold_models': kfold_models,
        'kfold_weights': kfold_weights,
        'kfold_val_f1': kfold_val_f1,
        'kfold_scaler': kfold_scaler,
        'best_dp_params': best_dp_params,
        'dp_grid_score': dp_grid_score,
        'two_stage_pipeline': two_stage_pipeline,
        'two_stage_val_f1': two_stage_val_f1,
    }
    return assets


# =====================================================================
#  3. 预测函数
# =====================================================================

def predict_single_model(assets, test_features, model_name):
    """用单个模型预测，返回 argmax 标签"""
    scaler = assets['scaler']
    feature_cols = assets['feature_cols']
    test_X = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
    test_X_scaled = scaler.transform(test_X)

    ws = assets.get('dl_window_size', 51)
    dl_model_types = assets.get('dl_model_types', [])

    if model_name in dl_model_types:
        proba = predict_enhanced_proba(
            assets['dl_models'][model_name], test_X_scaled,
            test_features['well_id'].values, ws
        )
    elif model_name in assets['tree_models']:
        proba = assets['tree_models'][model_name].predict_proba(test_X_scaled)
    else:
        raise ValueError(f"未知模型: {model_name}")
    return np.argmax(proba, axis=1)


def predict_weighted_ensemble(assets, test_features, model_subset=None):
    """加权集成预测，返回 argmax 标签"""
    scaler = assets['scaler']
    feature_cols = assets['feature_cols']
    final_weights = assets['final_weights']

    test_X = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
    test_X_scaled = scaler.transform(test_X)

    ws = assets.get('dl_window_size', 51)
    dl_model_types = assets.get('dl_model_types', [])

    if model_subset is None:
        model_subset = list(final_weights.keys())

    # 归一化子集权重
    subset_weights = {k: final_weights[k] for k in model_subset if k in final_weights}
    if not subset_weights:
        n = len(model_subset)
        subset_weights = {k: 1.0/n for k in model_subset}

    total_w = sum(subset_weights.values())
    subset_weights = {k: w/total_w for k, w in subset_weights.items()}

    final_proba = np.zeros((len(test_X_scaled), 4))
    for name, w in subset_weights.items():
        if name in dl_model_types:
            proba = predict_enhanced_proba(
                assets['dl_models'][name], test_X_scaled,
                test_features['well_id'].values, ws
            )
        elif name in assets['tree_models']:
            proba = assets['tree_models'][name].predict_proba(test_X_scaled)
        else:
            continue
        final_proba += proba * w

    return np.argmax(final_proba, axis=1)


def predict_ml_only(assets, test_features):
    """仅树模型集成（无 DP）"""
    ml_keys = [k for k in assets['final_weights']
               if k in assets['tree_models']]
    return predict_weighted_ensemble(assets, test_features, ml_keys)


def predict_ml_only_dp(assets, test_features):
    """仅树模型集成（带 DP 后处理）—— 优先使用 K-fold 预测"""
    feature_cols = assets['feature_cols']
    best_dp_params = assets.get('best_dp_params')

    kfold_models = assets.get('kfold_models')
    kfold_weights = assets.get('kfold_weights')

    if kfold_models is not None and kfold_weights is not None:
        # 使用 K-fold 预测（更稳定）
        test_feat_array = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
        kfold_scaler = assets.get('kfold_scaler')
        tree_proba, per_model = kfold_predict_proba(
            kfold_models, test_feat_array, kfold_weights, scaler=kfold_scaler
        )
        tree_preds = per_model
        dp_weights = kfold_weights
    else:
        # 回退到单次划分模型
        scaler = assets['scaler']
        test_X = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
        test_X_scaled = scaler.transform(test_X)
        final_weights = assets['final_weights']

        ml_weights = {k: v for k, v in final_weights.items() if k in assets['tree_models']}
        if sum(ml_weights.values()) > 0:
            total = sum(ml_weights.values())
            ml_weights = {k: v/total for k, v in ml_weights.items()}

        tree_preds = {}
        for name in ml_weights:
            tree_preds[name] = assets['tree_models'][name].predict_proba(test_X_scaled)
        dp_weights = ml_weights

    # 使用最佳 DP 参数（如果存在）
    dp_kwargs = best_dp_params or {}
    return advanced_post_process_v2(test_features, tree_preds, {}, dp_weights,
                                    **dp_kwargs)


def predict_dl_only_dp(assets, test_features, model_name):
    """单个 DL 模型 + DP 后处理"""
    scaler = assets['scaler']
    feature_cols = assets['feature_cols']
    ws = assets.get('dl_window_size', 51)
    test_X = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
    test_X_scaled = scaler.transform(test_X)
    proba = predict_enhanced_proba(
        assets['dl_models'][model_name], test_X_scaled,
        test_features['well_id'].values, ws
    )
    return advanced_post_process(test_features, {model_name: proba}, {}, {model_name: 1.0})


def predict_ml_ensemble_dp(assets, test_features):
    """ML ensemble with DP"""
    return predict_ml_only_dp(assets, test_features)


def predict_on_test(assets, test_features):
    """完整大混合 + DP 后处理 —— 优先使用 K-fold，使用 best_dp_params"""
    feature_cols = assets['feature_cols']
    best_dp_params = assets.get('best_dp_params')
    dp_kwargs = best_dp_params or {}

    kfold_models = assets.get('kfold_models')
    kfold_weights = assets.get('kfold_weights')

    if kfold_models is not None and kfold_weights is not None:
        # 使用 K-fold 树集成 + DP
        test_feat_array = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
        kfold_scaler = assets.get('kfold_scaler')
        tree_proba, per_model = kfold_predict_proba(
            kfold_models, test_feat_array, kfold_weights, scaler=kfold_scaler
        )
        return advanced_post_process_v2(test_features, per_model, {}, kfold_weights,
                                        **dp_kwargs)
    else:
        # 回退到单次划分模型
        scaler = assets['scaler']
        final_weights = assets['final_weights']
        test_X = np.nan_to_num(test_features[feature_cols].values, nan=0.0)
        test_X_scaled = scaler.transform(test_X)

        tree_preds = {}
        for name, model in assets['tree_models'].items():
            if name in final_weights:
                tree_preds[name] = model.predict_proba(test_X_scaled)

        dl_preds = {}
        dl_model_types = assets.get('dl_model_types', [])
        ws = assets.get('dl_window_size', 51)
        for mtype in dl_model_types:
            if mtype in final_weights:
                dl_preds[mtype] = predict_enhanced_proba(
                    assets['dl_models'][mtype], test_X_scaled,
                    test_features['well_id'].values, ws
                )

        return advanced_post_process_v2(test_features, tree_preds, dl_preds, final_weights,
                                        **dp_kwargs)


# =====================================================================
#  4. 辅助分析函数
# =====================================================================

def compute_per_well_f1(y_true, y_pred, well_ids):
    """按井计算 Macro F1"""
    scores = {}
    for wid in np.unique(well_ids):
        m = well_ids == wid
        scores[str(wid)] = f1_score(y_true[m], y_pred[m], average='macro', zero_division=0)
    return scores


def paired_ttest(scores_a, scores_b, label_a='A', label_b='B'):
    """配对 t 检验"""
    common = sorted(set(scores_a) & set(scores_b))
    a = [scores_a[k] for k in common]
    b = [scores_b[k] for k in common]
    n = len(a)
    if n < 2:
        return 0.0, 1.0, n, None, None
    mean_a, mean_b = np.mean(a), np.mean(b)
    diffs = np.array(a) - np.array(b)
    t_stat, p_val = scipy_stats.ttest_rel(a, b)
    return float(t_stat), float(p_val), n, mean_a, mean_b


def analyze_tolerance_offsets(y_true, y_pred, well_ids, tol_first=1, tol_rest=2):
    """容错偏移量分析"""
    results = {1: [], 2: [], 3: []}
    for wid in np.unique(well_ids):
        m  = well_ids == wid
        ix = np.where(m)[0]
        tl = y_true[m]
        pl = y_pred[m]
        for kp in [1, 2, 3]:
            tp = ix[tl == kp]
            if len(tp) == 0:
                continue
            true_pos = int(tp[0])
            tol = tol_first if kp == 1 else tol_rest
            pp = ix[pl == kp]
            if len(pp) > 0:
                off = int(abs(true_pos - pp[0]))
            else:
                off = None
            results[kp].append({
                'well_id': str(wid), 'true_pos': true_pos,
                'pred_pos': int(pp[0]) if len(pp) > 0 else None,
                'offset': off, 'tolerance': tol,
                'within_tol': off is not None and off <= tol,
            })
    return results


def offset_summary_rows(results):
    """生成偏移量汇总行"""
    kp_names = {1: '增斜(1)', 2: '稳斜(2)', 3: '降斜(3)'}
    rows = []
    for kp in [1, 2, 3]:
        n_true = len(results[kp])
        offsets = [r['offset'] for r in results[kp] if r['offset'] is not None]
        n_det  = len(offsets)
        within = sum(1 for r in results[kp] if r.get('within_tol'))
        le1  = sum(1 for o in offsets if o <= 1)
        le2  = sum(1 for o in offsets if o <= 2)
        le5  = sum(1 for o in offsets if o <= 5)
        le10 = sum(1 for o in offsets if o <= 10)
        rows.append({
            'name': kp_names[kp],
            'n_true': n_true, 'n_det': n_det,
            'avg_off': float(np.mean(offsets)) if offsets else float('nan'),
            'med_off': float(np.median(offsets)) if offsets else float('nan'),
            'std_off': float(np.std(offsets)) if offsets else float('nan'),
            'within_pct': within / n_true * 100 if n_true else 0.0,
            'le1_pct':  le1 / n_true * 100 if n_true else 0.0,
            'le2_pct':  le2 / n_true * 100 if n_true else 0.0,
            'le5_pct':  le5 / n_true * 100 if n_true else 0.0,
            'le10_pct': le10 / n_true * 100 if n_true else 0.0,
        })
    return rows


def format_md_table(rows, headers, align=None):
    """将列表字典格式化为 markdown 表格"""
    if align is None:
        align = ['---'] * len(headers)
    hdr = '| ' + ' | '.join(headers) + ' |'
    sep = '| ' + ' | '.join(align) + ' |'
    body = []
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h, '')
            if isinstance(v, float):
                vals.append(f'{v:.4f}')
            else:
                vals.append(str(v))
        body.append('| ' + ' | '.join(vals) + ' |')
    return hdr + '\n' + sep + '\n' + '\n'.join(body) + '\n'


# =====================================================================
#  5. 详细评分报告
# =====================================================================

def detailed_score_report(y_true, y_pred, well_ids_test, output_dir):
    """输出全方位评分报告"""
    report = {}

    report['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    report['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    report['micro_f1'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))

    per_class = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    class_names = ['0-无', '1-增斜', '2-稳斜', '3-降斜']
    report['per_class'] = {}
    for i, name in enumerate(class_names):
        report['per_class'][name] = {
            'precision': float(per_class[0][i]),
            'recall': float(per_class[1][i]),
            'f1': float(per_class[2][i]),
            'support': int(per_class[3][i]),
        }

    tol_f1 = macro_f1_with_tolerance(y_true, y_pred, well_ids_test,
                                     tolerance_first=Config.TOLERANCE_FIRST,
                                     tolerance_rest=Config.TOLERANCE_REST)
    report['tolerance_macro_f1'] = float(tol_f1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    report['confusion_matrix'] = cm.tolist()

    print(f"\n{'='*60}")
    print(f"  评 分 报 告")
    print(f"{'='*60}")
    print(f"  竞赛指标（容错 Macro F1）: {tol_f1:.4f}")
    print(f"  标准 Macro F1            : {report['macro_f1']:.4f}")
    print(f"  标准 Weighted F1         : {report['weighted_f1']:.4f}")
    print(f"  标准 Micro F1            : {report['micro_f1']:.4f}")
    print(f"{'-'*60}")
    print(f"  {'类别':>10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}  {'Support':>8}")
    print(f"{'-'*60}")
    for name, metrics in report['per_class'].items():
        print(f"  {name:>10}  {metrics['precision']:>10.4f}  {metrics['recall']:>10.4f}  {metrics['f1']:>10.4f}  {metrics['support']:>8d}")
    print(f"{'-'*60}")
    print(f"\n  混淆矩阵 (行=真实, 列=预测):")
    print(f"  {'':>8}  {'0-无':>6}  {'1-增':>6}  {'2-稳':>6}  {'3-降':>6}")
    for i, row_name in enumerate(class_names):
        row = f"  {row_name:>8}"
        for j in range(4):
            row += f"  {cm[i][j]:>5d}"
        print(row)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  评分报告已保存: {report_path}")

    return report


# =====================================================================
#  6. MD 报告生成
# =====================================================================

def generate_md_report(model_names, per_model_scores, ml_scores, full_scores,
                        assets, y_true, y_pred_full, well_ids_test, test_features,
                        output_dir):
    """生成完整的三表 MD 报告"""
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# 钻井轨迹造斜关键点识别 — 评估报告\n")
    lines.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ===== 表1：各模型容错测试集分数 =====
    lines.append("---\n")
    lines.append("## 表1：容错测试集评分\n\n")
    headers_table1 = ['模型', '容错 Macro F1', '标准 Macro F1', 'Weighted F1']
    align1 = ['---', '---:', '---:', '---:']
    rows1 = []
    for m_key, m_name in model_names.items():
        sc = per_model_scores[m_key]
        rows1.append({
            '模型': m_name,
            '容错 Macro F1': sc['tol_f1'],
            '标准 Macro F1': sc['std_f1'],
            'Weighted F1': sc['w_f1'],
        })

    # 添加混合模型行
    rows1.append({
        '模型': '混合ML (XGB+LGB+Cat)',
        '容错 Macro F1': ml_scores['tol_f1'],
        '标准 Macro F1': ml_scores['std_f1'],
        'Weighted F1': ml_scores['w_f1'],
    })
    rows1.append({
        '模型': '混合深度 (V3)',
        '容错 Macro F1': full_scores['dl_hybrid_tol'],
        '标准 Macro F1': full_scores['dl_hybrid_std'],
        'Weighted F1': full_scores['dl_hybrid_w'],
    })
    rows1.append({
        '模型': '大混合 (全模型)',
        '容错 Macro F1': full_scores['tol_f1'],
        '标准 Macro F1': full_scores['std_f1'],
        'Weighted F1': full_scores['w_f1'],
    })

    lines.append(format_md_table(rows1, headers_table1, align1))
    lines.append("\n> **说明**：容错 Macro F1 为竞赛官方指标，对第一个增斜点容错 ±1，其余关键点容错 ±2。\n")

    # ===== 表2：显著性检验 =====
    lines.append("---\n")
    lines.append("## 表2：配对 t 检验显著性结果\n\n")

    all_model_keys = list(model_names.keys())
    # 为每个模型计算每井 F1
    per_well_scores = {}
    for m_key in all_model_keys:
        pred = predict_single_model(assets, test_features, m_key)
        per_well_scores[m_key] = compute_per_well_f1(y_true, pred, well_ids_test)

    # 混合模型的每井 F1
    per_well_scores['ml_ensemble'] = compute_per_well_f1(
        y_true, ml_scores['pred'], well_ids_test)
    per_well_scores['dl_hybrid'] = compute_per_well_f1(
        y_true, full_scores['dl_hybrid_pred'], well_ids_test)
    per_well_scores['big_hybrid'] = compute_per_well_f1(
        y_true, y_pred_full, well_ids_test)

    compare_pairs = [
        ('big_hybrid', 'xgb', '大混合 vs XGBoost'),
        ('big_hybrid', 'lgb', '大混合 vs LightGBM'),
        ('big_hybrid', 'cat', '大混合 vs CatBoost'),
        ('big_hybrid', 'ml_ensemble', '大混合 vs 混合ML'),
        ('big_hybrid', 'dl_hybrid', '大混合 vs 混合深度'),
        ('ml_ensemble', 'dl_hybrid', '混合ML vs 混合深度'),
    ]

    headers_t2 = ['对比组', 'A均值', 'B均值', '差异', 't统计量', 'p值', '井数', '显著性']
    align2 = ['---', '---:', '---:', '---:', '---:', '---:', '---:', '---']
    rows2 = []

    for a_key, b_key, label in compare_pairs:
        if a_key not in per_well_scores or b_key not in per_well_scores:
            continue
        t_stat, p_val, n, mean_a, mean_b = paired_ttest(
            per_well_scores[a_key], per_well_scores[b_key], a_key, b_key)
        if n < 2:
            continue
        diff = mean_a - mean_b
        sig = '✅ 显著' if p_val < 0.05 else '❌ 不显著'
        rows2.append({
            '对比组': label,
            'A均值': mean_a,
            'B均值': mean_b,
            '差异': diff,
            't统计量': t_stat,
            'p值': p_val,
            '井数': n,
            '显著性': sig,
        })

    lines.append(format_md_table(rows2, headers_t2, align2))
    lines.append("\n> **说明**：配对 t 检验基于每井 Macro F1 得分。p < 0.05 表示差异具有统计学显著性。\n")

    # ===== 表3：物理容错偏移量分析 =====
    lines.append("---\n")
    lines.append("## 表3：物理容错偏移量分析（大混合模型）\n\n")

    offset_results = analyze_tolerance_offsets(y_true, y_pred_full, well_ids_test,
                                                Config.TOLERANCE_FIRST, Config.TOLERANCE_REST)
    offset_rows = offset_summary_rows(offset_results)

    headers_t3 = ['关键点类型', '真实数', '检测数', '平均偏移', '中位偏移',
                  '标准差', '容错内%', '≤1偏移%', '≤2偏移%', '≤5偏移%', '≤10偏移%']
    align3 = ['---', '---:', '---:', '---:', '---:', '---:',
              '---:', '---:', '---:', '---:', '---:']
    rows3 = []
    for r in offset_rows:
        rows3.append({
            '关键点类型': r['name'],
            '真实数': r['n_true'],
            '检测数': r['n_det'],
            '平均偏移': f'{r["avg_off"]:.2f}',
            '中位偏移': f'{r["med_off"]:.1f}',
            '标准差': f'{r["std_off"]:.2f}',
            '容错内%': f'{r["within_pct"]:.1f}%',
            '≤1偏移%': f'{r["le1_pct"]:.1f}%',
            '≤2偏移%': f'{r["le2_pct"]:.1f}%',
            '≤5偏移%': f'{r["le5_pct"]:.1f}%',
            '≤10偏移%': f'{r["le10_pct"]:.1f}%',
        })
    lines.append(format_md_table(rows3, headers_t3, align3))
    lines.append("\n> **说明**：偏移量为真实关键点位置与预测关键点位置的绝对差值（测点数）。\n")

    # ===== 融合权重 =====
    lines.append("---\n")
    lines.append("## 附：模型融合权重\n\n")
    lines.append(f"| 模型 | 融合权重 | 验证 F1 |\n")
    lines.append(f"| --- | ---:| ---:|\n")
    for name, w in sorted(assets['final_weights'].items(), key=lambda x: -x[1]):
        vs = assets['val_scores'].get(name, 0)
        lines.append(f"| {name} | {w:.4f} | {vs:.4f} |\n")

    md_content = ''.join(lines)
    md_path = os.path.join(output_dir, 'evaluation_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"\n  MD 报告已保存: {md_path}")
    return md_path


# =====================================================================
#  7. 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='钻井轨迹造斜关键点识别 - 评估脚本 V3')
    parser.add_argument('--test_ratio', type=float, default=0.15,
                        help='测试井比例 (默认: 0.15)')
    parser.add_argument('--output', type=str, default='eval_results',
                        help='输出目录 (默认: eval_results)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式：减少树模型轮数和DL训练轮数')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='train.csv 路径 (默认: ../data/train.csv)')
    parser.add_argument('--tuned', type=str, default=None,
                        help='使用调参结果 (optuna/ray)，从 tune_results/<name>_best_params.json 加载')
    parser.add_argument('--v4', action='store_true',
                        help='实验模式：额外训练两阶段DL + DP v2（K-fold已默认启用）')
    args = parser.parse_args()

    t_start = time.time()

    # ---- 加载数据 ----
    if args.data_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        args.data_path = os.path.join(base, '..', 'data', 'train.csv')

    print(f"加载数据: {args.data_path}")
    df = pd.read_csv(args.data_path)
    df.replace('N/A', np.nan, inplace=True)
    for col in ['XJS', 'JX', 'FW', 'LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    print(f"  总行数: {len(df)}, 井数: {df['转换后JH'].nunique()}")

    # ---- 划分井 ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_wells, test_wells = stratified_well_split(df, args.test_ratio, args.seed)
    train_df = df[df['转换后JH'].isin(train_wells)].copy()
    test_df  = df[df['转换后JH'].isin(test_wells)].copy()

    print(f"训练集: {len(train_df)} 行, {len(train_wells)} 口井")
    print(f"测试集: {len(test_df)} 行, {len(test_wells)} 口井")
    print(f"训练集关键点分布:\n{train_df['关键点'].value_counts().sort_index()}")
    print(f"测试集关键点分布:\n{test_df['关键点'].value_counts().sort_index()}")

    # ---- 快速模式 ----
    hyperparams = {}
    if args.quick:
        hyperparams.update({
            'tree_n_estimators': 500,
            'dl_epochs': 100,
        })
        print(f"\n⚡ 快速模式: tree_n_estimators={hyperparams['tree_n_estimators']}, "
              f"dl_epochs={hyperparams['dl_epochs']}")

    # ---- 加载调参结果 ----
    if args.tuned:
        tune_path = f'tune_results/{args.tuned}_best_params.json'
        if os.path.exists(tune_path):
            with open(tune_path) as f:
                tune_data = json.load(f)
            tune_params = tune_data.get('hyperparams', tune_data)
            tune_params = {k: v for k, v in tune_params.items() if k != 'best_ensemble_f1'}
            hyperparams.update(tune_params)
            print(f"\n🎯 加载调参结果: {tune_path}")
            print(f"   验证集 F1 (调参时): {tune_data.get('best_ensemble_f1', 'N/A')}")
        else:
            print(f"\n⚠  未找到调参文件: {tune_path}，使用默认参数")

    # ---- 训练 ----
    print(f"\n{'='*60}")
    print(f"  开始训练")
    print(f"{'='*60}")

    dl_types = ['hybrid_v3', 'lstm_only', 'transformer_only']

    assets = train_pipeline(train_df, hyperparams, dl_model_types=dl_types,
                            v4_mode=args.v4)

    # ---- 测试集特征工程 ----
    print("\n[6] 测试集特征工程 ...")
    test_labels_lookup = test_df[['id', '关键点']].copy()
    test_features = create_advanced_features_v2(test_df, is_train=False)

    for col in assets['feature_cols']:
        if col not in test_features.columns:
            test_features[col] = 0

    test_features = test_features.merge(test_labels_lookup, on='id', how='left')
    if 'label' not in test_features.columns and '关键点' in test_features.columns:
        test_features['label'] = test_features['关键点'].fillna(0).astype(int)

    well_ids_test = test_features['well_id'].values
    y_true = test_features['label'].values.astype(int)

    # ============ 6 个独立模型命名 ============
    model_names = {
        'xgb': 'XGBoost',
        'lgb': 'LightGBM',
        'cat': 'CatBoost',
        'hybrid_v3': 'V3混合 (LSTM+Transformer)',
        'lstm_only': 'LSTM-Only',
        'transformer_only': 'Transformer-Only',
    }

    dl_model_types = assets.get('dl_model_types', [])
    available_models = {k: v for k, v in model_names.items()
                        if k in assets['tree_models'] or k in dl_model_types}

    # ============ 表1：6 个独立模型 + 3 种混合 ============
    print(f"\n{'='*75}")
    print(f"  📊 表1：6 个独立模型 + 3 种混合容错评分")
    print(f"{'='*75}")
    print(f"  {'模型':>32}  {'容错MacroF1':>14}  {'标准MacroF1':>14}  {'WeightedF1':>14}")
    print(f"{'-'*75}")

    per_model_scores = {}
    for m_key, m_name in available_models.items():
        pred = predict_single_model(assets, test_features, m_key)
        tol_f1 = macro_f1_with_tolerance(y_true, pred, well_ids_test)
        std_f1 = f1_score(y_true, pred, average='macro', zero_division=0)
        w_f1   = f1_score(y_true, pred, average='weighted', zero_division=0)
        per_model_scores[m_key] = {'tol_f1': tol_f1, 'std_f1': std_f1, 'w_f1': w_f1}
        print(f"  {m_name:>32}  {tol_f1:>13.4f}   {std_f1:>13.4f}   {w_f1:>13.4f}")

    # ============ 混合 ML 模型 ============
    print(f"\n{'─'*75}")
    print(f"  📊 混合ML模型 (XGBoost + LightGBM + CatBoost)")
    print(f"{'─'*75}")

    pred_ml = predict_ml_only(assets, test_features)
    ml_tol_f1 = macro_f1_with_tolerance(y_true, pred_ml, well_ids_test)
    ml_std_f1 = f1_score(y_true, pred_ml, average='macro', zero_division=0)
    ml_w_f1   = f1_score(y_true, pred_ml, average='weighted', zero_division=0)
    print(f"  {'混合ML-不带DP':>32}  {ml_tol_f1:>13.4f}")
    pred_ml_dp = predict_ml_only_dp(assets, test_features)
    ml_dp_tol = macro_f1_with_tolerance(y_true, pred_ml_dp, well_ids_test)
    print(f"  {'混合ML-带DP':>32}  {ml_dp_tol:>13.4f}")

    ml_scores = {
        'tol_f1': ml_tol_f1, 'std_f1': ml_std_f1, 'w_f1': ml_w_f1,
        'pred': pred_ml, 'pred_dp': pred_ml_dp,
    }

    # ============ 混合深度模型 (V3) ============
    print(f"\n{'─'*75}")
    print(f"  📊 混合深度模型 (V3: Conv+LSTM+Transformer)")
    print(f"{'─'*75}")

    dl_hybrid_pred = predict_single_model(assets, test_features, 'hybrid_v3')
    dl_tol = macro_f1_with_tolerance(y_true, dl_hybrid_pred, well_ids_test)
    dl_std = f1_score(y_true, dl_hybrid_pred, average='macro', zero_division=0)
    dl_w   = f1_score(y_true, dl_hybrid_pred, average='weighted', zero_division=0)
    print(f"  {'混合深度-V3':>32}  {dl_tol:>13.4f}   {dl_std:>13.4f}   {dl_w:>13.4f}")

    # ============ 最终大混合模型 ============
    print(f"\n{'─'*75}")
    print(f"  📊 最终大混合模型 (K-fold 树集成 + DP)")
    print(f"{'─'*75}")
    # 大混合使用 predict_on_test, 优先用 K-fold
    pred_full_dp = predict_on_test(assets, test_features)
    full_dp_tol = macro_f1_with_tolerance(y_true, pred_full_dp, well_ids_test)
    print(f"  {'大混合-带DP(K-fold)':>32}  {full_dp_tol:>13.4f}")
    if assets.get('kfold_val_f1'):
        print(f"  {'K-fold CV平均F1':>32}  {np.mean(assets['kfold_val_f1']):>13.4f}")

    # 不考虑权重的大混合（等权重 XGB + LGB）
    pred_full = predict_weighted_ensemble(assets, test_features)
    full_tol_f1 = macro_f1_with_tolerance(y_true, pred_full, well_ids_test)
    full_std_f1 = f1_score(y_true, pred_full, average='macro', zero_division=0)
    full_w_f1   = f1_score(y_true, pred_full, average='weighted', zero_division=0)
    print(f"  {'大混合-不带DP(加权)':>32}  {full_tol_f1:>13.4f}")

    full_scores = {
        'tol_f1': full_tol_f1, 'std_f1': full_std_f1, 'w_f1': full_w_f1,
        'dl_hybrid_tol': dl_tol, 'dl_hybrid_std': dl_std, 'dl_hybrid_w': dl_w,
        'dl_hybrid_pred': dl_hybrid_pred,
        'pred': pred_full, 'pred_dp': pred_full_dp,
    }

    # ============ V4 信息（仅 --v4 时启用） ============
    v4_tol_f1 = None
    v4_pred = None
    if args.v4:
        print(f"\n{'─'*75}")
        print(f"  📊 V4 实验")
        print(f"{'─'*75}")
        # DP 网格搜索信息
        best_dp_params = assets.get('best_dp_params')
        dp_grid_score = assets.get('dp_grid_score')
        if best_dp_params:
            print(f"  DP 网格搜索最佳参数: {best_dp_params}")
            print(f"  DP 网格搜索验证 F1: {dp_grid_score:.4f}")

        # 两阶段 DL
        if assets.get('two_stage_pipeline'):
            v4_pred = v4_predict_on_test(
                assets, test_features,
                two_stage_pipeline=assets['two_stage_pipeline'],
                best_dp_params=best_dp_params
            )
            v4_tol_f1 = macro_f1_with_tolerance(y_true, v4_pred, well_ids_test)
            print(f"  {'两阶段DL+DP':>32}  {v4_tol_f1:>13.4f}")
            if assets.get('two_stage_val_f1'):
                print(f"  {'两阶段DL验证F1':>32}  {assets['two_stage_val_f1']:>13.4f}")

    # ============ 对比汇总 ============
    print(f"\n{'='*75}")
    print(f"  ⇨ 对比汇总")
    print(f"{'='*75}")
    print(f"  {'方案':>32}  {'容错MacroF1':>14}")
    print(f"{'-'*50}")
    for m_key, m_name in available_models.items():
        print(f"  {m_name:>32}  {per_model_scores[m_key]['tol_f1']:>13.4f}")
    print(f"  {'混合ML模型(ML-only)':>32}  {ml_tol_f1:>13.4f}")
    print(f"  {'混合深度(V3)':>32}  {dl_tol:>13.4f}")
    print(f"  {'K-fold+DP(大混合)':>32}  {full_dp_tol:>13.4f}")
    if v4_tol_f1 is not None:
        print(f"  {'两阶段DL实验':>32}  {v4_tol_f1:>13.4f}")
        print(f"  {'两阶段DL vs 大混合':>32}  {v4_tol_f1 - full_dp_tol:>+13.4f}")
    print(f"  {'大混合(带DP) vs 混合ML-DP':>32}  {full_dp_tol - ml_dp_tol:>+13.4f}")
    print(f"  {'大混合(带DP) vs 混合ML 差异':>32}  {full_dp_tol - ml_tol_f1:>+13.4f}")
    print(f"  {'大混合(带DP) vs 混合深度':>32}  {full_dp_tol - dl_tol:>+13.4f}")
    print(f"{'='*75}")

    # ============ 验证要求 ============
    print(f"\n{'='*50}")
    best_individual = max(per_model_scores.values(), key=lambda x: x['tol_f1'])
    print(f"  最佳独立模型: {best_individual['tol_f1']:.4f}")
    print(f"  混合ML(DP):   {ml_dp_tol:.4f}")
    print(f"  混合深度(V3): {dl_tol:.4f}")
    print(f"  大混合(K-fold+DP): {full_dp_tol:.4f}")

    checks = []
    checks.append(("大混合(DP) > 最佳独立", full_dp_tol >= best_individual['tol_f1']))
    checks.append(("大混合(DP) > 混合ML(DP)", full_dp_tol >= ml_dp_tol))
    checks.append(("大混合 > 混合深度", full_tol_f1 >= dl_tol))
    # V3 混合深度应强于 LSTM-Only 和 Transformer-Only
    if 'lstm_only' in per_model_scores and 'transformer_only' in per_model_scores:
        checks.append(("V3混合 > LSTM-Only", dl_tol >= per_model_scores['lstm_only']['tol_f1']))
        checks.append(("V3混合 > Transformer-Only", dl_tol >= per_model_scores['transformer_only']['tol_f1']))

    all_pass = True
    for desc, ok in checks:
        mark = "✅" if ok else "❌"
        all_pass = all_pass and ok
        print(f"  {mark} {desc}")

    if all_pass:
        print(f"\n  🎉 所有要求满足！大混合模型全面超越其他方案！")
    else:
        print(f"\n  ⚠  部分要求未满足，检查详细结果。")

    print(f"{'='*50}")

    # ---- 生成 MD 报告 ----
    md_path = generate_md_report(
        available_models, per_model_scores, ml_scores, full_scores,
        assets, y_true, pred_full_dp, well_ids_test, test_features,
        args.output
    )

    # ---- 保存 JSON ----
    report = {
        'per_model': {available_models[k]: v for k, v in per_model_scores.items()},
        'ml_only': {'tol_f1': ml_tol_f1, 'std_f1': ml_std_f1, 'w_f1': ml_w_f1},
        'dl_hybrid': {'tol_f1': dl_tol, 'std_f1': dl_std, 'w_f1': dl_w},
        'full_hybrid': {'tol_f1': full_tol_f1, 'std_f1': full_std_f1, 'w_f1': full_w_f1,
                        'dp_tol_f1': full_dp_tol},
        'v4': {'tol_f1': v4_tol_f1} if v4_tol_f1 is not None else None,
        'val_scores': assets['val_scores'],
        'final_weights': assets['final_weights'],
        'kfold_weights': assets.get('kfold_weights'),
        'test_wells': list(test_wells),
        'test_rows': len(test_features),
    }
    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, 'evaluation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  完整报告已保存: {report_path}")

    # ---- 保存预测结果 ----
    submission = pd.DataFrame({
        'id': test_features['id'],
        'well_id': test_features['well_id'],
        '真实关键点': y_true,
        '大混合预测': pred_full_dp,
    })
    if v4_pred is not None:
        submission['V4预测'] = v4_pred
    sub_path = os.path.join(args.output, 'predictions.csv')
    submission.to_csv(sub_path, index=False, encoding='utf-8-sig')
    print(f"  预测结果已保存: {sub_path}")

    # ---- 复制 MD 到 docs/ ----
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    docs_md_path = os.path.join(docs_dir, 'evaluation_report.md')
    with open(docs_md_path, 'w', encoding='utf-8') as f:
        f.write(open(md_path, encoding='utf-8').read())
    print(f"  MD 报告已复制到: {docs_md_path}")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"  评估完成！总耗时: {elapsed:.1f} 秒 ({elapsed/60:.1f} 分钟)")
    print(f"{'='*60}")

    return full_tol_f1


if __name__ == '__main__':
    main()
