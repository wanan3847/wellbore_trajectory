#!/usr/bin/env python3
"""
================================================================================
  Ray Tune 自动调参脚本 — 用 ASHA 调度器加速搜索
================================================================================

  功能:
    1. 使用 Ray Tune + ASHA (Async Successive Halving Algorithm) 调度器
    2. 搜索树模型 (XGBoost / LightGBM / CatBoost) 的最佳超参数
    3. 自动剪枝低效 Trial，节省时间
    4. 训练结束输出最佳参数配置

  用法:
    python src/tune_ray.py                            # 默认 50 次 trial
    python src/tune_ray.py --n_trials 100             # 更多搜索
    python src/tune_ray.py --quick                    # 快速验证（10次）
    python src/tune_ray.py --cpus_per_trial 2         # 每个 trial 用 2 核
    python src/tune_ray.py --no_local_mode            # 启用 Ray 分布式模式

  说明:
    - 特征工程只执行一次，缓存到 tune_cache/ 目录
    - 每个 trial 从缓存加载数据，重训练模型
    - DL Transformer 模型不包含在搜索中（太慢），单独训练
    - ASHA 调度器会自动停止效果差的 trial，加速搜索
    - 默认 local_mode=True（单进程调试），加 --no_local_mode 启用并行
"""

import os
import sys
import json
import time
import argparse
import warnings
import tempfile
import itertools
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score

import torch
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2 import (
    Config, create_advanced_features_v2, sample_hard_negatives,
    macro_f1_with_tolerance
)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# macOS 多 OpenMP 库兼容（XGBoost/LightGBM/CatBoost 各有自己的 libomp）
# 必须 OMP_NUM_THREADS=1 + n_jobs=1，否则 pthread_mutex_init 崩给你看
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# =====================================================================
#  1. 数据缓存（只执行一次，所有 Ray Tune trials 共享）
# =====================================================================

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tune_cache')


def preprocess_and_cache(data_path=None):
    """
    执行特征工程并缓存到磁盘。
    返回缓存文件路径。
    """
    t0 = time.time()
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, 'preprocessed_data.npz')
    meta_path  = os.path.join(CACHE_DIR, 'metadata.json')

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        print(f"  从缓存加载预处理数据: {cache_path}")
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  缓存信息: {meta['n_samples']} 样本, {meta['n_features']} 特征, "
              f"{meta['n_wells']} 井")
        return cache_path, meta_path

    print(f"{'='*60}")
    print(f"  特征工程预处理（一次性，缓存到 tune_cache/）")
    print(f"{'='*60}")

    if data_path is None:
        base = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(base, '..', 'data', 'train.csv')

    df = pd.read_csv(data_path)
    df.replace('N/A', np.nan, inplace=True)
    for col in ['XJS', 'JX', 'FW', 'LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"  原始数据: {len(df)} 行, {df['转换后JH'].nunique()} 口井")

    # 特征工程
    train_features = create_advanced_features_v2(df, is_train=True)
    print(f"  特征工程完成: {train_features.shape[1]} 列")

    # 负采样（默认参数，后面通过样本权重调整）
    sampled = sample_hard_negatives(
        train_features,
        window=Config.HARD_NEGATIVE_WINDOW,
        ratio=Config.NEGATIVE_SAMPLE_RATIO,
    )
    print(f"  负采样完成: {len(sampled)} 行")

    exclude_cols = ['id', 'well_id', 'label']
    feature_cols = [c for c in sampled.columns if c not in exclude_cols]

    X_raw = np.nan_to_num(sampled[feature_cols].values, nan=0.0)
    y = sampled['label'].values.astype(int)
    well_ids = sampled['well_id'].values
    well_list = np.unique(well_ids)

    # 方差过滤
    selector = VarianceThreshold(threshold=0.001)
    X = selector.fit_transform(X_raw)
    feature_cols_filtered = [f for f, m in zip(feature_cols, selector.get_support()) if m]

    print(f"  方差过滤后: {X.shape[1]} 个特征")
    print(f"  标签分布: 0={np.sum(y==0)}, 1={np.sum(y==1)}, "
          f"2={np.sum(y==2)}, 3={np.sum(y==3)}")

    # 保存
    np.savez_compressed(cache_path, X=X, y=y, well_ids=well_ids)
    metadata = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_wells': len(well_list),
        'well_list': list(well_list),
        'feature_cols': feature_cols_filtered,
        'n_classes': [int((y == i).sum()) for i in range(4)],
    }
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, ensure_ascii=False)

    print(f"  预处理耗时: {time.time()-t0:.1f} 秒")
    print(f"  缓存已保存: {cache_path}")

    return cache_path, meta_path


# =====================================================================
#  2. Ray Tune 训练函数
# =====================================================================

def train_with_config(config, cache_path=None, meta_path=None):
    """
    Ray Tune 的 trainable 函数。
    每个 trial 调用一次，从缓存加载数据 + 用给定超参数训练。
    """
    import ray
    from ray import tune

    # 加载缓存数据
    data = np.load(cache_path)
    X_all = data['X']
    y_all = data['y']
    well_ids_all = data['well_ids']

    with open(meta_path) as f:
        meta = json.load(f)
    well_list = meta['well_list']

    # ============ 超参数 ============
    cw_positive = config.get('cw_positive', 150)
    tree_lr     = config.get('tree_lr', 0.05)
    tree_depth  = config.get('tree_depth', 12)

    xgb_subsample = config.get('xgb_subsample', 1.0)
    xgb_colsample = config.get('xgb_colsample', 1.0)
    xgb_min_child = config.get('xgb_min_child', 1)
    xgb_gamma     = config.get('xgb_gamma', 0)
    xgb_reg_lambda = config.get('xgb_reg_lambda', 1.0)
    xgb_reg_alpha  = config.get('xgb_reg_alpha', 0.0)

    lgb_num_leaves = config.get('lgb_num_leaves', 64)
    lgb_subsample  = config.get('lgb_subsample', 1.0)
    lgb_colsample  = config.get('lgb_colsample', 1.0)
    lgb_min_child_samples = config.get('lgb_min_child_samples', 20)
    lgb_reg_lambda = config.get('lgb_reg_lambda', 0.0)
    lgb_reg_alpha  = config.get('lgb_reg_alpha', 0.0)

    cat_depth       = config.get('cat_depth', 10)
    cat_l2_leaf     = config.get('cat_l2_leaf', 3.0)
    cat_subsample   = config.get('cat_subsample', 1.0)
    cat_border_count = config.get('cat_border_count', 128)

    n_estimators = config.get('n_estimators', 500)

    cw = {0: 1, 1: cw_positive, 2: cw_positive, 3: cw_positive}

    # ============ 划分 ============
    np.random.seed(config.get('trial_seed', 42))
    shuffled = well_list.copy()
    np.random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * Config.TEST_SIZE))
    val_wells = shuffled[:n_val]
    train_wells = shuffled[n_val:]

    train_mask = np.isin(well_ids_all, train_wells)
    val_mask   = np.isin(well_ids_all, val_wells)

    if train_mask.sum() < 100 or val_mask.sum() < 50:
        tune.report(ensemble_f1=0.0, xgb_f1=0.0, lgb_f1=0.0, cat_f1=0.0)
        return

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_mask])
    X_val   = scaler.transform(X_all[val_mask])
    y_train = y_all[train_mask]
    y_val   = y_all[val_mask]
    well_ids_val = well_ids_all[val_mask]

    sample_weights = np.array([cw[int(l)] for l in y_train])

    # ============ XGBoost ============
    try:
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=tree_depth,
            learning_rate=tree_lr,
            subsample=xgb_subsample, colsample_bytree=xgb_colsample,
            min_child_weight=xgb_min_child, gamma=xgb_gamma,
            reg_lambda=xgb_reg_lambda, reg_alpha=xgb_reg_alpha,
            random_state=42, n_jobs=1, eval_metric='mlogloss',
            early_stopping_rounds=30, verbosity=0,
        )
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      sample_weight=sample_weights, verbose=False)
        xgb_pred = xgb_model.predict(X_val)
        xgb_f1 = macro_f1_with_tolerance(y_val, xgb_pred, well_ids_val)
    except Exception as e:
        xgb_f1 = 0.0
        xgb_model = None

    # ============ LightGBM ============
    try:
        lgb_model = lgb.LGBMClassifier(
            n_estimators=n_estimators, max_depth=tree_depth,
            learning_rate=tree_lr,
            num_leaves=lgb_num_leaves,
            subsample=lgb_subsample, colsample_bytree=lgb_colsample,
            min_child_samples=lgb_min_child_samples,
            reg_lambda=lgb_reg_lambda, reg_alpha=lgb_reg_alpha,
            class_weight=cw,
            random_state=42, n_jobs=1, verbose=-1,
        )
        lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
        lgb_pred = lgb_model.predict(X_val)
        lgb_f1 = macro_f1_with_tolerance(y_val, lgb_pred, well_ids_val)
    except Exception as e:
        lgb_f1 = 0.0
        lgb_model = None

    # ============ CatBoost ============
    try:
        cat_model = cb.CatBoostClassifier(
            iterations=n_estimators, depth=cat_depth,
            learning_rate=tree_lr,
            l2_leaf_reg=cat_l2_leaf, subsample=cat_subsample, bootstrap_type='Bernoulli',
            border_count=cat_border_count,
            class_weights=cw,
            random_state=42, verbose=False, early_stopping_rounds=30,
        )
        cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        cat_pred = cat_model.predict(X_val).flatten().astype(int)
        cat_f1 = macro_f1_with_tolerance(y_val, cat_pred, well_ids_val)
    except Exception as e:
        cat_f1 = 0.0
        cat_model = None

    # ============ 加权集成 ============
    scores = {'xgb': xgb_f1, 'lgb': lgb_f1, 'cat': cat_f1}
    weights = {}
    total_w = 0
    for k, v in scores.items():
        w = v ** 2
        weights[k] = w
        total_w += w

    if total_w > 0:
        ensemble_proba = np.zeros((len(X_val), 4))
        if xgb_f1 > 0 and xgb_model is not None:
            ensemble_proba += xgb_model.predict_proba(X_val) * weights['xgb']
        if lgb_f1 > 0 and lgb_model is not None:
            ensemble_proba += lgb_model.predict_proba(X_val) * weights['lgb']
        if cat_f1 > 0 and cat_model is not None:
            ensemble_proba += cat_model.predict_proba(X_val) * weights['cat']
        ensemble_proba /= total_w
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_f1 = macro_f1_with_tolerance(y_val, ensemble_pred, well_ids_val)
    else:
        ensemble_f1 = 0.0

    # 报告结果给 Ray Tune
    tune.report(
        ensemble_f1=float(ensemble_f1),
        xgb_f1=float(xgb_f1),
        lgb_f1=float(lgb_f1),
        cat_f1=float(cat_f1),
        n_train=int(train_mask.sum()),
        n_val=int(val_mask.sum()),
    )


# =====================================================================
#  3. 搜索空间定义
# =====================================================================

def get_search_space():
    """返回 Ray Tune 的搜索空间。"""
    from ray import tune

    return {
        # 类别权重
        'cw_positive': tune.randint(50, 300),

        # 树模型通用
        'tree_lr': tune.loguniform(0.01, 0.3),
        'tree_depth': tune.randint(6, 20),
        'n_estimators': 500,  # 固定，由 early stopping 控制

        # XGBoost
        'xgb_subsample': tune.uniform(0.6, 1.0),
        'xgb_colsample': tune.uniform(0.6, 1.0),
        'xgb_min_child': tune.randint(1, 10),
        'xgb_gamma': tune.uniform(0, 5.0),
        'xgb_reg_lambda': tune.loguniform(0.1, 10.0),
        'xgb_reg_alpha': tune.uniform(0.0, 5.0),

        # LightGBM
        'lgb_num_leaves': tune.randint(16, 128),
        'lgb_subsample': tune.uniform(0.6, 1.0),
        'lgb_colsample': tune.uniform(0.6, 1.0),
        'lgb_min_child_samples': tune.randint(10, 100),
        'lgb_reg_lambda': tune.uniform(0.0, 10.0),
        'lgb_reg_alpha': tune.uniform(0.0, 10.0),

        # CatBoost
        'cat_depth': tune.randint(6, 14),
        'cat_l2_leaf': tune.loguniform(1.0, 10.0),
        'cat_subsample': tune.uniform(0.6, 1.0),
        'cat_border_count': tune.randint(32, 255),

        # Trial 种子
        'trial_seed': tune.randint(0, 10000),
    }


# =====================================================================
#  4. 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Ray Tune 自动调参 - 超级混合模型')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='搜索次数 (默认: 50)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: 10次')
    parser.add_argument('--cpus_per_trial', type=int, default=2,
                        help='每个 trial 的 CPU 数 (默认: 2)')
    parser.add_argument('--gpu_per_trial', type=float, default=0,
                        help='每个 trial 的 GPU 数 (默认: 0)')
    parser.add_argument('--no_local_mode', action='store_true',
                        help='禁用 local_mode（启用 Ray 分布式）')
    parser.add_argument('--storage_path', type=str, default='~/ray_results',
                        help='Ray 结果存储路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--data_path', type=str, default=None,
                        help='train.csv 路径')
    args = parser.parse_args()

    t_global = time.time()

    # ---- 初始化 Ray ----
    import ray
    from ray import tune
    from ray.tune.schedulers import ASHAScheduler

    if not args.no_local_mode:
        ray.init(local_mode=True, ignore_reinit_error=True)
        print("  Ray 运行模式: local_mode (单进程调试)")
    else:
        ray.init(ignore_reinit_error=True)
        print("  Ray 运行模式: 分布式 (多进程并行)")

    # ---- 数据预处理（缓存） ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_path, meta_path = preprocess_and_cache(args.data_path)

    n_trials = 10 if args.quick else args.n_trials

    # ---- ASHA 调度器 ----
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=3,  # 最多 3 个 step（XGB/LGB/CAT）
        grace_period=1,
        reduction_factor=2,
        brackets=1,
    )

    # ---- 搜索空间 ----
    search_space = get_search_space()

    # ---- 构建可调用的训练函数 ----
    from functools import partial
    trainable = partial(
        train_with_config,
        cache_path=cache_path,
        meta_path=meta_path,
    )

    # ---- Tuner ----
    tuner = tune.Tuner(
        trainable,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric='ensemble_f1',
            mode='max',
            scheduler=asha_scheduler,
            num_samples=n_trials,
            max_concurrent_trials=4,
        ),
        run_config=tune.RunConfig(
            name='wellbore_ray_tune',
            storage_path=os.path.expanduser(args.storage_path),
            verbose=1,
        ),
    )

    print(f"\n{'='*60}")
    print(f"  开始 Ray Tune 调参")
    print(f"  Trial 次数: {n_trials}")
    print(f"  CPUS/Trial: {args.cpus_per_trial}")
    print(f"  ASHA 调度器: max_t=3, grace_period=1")
    print(f"{'='*60}\n")

    # ---- 运行 ----
    results = tuner.fit()

    # ============ 输出结果 ============
    elapsed_total = time.time() - t_global

    print(f"\n{'='*60}")
    print(f"  调 参 完 成")
    print(f"{'='*60}")
    print(f"  总耗时: {elapsed_total:.0f} 秒 ({elapsed_total/60:.1f} 分钟)")

    # 获取所有结果
    all_results = results.get_dataframe()
    print(f"  总 Trials: {len(all_results)}")

    if len(all_results) == 0:
        print("  没有完成的 trial，请检查错误日志")
        return

    # 最佳结果
    best_result = results.get_best_result(metric='ensemble_f1', mode='max')
    print(f"  最佳 Ensemble F1: {best_result.metrics['ensemble_f1']:.4f}")

    print(f"\n{'─'*60}")
    print(f"  最 佳 超 参 数")
    print(f"{'─'*60}")
    best_config = best_result.config
    for key, value in best_config.items():
        if key != 'trial_seed':
            print(f"    {key:25s} = {value}")

    print(f"\n{'─'*60}")
    print(f"  各模型得分:")
    print(f"{'─'*60}")
    for key in ['xgb_f1', 'lgb_f1', 'cat_f1', 'ensemble_f1']:
        val = best_result.metrics.get(key, 'N/A')
        print(f"    {key:25s} = {val}")

    # ---- 保存最佳参数 ----
    output_config = {k: v for k, v in best_config.items() if k != 'trial_seed'}
    output_config.update({
        'best_ensemble_f1': float(best_result.metrics['ensemble_f1']),
    })
    os.makedirs('tune_results', exist_ok=True)
    config_path = 'tune_results/ray_best_params.json'
    with open(config_path, 'w') as f:
        json.dump(output_config, f, indent=2)
    print(f"\n  最佳参数已保存: {config_path}")

    # ---- 可直接使用的配置 ----
    print(f"\n{'─'*60}")
    print(f"  可直接使用的配置:")
    print(f"{'─'*60}")
    print(f"params = {{")
    for key, value in output_config.items():
        if key == 'best_ensemble_f1':
            continue
        if isinstance(value, float):
            print(f"    '{key}': {value},")
        else:
            print(f"    '{key}': {value},")
    print(f"}}")

    # ---- Top-5 配置 ----
    print(f"\n{'─'*60}")
    print(f"  Top-5 Trial 配置:")
    print(f"{'─'*60}")
    df = all_results.sort_values('ensemble_f1', ascending=False)
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        print(f"  #{i+1}: ensemble_f1={row.get('ensemble_f1', 0):.4f}")

    # ---- 清理 ----
    ray.shutdown()

    print(f"\n{'='*60}")
    print(f"  调参完成！")
    print(f"  提示: 用 python src/evaluate.py 测试最佳参数效果")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
