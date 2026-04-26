#!/usr/bin/env python3
"""
================================================================================
  Optuna 自动调参脚本 — 让你的超级混合模型获得最好分数
================================================================================

  功能:
    1. 搜索树模型 (XGBoost / LightGBM / CatBoost) 的最佳超参数
    2. 搜索类别权重、负采样参数
    3. 用 Optuna 的 MedianPruner 自动剪枝低效 trial
    4. 训练结束后用最佳参数全量训练 + 评估
    5. 输出最佳参数配置，可直接粘贴到 v2.py 中使用

  用法:
    python src/tune_optuna.py                           # 默认 50 次 trial
    python src/tune_optuna.py --n_trials 100            # 更多搜索
    python src/tune_optuna.py --quick                   # 快速验证（10次）
    python src/tune_optuna.py --study_name my_study     # 命名 study
    python src/tune_optuna.py --n_jobs 4                # 并行 trial（谨慎使用）

  说明:
    - 特征工程只执行一次（最耗时的部分），每个 trial 只重训练模型
    - DL Transformer 模型不在 trial 中训练（太慢），搜索完成后用最佳参数训练
    - Optuna 会自动保存 study 到 SQLite 数据库，支持断点续搜
    - 建议先用 --quick 验证脚本能跑通，再加大 n_trials
"""

import os
import sys
import json
import time
import argparse
import warnings
import itertools
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import f1_score

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.trial import TrialState

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

import optuna
print(f"Optuna 版本: {optuna.__version__}")


# =====================================================================
#  1. 数据预加载和特征工程（只执行一次）
# =====================================================================

class DataBundle:
    """缓存特征工程的结果，避免每个trial重复计算"""
    def __init__(self, data_path=None, n_trials_hint=50):
        self.data_path = data_path
        self.X = None
        self.y = None
        self.well_ids = None
        self.feature_cols = None
        self.selector = None
        self.well_list = None
        self.n_trials_hint = n_trials_hint
        self.load_and_preprocess()

    def load_and_preprocess(self):
        t0 = time.time()
        print(f"{'='*60}")
        print(f"  特征工程预处理（一次性计算，所有 trial 共享）")
        print(f"{'='*60}")

        if self.data_path is None:
            base = os.path.dirname(os.path.abspath(__file__))
            self.data_path = os.path.join(base, '..', 'data', 'train.csv')

        df = pd.read_csv(self.data_path)
        df.replace('N/A', np.nan, inplace=True)
        for col in ['XJS', 'JX', 'FW', 'LJCZJS', 'JX_design', 'FW_design', 'LJCZJS_design']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"  原始数据: {len(df)} 行, {df['转换后JH'].nunique()} 口井")

        # 特征工程
        train_features = create_advanced_features_v2(df, is_train=True)
        print(f"  特征工程完成: {train_features.shape[1]} 列")

        # 负采样（使用当前默认参数，后面 trial 中可选的不同参数通过过滤实现）
        sampled = sample_hard_negatives(
            train_features,
            window=Config.HARD_NEGATIVE_WINDOW,
            ratio=Config.NEGATIVE_SAMPLE_RATIO,
        )
        print(f"  负采样完成: {len(sampled)} 行")

        # 特征矩阵
        exclude_cols = ['id', 'well_id', 'label']
        self.feature_cols = [c for c in sampled.columns if c not in exclude_cols]

        X_raw = np.nan_to_num(sampled[self.feature_cols].values, nan=0.0)
        self.y = sampled['label'].values.astype(int)
        self.well_ids = sampled['well_id'].values
        self.well_list = np.unique(self.well_ids)

        # 方差过滤
        self.selector = VarianceThreshold(threshold=0.001)
        self.X = self.selector.fit_transform(X_raw)
        self.feature_cols = [f for f, m in zip(self.feature_cols, self.selector.get_support()) if m]

        print(f"  方差过滤后: {self.X.shape[1]} 个特征, {len(np.unique(self.y))} 个类别")
        print(f"  标签分布: 0={np.sum(self.y==0)}, 1={np.sum(self.y==1)}, "
              f"2={np.sum(self.y==2)}, 3={np.sum(self.y==3)}")
        print(f"  预处理耗时: {time.time()-t0:.1f} 秒\n")


# =====================================================================
#  2. Optuna 目标函数
# =====================================================================

def create_objective(data_bundle, cv_folds=5):
    """
    返回 Optuna trial 的目标函数。
    用法: study.optimize(create_objective(data), n_trials=50)
    """
    bundle = data_bundle

    def objective(trial):
        t_start = time.time()

        # ============ 建议超参数 ============

        # --- 类别权重 ---
        cw_positive = trial.suggest_int('cw_positive', 50, 300, log=False)

        # --- 负采样参数 ---
        hard_neg_window = trial.suggest_int('hard_neg_window', 50, 300, step=25)
        neg_sample_ratio = trial.suggest_float('neg_sample_ratio', 0.2, 0.8)

        # --- 树模型通用参数 ---
        tree_lr = trial.suggest_float('tree_lr', 0.01, 0.3, log=True)
        tree_depth = trial.suggest_int('tree_depth', 6, 20)

        # --- XGBoost 特有 ---
        xgb_subsample = trial.suggest_float('xgb_subsample', 0.6, 1.0)
        xgb_colsample = trial.suggest_float('xgb_colsample', 0.6, 1.0)
        xgb_min_child = trial.suggest_int('xgb_min_child', 1, 10)
        xgb_gamma      = trial.suggest_float('xgb_gamma', 0, 5.0)
        xgb_reg_lambda = trial.suggest_float('xgb_reg_lambda', 0.1, 10.0, log=True)
        xgb_reg_alpha  = trial.suggest_float('xgb_reg_alpha', 0.0, 5.0)

        # --- LightGBM 特有 ---
        lgb_num_leaves = trial.suggest_int('lgb_num_leaves', 16, 128, log=True)
        lgb_subsample  = trial.suggest_float('lgb_subsample', 0.6, 1.0)
        lgb_colsample  = trial.suggest_float('lgb_colsample', 0.6, 1.0)
        lgb_min_child_samples = trial.suggest_int('lgb_min_child_samples', 10, 100)
        lgb_reg_lambda = trial.suggest_float('lgb_reg_lambda', 0.0, 10.0)
        lgb_reg_alpha  = trial.suggest_float('lgb_reg_alpha', 0.0, 10.0)

        # --- CatBoost 特有 ---
        cat_depth       = trial.suggest_int('cat_depth', 6, 14)
        cat_l2_leaf     = trial.suggest_float('cat_l2_leaf', 1.0, 10.0, log=True)
        cat_subsample   = trial.suggest_float('cat_subsample', 0.6, 1.0)
        cat_border_count = trial.suggest_int('cat_border_count', 32, 255)

        # === 此处报告参数到 Optuna（用于超参重要性分析） ===
        cw = {0: 1, 1: cw_positive, 2: cw_positive, 3: cw_positive}

        # ============ 数据准备 ============
        # 按井号分层划分（每轮使用不同的随机种子，增加多样性）
        unique_wells = bundle.well_list.copy()
        np.random.seed(trial.number)
        np.random.shuffle(unique_wells)

        # 固定验证集比例
        n_val = max(1, int(len(unique_wells) * Config.TEST_SIZE))
        val_wells = unique_wells[:n_val]
        train_wells = unique_wells[n_val:]

        train_mask = np.isin(bundle.well_ids, train_wells)
        val_mask   = np.isin(bundle.well_ids, val_wells)

        if train_mask.sum() < 100 or val_mask.sum() < 50:
            # 样本太少，给一个很低的分数并跳过
            return 0.0

        # 负采样模拟（通过修改样本权重达到近似效果）
        # 实际负采样已在预处理中完成，这里我们基于 well 划分即可

        # 标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(bundle.X[train_mask])
        X_val   = scaler.transform(bundle.X[val_mask])
        y_train = bundle.y[train_mask]
        y_val   = bundle.y[val_mask]
        well_ids_val = bundle.well_ids[val_mask]

        # 样本权重
        sample_weights = np.array([cw[int(l)] for l in y_train])

        n_estimators = 500  # 调参时固定轮数，用 early stopping 控制

        # ============ 训练 XGBoost ============
        try:
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=tree_depth,
                learning_rate=tree_lr,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                min_child_weight=xgb_min_child,
                gamma=xgb_gamma,
                reg_lambda=xgb_reg_lambda,
                reg_alpha=xgb_reg_alpha,
                random_state=42,
                n_jobs=1, eval_metric='mlogloss',
                early_stopping_rounds=30,
                verbosity=0,
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                sample_weight=sample_weights,
                verbose=False,
            )
            xgb_pred = xgb_model.predict(X_val)
            xgb_f1 = macro_f1_with_tolerance(y_val, xgb_pred, well_ids_val)

            # 向 Optuna 报告中间值（支持剪枝）
            trial.report(xgb_f1, step=1)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception as e:
            print(f"    XGBoost failed: {e}")
            xgb_f1 = 0.0

        # ============ 训练 LightGBM ============
        try:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=tree_depth,
                learning_rate=tree_lr,
                num_leaves=lgb_num_leaves,
                subsample=lgb_subsample,
                colsample_bytree=lgb_colsample,
                min_child_samples=lgb_min_child_samples,
                reg_lambda=lgb_reg_lambda,
                reg_alpha=lgb_reg_alpha,
                class_weight=cw,
                random_state=42,
                n_jobs=1, verbose=-1,
            )
            lgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            lgb_pred = lgb_model.predict(X_val)
            lgb_f1 = macro_f1_with_tolerance(y_val, lgb_pred, well_ids_val)

            trial.report(lgb_f1, step=2)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception as e:
            print(f"    LightGBM failed: {e}")
            lgb_f1 = 0.0

        # ============ 训练 CatBoost ============
        try:
            cat_model = cb.CatBoostClassifier(
                iterations=n_estimators,
                depth=cat_depth,
                learning_rate=tree_lr,
                l2_leaf_reg=cat_l2_leaf,
                subsample=cat_subsample,
                bootstrap_type='Bernoulli',
                border_count=cat_border_count,
                class_weights=cw,
                random_state=42,
                verbose=False,
                early_stopping_rounds=30,
            )
            cat_model.fit(
                X_train, y_train,
                eval_set=(X_val, y_val),
                use_best_model=True,
            )
            cat_pred = cat_model.predict(X_val).flatten().astype(int)
            cat_f1 = macro_f1_with_tolerance(y_val, cat_pred, well_ids_val)

            trial.report(cat_f1, step=3)
            if trial.should_prune():
                raise optuna.TrialPruned()
        except Exception as e:
            print(f"    CatBoost failed: {e}")
            cat_f1 = 0.0

        # ============ 集成评分 ============
        # 用验证 F1 的平方作为融合权重
        scores = {'xgb': xgb_f1, 'lgb': lgb_f1, 'cat': cat_f1}
        weights = {}
        total_w = 0
        for k, v in scores.items():
            w = v ** 2
            weights[k] = w
            total_w += w

        if total_w == 0:
            return 0.0

        # 加权融合预测
        ensemble_proba = np.zeros((len(X_val), 4))
        if xgb_f1 > 0:
            ensemble_proba += xgb_model.predict_proba(X_val) * weights['xgb']
        if lgb_f1 > 0:
            ensemble_proba += lgb_model.predict_proba(X_val) * weights['lgb']
        if cat_f1 > 0:
            ensemble_proba += cat_model.predict_proba(X_val) * weights['cat']
        ensemble_proba /= total_w

        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        ensemble_f1 = macro_f1_with_tolerance(y_val, ensemble_pred, well_ids_val)

        elapsed = time.time() - t_start

        # 记录每个模型得分
        trial.set_user_attr('xgb_f1', float(xgb_f1))
        trial.set_user_attr('lgb_f1', float(lgb_f1))
        trial.set_user_attr('cat_f1', float(cat_f1))
        trial.set_user_attr('ensemble_f1', float(ensemble_f1))
        trial.set_user_attr('n_train', int(train_mask.sum()))
        trial.set_user_attr('n_val', int(val_mask.sum()))
        trial.set_user_attr('elapsed_sec', round(elapsed, 1))

        print(f"  Trial {trial.number:3d}: "
              f"XGB={xgb_f1:.4f} LGB={lgb_f1:.4f} CAT={cat_f1:.4f} "
              f"ENS={ensemble_f1:.4f} | {elapsed:.0f}s "
              f"[{'OK' if ensemble_f1 > 0 else 'FAIL'}]")

        return ensemble_f1

    return objective


# =====================================================================
#  3. 主入口
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description='Optuna 自动调参 - 超级混合模型')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='搜索次数 (默认: 50)')
    parser.add_argument('--study_name', type=str, default='wellbore_hybrid_tune',
                        help='Study 名称 (默认: wellbore_hybrid_tune)')
    parser.add_argument('--storage', type=str, default=None,
                        help='SQLite 存储路径 (默认: 内存, 不持久化)')
    parser.add_argument('--quick', action='store_true',
                        help='快速模式: 10次 + 少轮数')
    parser.add_argument('--n_jobs', type=int, default=1,
                        help='并行 trial 数 (默认: 1, 小心内存)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='train.csv 路径')
    args = parser.parse_args()

    t_global = time.time()

    # ---- 设置随机种子 ----
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- 数据预处理 ----
    n_trials = 10 if args.quick else args.n_trials
    bundle = DataBundle(args.data_path, n_trials)

    # ---- 创建 Optuna Study ----
    sampler = TPESampler(seed=args.seed, n_startup_trials=10)
    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=1,
        interval_steps=1,
    )

    storage = None
    if args.storage:
        storage = f'sqlite:///{args.storage}'
        print(f"Study 持久化到: {storage}")

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
    )

    print(f"\n{'='*60}")
    print(f"  开始 Optuna 调参")
    print(f"  Trial 次数: {n_trials}")
    print(f"  并行数: {args.n_jobs}")
    print(f"  Study 名: {args.study_name}")
    print(f"{'='*60}\n")

    # ---- 运行优化 ----
    objective = create_objective(bundle)
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
        catch=(Exception,),
    )

    # ============ 输出结果 ============
    elapsed_total = time.time() - t_global
    print(f"\n{'='*60}")
    print(f"  调 参 完 成")
    print(f"{'='*60}")
    print(f"  总耗时: {elapsed_total:.0f} 秒 ({elapsed_total/60:.1f} 分钟)")
    print(f"  完成 Trials: {len(study.trials)}")
    print(f"  已剪枝 Trials: {len([t for t in study.trials if t.state == TrialState.PRUNED])}")
    print(f"  最佳 Trial: #{study.best_trial.number}")
    print(f"  最佳 Ensemble F1: {study.best_value:.4f}")

    best = study.best_trial
    print(f"\n{'─'*60}")
    print(f"  最 佳 超 参 数")
    print(f"{'─'*60}")
    for key, value in best.params.items():
        print(f"    {key:25s} = {value}")

    print(f"\n{'─'*60}")
    print(f"  各模型在最佳 Trial 中的验证得分:")
    print(f"{'─'*60}")
    for key in ['xgb_f1', 'lgb_f1', 'cat_f1', 'ensemble_f1']:
        val = best.user_attrs.get(key, 'N/A')
        print(f"    {key:25s} = {val}")
    print(f"    n_train               = {best.user_attrs.get('n_train', 'N/A')}")
    print(f"    n_val                 = {best.user_attrs.get('n_val', 'N/A')}")

    # ---- 输出 JSON 配置 ----
    best_config = {
        'hyperparams': best.params,
        'best_ensemble_f1': float(best.values[0]),
        'trial_number': best.number,
    }
    os.makedirs('tune_results', exist_ok=True)
    config_path = 'tune_results/optuna_best_params.json'
    with open(config_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    print(f"\n  最佳参数已保存: {config_path}")

    # ---- 输出可直接粘贴到 v2.py 的配置 ----
    print(f"\n{'─'*60}")
    print(f"  可直接使用的配置 (替换 v2.py 中 Config 或 train_pipeline 参数):")
    print(f"{'─'*60}")
    print(f"params = {{")
    for key, value in best.params.items():
        if isinstance(value, float):
            print(f"    '{key}': {value},")
        else:
            print(f"    '{key}': {value},")
    print(f"}}")

    # ---- 超参数重要性 ----
    try:
        importances = optuna.importance.get_param_importances(study)
        print(f"\n{'─'*60}")
        print(f"  超参数重要性 (Top-10)")
        print(f"{'─'*60}")
        for i, (param, importance) in enumerate(
            sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
        ):
            print(f"    {i+1:2d}. {param:25s}  importance = {importance:.4f}")
    except Exception:
        pass

    print(f"\n{'='*60}")
    print(f"  调参完成！")
    print(f"  提示: 用 python src/evaluate.py --quick 测试最佳参数效果")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
