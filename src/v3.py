#!/usr/bin/env python3
"""
v3 — 两阶段流水线 + K折集成 + KP3增强 + DP v2
目标：从 0.71 提升至 0.80+ 容错 Macro F1
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tqdm import tqdm
import sys, os, copy, warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v2 import (
    Config, macro_f1_with_tolerance,
    dp_post_process_v2, advanced_post_process_v2,
    advanced_post_process  # 原始 DP (v1)
)
from dl_improved import (
    DrillingWindowDataset, PositionalEncoding
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================== KP3 数据增强 ====================

def augment_kp3(features_df, labels, well_ids, n_shift=3, noise_std=0.01):
    """
    对 KP3 样本来做移位+噪声增强。
    每个真实 KP3 生成 2*n_shift 个副本 ±1..n_shift 偏移 + 噪声。
    Returns: 增强后的 (X_aug, y_aug, well_ids_aug)
    """
    kp3_mask = labels == 3
    if kp3_mask.sum() == 0:
        return features_df.values, labels, well_ids

    X = features_df.values if isinstance(features_df, pd.DataFrame) else features_df
    X_aug_list, y_aug_list, wid_aug_list = [], [], []
    n_features = X.shape[1]

    kp3_indices = np.where(kp3_mask)[0]
    for idx in kp3_indices:
        for shift in range(-n_shift, n_shift + 1):
            if shift == 0:
                continue
            shifted = idx + shift
            if 0 <= shifted < len(X):
                sample = X[shifted].copy().astype(np.float32)
                noise = np.random.randn(n_features).astype(np.float32) * noise_std
                sample = sample + noise
                X_aug_list.append(sample)
                y_aug_list.append(3)
                wid_aug_list.append(well_ids[shifted] if well_ids is not None else 0)

    if len(X_aug_list) == 0:
        return X, labels, well_ids

    X_aug = np.stack(X_aug_list, axis=0)
    y_aug = np.array(y_aug_list, dtype=np.int64)
    wid_aug = np.array(wid_aug_list)

    X_out = np.concatenate([X, X_aug], axis=0)
    y_out = np.concatenate([labels, y_aug], axis=0)
    wid_out = np.concatenate([well_ids, wid_aug], axis=0) if well_ids is not None else None

    return X_out, y_out, wid_out


# ==================== 阶段1：关键点检测（二分类） ====================

class Stage1Detector(nn.Module):
    """
    二分类关键点检测器 — 区分"是否为关键点"（类别1/2/3 vs 0）
    基于窗口输入 [batch, n_features, window_size]
    """
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(
            64, hidden_dim, 2, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        # 全局注意力池化
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2) * 0.02)
        self.scale = (hidden_dim * 2) ** -0.5

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # 二分类：2-class logits
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)

        lstm_out, _ = self.lstm(x)

        # AttentionPooling
        scores = torch.matmul(self.query, lstm_out.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, lstm_out).squeeze(1)

        return self.fc(pooled)  # [batch, 2]


class Stage2Classifier(nn.Module):
    """
    关键点分类器 — 区分 KP1/KP2/KP3/KP3
    只在候选窗口上运行（由 Stage1 筛选后）
    """
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 num_classes=4, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.lstm = nn.LSTM(
            64, hidden_dim, 2, batch_first=True,
            bidirectional=True, dropout=dropout
        )

        # 位置编码 + Transformer
        self.pos_encoder = PositionalEncoding(hidden_dim * 2, dropout=dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim * 2, nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, 2)

        # 注意力池化
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim * 2) * 0.02)
        self.scale = (hidden_dim * 2) ** -0.5

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

        scores = torch.matmul(self.query, tf_out.transpose(1, 2)) * self.scale
        weights = F.softmax(scores, dim=-1)
        pooled = torch.matmul(weights, tf_out).squeeze(1)

        return self.fc(pooled)


# ==================== 两阶段训练流水线 ====================

class TwoStagePipeline:
    """
    两阶段 DL 流水线：
    - 阶段1：二分类关键点检测（WindowDataset, FocalLoss）
    - 阶段2：四分类（KP0/1/2/3）
    - 推理时检测概率 gating 分类概率
    """
    def __init__(self, n_features, window_size=51, hidden_dim=128,
                 lr=0.001, batch_size=256, dropout=0.3):
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.batch_size = batch_size
        self.dropout = dropout
        self.detector = None
        self.classifier = None

    def _make_binary_labels(self, y):
        """4-class → binary (0=non-kp, 1=any kp)"""
        return (np.asarray(y) > 0).astype(np.int64)

    def train(self, X_train, y_train, well_ids_train,
              X_val, y_val, well_ids_val,
              epochs=100, patience=20, verbose=True):
        """训练两阶段模型，返回最佳验证 F1"""
        n_classes = len(np.unique(np.concatenate([y_train, y_val])))

        # === 阶段1：检测器 ===
        if verbose:
            print("    [Stage1] 训练关键点检测器...")
        self.detector = Stage1Detector(
            self.n_features, self.window_size, self.hidden_dim, self.dropout
        ).to(DEVICE)

        y_bin_train = self._make_binary_labels(y_train)
        y_bin_val = self._make_binary_labels(y_val)

        det_cw = torch.FloatTensor([1.0, 30.0]).to(DEVICE)
        det_criterion = nn.CrossEntropyLoss(weight=det_cw)

        det_optim = torch.optim.AdamW(
            self.detector.parameters(), lr=self.lr, weight_decay=1e-4
        )
        det_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            det_optim, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )

        train_ds = DrillingWindowDataset(X_train, y_bin_train, well_ids_train, self.window_size)
        val_ds = DrillingWindowDataset(X_val, y_bin_val, well_ids_val, self.window_size)
        train_ld = DataLoader(train_ds, self.batch_size, shuffle=True)
        val_ld = DataLoader(val_ds, self.batch_size)

        sorted_y_bin_val = val_ds.y
        best_det_f1 = 0.0
        stale = 0

        for ep in range(epochs):
            self.detector.train()
            loss_sum = 0.0
            for bx, by in train_ld:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                det_optim.zero_grad()
                logits = self.detector(bx)
                loss = det_criterion(logits, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.detector.parameters(), 5.0)
                det_optim.step()
                loss_sum += loss.item()

            self.detector.eval()
            preds = []
            with torch.no_grad():
                for bx, _ in val_ld:
                    logits = self.detector(bx.to(DEVICE))
                    preds.extend(torch.argmax(logits, dim=1).cpu().numpy())

            f1 = f1_score(sorted_y_bin_val, np.array(preds), average='binary', zero_division=0)
            det_sched.step(f1)

            if f1 > best_det_f1:
                best_det_f1 = f1
                self._best_detector = copy.deepcopy(self.detector.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        self.detector.load_state_dict(self._best_detector)
        if verbose:
            print(f"    [Stage1] 最佳检测 F1={best_det_f1:.4f}")

        # === 阶段2：分类器 ===
        if verbose:
            print("    [Stage2] 训练关键点分类器...")
        self.classifier = Stage2Classifier(
            self.n_features, self.window_size, self.hidden_dim,
            num_classes=n_classes, dropout=self.dropout
        ).to(DEVICE)

        cls_cw = torch.FloatTensor([1.0, 10.0, 15.0, 30.0][:n_classes]).to(DEVICE)
        cls_criterion = nn.CrossEntropyLoss(weight=cls_cw)

        cls_optim = torch.optim.AdamW(
            self.classifier.parameters(), lr=self.lr, weight_decay=1e-4
        )
        cls_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            cls_optim, mode='max', factor=0.5, patience=5, min_lr=1e-5
        )

        cls_train_ds = DrillingWindowDataset(X_train, y_train, well_ids_train, self.window_size)
        cls_val_ds = DrillingWindowDataset(X_val, y_val, well_ids_val, self.window_size)
        cls_train_ld = DataLoader(cls_train_ds, self.batch_size, shuffle=True)
        cls_val_ld = DataLoader(cls_val_ds, self.batch_size)

        sorted_y_val = cls_val_ds.y
        sorted_well_ids_val = cls_val_ds._well_ids
        best_cls_f1 = 0.0
        stale = 0

        for ep in range(epochs):
            self.classifier.train()
            loss_sum = 0.0
            for bx, by in cls_train_ld:
                bx, by = bx.to(DEVICE), by.to(DEVICE)
                cls_optim.zero_grad()
                loss = cls_criterion(self.classifier(bx), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), 5.0)
                cls_optim.step()
                loss_sum += loss.item()

            self.classifier.eval()
            preds = []
            with torch.no_grad():
                for bx, _ in cls_val_ld:
                    out = self.classifier(bx.to(DEVICE))
                    preds.extend(torch.argmax(out, dim=1).cpu().numpy())

            f1 = macro_f1_with_tolerance(sorted_y_val, np.array(preds), sorted_well_ids_val)
            cls_sched.step(f1)

            if f1 > best_cls_f1:
                best_cls_f1 = f1
                self._best_classifier = copy.deepcopy(self.classifier.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break

        self.classifier.load_state_dict(self._best_classifier)
        if verbose:
            print(f"    [Stage2] 最佳分类 F1={best_cls_f1:.4f}")

        return best_cls_f1

    def predict(self, X_test, well_ids_test):
        """
        推理：检测器 gating 分类器
        Returns: (detection_probas [n], class_probas [n, 4])
        """
        self.detector.eval()
        self.classifier.eval()

        ds = DrillingWindowDataset(X_test, y=None, well_ids=well_ids_test, window_size=self.window_size)
        ld = DataLoader(ds, batch_size=256)

        det_probas = []
        cls_probas = []
        with torch.no_grad():
            for bx in ld:
                data = bx[0] if isinstance(bx, list) else bx
                data = data.to(DEVICE)

                # 检测器输出
                det_logits = self.detector(data)
                det_probs = F.softmax(det_logits.view(-1, 2), dim=1)[:, 1].cpu().numpy()
                det_probas.append(det_probs)

                # 分类器输出
                cls_logits = self.classifier(data)
                cls_probs = F.softmax(cls_logits, dim=1).cpu().numpy()
                cls_probas.append(cls_probs)

        return np.concatenate(det_probas), np.vstack(cls_probas)


# ==================== K折树模型集成 ====================

def kfold_tree_ensemble(X_train, y_train, well_ids_train,
                        tree_params=None, n_folds=5, random_seed=42,
                        verbose=True):
    """
    按井分层的 K 折交叉验证训练树模型。
    Returns:
      cv_models: {'xgb': [model_1, ..., model_k], 'lgb': [...], 'cat': [...]}
      oof_preds: {'xgb': np.array, 'lgb': np.array, 'cat': np.array}
      ensemble_weights: {'xgb': w, 'lgb': w, 'cat': w} (OOF 调优)
      val_f1_by_fold: [f1_1, ..., f1_k]
    """
    if tree_params is None:
        tree_params = {}

    # 按井分组，构建分层标签（基于是否包含 KP3）
    well_df = pd.DataFrame({'well_id': well_ids_train, 'label': y_train})
    well_info = well_df.groupby('well_id')['label'].apply(lambda x: int(3 in x.values)).reset_index()
    well_ids_unique = well_info['well_id'].values
    strat_labels = well_info['label'].values  # 1 if KP3 in well, else 0

    n_wells = len(well_ids_unique)
    n_splits = min(n_folds, n_wells)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)

    # Map well_id → row index in full dataset
    well_to_rows = {}
    for i, wid in enumerate(well_ids_train):
        well_to_rows.setdefault(wid, []).append(i)

    cv_models = {'xgb': [], 'lgb': [], 'cat': []}
    oof_preds = {'xgb': np.zeros(len(X_train)), 'lgb': np.zeros(len(X_train)),
                 'cat': np.zeros(len(X_train))}
    oof_preds_4c = {'xgb': np.zeros((len(X_train), 4)),
                    'lgb': np.zeros((len(X_train), 4)),
                    'cat': np.zeros((len(X_train), 4))}
    val_f1_list = []

    default_xgb = {
        'n_estimators': 400, 'max_depth': 10, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_weight': 5, 'gamma': 2.0,
        'reg_lambda': 2.0, 'reg_alpha': 2.0,
        'eval_metric': 'mlogloss', 'use_label_encoder': False,
        'random_state': random_seed, 'verbosity': 0, 'n_jobs': -1,
        'early_stopping_rounds': 20,
    }
    default_xgb.update(tree_params.get('xgb', {}))

    default_lgb = {
        'n_estimators': 400, 'num_leaves': 31, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'min_child_samples': 20,
        'reg_lambda': 2.0, 'reg_alpha': 1.0,
        'class_weight': {0: 1, 1: 40, 2: 60, 3: 100},
        'random_state': random_seed, 'verbose': -1, 'n_jobs': -1,
    }
    default_lgb.update(tree_params.get('lgb', {}))

    default_cat = {
        'n_estimators': 400, 'depth': 8, 'learning_rate': 0.05,
        'l2_leaf_reg': 3.0,
        'class_weights': {0: 1, 1: 40, 2: 60, 3: 100},
        'random_seed': random_seed, 'verbose': False,
    }
    default_cat.update(tree_params.get('cat', {}))

    # 全局归一化（所有 fold 共用同一个 scaler，保持预测一致性）
    global_scaler = StandardScaler()
    X_train_scaled = global_scaler.fit_transform(X_train)

    for fold, (train_well_idx, val_well_idx) in enumerate(skf.split(well_ids_unique, strat_labels)):
        train_wells = well_ids_unique[train_well_idx]
        val_wells = well_ids_unique[val_well_idx]

        train_rows = np.concatenate([well_to_rows[w] for w in train_wells])
        val_rows = np.concatenate([well_to_rows[w] for w in val_wells])

        X_tr_s = X_train_scaled[train_rows]
        y_tr = y_train[train_rows]
        X_val_s = X_train_scaled[val_rows]
        y_val = y_train[val_rows]
        wid_val = well_ids_train[val_rows]

        # XGBoost
        model_xgb = xgb.XGBClassifier(**default_xgb)
        model_xgb.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)])
        pred_xgb = model_xgb.predict_proba(X_val_s)
        oof_preds['xgb'][val_rows] = np.argmax(pred_xgb, axis=1)
        oof_preds_4c['xgb'][val_rows] = pred_xgb
        cv_models['xgb'].append(model_xgb)

        # LightGBM
        model_lgb = lgb.LGBMClassifier(**default_lgb)
        model_lgb.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], callbacks=[lgb.early_stopping(20)])
        pred_lgb = model_lgb.predict_proba(X_val_s)
        oof_preds['lgb'][val_rows] = np.argmax(pred_lgb, axis=1)
        oof_preds_4c['lgb'][val_rows] = pred_lgb
        cv_models['lgb'].append(model_lgb)

        # CatBoost
        model_cat = cb.CatBoostClassifier(**default_cat)
        model_cat.fit(X_tr_s, y_tr, eval_set=(X_val_s, y_val), verbose=False)
        pred_cat = model_cat.predict_proba(X_val_s)
        oof_preds['cat'][val_rows] = np.argmax(pred_cat, axis=1)
        oof_preds_4c['cat'][val_rows] = pred_cat
        cv_models['cat'].append(model_cat)

        # Evaluate fold
        fold_ensemble = (pred_xgb + pred_lgb + pred_cat) / 3
        fold_pred = np.argmax(fold_ensemble, axis=1)
        fold_f1 = macro_f1_with_tolerance(y_val, fold_pred, wid_val)
        val_f1_list.append(fold_f1)

        if verbose:
            print(f"    [K-Fold {fold+1}/{n_splits}] "
                  f"Train: {len(train_wells)} wells, Val: {len(val_wells)} wells, "
                  f"ValF1={fold_f1:.4f}")

    # --- OOF weight optimization ---
    oof_preds_proba = oof_preds_4c
    y_train_adjusted = y_train

    # Try different ensemble weight combinations
    def eval_oof_weights(xgb, lgb, cat):
        blended = oof_preds_proba['xgb'] * xgb + oof_preds_proba['lgb'] * lgb + oof_preds_proba['cat'] * cat
        preds = np.argmax(blended, axis=1)
        return macro_f1_with_tolerance(y_train_adjusted, preds, well_ids_train)

    best = {'xgb': 1/3, 'lgb': 1/3, 'cat': 1/3}
    best_f1 = 0

    # Grid search over weights (step 0.1)
    for w1 in np.arange(0, 1.05, 0.1):
        for w2 in np.arange(0, 1.05 - w1, 0.1):
            w3 = round(1 - w1 - w2, 1)
            if w3 < 0:
                continue
            f1_val = eval_oof_weights(w1, w2, w3)
            if f1_val > best_f1:
                best_f1 = f1_val
                best = {'xgb': w1, 'lgb': w2, 'cat': w3}

    # Also try single-model and top-2
    for name in ['xgb', 'lgb', 'cat']:
        w = {n: 0.0 for n in ['xgb', 'lgb', 'cat']}
        w[name] = 1.0
        f1_val = eval_oof_weights(**w)
        if f1_val > best_f1:
            best_f1 = f1_val
            best = w

    if verbose:
        print(f"    [OOF权重] xgb={best['xgb']:.1f} lgb={best['lgb']:.1f} "
              f"cat={best['cat']:.1f} → OOF F1={best_f1:.4f}")
        print(f"    [K-Fold] 平均验证 F1={np.mean(val_f1_list):.4f} ± {np.std(val_f1_list):.4f}")

    return cv_models, oof_preds_4c, best, val_f1_list, global_scaler


def kfold_predict_proba(cv_models, X_test, weights, scaler=None):
    """
    对测试集使用 K 折模型平均预测。
    weights: {'xgb': w, 'lgb': w, 'cat': w} — OOF 调优后的权重
    scaler: 训练时使用的 StandardScaler（来自 kfold_tree_ensemble）
    Return: (blended_proba [n, 4], per_model_probas dict)
    """
    preds = {}
    blended = np.zeros((X_test.shape[0], 4))
    if scaler is not None:
        X_test_s = scaler.transform(X_test)
    else:
        X_test_s = StandardScaler().fit_transform(X_test)

    for name in ['xgb', 'lgb', 'cat']:
        models = cv_models.get(name, [])
        if not models or weights.get(name, 0) == 0:
            preds[name] = np.zeros((X_test.shape[0], 4))
            continue
        # Average predictions across folds
        fold_preds = []
        for model in models:
            # Re-scale test data per fold (or just fit once globally)
            fold_preds.append(model.predict_proba(X_test_s))
        mean_pred = np.mean(fold_preds, axis=0)
        preds[name] = mean_pred
        blended += mean_pred * weights[name]

    return blended, preds


# ==================== v4 预测流水线 ====================

def v4_predict_on_test(assets, test_features, two_stage_pipeline=None):
    """
    v4 流水线：K-fold tree ensemble + 原始 DP
    两阶段 DL 效果太差（F1≈0.25），暂时去掉避免拖累。
    """
    kfold_models = assets.get('kfold_models')
    kfold_weights = assets.get('kfold_weights')
    tree_preds = {}
    dl_preds = {}

    # K-fold tree predictions
    if kfold_models is not None:
        test_feat_array = test_features[assets['feature_cols']].values
        kfold_scaler = assets.get('kfold_scaler')
        blended, per_model = kfold_predict_proba(
            kfold_models, test_feat_array, kfold_weights, scaler=kfold_scaler
        )
        tree_preds = per_model
        tree_proba = blended
        # 使用 K-fold 权重作为 DP 的 weights
        dp_weights = kfold_weights
    else:
        # Fallback to existing tree preds
        scaler = assets.get('scaler')
        test_scaled = scaler.transform(test_features[assets['feature_cols']].values)
        for name in ['xgb', 'lgb', 'cat']:
            if name in assets.get('tree_models', {}):
                proba = assets['tree_models'][name].predict_proba(test_scaled)
                tree_preds[name] = proba
        dp_weights = assets.get('final_weights', {})
        tree_proba = np.zeros((len(test_features), 4))
        for name in ['xgb', 'lgb', 'cat']:
            if name in tree_preds and tree_preds[name] is not None and name in dp_weights:
                tree_proba += tree_preds[name] * dp_weights[name]

    # 使用原始 DP (v1) 后处理 — 已验证能提升 +0.134
    final_pred = advanced_post_process(
        test_features, tree_preds, dl_preds, dp_weights
    )

    return final_pred
