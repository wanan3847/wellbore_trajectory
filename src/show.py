import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime

# ==================== 1. 模型架构定义 ====================
class HybridAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, num_classes=4):
        super(HybridAttentionModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim * 2, num_heads=8, dropout=0.1, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), 
            nn.LayerNorm(64), 
            nn.ReLU(), 
            nn.Dropout(0.3), 
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        return self.fc(out)

# ==================== 2. 资源与数据保存逻辑 ====================
def save_results_to_local(result_df):
    save_path = "submission_results.csv"
    if not os.path.exists(save_path):
        result_df.to_csv(save_path, index=False, encoding='utf-8-sig')
    else:
        result_df.to_csv(save_path, mode='a', header=False, index=False, encoding='utf-8-sig')
    return True

@st.cache_resource
def load_all_assets_v3():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pkl_path = os.path.join(current_dir, "models", "drilling_model_full_v3.pkl")

    if not os.path.exists(pkl_path):
        st.error(f"❌ 找不到模型文件：{pkl_path}")
        return None

    assets = joblib.load(pkl_path)
    input_dim = len(assets['feature_cols'])
    dl_model = HybridAttentionModel(input_dim)

    if 'dl_model_state' in assets:
        dl_model.load_state_dict(assets['dl_model_state'])
    else:
        pth_path = os.path.join(current_dir, "..", "models", "best_hybrid_model.pth")
        dl_model.load_state_dict(torch.load(pth_path, map_location='cpu'))

    dl_model.eval()
    assets['transformer'] = dl_model
    return assets

# ==================== 3. 特征工程 ====================
def create_ui_features_v3_local(group, feature_cols_required):
    df = group.sort_values('depth').reset_index(drop=True)
    f = {'XJS': df['depth'].values, 'JX': df['inclination'].values,
         'FW': df['azimuth'].values, 'LJCZJS': df['tvd'].values}

    jx_series = pd.Series(f['JX'])
    jx_diff1 = jx_series.diff().fillna(0)
    f['JX_diff_1'] = jx_diff1.values
    f['JX_diff_2'] = jx_diff1.diff().fillna(0)

    for w in [10, 20, 30]:
        f[f'JX_mean_{w}'] = jx_series.rolling(w, min_periods=1, center=True).mean().values
        f[f'JX_std_{w}'] = jx_series.rolling(w, min_periods=1, center=True).std().fillna(0).values
        f[f'JX_diff_mean_{w}'] = jx_diff1.rolling(w, min_periods=1, center=True).mean().values

    res_df = pd.DataFrame(f)
    for col in feature_cols_required:
        if col not in res_df.columns: res_df[col] = 0
    return res_df[feature_cols_required]

# ==================== 4. 核心推理逻辑 ====================
def run_enhanced_dp_logic(group, probas):
    results = {"增斜点": None, "稳斜点": None, "降斜点": None}
    MIN_GAP = 20

    def get_smart_idx(p_series, offset=0):
        if len(p_series) == 0 or np.all(np.isnan(p_series)): return 0
        peak_idx = np.argmax(p_series)
        window_size = 5
        start = max(0, peak_idx - window_size)
        end = min(len(p_series), peak_idx + window_size + 1)
        weights = p_series[start:end]
        sum_w = np.nansum(weights)
        if sum_w <= 0: return int(peak_idx) + offset
        centroid_idx = np.average(np.arange(start, end), weights=weights)
        return int(round(centroid_idx)) + offset if not np.isnan(centroid_idx) else int(peak_idx) + offset

    probas = np.nan_to_num(probas, nan=0.0)
    idx1 = get_smart_idx(probas[:, 1])
    results["增斜点"] = group.iloc[idx1]

    if idx1 + MIN_GAP < len(probas):
        idx2 = get_smart_idx(probas[idx1 + MIN_GAP:, 2], idx1 + MIN_GAP)
        results["稳斜点"] = group.iloc[idx2]
        if idx2 + MIN_GAP < len(probas):
            idx3 = get_smart_idx(probas[idx2 + MIN_GAP:, 3], idx2 + MIN_GAP)
            results["降斜点"] = group.iloc[idx3]
    return results

# ==================== 5. 主界面布局 ====================
assets = load_all_assets_v3()

with st.sidebar:
    st.header("🛡️ 专家控制中心")
    mode = st.radio("预测模式", ["四模型权重聚合", "单引擎深度查看"], key="mode_v3")

    if mode == "四模型权重聚合":
        st.subheader("模型权重分配")
        w_lgb = st.slider("LGB 权重", 0.0, 1.0, 0.35)
        w_tf = st.slider("Transformer 权重", 0.0, 1.0, 0.30)
        w_xgb = st.slider("XGB 权重", 0.0, 1.0, 0.20)
        w_cat = st.slider("CAT 权重", 0.0, 1.0, 0.15)
        tw = w_lgb + w_tf + w_xgb + w_cat
        ws = {"lgb": w_lgb/tw, "tf": w_tf/tw, "xgb": w_xgb/tw, "cat": w_cat/tw} if tw > 0 else {"lgb":0.4, "tf":0.3, "xgb":0.2, "cat":0.1}
    else:
        sel_engine = st.selectbox("选择引擎", ["LightGBM", "Transformer", "XGBoost", "CatBoost"])

    # 从本地文件加载井号
    val_data_path = os.path.join(os.path.dirname(__file__), "..", "data", "validation_without_label.csv")
    if os.path.exists(val_data_path):
        df_all_val = pd.read_csv(val_data_path)
        well_list = df_all_val['well_id'].unique().tolist()
    else:
        st.error(f"找不到数据文件: {val_data_path}")
        well_list = []
        
    selected_well = st.selectbox("🎯 选择监测井号", well_list)

st.title("🚀 钻井轨迹造斜关键点 AI 推演系统 v3")

if selected_well and assets:
    df_raw = df_all_val[df_all_val['well_id'] == selected_well].sort_values('depth').copy()
    X_ui = create_ui_features_v3_local(df_raw, assets['feature_cols'])
    X_scaled = assets['scaler'].transform(X_ui)

    # 推理
    p_lgb = assets['tree_models']['lgb'].predict_proba(X_scaled)
    p_xgb = assets['tree_models']['xgb'].predict_proba(X_scaled)
    p_cat = assets['tree_models']['cat'].predict_proba(X_scaled)
    with torch.no_grad():
        input_tensor = torch.FloatTensor(X_scaled).unsqueeze(1)
        p_tf = F.softmax(assets['transformer'](input_tensor), dim=1).numpy()

    if mode == "四模型权重聚合":
        final_p = (p_lgb * ws['lgb'] + p_tf * ws['tf'] + p_xgb * ws['xgb'] + p_cat * ws['cat'])
        display_title = f"井号 {selected_well} 复合推演视图"
    else:
        final_p = {"LightGBM": p_lgb, "Transformer": p_tf, "XGBoost": p_xgb, "CatBoost": p_cat}[sel_engine]
        display_title = f"井号 {selected_well} 引擎视图: {sel_engine}"

    kps = run_enhanced_dp_logic(df_raw, final_p)

    # 可视化
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=df_raw['depth'], y=df_raw['inclination'], name="实测井斜角", line=dict(color='#2c3e50', width=3)))

    clrs = {1: 'rgba(46, 204, 113, 0.2)', 2: 'rgba(52, 152, 219, 0.2)', 3: 'rgba(231, 76, 60, 0.2)'}
    for i in [1, 2, 3]:
        fig.add_trace(go.Scatter(x=df_raw['depth'], y=final_p[:, i], name=f"点{i}置信度", fill='tozeroy', line=dict(width=0), fillcolor=clrs[i]), secondary_y=True)

    kp_style = [("增斜点", 'red', 'star'), ("稳斜点", 'blue', 'diamond'), ("降斜点", 'purple', 'triangle-down')]
    for key, color, sym in kp_style:
        if kps[key] is not None:
            fig.add_trace(go.Scatter(x=[kps[key]['depth']], y=[kps[key]['inclination']], mode='markers+text', name=key, text=[key], textposition="top center", marker=dict(color=color, size=15, symbol=sym)))

    fig.update_layout(height=650, title=display_title, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # 结果明细
    with st.expander("🔍 引擎置信度明细", expanded=True):
        for name, idx in [("增斜", 1), ("稳斜", 2), ("降斜", 3)]:
            chart_data = pd.DataFrame({
                "Depth": df_raw['depth'], "LGB": p_lgb[:, idx], "TF": p_tf[:, idx], "XGB": p_xgb[:, idx], "CAT": p_cat[:, idx]
            }).set_index("Depth")
            st.write(f"### 🚩 {name}过程置信度")
            st.line_chart(chart_data)

    # 导出
    st.subheader("💾 结果同步")
    config_str = f"聚合" if mode == "四模型权重聚合" else f"单引擎:{sel_engine}"
    submission_df = pd.DataFrame({
        'well_id': [selected_well],
        'build_depth': [kps['增斜点']['depth'] if kps['增斜点'] is not None else None],
        'hold_depth': [kps['稳斜点']['depth'] if kps['稳斜点'] is not None else None],
        'drop_depth': [kps['降斜点']['depth'] if kps['降斜点'] is not None else None],
        'config': [config_str],
        'time': [datetime.datetime.now()]
    })
    
    st.table(submission_df)
    if st.button("📤 保存结果到本地 CSV", type="primary", use_container_width=True):
        save_results_to_local(submission_df)
        st.success("结果已追加至 submission_results.csv")
        st.balloons()