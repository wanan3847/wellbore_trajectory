评估流程解读
该评估脚本对 6 个独立模型 + 3 种混合方案 进行系统评分，按以下流程执行：

1. 数据划分
stratified_well_split 按井号（而非按行）分层抽样，确保测试井和训练井在关键点类别分布上一致。避免同井数据泄露。

2. 训练流水线 train_pipeline
在第 1 阶段用内部验证集（从训练井中再划分）训练所有模型并计算验证集 Score² 权重，第 2 阶段用全量训练井数据 retrain 最终模型。

3. 6 个独立模型评分
主入口 main() 遍历：

模型 key	名称	预测方式
xgb	XGBoost	predict_proba (tree)
lgb	LightGBM	predict_proba (tree)
cat	CatBoost	predict_proba (tree)
hybrid_v3	V3混合(Conv+LSTM+Transformer)	predict_enhanced_proba (DL)
lstm_only	LSTM-Only	predict_enhanced_proba (DL)
transformer_only	Transformer-Only	predict_enhanced_proba (DL)
每个模型输出 3 个指标（detailed_score_report）：

容错 Macro F1 — 竞赛官方指标，首个增斜点容错 ±1，其余±2
标准 Macro F1 — 标准 macro F1
加权 F1 — 按类别加权的 F1
4. 混合模型评分
混合ML (predict_ml_only, L969): 对 XGB+LGB+Cat 三个树模型的预测概率做 Score² 加权平均。

混合深度 V3 (L988): 直接取 hybrid_v3 DL 模型本身的预测（单模型就已是 Conv+LSTM+Transformer 混合结构）。

大混合/全模型 (L999-L1006): 分两个版本：

不带 DP: predict_weighted_ensemble — 所有 6 个模型的概率用验证集 Score² 权重加权平均
带 DP: predict_on_test — 加权融合后，再经 advanced_post_process 做领域规则后处理（如物理约束平滑），这是最终提交版本
5. 权重机制
融合权重在 train_pipeline 中计算：


weights_val = {name: score ** 2 for name, score in val_scores.items()}
Score² 加权让强模型贡献更大、弱模型自然衰减，无需硬阈值排除。

6. 三表统计
表1: 各模型容错测试集评分排名
表2: 配对 t 检验（按井级 Macro F1）检验大混合相比各对比组是否统计显著
表3: 物理容错偏移量分析（各关键点的平均偏移、中位偏移、容错内比例等）
7. 验证断言
L1032-L1058 自动检查：

大混合 > 最佳独立模型
大混合 > 混合ML
大混合 > 混合深度(V3)
V3 混合 > LSTM-Only
V3 混合 > Transformer-Only