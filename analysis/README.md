# analysis — CRISPR 编辑效率影响因素发现与统计分析引擎

> 状态: 已在 DeepCRISPR 1344 组真实批上全链路跑通 (13 个分析任务全部 completed)。
> 引擎对训练系统 (train.py / data_digging.py / core/) **只读**; 训练流程未因分析需求改动。

## 1. 职责
把 1344 组 sgRNA 编辑效率实验结果转化为可交付产物:
Data QC → Prediction/Generalization → Environment Factor → Sequence/Motif →
Statistical Evidence → 报告与图表。

## 2. 术语纪律 (核心红线)
- **Effect**: ΔR²/ΔMAE/ΔRMSE/系数/ISM effect/enrichment effect
- **Importance/Attribution**: SHAP/IG/ISM/Attention/Gain (非 p 值; 本引擎 7 方法白名单)
- **Statistical Evidence**: 仅来自明确检验/CI (regression p/FDR, ANOVA, enrichment p/FDR,
  permutation/bootstrap)
- SNR≥2.5 + 足够效应量 → "Strong attribution", **不称为 statistical significance**;
  对 attribution 做检验必须先建 permutation/bootstrap null, 再 FDR
  连同 `analysis.evidence` / `analysis.cellline` 两个子包一并移出, 不再产出。

## 3. 模块结构
```text
analysis/
├── __init__.py          版本与分层说明
├── config.py            SNR/FDR/consensus/ANOVA/motif/QC 阈值集中配置 (seed 固定可复现)
├── schemas.py           统一实验记录类型 (ExperimentRecord)
├── plans.py             AnalysisPlan / Capabilities / ExecutionPlan (selected/available/status)
├── pipeline.py          编排入口 (python -m analysis.pipeline)
├── collect_results.py   逐 run 指标汇总 → train_data/
├── importance_extraction.py  归因白名单化 → feature_importance/ + key_regulatory_biomarkers.csv
├── prediction.py        model×split 汇总 + LOCO
├── leakage.py           身份类 (canonical/revcomp) 与重叠审计工具
├── data/                loaders 统一实验表 + validation 同 cohort 一致性
├── stats/               multiple_testing(BH-FDR) / bootstrap / hypothesis_tests / tasks
├── environment/         条件增量 / 主效应 / 交互 (同 seed 配对, 跨 seed 永不配对)
├── attribution/         whitelist 字段 → 统一 attribution 表
├── sequence/motif/      motif 提取 / 聚类 / 富集 (core / iupac / pipeline, 只产数据表)
├── visualization/       render_all(统一表) → figures/environment + figures/plots
├── reports/             三份交付报告 (overview / data_quality / anomaly_report) 正文构造
├── candidates/          WT 位点候选挑选 (序列–效率方向性检验)
└── crispron_validation/ CNN7 × CRISPRon 的 Pos18 C→A 一致性验证
                         (windows / crispron / model / table / pipeline / __main__)
```

## 4. 当前实现范围 (全部经 DeepCRISPR 1344 真实批验证)
- 统一实验表 (1344 行全部有效) + 同 seed 配对一致性校验 (0 矛盾) + anomaly_report.csv
- prediction (model×split×cell 汇总 + LOCO) / environment 条件 ΔR² + 主效应 + 交互 + 析因 DAG
- attribution 统一表 (331200 行: linear/xgb/mlp/cnn/transformer 白名单方法)
- motif 候选 / 实例 / 富集三表 (656 motif, 186737 条实例, Fisher 精确检验)
- 统计: 配对 per-sample bootstrap + 符号翻转置换检验 + Type-II 边际 ANOVA, 族内 BH-FDR
- 交付: 3 份报告 + 20 张表 CSV + 5 张 train_data + 7 个归因 md + key_regulatory_biomarkers.csv
  + figures/ 198 张 PNG + plan/status/execution_log JSON
- 未实现任务 → unavailable+reason, 绝不伪造; 不可用输入一律如实标注

## 5. CLI 与测试
```bash
python -m analysis.pipeline --batch-dir <batch> --output <out>
python -m analysis.pipeline --batch-dir <batch> --analysis-plan plan.json
python -m analysis.collect_results --batch-dir <batch>
python analysis/importance_extraction.py --batch_dir <batch>
python -m analysis.crispron_validation                    # CNN7 × CRISPRon 一致性验证
python rule_discovery.py --data-set DeepCRISPR            # 赛道二规律发现交付物
```

## 6. 兼容说明
- `analysis/` 根目录的 `collect_results.py` / `importance_extraction.py` 既是库又是脚本,
  由 `analysis.pipeline` 与命令行共同使用; 引擎对训练产物**单向消费**。
- 所有阈值集中在 `analysis/config.py`, 改配置即可调整判定口径, 不需要改核心逻辑。
