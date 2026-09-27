# 08 Anomaly Report

## 类型说明

- metric_inconsistency: 同一 cohort (同 seed 配对) 下 ΔR² 与 ΔRMSE 同号 (R²=1-SSE/SST, RMSE=√(SSE/n) 的理论矛盾标记, 不删除实验)
- numerical_instability_excluded_from_evidence: 12 上下文行 (|ΔR²| 超阈值, 不参与跨模型聚合)

## 明细
_无 metric inconsistency_


> 完整 machine-readable 见 tables/anomaly_report.csv。
