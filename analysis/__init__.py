"""analyse — CRISPR 编辑效率影响因素发现与证据整合分析引擎。

分层职责：
    analysis.config      显著性/统计/QC 阈值等集中配置
    analysis.schemas     统一数据记录 (ExperimentRecord)
    analysis.data        批量结果只读加载、字段适配、校验 (训练系统只读)
    analysis.stats       Effect/CI/检验/FDR family/Bootstrap 等规范实现
    analysis.plans       AnalysisPlan / Validator / ExecutionPlan / 状态
    analysis.pipeline    编排入口 (GUI/CLI 只提供 AnalysisPlan)

已移出（人为证据标签体系）: `analysis.evidence`（Evidence Tier / 假设生成）与
`analysis.cellline`（细胞系一致性标签）。**保留**：§1 显著性/星级 与 §5.4 FDR family。

铁律: 本包对训练系统 (src/, predict.py 等) 只读; 不反向 import 训练模块。
"""

__version__ = "0.1.0"
