"""analysis.reports.markdown_report — 三份交付报告的 markdown 正文构造（纯函数）。

交付结构（2026-09-25 定稿）：`summary/<数据集>/reports/` 只保留
    overview.md  ·  data_quality.md  ·  anomaly_report.md

其余报告（02_prediction_generalization / 03_environment_effects /
04_sequence_and_motifs / 07_biological_hypotheses / environment_dag_report）已移出交付范围，
对应 build_* 函数一并删除；它们承载的**数据**仍以 CSV 交付
（prediction_summary / environment_* / motif_* 等）。

05_cellline_heterogeneity 与 06_evidence_integration 连同其数据表
（cellline_effects.csv / evidence_matrix.csv）一起移出——那是人为证据分级/标签体系。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from analysis.plans import AnalysisPlan

#: 当"指标表"与"目录扫描"两条路都拿不到数据时，异常/一致性检测**无法进行**。
#: 这种缺口不能只 print 一行了事——必须写进交付报告，否则报告里的"_无_"会被
#: 误读成"检查过了，没问题"。见 analysis/pipeline.py 的 detection_note。
ANOMALY_DETECTION_SKIPPED = "[!] 未实现检测跳过"


def _fmt(v: Any, round_digits: int = 4) -> str:
    """单个单元格的 markdown 文本表示。

    * ``None`` / NaN → ``nan``（不伪装成 0）
    * bool        → ``True`` / ``False``
    * 数值        → 定点 ``round_digits`` 位（round_digits=0 时整数不带小数点）
    * 其余        → ``str(v)``
    """
    if v is None:
        return "nan"
    try:
        if pd.isna(v):
            return "nan"
    except (TypeError, ValueError):
        pass
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.{round_digits}f}"
    try:
        return f"{float(v):.{round_digits}f}"
    except (TypeError, ValueError):
        return str(v)


def _md_table(df: pd.DataFrame, round_digits: int = 4) -> str:
    """DataFrame → markdown 表：列名原样，分隔行统一右对齐 (``---:``)。"""
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join("---:" for _ in cols) + " |"]
    for _, row in df.iterrows():
        out.append("| " + " | ".join(_fmt(v, round_digits) for v in row) + " |")
    return "\n".join(out)


def build_overview_md(
    coverage: Dict[str, Any],
    plan: AnalysisPlan,
    status: Dict[str, Any],
    anomaly_rows: pd.DataFrame,
) -> str:
    lines: List[str] = []
    lines.append("# 00 Overview\n")
    lines.append(f"- experiment count: **{coverage.get('experiment_count')}**")
    lines.append(f"- valid experiments: **{coverage.get('valid_count')}**")
    def _join(items) -> str:
        """覆盖度行的列表拼接：非字符串取值（脏数据）也要能出报告，不因 join 崩掉。"""
        return ", ".join(str(x) for x in items) or "—"

    lines.append(f"- models: {_join(coverage.get('models', []))}")
    lines.append(f"- cell lines: {_join(coverage.get('cell_lines', []))}")
    lines.append(f"- environments: {len(coverage.get('environments', []))}")
    lines.append(f"- splits: {_join(coverage.get('splits', []))}")
    lines.append(f"- metric inconsistency (ΔR²/ΔRMSE 同号) rows: **{len(anomaly_rows)}**")

    lines.append("\n## Analysis Plan (实际选择)\n")
    for t in status.get("tasks", []):
        mark = {
            "completed": "✓", "pending": "☐", "skipped": "⊘",
            "unavailable": "⚠", "failed": "✗", "running": "⟳",
        }.get(t.get("status"), "○")
        reason = f"  ({t.get('reason')})" if t.get("reason") else ""
        lines.append(f"- {mark} {t['task_id']} [selected={t['selected']}, "
                     f"available={t['available']}, status={t['status']}]{reason}")
    lines.append("\n## 执行说明\n")
    lines.append("- 本报告由统一 experiment table 生成; 所有图/表引用同一数据源。")
    lines.append("- PFI/LOFO 等若训练端未生成, 一律标记 Unavailable, 不回头修改训练系统。")
    return "\n".join(lines) + "\n"


def build_data_quality_md(coverage: dict, table: pd.DataFrame, anomalies: pd.DataFrame,
                          detection_note: Optional[str] = None) -> str:
    lines = ["# 01 Data Quality (批次级)\n"]
    lines.append(f"- experiments: {coverage.get('experiment_count')} | valid (R² finite): "
                 f"{coverage.get('valid_count')}")
    if table is not None and "cell_line" in table.columns and "environment" in table.columns:
        bal = table.groupby(["cell_line", "environment"]).size().unstack(fill_value=0)
        bal = bal.reset_index()
        # mixed 划分在训练端把 cell_line 记作字面量 "none"；在交付报告里显示为
        # "mixed" 才符合语义（纯显示层改名，不动 tables/ 里的原始取值）。
        bal["cell_line"] = bal["cell_line"].map(
            lambda v: "mixed" if str(v).strip().lower() == "none" else v)
        if len(bal.columns) > 1:      # 无 environment 列（空批次）时不画空表
            lines.append("\n## cell-line × environment 覆盖\n")
            lines.append(_md_table(bal, round_digits=0))
    lines.append("\n## Metric inconsistency (同 cohort ΔR²/ΔRMSE 同号)\n")
    if detection_note:
        lines.append(detection_note)
    elif anomalies is None or anomalies.empty:
        lines.append("_无_")
    else:
        lines.append(_md_table(anomalies, round_digits=6))
    lines.append("\n> Eligible/Limited/Ineligible 的逐环境判定由训练前的数据质控流程输出"
                 "（本仓库未随附该程序）;")
    lines.append("此处为批次级状态 (训练后只读)。\n")
    return "\n".join(lines)


def build_anomaly_md(anomalies: pd.DataFrame, matrix_excluded: int = 0,
                     detection_note: Optional[str] = None) -> str:
    lines = ["# 08 Anomaly Report\n"]
    lines.append("## 类型说明\n")
    lines.append("- metric_inconsistency: 同一 cohort (同 seed 配对) 下 ΔR² 与 ΔRMSE 同号 "
                 "(R²=1-SSE/SST, RMSE=√(SSE/n) 的理论矛盾标记, 不删除实验)")
    lines.append(f"- numerical_instability_excluded_from_evidence: {matrix_excluded} 上下文行 "
                 "(|ΔR²| 超阈值, 不参与跨模型聚合)\n")
    if detection_note:
        lines.append(f"## 明细\n{detection_note}\n")
    elif anomalies is None or anomalies.empty:
        lines.append("## 明细\n_无 metric inconsistency_\n")
    else:
        lines.append("## 明细 (METRIC_INCONSISTENCY)\n")
        lines.append(_md_table(anomalies, round_digits=6))
    lines.append("\n> 完整 machine-readable 见 tables/anomaly_report.csv。\n")
    return "\n".join(lines)
