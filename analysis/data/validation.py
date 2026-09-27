"""analysis.data.validation — 结果校验。

重点: 同一 test cohort 下 R2 与 RMSE 存在解析关系 (R²=1-SSE/SST, RMSE=√(SSE/n)),
因此 ΔR² 与 ΔRMSE 不应同号 (除非样本组成不同)。此检查只"标记不一致",
不删除任何实验 (Metric inconsistency != 实验无效)。
"""
from __future__ import annotations

from typing import Dict, List

import pandas as pd


def metric_consistency_flags(delta_r2: float, delta_rmse: float, tol: float = 1e-9) -> List[str]:
    """同一 cohort 下 R2/RMSE 同向变化 = 指标不一致 (理论矛盾)。

    返回标签: metric_inconsistency_same_increase / same_decrease / ok。
    """
    flags: List[str] = []
    same_increase = delta_r2 > tol and delta_rmse > tol
    same_decrease = delta_r2 < -tol and delta_rmse < -tol
    if same_increase:
        flags.append("metric_inconsistency_same_increase")
    elif same_decrease:
        flags.append("metric_inconsistency_same_decrease")
    return flags


def validate_metric_consistency(table: pd.DataFrame, tol: float = 1e-9) -> pd.DataFrame:
    """对 sequence 基线外的每行相对同 (split, cell_line, model) sequence 计算 delta 并标记。"""
    if table.empty or "environment" not in table.columns:
        return pd.DataFrame(columns=["row", "flags", "delta_r2", "delta_rmse"])
    work = table.copy()
    baseline = {}
    for _, r in work.iterrows():
        if str(r.get("environment", "")).lower() == "sequence":
            # 同 seed 配对: mixed 多 seed 绝不跨 seed 比较 (paired cohort 原则)
            seed = str(r.get("random_seed", "") or "none")
            key = (str(r["split_type"]), str(r["cell_line"]), str(r["model"]), seed)
            baseline[key] = (r["R2"], r["RMSE"])
    rows: List[Dict] = []
    for idx, r in work.iterrows():
        env = str(r.get("environment", "")).lower()
        if env == "sequence":
            continue
        seed = str(r.get("random_seed", "") or "none")
        base = baseline.get((str(r["split_type"]), str(r["cell_line"]), str(r["model"]), seed))
        if base is None or pd.isna(r["R2"]) or pd.isna(r["RMSE"]):
            continue
        d_r2 = float(r["R2"] - base[0])
        d_rmse = float(r["RMSE"] - base[1])
        flags = metric_consistency_flags(d_r2, d_rmse, tol)
        if flags:
            rows.append({
                "row": int(idx),
                "experiment": str(r.get("run_name", idx)),
                "flags": ";".join(flags),
                "delta_r2": round(d_r2, 6),
                "delta_rmse": round(d_rmse, 6),
            })
    if not rows:
        return pd.DataFrame(columns=["row", "experiment", "flags", "delta_r2", "delta_rmse"])
    return pd.DataFrame(rows)
