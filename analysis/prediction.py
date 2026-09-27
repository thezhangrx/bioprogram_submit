"""analysis.prediction — 预测/泛化性能分析 (只读统一表)。"""
from __future__ import annotations

from typing import Dict

import pandas as pd

METRICS = ["R2", "MAE", "RMSE", "Pearson", "Spearman"]


def performance_by_model_split(table: pd.DataFrame) -> pd.DataFrame:
    """模型 × split 的指标均值/标准差与实验数。"""
    cols = ["model", "split_type"] + METRICS
    cols = [c for c in cols if c in table.columns]
    agg = table.groupby(["model", "split_type"], dropna=False)[
        [c for c in METRICS if c in table.columns]
    ].agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(str(x).strip("_") for x in c).strip("_") if isinstance(c, tuple)
                   else str(c) for c in agg.columns]
    n = table.groupby(["model", "split_type"], dropna=False).size().rename("n_experiments")
    return agg.merge(n, on=["model", "split_type"], how="left")


LOCO_COLUMNS = ["model", "cell_line", "R2_mean", "R2_median", "MAE_mean", "MAE_median",
                "RMSE_mean", "RMSE_median", "Pearson_mean", "Pearson_median",
                "Spearman_mean", "Spearman_median", "n_experiments", "n_valid"]


def loco_performance(table: pd.DataFrame, unstable: float = 10.0) -> pd.DataFrame:
    """LOCO (split='all' 留一细胞系跨域) 泛化表。

    D6 修复: 与主统计流程使用同一 numerical validity contract —— |R²| >= unstable
    (默认 10.0 = AnalysisConfig.consensus.unstable_effect_threshold) 的发散 run
    不进入均值/中位数；同时输出 n_experiments（全部）与 n_valid（过滤后），
    避免出现 −2.85e19 之类的发散值直接进入论文图表。
    """
    full = table[table["split_type"].str.lower() == "all"]
    if full.empty:
        return pd.DataFrame(columns=LOCO_COLUMNS)
    sub = full.copy()
    r2 = pd.to_numeric(sub.get("R2"), errors="coerce")
    sub = sub[r2.abs() < float(unstable)]
    cols = [c for c in METRICS if c in sub.columns]
    if sub.empty or not cols:
        return pd.DataFrame(columns=LOCO_COLUMNS)
    out = sub.groupby(["model", "cell_line"], dropna=False)[cols].agg(["mean", "median"]).reset_index()
    out.columns = ["_".join([str(x) for x in c if x]) if isinstance(c, tuple) else str(c)
                   for c in out.columns]
    out = out.rename(columns={"model_": "model", "cell_line_": "cell_line"})
    n_all = full.groupby(["model", "cell_line"], dropna=False).size().rename("n_experiments")
    n_ok = sub.groupby(["model", "cell_line"], dropna=False).size().rename("n_valid")
    return out.merge(n_all, on=["model", "cell_line"], how="left").merge(
        n_ok, on=["model", "cell_line"], how="left")
