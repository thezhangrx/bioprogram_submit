"""analysis.environment.incremental_effect — 条件 ΔR² / 主效应 (Paired baseline 原则)。

设计:
  16 组合 = 4 环境因子的 2^4 全因子。
  ΔR²(e | S) = metric(S+e) - metric(S), 且 (S+e, S) 必须来自同一
  (split_type, cell_line, model, seed) 分组 —— 同 eligible cohort / 同 test indices。
  不同 seed 绝不跨 seed 配对。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

ALL_ENVIRONMENTS = {"ctcf", "dnase", "h3k4me3", "rrbs"}
GROUP_COLS = ["split_type", "cell_line", "model", "random_seed"]
METRIC_PAIRS = [("R2", "delta_r2"), ("MAE", "delta_mae"), ("RMSE", "delta_rmse")]


def parse_environment_set(environment: str) -> Set[str]:
    """环境组合名 -> 环境因子集合。'sequence'->(), 'all'->4 因子。"""
    env = str(environment).strip().lower()
    if env == "sequence":
        return set()
    if env == "all":
        return set(ALL_ENVIRONMENTS)
    parts = [p for p in env.split("_") if p]
    if parts and parts[0] == "sequence":
        parts = parts[1:]
    return {p for p in parts if p in ALL_ENVIRONMENTS}


def combo_name(env_set: Set[str]) -> str:
    if not env_set:
        return "sequence"
    if set(env_set) == ALL_ENVIRONMENTS:
        return "all"
    return "sequence_" + "_".join(sorted(env_set))


def _row_env_index(rows_by_env: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Set[str]]]:
    env_sets: Dict[str, Set[str]] = {}
    env_index: Dict[str, pd.DataFrame] = {}
    for env_name, sub in rows_by_env.items():
        env_index[env_name] = sub
        env_sets[env_name] = parse_environment_set(env_name)
    return env_index, env_sets


def compute_conditional_increments(table: pd.DataFrame) -> pd.DataFrame:
    """
    对每个 (split, cell, model, seed) 分组, 求所有可配对 (S, S+e) 的增量。
    返回长表: 每行 = 一次配对增量 (same cohort)。
    """
    if table.empty:
        return pd.DataFrame(columns=(GROUP_COLS + ["environment_added", "background",
                                                   "background_set", "delta_r2",
                                                   "delta_mae", "delta_rmse"]))
    rows: List[Dict] = []
    for _, group in table.groupby([c for c in GROUP_COLS if c in table.columns], dropna=False):
        index, env_sets = _row_env_index({name: g for name, g in group.groupby("environment", dropna=False)})
        for background_name, bg_row in index.items():
            bg_set = env_sets[background_name]
            for candidate, cand_row in index.items():
                cand_set = env_sets[candidate]
                added = cand_set - bg_set
                if len(added) != 1 or not bg_set.issubset(cand_set):
                    continue
                if len(bg_row) != 1 or len(cand_row) != 1:
                    continue  # 同 key 出现多行说明上游表未聚合 -> 拒绝 (不做隐式平均)
                b = bg_row.iloc[0]
                c = cand_row.iloc[0]
                if any(pd.isna(b.get(m)) or pd.isna(c.get(m)) for m, _ in METRIC_PAIRS):
                    continue
                row: Dict = {
                    "split_type": b["split_type"], "cell_line": b["cell_line"],
                    "model": b["model"], "random_seed": b.get("random_seed"),
                    "environment_added": sorted(added)[0],
                    "background": background_name,
                    "background_set": ",".join(sorted(bg_set)),
                }
                for metric, delta_name in METRIC_PAIRS:
                    row[delta_name] = float(c[metric] - b[metric])
                rows.append(row)
    if not rows:
        return pd.DataFrame(columns=(GROUP_COLS + ["environment_added", "background",
                                                   "background_set", "delta_r2",
                                                   "delta_mae", "delta_rmse"]))
    return pd.DataFrame(rows)


def summarize_conditional(raw: pd.DataFrame) -> pd.DataFrame:
    """按 (split, cell, model, add, background) 聚合 (跨 seed 均值, 报告 n 配对)。"""
    if raw.empty:
        # 空表也必须带列契约：否则写出的是空行 CSV，下游 read_csv 直接报错。
        return pd.DataFrame(columns=[
            "split_type", "cell_line", "model", "environment_added", "background",
            "background_set", "delta_r2_mean", "delta_mae_mean", "delta_rmse_mean", "n_paired"])
    keys = ["split_type", "cell_line", "model", "environment_added", "background", "background_set"]
    agg = raw.groupby(keys, dropna=False).agg(
        delta_r2_mean=("delta_r2", "mean"),
        delta_mae_mean=("delta_mae", "mean"),
        delta_rmse_mean=("delta_rmse", "mean"),
        n_paired=("delta_r2", "size"),
    ).reset_index()
    for col in ("delta_r2_mean", "delta_mae_mean", "delta_rmse_mean"):
        agg[col] = agg[col].round(6)
    return agg


def compute_main_effects(conditional_raw: pd.DataFrame) -> pd.DataFrame:
    """
    主效应 e = 对一切不含 e 的背景 S 的 Δ(e|S) 平均 (跨 seed 后再跨背景平均)。
    输出每 (split, cell, model, environment) 主效应均值与配对背景数。
    """
    if conditional_raw.empty:
        return pd.DataFrame(columns=[
            "split_type", "cell_line", "model", "environment", "main_r2_delta",
            "main_mae_delta", "main_rmse_delta", "n_seeds", "n_backgrounds_avg"])
    keys = ["split_type", "cell_line", "model", "random_seed", "environment_added"]
    per_seed = conditional_raw.groupby(keys, dropna=False).agg(
        delta_r2=("delta_r2", "mean"),
        delta_mae=("delta_mae", "mean"),
        delta_rmse=("delta_rmse", "mean"),
        n_backgrounds=("delta_r2", "size"),
    ).reset_index()
    out = per_seed.groupby(["split_type", "cell_line", "model", "environment_added"],
                           dropna=False).agg(
        main_r2_delta=("delta_r2", "mean"),
        main_mae_delta=("delta_mae", "mean"),
        main_rmse_delta=("delta_rmse", "mean"),
        n_seeds=("delta_r2", "size"),
        n_backgrounds_avg=("n_backgrounds", "mean"),
    ).reset_index().rename(columns={"environment_added": "environment"})
    for col in ("main_r2_delta", "main_mae_delta", "main_rmse_delta"):
        out[col] = out[col].round(6)
    return out


def compute_pair_interactions(conditional_raw: pd.DataFrame) -> pd.DataFrame:
    """
    预测/统计交互占位实现: I(a,b) 用标准 2^4 正交对比平均:
      I = E_{S 不含 a,b} [ (Δ(a|S+b) - Δ(a|S)) ]   (全因子下等价 1/2 对比)
    返回每 (split, cell, model, 背景残基 S0=不含 a,b 的背景上的 b)…简化输出
    (split, cell, model, factor_a, factor_b, interaction_r2, n_pairs)。
    """
    if conditional_raw.empty:
        return pd.DataFrame(columns=["split_type", "cell_line", "model", "random_seed",
                                     "factor_a", "factor_b", "interaction_r2",
                                     "n_with_b", "n_without_b"])
    records: List[Dict] = []
    for (split, cell, model, seed), group in conditional_raw.groupby(
            ["split_type", "cell_line", "model", "random_seed"], dropna=False):
        by_a: Dict[str, Dict[str, float]] = {}
        for _, r in group.iterrows():
            by_a.setdefault(str(r["environment_added"]), {})[str(r["background"])] = float(r["delta_r2"])
        add_envs = list(by_a.keys())
        for i, a in enumerate(add_envs):
            for b in add_envs[i + 1:]:
                # 在含 b 的背景集合与不含 b 的背景集合上分别取 a 的条件增量
                # 注意: by_a[a] 的键是背景组合名 ("sequence_ctcf"), 必须解析成集合再判断 b,
                # 不能把组合名直接当集合 (旧实现误用 str(bg).split(",") -> 交互恒为空)。
                vals_b0 = [v for bg_name, v in by_a[a].items()
                           if b not in parse_environment_set(bg_name)]
                vals_b1 = [v for bg_name, v in by_a[a].items()
                           if b in parse_environment_set(bg_name)]
                if not vals_b0 or not vals_b1:
                    continue
                interaction = float(np.mean(vals_b1) - np.mean(vals_b0))
                records.append({
                    "split_type": split, "cell_line": cell, "model": model,
                    "random_seed": seed, "factor_a": a, "factor_b": b,
                    "interaction_r2": round(interaction, 6),
                    "n_with_b": len(vals_b1), "n_without_b": len(vals_b0),
                })
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)
