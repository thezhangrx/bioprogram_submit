"""analysis.stats.tasks — 把已实现的统计工具接到真实 effect 上 (bootstrap / permutation / FDR / ANOVA)。

设计边界 (严格遵守):
  * 只**读取**已有训练产物, 不重新训练, 不改动 R²/MAE/RMSE/XAI 任何数值;
  * paired ΔR² bootstrap 使用**同一 eligible cohort / 同一 test 样本**的 per-sample 预测
    (`<batch>/<run_name>/<model>_predictions.csv`), 符合 Paired baseline 原则;
  * SNR 不是 p 值, FDR 只对**真正产生 p 值的检验**执行, 不同科学问题绝不并入同一 family;
  * 数据不足/缺 artifact -> status=unavailable + reason, 绝不用 0 或 NaN 冒充结果;
  * METRIC_INCONSISTENCY 只标记 warning, 不删除 effect。

输出表 (由 pipeline 落盘):
    tables/bootstrap_results.csv
    tables/permutation_results.csv
    tables/anova_results.csv
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig
from analysis.environment.incremental_effect import (ALL_ENVIRONMENTS, combo_name,
                                                    parse_environment_set)
from analysis.stats.bootstrap import bootstrap_paired_metrics_ci_fast
from analysis.stats.hypothesis_tests import (permutation_test_for_effect,
                                            permutation_test_for_effect_fast,
                                            factorial_anova)
from analysis.stats.multiple_testing import bh_fdr

BOOTSTRAP_COLUMNS = ["split_type", "cell_line", "model", "random_seed",
                     "parent_combination", "child_combination", "added_environment",
                     "metric", "estimate", "ci_low", "ci_high", "excludes_zero",
                     "n_bootstrap", "seed", "alpha", "n_samples", "status", "reason"]
PERMUTATION_COLUMNS = ["split_type", "cell_line", "model", "random_seed", "test_type",
                       "parent_combination", "child_combination", "added_environment",
                       "factor", "null_hypothesis", "alternative", "observed_effect",
                       "p_value", "n_permutations", "seed", "n_values",
                       "n_values_excluded", "family_key", "status", "reason"]
ANOVA_COLUMNS = ["model_scope", "split_type", "cell_line", "model", "factor", "effect",
                 "F_statistic", "p_value", "df_num", "df_den", "effect_size",
                 "ci_low", "ci_high", "n_obs", "family_key", "status", "reason",
                 # 与 permutation_results 同口径的族内 BH-FDR（族规模 < min_family_size 时留空）
                 "FDR", "fdr_family", "fdr_status"]

METRICS = ("R2", "MAE", "RMSE")


# ---------------------------------------------------------------------------
# 1) per-sample 预测读取 (只读已有产物)
# ---------------------------------------------------------------------------
def predictions_path(batch_dir: str | Path, run_name: str) -> Optional[Path]:
    """<batch>/<run_name>/*_predictions.csv (排除 validation)。"""
    exp_dir = Path(batch_dir) / str(run_name)
    if not exp_dir.is_dir():
        return None
    cands = [p for p in sorted(exp_dir.glob("*_predictions.csv")) if "validation" not in p.name]
    return cands[0] if cands else None


def load_run_pairs(batch_dir: str | Path, run_name: str,
                   cache: Optional[Dict[str, Optional[np.ndarray]]] = None
                   ) -> Optional[np.ndarray]:
    """读取某实验的 test 预测 -> (n, 2) = [y_true, y_pred]; 不可用返回 None。"""
    key = str(run_name)
    if cache is not None and key in cache:
        return cache[key]
    arr: Optional[np.ndarray] = None
    path = predictions_path(batch_dir, run_name)
    if path is not None:
        try:
            df = pd.read_csv(path)
            if {"y_true", "y_pred"}.issubset(df.columns):
                vals = df[["y_true", "y_pred"]].apply(pd.to_numeric, errors="coerce")
                vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
                if len(vals) >= 3:
                    arr = vals.to_numpy(dtype=float)
        except Exception:  # noqa: BLE001 - 缺文件/坏文件 -> unavailable
            arr = None
    if cache is not None:
        cache[key] = arr
    return arr


def _valid_row(row: pd.Series, threshold: float) -> bool:
    """实验有效性 (与 factorial DAG 同规则): 指标有限且未发散, 相关未越界。"""
    for metric in ("R2", "RMSE", "MAE"):
        try:
            v = float(row.get(metric))
        except (TypeError, ValueError):
            return False
        if not np.isfinite(v) or abs(v) > float(threshold):
            return False
    for metric in ("Pearson", "Spearman"):
        try:
            v = float(row.get(metric))
        except (TypeError, ValueError):
            continue
        if np.isfinite(v) and abs(v) > 1.5:
            return False
    return True


def _lattice_pairs(table: pd.DataFrame, threshold: float) -> List[Dict]:
    """枚举所有可配对的 (S, S+e) 边 (同 split/cell/model/seed), 含实验 run_name。"""
    out: List[Dict] = []
    group_cols = [c for c in ("split_type", "cell_line", "model", "random_seed")
                  if c in table.columns]
    for key, group in table.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        ident = dict(zip(group_cols, key))
        by_env: Dict[str, List[pd.Series]] = {}
        for _, row in group.iterrows():
            by_env.setdefault(str(row["environment"]).lower(), []).append(row)
        for parent_name, prows in by_env.items():
            if len(prows) != 1:
                continue
            pset = parse_environment_set(parent_name)
            for env in ALL_ENVIRONMENTS - pset:
                child_name = combo_name(pset | {env})
                crows = by_env.get(child_name, [])
                if len(crows) != 1:
                    continue
                ident_row = {
                    **ident,
                    "parent_combination": parent_name,
                    "child_combination": child_name,
                    "added_environment": env,
                    "parent_run": str(prows[0].get("run_name", "")),
                    "child_run": str(crows[0].get("run_name", "")),
                    "parent_valid": _valid_row(prows[0], threshold),
                    "child_valid": _valid_row(crows[0], threshold),
                }
                out.append(ident_row)
    return out


# ---------------------------------------------------------------------------
# 2) Paired ΔR² / ΔMAE / ΔRMSE bootstrap (同一 test 样本)
# ---------------------------------------------------------------------------
def bootstrap_environment_edges(batch_dir: str | Path, table: pd.DataFrame,
                                config: Optional[AnalysisConfig] = None,
                                cache: Optional[Dict[str, Optional[np.ndarray]]] = None,
                                max_pairs: Optional[int] = None) -> pd.DataFrame:
    cfg = config or AnalysisConfig()
    thr = float(cfg.consensus.unstable_effect_threshold)
    cache = {} if cache is None else cache
    rows: List[Dict] = []
    pairs = _lattice_pairs(table, thr)
    if max_pairs:
        pairs = pairs[:int(max_pairs)]
    for pair in pairs:
        base = {k: pair[k] for k in ("split_type", "cell_line", "model", "random_seed",
                                     "parent_combination", "child_combination",
                                     "added_environment")}
        if not (pair["parent_valid"] and pair["child_valid"]):
            rows.append({**base, "metric": None, "estimate": None, "ci_low": None,
                         "ci_high": None, "excludes_zero": None, "n_bootstrap": 0,
                         "seed": cfg.bootstrap_seed, "alpha": cfg.bootstrap_alpha,
                         "n_samples": 0, "status": "unavailable",
                         "reason": "invalid_experiment_excluded (diverged/missing metric)"})
            continue
        a = load_run_pairs(batch_dir, pair["parent_run"], cache)
        b = load_run_pairs(batch_dir, pair["child_run"], cache)
        if a is None or b is None:
            rows.append({**base, "metric": None, "estimate": None, "ci_low": None,
                         "ci_high": None, "excludes_zero": None, "n_bootstrap": 0,
                         "seed": cfg.bootstrap_seed, "alpha": cfg.bootstrap_alpha,
                         "n_samples": 0, "status": "unavailable",
                         "reason": "prediction_artifact_missing"})
            continue
        if a.shape[0] != b.shape[0]:
            rows.append({**base, "metric": None, "estimate": None, "ci_low": None,
                         "ci_high": None, "excludes_zero": None, "n_bootstrap": 0,
                         "seed": cfg.bootstrap_seed, "alpha": cfg.bootstrap_alpha,
                         "n_samples": 0, "status": "unavailable",
                         "reason": f"cohort_mismatch (n_parent={a.shape[0]}, n_child={b.shape[0]})"})
            continue
        results = bootstrap_paired_metrics_ci_fast(
            a, b, metrics=METRICS, n_iterations=cfg.bootstrap_iterations,
            seed=cfg.bootstrap_seed, alpha=cfg.bootstrap_alpha)
        for metric in METRICS:
            res = results[metric]
            rows.append({
                **base, "metric": metric,
                "estimate": res.estimate if res.available else None,
                "ci_low": res.ci_low if res.available else None,
                "ci_high": res.ci_high if res.available else None,
                "excludes_zero": res.excludes_zero if res.available else None,
                "n_bootstrap": res.n_iterations if res.available else 0,
                "seed": cfg.bootstrap_seed, "alpha": cfg.bootstrap_alpha,
                "n_samples": int(a.shape[0]),
                "status": "ok" if res.available else "unavailable",
                "reason": "" if res.available else "insufficient paired samples (<3)",
            })
    return pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)


def bootstrap_main_effects(main_df: pd.DataFrame,
                           config: Optional[AnalysisConfig] = None) -> pd.DataFrame:
    """**Model-Level Bootstrap Interval (n = 7 model configurations)**。

    正式名称与解释 (R6): 重采样单元是**模型配置**, 不是单条 sgRNA。
    该区间刻画的是 effect 在**不同模型归纳偏置之间的一致性/稳定性** (cross-model robustness),
    而**不是** guide 总体的抽样不确定性; 7 个模型共享同一份数据/标签/预处理/划分,
    因此不可视为 independent biological replicates, n=7 下百分位 bootstrap 也无严格覆盖保证。
    输出中的 `n_models` 显式记录口径（`estimate_basis` / `interval_type` 两个常量列已移出，
    改在 README 的 `R2.results/Summary/<batch>文件说明` 里说明）。
    """
    from analysis.stats.bootstrap import bootstrap_ci

    cfg = config or AnalysisConfig()
    if main_df is None or main_df.empty:
        return pd.DataFrame(columns=["feature", "estimate", "ci_low", "ci_high", "excludes_zero",
                                     "n_bootstrap", "seed", "n_models", "status"])
    rows: List[Dict] = []
    thr = float(cfg.consensus.unstable_effect_threshold)
    for factor, sub in main_df.groupby("environment", dropna=False):
        # 与 evidence matrix 的 overall_effect 同口径: 先按模型求均值 (等权), 再对模型 bootstrap。
        # 若直接对全部 (模型×split×cell) 行 bootstrap, CI 可能与点估计不一致 (网格不均衡)。
        per_model = pd.to_numeric(sub["main_r2_delta"], errors="coerce")
        per_model = per_model[per_model.abs() < thr]
        vals = (sub.loc[per_model.index].groupby("model")["main_r2_delta"].mean()
                .dropna().tolist())
        if len(vals) < 3:
            rows.append({"feature": factor, "estimate": (float(np.mean(vals)) if vals else None),
                         "ci_low": None, "ci_high": None, "excludes_zero": None,
                         "n_bootstrap": 0, "seed": cfg.bootstrap_seed, "n_models": len(vals),
                         "status": "unavailable",
                         "reason": "需要 >=3 个模型的 main effect 才能 bootstrap"})
            continue
        res = bootstrap_ci(vals, estimator=lambda x: float(np.mean(x)),
                           n_iterations=cfg.bootstrap_iterations,
                           seed=cfg.bootstrap_seed, alpha=cfg.bootstrap_alpha)
        rows.append({"feature": factor, "estimate": res.estimate, "ci_low": res.ci_low,
                     "ci_high": res.ci_high, "excludes_zero": res.excludes_zero,
                     "n_bootstrap": res.n_iterations, "seed": cfg.bootstrap_seed,
                     "n_models": len(vals),
                     "status": "ok" if res.available else "unavailable", "reason": ""})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) Permutation test (只在有明确 null hypothesis 的 effect 上执行)
# ---------------------------------------------------------------------------
def permutation_environment_edges(batch_dir: str | Path, table: pd.DataFrame,
                                  config: Optional[AnalysisConfig] = None,
                                  cache: Optional[Dict[str, Optional[np.ndarray]]] = None
                                  ) -> pd.DataFrame:
    """H0: 加入环境 e 不改变 per-sample 平方误差 (mean loss difference = 0)。

    effect_values = SE_parent_i − SE_child_i (同一 test 样本), H1: mean > 0。
    """
    cfg = config or AnalysisConfig()
    thr = float(cfg.consensus.unstable_effect_threshold)
    cache = {} if cache is None else cache
    rows: List[Dict] = []
    for pair in _lattice_pairs(table, thr):
        base = {k: pair[k] for k in ("split_type", "cell_line", "model", "random_seed")}
        ident = {**base, "test_type": "environment_edge",
                 "parent_combination": pair["parent_combination"],
                 "child_combination": pair["child_combination"],
                 "added_environment": pair["added_environment"], "factor": None,
                 "null_hypothesis": ("mean per-sample loss difference (SE_parent - SE_child) = 0 "
                                     "on the same test samples (sign-flip randomization)"),
                 "alternative": "greater (adding the environment reduces squared error)",
                 "family_key": f"environment_edge|{base['split_type']}|{base['cell_line']}|{base['model']}"}
        if not (pair["parent_valid"] and pair["child_valid"]):
            rows.append({**ident, "observed_effect": None, "p_value": None,
                         "n_permutations": 0, "seed": cfg.bootstrap_seed, "n_values": 0,
                         "status": "unavailable", "reason": "invalid_experiment_excluded"})
            continue
        a = load_run_pairs(batch_dir, pair["parent_run"], cache)
        b = load_run_pairs(batch_dir, pair["child_run"], cache)
        if a is None or b is None or a.shape[0] != b.shape[0]:
            rows.append({**ident, "observed_effect": None, "p_value": None,
                         "n_permutations": 0, "seed": cfg.bootstrap_seed, "n_values": 0,
                         "status": "unavailable",
                         "reason": "prediction_artifact_missing" if (a is None or b is None)
                                   else "cohort_mismatch"})
            continue
        se_a = (a[:, 0] - a[:, 1]) ** 2
        se_b = (b[:, 0] - b[:, 1]) ** 2
        diff = se_a - se_b
        res = permutation_test_for_effect_fast(
            diff, null_effect=0.0, n_permutations=cfg.permutation_iterations,
            seed=cfg.bootstrap_seed, one_sided="greater")
        rows.append({**ident, "observed_effect": res.observed, "p_value": res.p_value,
                     "n_permutations": res.n_permutations, "seed": cfg.bootstrap_seed,
                     "n_values": int(diff.size),
                     "status": "ok" if res.null_stats_available and np.isfinite(res.p_value)
                               else "unavailable",
                     "reason": "" if res.null_stats_available else "insufficient paired samples"})
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


def permutation_main_effects(cond_summary: pd.DataFrame,
                             config: Optional[AnalysisConfig] = None) -> pd.DataFrame:
    """H0: factor 的条件增量在各背景下均值为 0 (two-sided), 单位 = 背景×seed 的条件增量。"""
    cfg = config or AnalysisConfig()
    if cond_summary is None or cond_summary.empty:
        return pd.DataFrame(columns=PERMUTATION_COLUMNS)
    rows: List[Dict] = []
    for keys, sub in cond_summary.groupby(["split_type", "cell_line", "model",
                                           "environment_added"], dropna=False):
        split, cell, model, factor = keys
        # R1 修复: 与证据矩阵 / bootstrap CI / 边级与交互置换保持一致,
        # 在 permutation inference 之前剔除 numerical divergence 与 NaN/inf 记录。
        # 判据 = consensus.unstable_effect_threshold (同一参数, 不新增阈值)。
        thr = float(cfg.consensus.unstable_effect_threshold)
        raw = pd.to_numeric(sub["delta_r2_mean"], errors="coerce")
        finite = raw[np.isfinite(raw)]
        vals = finite[finite.abs() < thr].tolist()
        n_excluded = int(len(finite) - len(vals))
        ident = {"split_type": split, "cell_line": cell, "model": model, "random_seed": None,
                 "test_type": "environment_main_effect", "parent_combination": None,
                 "child_combination": None, "added_environment": factor, "factor": factor,
                 "null_hypothesis": ("mean conditional ΔR² over backgrounds = 0 "
                                     "(sign-flip randomization over background values)"),
                 "alternative": "two-sided",
                 "family_key": f"environment_main|{split}|{cell}|{model}"}
        if len(vals) < 3:
            rows.append({**ident, "observed_effect": float(np.mean(vals)) if vals else None,
                         "p_value": None, "n_permutations": 0, "seed": cfg.bootstrap_seed,
                         "n_values": len(vals), "n_values_excluded": n_excluded,
                         "status": "unavailable",
                         "reason": ("需要 >=3 个背景/seed 条件增量"
                                    + (f" (已剔除 {n_excluded} 个发散/无效值)"
                                       if n_excluded else ""))})
            continue
        res = permutation_test_for_effect_fast(vals, null_effect=0.0,
                                               n_permutations=cfg.permutation_iterations,
                                               seed=cfg.bootstrap_seed, one_sided="two_sided")
        rows.append({**ident, "observed_effect": res.observed, "p_value": res.p_value,
                     "n_permutations": res.n_permutations, "seed": cfg.bootstrap_seed,
                     "n_values": len(vals), "n_values_excluded": n_excluded,
                     "status": "ok", "reason": ""})
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


def permutation_interactions(batch_dir: str | Path, table: pd.DataFrame,
                             config: Optional[AnalysisConfig] = None,
                             cache: Optional[Dict[str, Optional[np.ndarray]]] = None
                             ) -> pd.DataFrame:
    """H0: 交互 contrast I(a,b) 的 per-sample 均值为 0 (two-sided)。

    per-sample interaction: d_i(S0) = (SE_{S0+b} - SE_{S0+b+a}) - (SE_{S0} - SE_{S0+a}),
    对所有不含 a,b 的背景 S0 取样本级均值后再检验 (样本级配对, 非把 R² 当独立重复)。
    """
    cfg = config or AnalysisConfig()
    thr = float(cfg.consensus.unstable_effect_threshold)
    cache = {} if cache is None else cache
    rows: List[Dict] = []
    group_cols = [c for c in ("split_type", "cell_line", "model", "random_seed")
                  if c in table.columns]
    factors = sorted(ALL_ENVIRONMENTS)
    for key, group in table.groupby(group_cols, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        ident = dict(zip(group_cols, key))
        by_env = {str(r["environment"]).lower(): r for _, r in group.iterrows()}
        valid = {name: _valid_row(row, thr) for name, row in by_env.items()}
        for i, a_env in enumerate(factors):
            for b_env in factors[i + 1:]:
                diffs: List[np.ndarray] = []
                for s0_size in range(0, len(factors) - 1):
                    from itertools import combinations
                    for s0 in combinations([f for f in factors if f not in (a_env, b_env)], s0_size):
                        names = {
                            "S0": combo_name(set(s0)),
                            "S0a": combo_name(set(s0) | {a_env}),
                            "S0b": combo_name(set(s0) | {b_env}),
                            "S0ab": combo_name(set(s0) | {a_env, b_env}),
                        }
                        if not all(n in by_env and valid[n] for n in names.values()):
                            continue
                        arrays = {k: load_run_pairs(batch_dir, str(by_env[n].get("run_name", "")), cache)
                                  for k, n in names.items()}
                        if any(v is None for v in arrays.values()):
                            continue
                        sizes = {v.shape[0] for v in arrays.values()}
                        if len(sizes) != 1:
                            continue
                        se = {k: (v[:, 0] - v[:, 1]) ** 2 for k, v in arrays.items()}
                        diffs.append((se["S0b"] - se["S0ab"]) - (se["S0"] - se["S0a"]))
                base = {"split_type": ident.get("split_type"), "cell_line": ident.get("cell_line"),
                        "model": ident.get("model"), "random_seed": ident.get("random_seed"),
                        "test_type": "environment_interaction", "parent_combination": None,
                        "child_combination": None, "added_environment": None,
                        "factor": f"{a_env}*{b_env}",
                        "null_hypothesis": (f"mean per-sample interaction I({a_env},{b_env}) = 0 "
                                            "(sign-flip randomization over pooled background contrasts)"),
                        "alternative": "two-sided",
                        "family_key": (f"environment_interaction|{ident.get('split_type')}|"
                                       f"{ident.get('cell_line')}|{ident.get('model')}")}
                if not diffs:
                    rows.append({**base, "observed_effect": None, "p_value": None,
                                 "n_permutations": 0, "seed": cfg.bootstrap_seed, "n_values": 0,
                                 "status": "unavailable",
                                 "reason": "缺少可配对的 4 个组合预测 (S0, S0+a, S0+b, S0+a+b)"})
                    continue
                vals = np.mean(np.vstack(diffs), axis=0)
                res = permutation_test_for_effect_fast(
                    vals, null_effect=0.0, n_permutations=cfg.permutation_iterations,
                    seed=cfg.bootstrap_seed, one_sided="two_sided")
                rows.append({**base, "observed_effect": res.observed, "p_value": res.p_value,
                             "n_permutations": res.n_permutations, "seed": cfg.bootstrap_seed,
                             "n_values": int(vals.size), "status": "ok", "reason": ""})
    return pd.DataFrame(rows, columns=PERMUTATION_COLUMNS)


# ---------------------------------------------------------------------------
# 4) BH-FDR (按 family, 不混问题)
# ---------------------------------------------------------------------------
def apply_fdr(results: pd.DataFrame, config: Optional[AnalysisConfig] = None) -> pd.DataFrame:
    """对带 p_value 的检验结果按 `family_key` 分别做 BH-FDR。"""
    cfg = config or AnalysisConfig()
    if results is None or results.empty:
        # 通用空表分支：不硬编码 permutation 的列，直接沿用输入表的列 + 3 个校正列
        cols = list(results.columns) if results is not None else []
        for col in ("FDR", "fdr_family", "fdr_status"):
            if col not in cols:
                cols.append(col)
        return pd.DataFrame(columns=cols)
    out = results.copy()
    out["fdr_family"] = out["family_key"]
    out["FDR"] = np.nan
    out["fdr_status"] = "not_applicable"
    for family, sub in out.groupby("fdr_family", dropna=False):
        finite = sub["p_value"].notna()
        if int(finite.sum()) < int(cfg.fdr.min_family_size):
            out.loc[sub.index, "fdr_status"] = (
                f"not_applicable (family size {int(finite.sum())} < {cfg.fdr.min_family_size})")
            continue
        q = bh_fdr(sub.loc[finite, "p_value"].tolist())
        out.loc[sub.index[finite], "FDR"] = q
        out.loc[sub.index, "fdr_status"] = "ok"
    return out


# ---------------------------------------------------------------------------
# 5) ANOVA (blocked factorial + per-group 加性主效应)
# ---------------------------------------------------------------------------
def _binary_factors(table: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = table.copy()
    df["_env_set"] = df["environment"].map(lambda s: parse_environment_set(str(s)))
    factors = sorted(ALL_ENVIRONMENTS)
    for f in factors:
        df[f] = df["_env_set"].map(lambda s, f=f: 1 if f in s else 0)
    return df, factors


def run_anova_tasks(table: pd.DataFrame, config: Optional[AnalysisConfig] = None) -> pd.DataFrame:
    """全局 blocked factorial ANOVA (含交互) + 每个 (split, cell, model) 的加性主效应 ANOVA。"""
    cfg = config or AnalysisConfig()
    a_cfg = cfg.anova
    rows: List[Dict] = []
    if table is None or table.empty:
        return pd.DataFrame(columns=ANOVA_COLUMNS)
    df, factors = _binary_factors(table)
    thr = float(cfg.consensus.unstable_effect_threshold)
    r2 = pd.to_numeric(df["R2"], errors="coerce")
    df = df[(r2.notna()) & (r2.abs() < thr)].copy()

    def _emit(terms, scope: str, split, cell, model, family_key: str) -> None:
        for t in terms:
            rows.append({"model_scope": scope, "split_type": split, "cell_line": cell,
                         "model": model, "factor": t.factor, "effect": t.effect,
                         "F_statistic": t.F_statistic, "p_value": t.p_value,
                         "df_num": t.df_num, "df_den": t.df_den,
                         "effect_size": t.effect_size, "ci_low": t.ci_low,
                         "ci_high": t.ci_high, "n_obs": t.n_obs,
                         "family_key": family_key, "status": t.status, "reason": t.reason})

    blocks = [b for b in a_cfg.block_factors if b in df.columns]
    if len(df) >= int(a_cfg.min_observations):
        terms = factorial_anova(df, response="R2", factors=factors, block_factors=blocks,
                                include_interactions=bool(a_cfg.include_interactions),
                                min_observations=int(a_cfg.min_observations),
                                min_residual_df=int(a_cfg.min_residual_df),
                                n_iterations=int(a_cfg.ci_iterations), seed=cfg.bootstrap_seed,
                                model_scope="blocked_factorial", alpha=cfg.bootstrap_alpha)
        _emit(terms, "blocked_factorial", "ALL", "ALL", "ALL", "environment_anova|global")
    else:
        rows.append({"model_scope": "blocked_factorial", "split_type": "ALL", "cell_line": "ALL",
                     "model": "ALL", "factor": "(none)", "effect": None, "F_statistic": None,
                     "p_value": None, "df_num": None, "df_den": None, "effect_size": None,
                     "ci_low": None, "ci_high": None, "n_obs": int(len(df)),
                     "family_key": "environment_anova|global", "status": "unavailable",
                     "reason": f"insufficient_data: n={len(df)} < {a_cfg.min_observations}"})

    if a_cfg.run_per_group:
        for (split, cell, model), sub in df.groupby(["split_type", "cell_line", "model"],
                                                    dropna=False):
            terms = factorial_anova(sub, response="R2", factors=factors, block_factors=[],
                                    include_interactions=False,
                                    min_observations=int(a_cfg.min_observations),
                                    min_residual_df=int(a_cfg.min_residual_df),
                                    n_iterations=0, seed=cfg.bootstrap_seed,
                                    model_scope="per_group_additive_main", alpha=cfg.bootstrap_alpha)
            _emit(terms, "per_group_additive_main", split, cell, model,
                  f"environment_anova_group|{split}|{cell}|{model}")
    out = pd.DataFrame(rows, columns=ANOVA_COLUMNS)
    # 按 family_key 做 BH-FDR（与 permutation_results 同口径）：
    #   environment_anova|global                     -> 10 个 term 一族
    #   environment_anova_group|<split>|<cell>|<model> -> 该组 4 个主效应一族
    # 族内有效 p < FdrFamilyConfig.min_family_size(=2) 时 FDR 留空并记 not_applicable（不伪造）。
    return apply_fdr(out, config=cfg)


# ---------------------------------------------------------------------------
# 6) 能力探测 (供 validator / status)
# ---------------------------------------------------------------------------


def bootstrap_cellline_effects(main_df: pd.DataFrame,
                               config: Optional[AnalysisConfig] = None) -> pd.DataFrame:
    """每个 (split_type, factor, cell_line) 的跨模型 main effect CI (用于 cell-line CI overlap)。

    单位 = 模型 (n=模型数); 口径说明见 README 的 `R2` 一节（原 `estimate_basis` 列已移出）。
    """
    from analysis.stats.bootstrap import bootstrap_ci

    cfg = config or AnalysisConfig()
    cols = ["split_type", "cell_line", "factor", "estimate", "ci_low", "ci_high",
            "excludes_zero", "n_bootstrap", "n_models", "status"]
    if main_df is None or main_df.empty:
        return pd.DataFrame(columns=cols)
    thr = float(cfg.consensus.unstable_effect_threshold)
    rows: List[Dict] = []
    for (split, cell, factor), sub in main_df.groupby(["split_type", "cell_line", "environment"],
                                                      dropna=False):
        vals = pd.to_numeric(sub["main_r2_delta"], errors="coerce").dropna()
        vals = [float(v) for v in vals if abs(float(v)) < thr]
        if len(vals) < 3:
            rows.append({"split_type": split, "cell_line": cell, "factor": factor,
                         "estimate": float(np.mean(vals)) if vals else None, "ci_low": None,
                         "ci_high": None, "excludes_zero": None, "n_bootstrap": 0,
                         "n_models": len(vals), "status": "unavailable"})
            continue
        res = bootstrap_ci(vals, estimator=lambda x: float(np.mean(x)),
                           n_iterations=cfg.bootstrap_iterations, seed=cfg.bootstrap_seed,
                           alpha=cfg.bootstrap_alpha)
        rows.append({"split_type": split, "cell_line": cell, "factor": factor,
                     "estimate": res.estimate, "ci_low": res.ci_low, "ci_high": res.ci_high,
                     "excludes_zero": res.excludes_zero, "n_bootstrap": res.n_iterations,
                     "n_models": len(vals),
                     "status": "ok" if res.available else "unavailable"})
    return pd.DataFrame(rows, columns=cols)
