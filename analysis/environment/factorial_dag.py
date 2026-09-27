"""analysis.environment.factorial_dag — Environment Factorial DAG (2^4 lattice) 的数据层。

科学定义 (与报告/图中措辞保持一致):
    Node : 一个 environment combination (理论上 2^4 = 16 个)
    Edge : 从已有环境集合 S 增加一个新环境 e, 即 S -> S ∪ {e} (理论上 32 条)
    Edge value: conditional incremental effect  ΔR²_{e|S} = R²(S∪{e}) - R²(S)
    Primary effect: ΔR²          Direction: positive / negative
    Uncertainty: bootstrap CI (存在才填, 绝不伪造)

关键结构性质:
    * 每个 combination 只有一个节点, 但可以有多个父节点/入边
      (CTCF→CTCF+DNase 与 DNase→CTCF+DNase 同时存在);
    * 缺失/异常只改变 node.status / edge.status / warning, 不删除理论结构;
    * METRIC_INCONSISTENCY (ΔR² 与 ΔRMSE 同向) 只是 warning, 边仍然保留。

数据流 (禁止 visualization 自己扫描实验目录):
    existing results -> analysis.data.loaders (normalization)
        -> tables/environment_nodes.csv + tables/environment_edges.csv
        -> analysis.visualization.factorial_dag (纯 DataFrame 渲染)
"""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.environment.incremental_effect import (ALL_ENVIRONMENTS, combo_name,
                                                    compute_conditional_increments,
                                                    parse_environment_set)

# 规范序 (与 incremental_effect/combo_name 一致: 排序后拼接)
ENV_FACTORS: Tuple[str, ...] = ("ctcf", "dnase", "h3k4me3", "rrbs")
FACTOR_LABELS: Dict[str, str] = {"ctcf": "CTCF", "dnase": "DNase",
                                 "h3k4me3": "H3K4me3", "rrbs": "RRBS"}
NODE_METRICS: Tuple[str, ...] = ("R2", "RMSE", "MAE", "Pearson", "Spearman")
DELTA_COLUMNS: Tuple[str, ...] = ("delta_r2", "delta_rmse", "delta_mae",
                                  "delta_pearson", "delta_spearman")

NODE_COLUMNS = ["split_type", "cell_line", "model", "combination", "environment_count",
                "CTCF", "DNase", "H3K4me3", "RRBS",
                "r2", "rmse", "mae", "pearson", "spearman",
                "sample_count", "eligible_count", "n_invalid",
                "status", "warning"]

EDGE_COLUMNS = ["split_type", "cell_line", "model",
                "parent_combination", "child_combination", "added_environment",
                "parent_r2", "child_r2", "delta_r2",
                "parent_rmse", "child_rmse", "delta_rmse",
                "parent_mae", "child_mae", "delta_mae",
                "delta_pearson", "delta_spearman",
                "ci_low", "ci_high", "ci_excludes_zero", "n_bootstrap", "bootstrap_status",
                "n_paired", "status", "warning"]

DAG_MD_MARKER = "<!-- FACTORIAL_DAG_SECTION -->"


# ---------------------------------------------------------------------------
# 1) 理论结构 (不依赖任何实验数据; 用于完整性校验)
# ---------------------------------------------------------------------------
def theoretical_nodes() -> List[Dict]:
    """16 个理论 combination (level = 启用的环境因子个数 0..4)。"""
    out: List[Dict] = []
    for k in range(len(ENV_FACTORS) + 1):
        for subset in itertools.combinations(ENV_FACTORS, k):
            combo = combo_name(set(subset))
            row = {"combination": combo, "environment_count": k}
            for f in ENV_FACTORS:
                row[FACTOR_LABELS[f]] = bool(f in subset)
            out.append(row)
    return out


def theoretical_edges() -> List[Dict]:
    """32 条理论有向边 S -> S∪{e} (每个组合的每个合法单因素增加关系)。"""
    out: List[Dict] = []
    for k in range(len(ENV_FACTORS)):
        for subset in itertools.combinations(ENV_FACTORS, k):
            s = set(subset)
            for e in ENV_FACTORS:
                if e in s:
                    continue
                out.append({"parent_combination": combo_name(s),
                            "child_combination": combo_name(s | {e}),
                            "added_environment": e})
    return out


def parents_of(combination: str) -> List[str]:
    """组合的全部直接父组合 (少一个环境), 按规范序。"""
    s = parse_environment_set(combination)
    if not s:
        return []
    return [combo_name(s - {e}) for e in sorted(s)]


def combination_label(combination: str) -> str:
    """人类可读标签: sequence-only / CTCF + DNase / ALL (4 factors)。"""
    s = parse_environment_set(combination)
    if not s:
        return "sequence-only"
    if len(s) == len(ENV_FACTORS):
        return "ALL (4 factors)"
    return " + ".join(FACTOR_LABELS[f] for f in ENV_FACTORS if f in s)


# ---------------------------------------------------------------------------
# 2) 观测数据 -> nodes / edges
# ---------------------------------------------------------------------------
def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def experiment_validity(row: pd.Series, threshold: float = 10.0) -> Tuple[bool, str]:
    """单实验有效性: 缺失 / 发散 / 相关系数越界 -> (False, reason)。"""
    for metric in ("R2", "RMSE", "MAE"):
        v = row.get(metric)
        if not _finite(v):
            return False, f"{metric}_missing"
        if abs(float(v)) > float(threshold):
            return False, f"{metric}_diverged(|{metric}|>{threshold:g})"
    for metric in ("Pearson", "Spearman"):
        v = row.get(metric)
        if _finite(v) and abs(float(v)) > 1.5:
            return False, f"{metric}_out_of_range"
    return True, ""


def _groups(table: pd.DataFrame) -> List[Tuple]:
    keys = ["split_type", "cell_line", "model"]
    if all(k in table.columns for k in keys):
        return list(table.groupby(keys, dropna=False))
    return [((None, None, None), table)]


def build_environment_nodes(table: pd.DataFrame, threshold: float = 10.0) -> pd.DataFrame:
    """每个 (split, cell_line, model, combination) 一行 —— **包含理论存在但数据缺失的组合**。

    缺失不删除节点: `status='unavailable'`, 指标为 NaN。
    """
    rows: List[Dict] = []
    for (split, cell, model), group in _groups(table):
        for tnode in theoretical_nodes():
            combo = tnode["combination"]
            sub = group[group["environment"].astype(str).str.lower() == combo]
            valid_rows, reasons = [], []
            for _, r in sub.iterrows():
                ok, why = experiment_validity(r, threshold)
                (valid_rows.append(r) if ok else reasons.append(why))
            if valid_rows:
                vdf = pd.DataFrame(valid_rows)
                means = {m: (float(pd.to_numeric(vdf[m], errors="coerce").mean())
                             if m in vdf.columns else float("nan")) for m in NODE_METRICS}
                status = "ok"
                warning = ("invalid_experiments_excluded:" + ",".join(sorted(set(reasons)))
                           if reasons else "")
            else:
                means = {m: float("nan") for m in NODE_METRICS}
                status = "unavailable"
                warning = ("no experiment for this combination" if len(sub) == 0
                           else "all experiments invalid:" + ",".join(sorted(set(reasons))))
            row = {"split_type": split, "cell_line": cell, "model": model,
                   "combination": combo, "environment_count": tnode["environment_count"]}
            for f in ENV_FACTORS:
                row[FACTOR_LABELS[f]] = tnode[FACTOR_LABELS[f]]
            row.update({"r2": means["R2"], "rmse": means["RMSE"], "mae": means["MAE"],
                        "pearson": means["Pearson"], "spearman": means["Spearman"],
                        "sample_count": int(len(sub)), "eligible_count": int(len(valid_rows)),
                        "n_invalid": int(len(sub) - len(valid_rows)),
                        "status": status, "warning": warning})
            rows.append(row)
    out = pd.DataFrame(rows)
    return out[[c for c in NODE_COLUMNS if c in out.columns]]


def _edge_deltas(parent: pd.Series, child: pd.Series) -> Dict[str, float]:
    """一条边的 5 个配对增量 (缺失 -> NaN, 不伪造)。"""
    pairs = {"delta_r2": ("R2", None), "delta_rmse": ("RMSE", None), "delta_mae": ("MAE", None),
             "delta_pearson": ("Pearson", None), "delta_spearman": ("Spearman", None)}
    out: Dict[str, float] = {}
    for name, (metric, _) in pairs.items():
        pv, cv = parent.get(metric), child.get(metric)
        out[name] = float(cv) - float(pv) if (_finite(pv) and _finite(cv)) else float("nan")
    return out


def build_environment_edges(table: pd.DataFrame, threshold: float = 10.0,
                            nodes: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """32 条理论边的观测值; 每条边独立配对同 cohort (split, cell, model, seed)。"""
    rows: List[Dict] = []
    node_lookup = {}
    if nodes is not None and not nodes.empty:
        for _, n in nodes.iterrows():
            node_lookup[(n["split_type"], n["cell_line"], n["model"], n["combination"])] = n

    for (split, cell, model), group in _groups(table):
        by_env: Dict[str, List[pd.Series]] = {}
        for _, r in group.iterrows():
            by_env.setdefault(str(r["environment"]).lower(), []).append(r)

        for tedge in theoretical_edges():
            parent_combo = tedge["parent_combination"]
            child_combo = tedge["child_combination"]
            added = tedge["added_environment"]

            parent_rows = by_env.get(parent_combo, [])
            child_rows = by_env.get(child_combo, [])

            def _valid_map(rs: Sequence[pd.Series]) -> Tuple[Dict, List[str]]:
                keep, bad = {}, []
                for r in rs:
                    ok, why = experiment_validity(r, threshold)
                    if not ok:
                        bad.append(why)
                        continue
                    seed = r.get("random_seed")
                    if seed in keep:
                        bad.append("duplicate_seed")
                        continue
                    keep[seed] = r
                return keep, bad

            pvalid, pbad = _valid_map(parent_rows)
            cvalid, cbad = _valid_map(child_rows)
            paired = sorted(set(pvalid) & set(cvalid), key=lambda s: str(s))

            warnings: List[str] = []
            if pbad or cbad:
                warnings.append("INVALID_EXPERIMENTS_EXCLUDED")
            row = {"split_type": split, "cell_line": cell, "model": model,
                   "parent_combination": parent_combo, "child_combination": child_combo,
                   "added_environment": added}
            if not paired:
                if not parent_rows:
                    reason = f"parent_unavailable({parent_combo})"
                elif not child_rows:
                    reason = f"child_unavailable({child_combo})"
                elif not pvalid:
                    reason = "parent_all_invalid"
                elif not cvalid:
                    reason = "child_all_invalid"
                else:
                    reason = "no_paired_seed"
                row.update({k: float("nan") for k in
                            ("parent_r2", "child_r2", "delta_r2", "parent_rmse", "child_rmse",
                             "delta_rmse", "parent_mae", "child_mae", "delta_mae",
                             "delta_pearson", "delta_spearman", "ci_low", "ci_high",
                             "ci_excludes_zero", "n_bootstrap")})
                row["bootstrap_status"] = "unavailable"
                row.update({"n_paired": 0, "status": "unavailable",
                            "warning": ";".join(warnings + [reason, "CI_UNAVAILABLE"])})
                rows.append(row)
                continue

            deltas = [_edge_deltas(pvalid[s], cvalid[s]) for s in paired]
            ddf = pd.DataFrame(deltas)
            inconsistent = any(
                (_finite(d["delta_r2"]) and _finite(d["delta_rmse"])) and
                ((d["delta_r2"] > 0 and d["delta_rmse"] > 0) or
                 (d["delta_r2"] < 0 and d["delta_rmse"] < 0))
                for d in deltas)
            if inconsistent:
                warnings.append("METRIC_INCONSISTENCY")
            warnings.append("CI_UNAVAILABLE")   # bootstrap runner 未接通: 不伪造 CI
            row.update({
                "parent_r2": float(pd.to_numeric(pd.Series([pvalid[s].get("R2") for s in paired]),
                                                errors="coerce").mean()),
                "child_r2": float(pd.to_numeric(pd.Series([cvalid[s].get("R2") for s in paired]),
                                               errors="coerce").mean()),
                "parent_rmse": float(pd.to_numeric(pd.Series([pvalid[s].get("RMSE") for s in paired]),
                                                  errors="coerce").mean()),
                "child_rmse": float(pd.to_numeric(pd.Series([cvalid[s].get("RMSE") for s in paired]),
                                                 errors="coerce").mean()),
                "parent_mae": float(pd.to_numeric(pd.Series([pvalid[s].get("MAE") for s in paired]),
                                                 errors="coerce").mean()),
                "child_mae": float(pd.to_numeric(pd.Series([cvalid[s].get("MAE") for s in paired]),
                                                errors="coerce").mean()),
            })
            for col in DELTA_COLUMNS:
                row[col] = (float(ddf[col].mean()) if col in ddf.columns and
                            ddf[col].notna().any() else float("nan"))
            row.update({"ci_low": float("nan"), "ci_high": float("nan"),
                        "ci_excludes_zero": None, "n_bootstrap": 0,
                        "bootstrap_status": "not_run",
                        "n_paired": int(len(paired)), "status": "ok",
                        "warning": ";".join(warnings)})
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    return out[[c for c in EDGE_COLUMNS if c in out.columns]]


def merge_edge_ci(edges: pd.DataFrame, ci_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """合入真实的 bootstrap CI (analysis.stats.tasks.edge_ci_table 输出); 没有就保持 NaN。

    绝不伪造 CI: CI 缺失时保持 NaN + warning=CI_UNAVAILABLE。
    """
    out = edges.copy()
    if ci_table is None or ci_table.empty:
        return out
    keys = ["split_type", "cell_line", "model", "parent_combination", "child_combination"]
    if not all(k in ci_table.columns for k in keys):
        return out
    if not {"ci_low", "ci_high"}.issubset(ci_table.columns):
        return out
    extra = [c for c in ("ci_excludes_zero", "n_bootstrap", "bootstrap_status")
             if c in ci_table.columns]
    ci = ci_table[keys + ["ci_low", "ci_high"] + extra].copy()
    drop = [c for c in ["ci_low", "ci_high"] + extra if c in out.columns]
    out = out.drop(columns=drop).merge(ci, on=keys, how="left")
    has_ci = out["ci_low"].notna() | out["ci_high"].notna()
    warnings = (out["warning"].fillna("") if "warning" in out.columns
                else pd.Series("", index=out.index)).astype(str)
    with_ci = warnings.str.replace("CI_UNAVAILABLE", "CI_AVAILABLE", regex=False)
    out["warning"] = np.where(has_ci.to_numpy(), with_ci, warnings).tolist()
    for col in ("n_bootstrap",):
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    if "bootstrap_status" in out.columns:
        out["bootstrap_status"] = out["bootstrap_status"].fillna("unavailable")
    return out


def load_edge_ci(tables_dir: str | Path) -> Optional[pd.DataFrame]:
    """从**保留下来的** `bootstrap_results.csv` 抽出 ΔR² 的边级 CI。

    2026-09-25：原先读取独立的 `environment_bootstrap.csv`（它只是
    `bootstrap_results.csv[metric=='R2']` 的 10 列投影，属完全冗余）已删除，
    改为在此处**就地过滤 + 改名**，语义与原 `stats.tasks.edge_ci_table()` 完全一致：

        metric == "R2"  ->  excludes_zero→ci_excludes_zero, status→bootstrap_status

    找不到文件或没有 R2 行时返回 None（调用方保持 CI 列为 NaN 并标 CI_UNAVAILABLE，
    绝不伪造 CI）。
    """
    root = Path(tables_dir)
    for name in ("bootstrap_results.csv",):
        p = root / name
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if df.empty or "metric" not in df.columns:
            continue
        r2 = df[df["metric"] == "R2"].copy()
        if r2.empty:
            continue
        r2 = r2.rename(columns={"excludes_zero": "ci_excludes_zero",
                                "status": "bootstrap_status"})
        keep = ["split_type", "cell_line", "model", "parent_combination",
                "child_combination", "ci_low", "ci_high", "ci_excludes_zero",
                "n_bootstrap", "bootstrap_status"]
        return r2[[c for c in keep if c in r2.columns]]
    return None


# ---------------------------------------------------------------------------
# 3) Environment 交互 (ablation view 已移出交付范围 2026-09-25)
# ---------------------------------------------------------------------------
def build_environment_interactions(table: pd.DataFrame) -> pd.DataFrame:
    """成对交互 (基于 factorial edges, 不从任何 canonical tree 猜测)。

    I(a,b) = 在含 b 背景上的 Δ(a|·) 均值 − 在不含 b 背景上的 Δ(a|·) 均值 (跨 seed 平均)。
    """
    raw = compute_conditional_increments(table)
    if raw.empty:
        return pd.DataFrame(columns=["split_type", "cell_line", "model", "factor_a", "factor_b",
                                     "interaction_r2", "n_seeds", "n_with_b", "n_without_b"])
    from analysis.environment.incremental_effect import compute_pair_interactions
    per_seed = compute_pair_interactions(raw)
    if per_seed.empty:
        return pd.DataFrame(columns=["split_type", "cell_line", "model", "factor_a", "factor_b",
                                     "interaction_r2", "n_seeds", "n_with_b", "n_without_b"])
    agg = per_seed.groupby(["split_type", "cell_line", "model", "factor_a", "factor_b"],
                           dropna=False).agg(
        interaction_r2=("interaction_r2", "mean"),
        n_seeds=("interaction_r2", "size"),
        n_with_b=("n_with_b", "mean"),
        n_without_b=("n_without_b", "mean"),
    ).reset_index()
    agg["interaction_r2"] = agg["interaction_r2"].round(6)
    return agg


# ---------------------------------------------------------------------------
# 4) 报告
# ---------------------------------------------------------------------------
def build_dag_report(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """每个 (split, cell, model) 一行的完整性/缺失/异常统计。"""
    rows: List[Dict] = []
    if nodes is None or nodes.empty:
        return pd.DataFrame(columns=["split_type", "cell_line", "model",
                                     "theoretical_nodes", "observed_nodes", "unavailable_nodes",
                                     "theoretical_edges", "valid_edges", "unavailable_edges",
                                     "metric_inconsistency_edges", "invalid_experiments",
                                     "unavailable_combinations"])
    n_theo_nodes, n_theo_edges = len(theoretical_nodes()), len(theoretical_edges())
    for (split, cell, model), nsub in nodes.groupby(["split_type", "cell_line", "model"],
                                                    dropna=False):
        esub = edges[(edges["split_type"] == split) & (edges["cell_line"] == cell) &
                     (edges["model"] == model)] if not edges.empty else edges
        n_ok_nodes = int((nsub["status"] == "ok").sum())
        # 理论边数是「每个 (parent, child) 一条」；而 edges 表在 mixed 划分下**每个种子一行**
        # （4 seed -> 4 行/边）。所以这里必须**按 (parent, child) 去重**再与理论值比较，
        # 否则 valid_edges 会超过 theoretical_edges，unavailable_edges 变成负数
        # （修前 mixed 为 128-32 = -96）。
        edge_keys = [k for k in ("parent_combination", "child_combination") if k in esub.columns]
        if esub.empty:
            n_ok_edges = n_bad_edges = 0
        else:
            ok_edges = esub[esub["status"] == "ok"]
            n_ok_edges = (int(ok_edges[edge_keys].drop_duplicates().shape[0])
                          if edge_keys else int(len(ok_edges)))
            bad = esub["warning"].fillna("").str.contains("METRIC_INCONSISTENCY")
            bad_edges = esub[bad]
            n_bad_edges = (int(bad_edges[edge_keys].drop_duplicates().shape[0])
                           if edge_keys else int(len(bad_edges)))
        rows.append({
            "split_type": split, "cell_line": cell, "model": model,
            "theoretical_nodes": n_theo_nodes, "observed_nodes": n_ok_nodes,
            "unavailable_nodes": n_theo_nodes - n_ok_nodes,
            "theoretical_edges": n_theo_edges, "valid_edges": n_ok_edges,
            "unavailable_edges": n_theo_edges - n_ok_edges,
            "metric_inconsistency_edges": n_bad_edges,
            "invalid_experiments": int(nsub["n_invalid"].sum()),
            "unavailable_combinations": ", ".join(
                nsub.loc[nsub["status"] != "ok", "combination"].astype(str).tolist()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5) 编排 (数据 -> 表 -> 渲染)
# ---------------------------------------------------------------------------
def build_factorial_dag_tables(batch_dir: str | Path, threshold: Optional[float] = None,
                               tables_dir: Optional[str | Path] = None
                               ) -> Dict[str, pd.DataFrame]:
    """从统一 normalized experiment table 生成 nodes/edges/interactions。"""
    from analysis.config import AnalysisConfig
    from analysis.data.loaders import load_experiment_table

    thr = float(threshold if threshold is not None else
                AnalysisConfig().consensus.unstable_effect_threshold)
    table = load_experiment_table(batch_dir)
    nodes = build_environment_nodes(table, threshold=thr)
    edges = build_environment_edges(table, threshold=thr, nodes=nodes)
    from analysis.data.loaders import resolve_summary_root
    ci_dir = (Path(tables_dir) if tables_dir
              else resolve_summary_root(batch_dir) / "tables")
    edges = merge_edge_ci(edges, load_edge_ci(ci_dir))
    interactions = build_environment_interactions(table)
    return {"nodes": nodes, "edges": edges,
            "interactions": interactions, "threshold": thr, "table": table}


def write_factorial_dag_artifacts(batch_dir: str | Path, tables_dir: str | Path,
                                  threshold: Optional[float] = None) -> Dict[str, str]:
    """写出 environment_{nodes,edges,interactions,dag_report}.csv。

    2026-09-25 定稿：`environment_ablation.csv` 与 `environment_dag_report.md`
    已移出交付范围，不再产出（ablation 程序与 DAG 报告 builder 一并删除）。
    """
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    built = build_factorial_dag_tables(batch_dir, threshold=threshold, tables_dir=tables_dir)
    nodes, edges = built["nodes"], built["edges"]
    report = build_dag_report(nodes, edges)

    paths = {
        "nodes": str(tables_dir / "environment_nodes.csv"),
        "edges": str(tables_dir / "environment_edges.csv"),
        "interactions": str(tables_dir / "environment_interactions.csv"),
        "report_csv": str(tables_dir / "environment_dag_report.csv"),
    }
    nodes.to_csv(paths["nodes"], index=False)
    edges.to_csv(paths["edges"], index=False)
    built["interactions"].to_csv(paths["interactions"], index=False)
    report.to_csv(paths["report_csv"], index=False)

    return paths

