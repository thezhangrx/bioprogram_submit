"""analysis.visualization.factorial_dag — Environment Factorial DAG 渲染 (纯 DataFrame)。

只消费 tables/environment_nodes.csv + tables/environment_edges.csv,
**不扫描实验目录, 不重新解析 metadata** (数据流见 analysis.environment.factorial_dag)。

主图: figures/03_environment/factorial_dag/{split}_{cell}_{model}.png
      Node = environment combination; Edge = +一个环境因子 (conditional ΔR²)
      颜色 = effect direction (green>0 / red<0 / grey=unavailable), **不是显著性**
辅助: figures/03_environment/conditional_delta_r2/{split}_{cell}_{model}.png
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import sys
from pathlib import Path

# --- 项目根引导: 保证从任意工作目录直接运行/被导入都能解析 analysis、core ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

from analysis.environment.factorial_dag import (  # noqa: E402
    ENV_FACTORS, FACTOR_LABELS, combination_label, parents_of)

BOX_W, BOX_H = 2.6, 0.92
GAP_X = BOX_W + 0.55
LEVEL_GAP_Y = 2.15
COLOR_POS, COLOR_NEG, COLOR_ZERO, COLOR_NA = "#2e7d32", "#c62828", "#f9a825", "#9e9e9e"
WARN_MARK = "\u26a0"


def _safe_key(split, cell, model) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", f"{split}_{cell}_{model}").strip("_")


def _box_color(r2, lo, hi) -> str:
    """节点填充色: 以 R² 做中性渐变 (不代表显著性)。"""
    if r2 is None or not np.isfinite(r2):
        return "#eceff1"
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return "#f5f5f5"
    frac = float(np.clip((r2 - lo) / (hi - lo), 0.0, 1.0))
    cmap = plt.get_cmap("Blues")
    rgba = cmap(0.18 + 0.42 * frac)
    return mcolors.to_hex(rgba)


def _layout(combos_by_level: Dict[int, List[str]],
            parents: Dict[str, List[str]]) -> Dict[str, Tuple[float, float]]:
    """Sugiyama-lite: 层内 barycenter 排序 + 均匀铺开 (correctness > symmetry)。"""
    levels = sorted(combos_by_level)
    pos: Dict[str, Tuple[float, float]] = {}
    order = {lvl: sorted(combos_by_level[lvl]) for lvl in levels}
    x = {}
    for lvl in levels:
        n = len(order[lvl])
        for i, c in enumerate(order[lvl]):
            x[c] = (i - (n - 1) / 2.0) * GAP_X

    def sweep(upward: bool) -> None:
        seq = levels[1:] if upward else list(reversed(levels[:-1]))
        for lvl in seq:
            neighbour_lvl = lvl - 1 if upward else lvl + 1
            anchor = {}
            for c in order[lvl]:
                rel = parents[c] if upward else [k for k, v in parents.items() if c in v]
                vals = [x[p] for p in rel if p in x]
                anchor[c] = float(np.mean(vals)) if vals else x[c]
            order[lvl] = sorted(order[lvl], key=lambda c: (anchor[c], c))
            n = len(order[lvl])
            # 保持整体质心, 避免层间整体漂移
            shift = float(np.mean([x[c] for c in order[lvl]])) if order[lvl] else 0.0
            for i, c in enumerate(order[lvl]):
                x[c] = (i - (n - 1) / 2.0) * GAP_X + shift * 0.0

    for _ in range(4):
        sweep(True)
        sweep(False)

    for lvl in levels:
        n = len(order[lvl])
        for i, c in enumerate(order[lvl]):
            y = -float(lvl) * LEVEL_GAP_Y
            pos[c] = ((i - (n - 1) / 2.0) * GAP_X, y)
    return pos


def render_factorial_dag(nodes: pd.DataFrame, edges: pd.DataFrame,
                         figures_dir: str | Path,
                         orientation: str = "sequence_top",
                         groups: Optional[Sequence[Tuple]] = None) -> List[str]:
    """渲染 Environment Factorial DAG (每个 split/cell/model 一张)。

    orientation: "sequence_top" (level 0 在上, 与 lattice 的层级定义一致)
                 或 "all_top" (经典 lattice 方向, ALL 在上)。
    """
    out_dir = Path(figures_dir) / "factorial_dag"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if nodes is None or nodes.empty:
        return written

    keys = ["split_type", "cell_line", "model"]
    group_list = list(nodes.groupby(keys, dropna=False)) if groups is None else [
        (g, nodes[(nodes["split_type"] == g[0]) & (nodes["cell_line"] == g[1]) &
                  (nodes["model"] == g[2])]) for g in groups]

    for (split, cell, model), nsub in group_list:
        if nsub.empty or (nsub["status"] == "ok").sum() == 0:
            continue
        esub = edges[(edges["split_type"] == split) & (edges["cell_line"] == cell) &
                     (edges["model"] == model)] if not edges.empty else edges.iloc[0:0]

        combos_by_level: Dict[int, List[str]] = {}
        info: Dict[str, Dict] = {}
        for _, r in nsub.iterrows():
            lvl = int(r["environment_count"])
            combos_by_level.setdefault(lvl, []).append(str(r["combination"]))
            info[str(r["combination"])] = r.to_dict()
        parents = {c: parents_of(c) for c in info}
        pos = _layout(combos_by_level, parents)
        if orientation == "all_top":
            max_lvl = max(combos_by_level)
            pos = {c: (xy[0], -float(max_lvl - int(info[c]["environment_count"])) * LEVEL_GAP_Y)
                   for c, xy in pos.items()}

        r2_vals = [float(info[c]["r2"]) for c in info if np.isfinite(info[c]["r2"])]
        lo, hi = (min(r2_vals), max(r2_vals)) if r2_vals else (np.nan, np.nan)
        abs_d = pd.to_numeric(esub.loc[esub["status"] == "ok", "delta_r2"],
                              errors="coerce").abs().dropna()
        scale = float(abs_d.quantile(0.90)) if len(abs_d) else 1.0
        scale = scale if scale > 0 else 1.0

        fig_w = max(13.0, (max(len(v) for v in combos_by_level.values()) - 1) * GAP_X + 6.0)
        n_levels = max(combos_by_level) + 1
        fig_h = max(10.0, (n_levels - 1) * LEVEL_GAP_Y + 4.6)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
        ax.axis("off")

        has_warn = False
        for _, e in esub.iterrows():
            p, c = str(e["parent_combination"]), str(e["child_combination"])
            if p not in pos or c not in pos:
                continue
            x0, y0 = pos[p]
            x1, y1 = pos[c]
            d = e.get("delta_r2")
            warning = str(e.get("warning") or "")
            warn = "METRIC_INCONSISTENCY" in warning
            has_warn = has_warn or warn
            if not np.isfinite(d):
                color, lw, ls = COLOR_NA, 0.8, (0, (3, 2))
            else:
                color = COLOR_POS if d > 0 else (COLOR_NEG if d < 0 else COLOR_ZERO)
                lw = 0.7 + 2.3 * float(min(abs(d) / scale, 1.0))
                ls = (0, (4, 2)) if warn else "-"
            ax.plot([x0, x1], [y0 - BOX_H / 2, y1 + BOX_H / 2], color=color, lw=lw,
                    ls=ls, alpha=0.85, zorder=2, solid_capstyle="round")
            # 边标签: +ENV / ΔR² (+ ⚠); 沿边错开高度以减少重叠
            if np.isfinite(d):
                plist = sorted(parents_of(c))
                t = 0.38 + 0.11 * (plist.index(p) if p in plist else 0)
                t = min(t, 0.72)
                lx, ly = x0 + (x1 - x0) * t, (y0 - BOX_H / 2) + ((y1 + BOX_H / 2) - (y0 - BOX_H / 2)) * t
                label = f"+{FACTOR_LABELS.get(str(e['added_environment']), str(e['added_environment']))}\nΔR² {float(d):+.3f}"
                if warn:
                    label += f" {WARN_MARK}"
                ax.text(lx, ly, label, ha="center", va="center", fontsize=4.9, color="#212121",
                        zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                            ec=color, lw=0.4, alpha=0.85))
            else:
                lx, ly = (x0 + x1) / 2.0, (y0 + y1) / 2.0
                ax.text(lx, ly, f"+{FACTOR_LABELS.get(str(e['added_environment']), '?')}\nunavailable",
                        ha="center", va="center", fontsize=4.6, color="#757575", zorder=5,
                        bbox=dict(boxstyle="round,pad=0.10", fc="white", ec=COLOR_NA, lw=0.3))

        for combo, xy in pos.items():
            r = info[combo]
            x, y = xy
            face = _box_color(r.get("r2"), lo, hi)
            lvl = int(r["environment_count"])
            edge_color = "#37474f" if lvl == 0 else "#546e7a"
            lw_box = 1.8 if lvl == 0 else 0.9
            ax.add_patch(FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H,
                                        boxstyle="round,pad=0.02,rounding_size=0.12",
                                        facecolor=face, edgecolor=edge_color, linewidth=lw_box,
                                        zorder=4))
            name = combination_label(combo)
            if str(r.get("status")) == "ok":
                body = (f"{name}\nR² {r['r2']:.3f}  RMSE {r['rmse']:.3f}\n"
                        f"MAE {r['mae']:.3f}  n={int(r['eligible_count'])}")
                if r.get("n_invalid"):
                    body += " *"
            else:
                body = f"{name}\nR²: unavailable\nobserved n={int(r['sample_count'])}"
            ax.text(x, y, body, ha="center", va="center", fontsize=6.2, zorder=6,
                    color="#1a1a1a" if str(r.get("status")) == "ok" else "#616161")

        ax.set_xlim(-fig_w / 2.0 + 0.6, fig_w / 2.0 - 0.6)
        y_min = min(y for _, y in pos.values()) - BOX_H
        y_max = max(y for _, y in pos.values()) + BOX_H + 1.5
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Environment Factorial DAG (2^{len(ENV_FACTORS)} lattice) | "
                     f"{str(split).upper()} | {str(cell).upper()} | {model}",
                     fontsize=12, fontweight="bold", pad=18)
        n_ok_edges = int((esub["status"] == "ok").sum()) if not esub.empty else 0
        ax.text(0, y_max - 0.35,
                "Node = environment combination | directed edge = adding one environmental factor | "
                f"edge value = conditional ΔR² (paired cohort) | nodes {int((nsub['status'] == 'ok').sum())}/"
                f"{len(nsub)} observed, edges {n_ok_edges}/{len(esub)} valid of theory | "
                f"colour = effect direction (NOT significance){' | ' + WARN_MARK + ' metric inconsistency' if has_warn else ''}",
                ha="center", fontsize=8, color="#37474f")
        ax.text(0, y_max - 0.72,
                "* = invalid experiments excluded from this node (diverged / missing / correlation out of range); "
                "grey dashed = unavailable (kept in the theoretical factorial structure)",
                ha="center", fontsize=7, color="#78909c")

        handles = [Patch(facecolor="#dce9f5", edgecolor="#546e7a", label="node (fill = R² level)"),
                   plt.Line2D([], [], color=COLOR_POS, lw=2, label="ΔR² > 0"),
                   plt.Line2D([], [], color=COLOR_NEG, lw=2, label="ΔR² < 0"),
                   plt.Line2D([], [], color=COLOR_ZERO, lw=2, label="ΔR² = 0"),
                   plt.Line2D([], [], color=COLOR_NA, lw=1, ls=(0, (3, 2)), label="unavailable"),
                   plt.Line2D([], [], color="#c62828", lw=1, ls=(0, (4, 2)),
                              label=f"{WARN_MARK} METRIC_INCONSISTENCY (edge kept)")]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
                  fontsize=8, frameon=False)

        fig.tight_layout()
        path = out_dir / f"{_safe_key(split, cell, model)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def render_conditional_delta_r2(edges: pd.DataFrame, figures_dir: str | Path) -> List[str]:
    """辅助图: 每个 group 的全部 conditional ΔR² 排序条形图 (含 unavailable 计数)。"""
    out_dir = Path(figures_dir) / "conditional_delta_r2"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if edges is None or edges.empty:
        return written

    keys = ["split_type", "cell_line", "model"]
    for (split, cell, model), esub in edges.groupby(keys, dropna=False):
        ok = esub[esub["status"] == "ok"].copy()
        if ok.empty:
            continue
        ok["_label"] = [f"{combination_label(p)}  +{FACTOR_LABELS.get(a, a)}"
                        for p, a in zip(ok["parent_combination"], ok["added_environment"])]
        ok = ok.sort_values("delta_r2")
        colors = [COLOR_POS if d > 0 else (COLOR_NEG if d < 0 else COLOR_ZERO)
                  for d in ok["delta_r2"]]
        fig_h = max(4.5, 0.34 * len(ok) + 2.2)
        fig, ax = plt.subplots(figsize=(10.5, fig_h), dpi=150)
        y = np.arange(len(ok))
        ax.barh(y, ok["delta_r2"], color=colors, alpha=0.85, edgecolor="#455a64", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(ok["_label"], fontsize=7)
        for yi, d in zip(y, ok["delta_r2"]):
            ax.annotate(f"{float(d):+.4f}", (float(d), yi), textcoords="offset points",
                        xytext=(4 if d >= 0 else -4, 0), va="center",
                        ha="left" if d >= 0 else "right", fontsize=6)
        ax.margins(x=0.16)
        ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.7)
        n_na = int((esub["status"] != "ok").sum())
        n_inc = int(esub["warning"].fillna("").str.contains("METRIC_INCONSISTENCY").sum())
        ax.set_title(f"Conditional ΔR² per environment edge | {str(split).upper()} | "
                     f"{str(cell).upper()} | {model}\nvalid edges {len(ok)}/{len(esub)}"
                     + (f" | unavailable {n_na}" if n_na else "")
                     + (f" | {WARN_MARK} inconsistency {n_inc}" if n_inc else ""),
                     fontsize=10)
        ax.set_xlabel("ΔR² = R²(S ∪ {e}) − R²(S)   (paired cohort; direction, not significance)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{_safe_key(split, cell, model)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written


def render_environment_interactions(interactions: pd.DataFrame,
                                    figures_dir: str | Path) -> List[str]:
    """成对交互 I(a,b) (基于 factorial edges, 不来自任何 canonical tree)。"""
    out_dir = Path(figures_dir) / "interactions"
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[str] = []
    if interactions is None or interactions.empty:
        return written
    keys = ["split_type", "cell_line", "model"]
    for (split, cell, model), sub in interactions.groupby(keys, dropna=False):
        sub = sub.sort_values("interaction_r2")
        labels = [f"{FACTOR_LABELS.get(a, a)} × {FACTOR_LABELS.get(b, b)}"
                  for a, b in zip(sub["factor_a"], sub["factor_b"])]
        colors = [COLOR_POS if d > 0 else (COLOR_NEG if d < 0 else COLOR_ZERO)
                  for d in sub["interaction_r2"]]
        fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
        y = np.arange(len(sub))
        ax.barh(y, sub["interaction_r2"], color=colors, alpha=0.85,
                edgecolor="#455a64", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        for yi, d in zip(y, sub["interaction_r2"]):
            ax.annotate(f"{float(d):+.4f}", (float(d), yi), textcoords="offset points",
                        xytext=(4 if d >= 0 else -4, 0), va="center",
                        ha="left" if d >= 0 else "right", fontsize=6.5)
        ax.margins(x=0.16)
        ax.axvline(0, color="k", lw=0.9, ls="--", alpha=0.7)
        ax.set_title(f"Pairwise environment interaction (factorial) | {str(split).upper()} | "
                     f"{str(cell).upper()} | {model}\n"
                     f"I(a,b) = Δ(a|S+b) − Δ(a|S) averaged over backgrounds (paired seeds)",
                     fontsize=9.5)
        ax.set_xlabel("interaction ΔR² (not a significance test)")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        path = out_dir / f"{_safe_key(split, cell, model)}.png"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(str(path))
    return written
