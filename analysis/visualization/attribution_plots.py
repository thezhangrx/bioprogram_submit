"""attribution_plots — 位置归因热图（position × channel，per method；magnitude）。

**输出路径**：`<batch>/figures/plots/`（每个 attribution method 一张 PNG）。
即：调用方传入 figures 根目录，本模块自己在其下建 `plots/`。

形状：index = channel（DeepCRISPR 8 个：A/C/G/T/CTCF/Dnase/H3K4me3/RRBS），
      columns = position（1-based，23 个）→ 一张 8 × 23 的热图。
数值：`attribution_summary.csv` 的 `importance`，按 (channel, position) 取均值。

显示规则（**只决定"画什么"，不参与任何统计判定**）
------------------------------------------------
1. **不显著的格子不显示**：格子对应的 feature 若
   ``FDR >= 0.10``（linear）或 ``SNR < 0.8``（非线性），该色块**留白**而不是画出来。
   判据直接取原始数值，**不调用星级函数**（``get_sig_symbol`` /
   ``get_snr_significance_code``），避免把"画图"和"贴标签"耦在一起。
   数值来自 `feature_importance/key_regulatory_biomarkers.csv`。
2. **色域用稳健分位**，不用极值撑开：``vmin/vmax`` 取已显示格子的
   ``robust_low_quantile / robust_high_quantile`` 分位（默认 1% / 99%）。
   超出色域的格子**直接涂黑**（``cmap.set_over/under``），
   因此像 ``linear_coefficient`` 里 ±4.5e11 的异常系数不会把正常量级的格子压成一片同色。
3. 留白的格子**保留网格位置**，所以仍能看出"哪个位点/通道不显著"。

**输入**：
  * `tables/attribution_summary.csv`（或内存中的同一张表）；
  * `feature_importance/key_regulatory_biomarkers.csv`（显著性/稳健性；可选，
    缺失时不做显示门并在日志中说明）。
**产出**：`figures/plots/position_attribution_<method>.png`，method ∈
      cnn_ig / cnn_ism / mlp_ig / transformer_attention /
      xgboost_gain / xgboost_treeshap / linear_coefficient

**行数为何不总是 8**：环境通道是**按需置零**而非删除，所以 7 个方法里只有
`linear_coefficient` 少一行 —— 因为线性模型剔除全部 `*_T` 参照通道
（哑变量陷阱防护，见 README §16.1），因此**没有 `T` 行**，是 **7 × 23**；
其余 6 个方法均为 **8 × 23**。若某方法的某通道全为 0/缺失，该行会自然消失。
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

# --- 项目根引导: 保证从任意工作目录直接运行/被导入都能解析 analysis、core ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

from analysis.visualization.core import save_figure  # noqa: E402

LOGGER = logging.getLogger("attribution_plots")

#: 通道顺序，**全部大写归一**。
#: 注意大小写的实际分布：`attribution_summary.csv` 的 `channel` 列是**混合大小写**
#: （`CTCF` / `Dnase` / `H3K4me3` / `RRBS`，见 2026-09-26 实测），
#: 与本表的大写写法**不一致**。原实现直接与本表 `.isin()` 比对，
#: 导致 `Dnase` / `H3K4me3` 两个通道被**静默丢弃**（热图只剩 6 行而不是 8 行）。
#: 修法：本表统一用大写，且比对前先对数据做一次 `.str.upper()` 归一（大小写无关）。
CHANNEL_ORDER = ["A", "C", "G", "T", "CTCF", "DNASE", "H3K4ME3", "RRBS"]

#: 轴标签的显示写法（只影响图上的文字，不影响匹配）
CHANNEL_DISPLAY = {"DNASE": "Dnase", "H3K4ME3": "H3k4me3"}

#: 交付结构里本主题的固定子目录名：`<figures>/plots/`
OUTPUT_SUBDIR = "plots"

#: attribution method -> 显著性特征库里对应的模型名。
#: CNN 的三个 kernel 变体共用一个 method（`cnn_ig` / `cnn_ism`），
#: 因为特征库是按 `cnn33/53/73` 分开记的，任一变体达标即可显示。
METHOD_TO_SIGNIFICANCE_MODELS: Mapping[str, Tuple[str, ...]] = {
    "linear_coefficient": ("linear",),
    "xgboost_gain": ("xgboost",),
    "xgboost_treeshap": ("xgboost",),
    "mlp_ig": ("mlp",),
    "transformer_attention": ("transformer",),
    "cnn_ig": ("cnn33", "cnn53", "cnn73"),
    "cnn_ism": ("cnn33", "cnn53", "cnn73"),
}


@dataclass(frozen=True)
class AttributionPlotStyle:
    """位置归因热图的**显示**参数（与统计判定无关）。

    想调整"什么该画、什么该涂黑"改这里即可；也可以在调用 ``render`` 时
    用 ``style=replace(AttributionPlotStyle(), ...)`` 传一份临时样式。
    """

    # ---- 显示门：不达标的格子留白 ----
    #: True = 启用显示门；False = 全部格子都画（排障用）
    require_significance: bool = True
    #: linear：FDR >= 该值 -> 不显示（默认 0.10，与星级最低档 `.` 的边界一致）
    hide_fdr_at_or_above: float = 0.10
    #: 非线性：SNR < 该值 -> 不显示（默认 0.8，与星级最低档 `.` 的边界一致）
    hide_snr_below: float = 0.8
    #: 同一 feature 有多个上下文时的口径：
    #:   "any"  = 任一上下文达标即显示（存在性口径，默认；本项目其它统计门也用存在性）
    #:   "all"  = 全部上下文都达标才显示
    #:   "mean" = 达标上下文占比 >= 0.5 才显示
    #: 注：交付批次里上游特征库是**逐 feature** 过滤的，三种口径结果相同。
    gate_aggregation: str = "any"

    # ---- 色域与异常值 ----
    cmap: str = "YlGnBu"
    #: **异常值闸门**：|值| >= 该阈值的格子**直接涂黑**，并且**不参与色域计算**。
    #: 默认 10.0，与 `analysis.config.consensus.unstable_effect_threshold` 同口径
    #: （§5.1 数值稳定性：|Δ| >= 10 判为数值不稳定）。
    #: 这正是 `linear_coefficient` 需要的：30,912 个系数里有 14% ≥ 10、4% ≥ 1e9，
    #: 会把 (channel, position) 的**均值**从 ~1e-2 抬到 ~1e9，导致整张图一片同色。
    #: 设 None 关闭该闸门（退回"只按分位截断"）。
    outlier_abs_threshold: Optional[float] = 10.0
    #: 色域取**非异常**格子的分位（稳健），而不是 min/max
    robust_low_quantile: float = 0.01
    robust_high_quantile: float = 0.99
    #: 越界格子的颜色（"直接显示为黑色"）
    outlier_color: str = "black"
    #: 在标题里注明色域与越界格数，避免读者误判色标
    annotate_range: bool = True

    def __post_init__(self) -> None:
        if self.gate_aggregation not in ("any", "all", "mean"):
            raise ValueError("gate_aggregation 只能是 'any' / 'all' / 'mean'")
        if not 0.0 <= self.robust_low_quantile < self.robust_high_quantile <= 1.0:
            raise ValueError("robust_low_quantile / robust_high_quantile 必须满足 0<=lo<hi<=1")
        if self.outlier_abs_threshold is not None and float(self.outlier_abs_threshold) <= 0:
            raise ValueError("outlier_abs_threshold 必须 > 0 或为 None")


#: 交付默认样式
DEFAULT_STYLE = AttributionPlotStyle()


class SignificanceGate:
    """把「显著性/稳健性特征库」变成"哪些 (channel, position) 该显示"的白名单。

    输入表可以是 `feature_importance/key_regulatory_biomarkers.csv`
    （中文表头：`模型(model)` / `特征名(feature)` / `信噪比(SNR/t_stat)` / `FDR校正q值`），
    也可以是任何含等价英文列的 DataFrame。

    判据**直接用原始数值**：
      * model 属于 ``linear-family``（见 ``LINEAR_MODELS``）-> 看 ``fdr``
      * 其余 -> 看 ``snr``
    **不调用** `get_sig_symbol` / `get_snr_significance_code`，即不借用星级体系。
    """

    #: 走 FDR 判据的模型名（其余走 SNR）
    LINEAR_MODELS: Tuple[str, ...] = ("linear",)

    _ALIASES: Mapping[str, Sequence[str]] = {
        "model": ("model", "模型(model)", "模型"),
        "feature": ("feature", "特征名(feature)", "特征名"),
        "snr": ("snr", "信噪比(SNR/t_stat)", "信噪比"),
        "fdr": ("fdr", "FDR校正q值", "FDR", "q_value"),
    }

    def __init__(self, style: AttributionPlotStyle = DEFAULT_STYLE) -> None:
        self.style = style
        self._keep: Dict[str, Set[Tuple[str, float]]] = {}
        self.available = False
        self.reason = ""

    # ------------------------------------------------------------------
    @classmethod
    def _column(cls, df: pd.DataFrame, key: str) -> Optional[str]:
        for name in cls._ALIASES[key]:
            if name in df.columns:
                return name
        return None

    def fit(self, significance_table: Optional[pd.DataFrame]) -> "SignificanceGate":
        if significance_table is None or significance_table.empty:
            self.reason = ("缺少显著性特征库（key_regulatory_biomarkers.csv），"
                           "本次不做显示门：所有格子都会画出")
            return self
        col = {k: self._column(significance_table, k) for k in self._ALIASES}
        if not col["model"] or not col["feature"]:
            self.reason = (f"显著性表缺少模型/特征列（实际列：{list(significance_table.columns)[:8]}…），"
                           "本次不做显示门")
            return self

        work = pd.DataFrame({
            "model": significance_table[col["model"]].astype(str).str.strip().str.lower(),
            "feature": significance_table[col["feature"]].astype(str).str.strip(),
        })
        work["snr"] = pd.to_numeric(
            significance_table[col["snr"]], errors="coerce") if col["snr"] else np.nan
        work["fdr"] = pd.to_numeric(
            significance_table[col["fdr"]], errors="coerce") if col["fdr"] else np.nan
        # feature 形如 pos13_H3K4me3 / pos1_A -> (channel, 1-based position)
        work["channel"] = work["feature"].str.rsplit("_", n=1).str[-1].str.upper()
        work["position"] = pd.to_numeric(
            work["feature"].str.extract(r"^pos(\d+)_", expand=False), errors="coerce")
        work = work.dropna(subset=["position"])
        if work.empty:
            self.reason = "显著性表的 feature 列无法解析成 pos<N>_<Channel>，本次不做显示门"
            return self

        is_linear = work["model"].isin([m.lower() for m in self.LINEAR_MODELS])
        style = self.style
        # 达标判据：linear 看 FDR（< 阈值），其余看 SNR（>= 阈值）。
        # **不调用星级函数** —— 这里只是两个明确的数值比较。
        passed = pd.Series(False, index=work.index)
        lin = is_linear & work["fdr"].notna()
        passed.loc[lin] = work.loc[lin, "fdr"] < float(style.hide_fdr_at_or_above)
        non = (~is_linear) & work["snr"].notna()
        passed.loc[non] = work.loc[non, "snr"] >= float(style.hide_snr_below)
        work["passed"] = passed

        for method, models in METHOD_TO_SIGNIFICANCE_MODELS.items():
            sub = work[work["model"].isin([m.lower() for m in models])]
            if sub.empty:
                continue
            grouped = sub.groupby(["channel", "position"])["passed"]
            if style.gate_aggregation == "all":
                keep = grouped.all()
            elif style.gate_aggregation == "mean":
                keep = grouped.mean() >= 0.5
            else:
                keep = grouped.any()
            self._keep[method] = {(str(ch), float(pos))
                                  for (ch, pos), ok in keep.items() if bool(ok)}

        self.available = True
        self.reason = (f"显示门已启用：FDR >= {style.hide_fdr_at_or_above:g}（linear）"
                       f" 或 SNR < {style.hide_snr_below:g}（非线性）的格子留白；"
                       f"口径 = {style.gate_aggregation}")
        return self

    # ------------------------------------------------------------------
    def keep_mask(self, method: str, index: Iterable[str],
                  columns: Iterable[float]) -> Optional[np.ndarray]:
        """返回 (len(index), len(columns)) 的布尔矩阵；未知 method 返回 None。"""
        keep = self._keep.get(str(method))
        if keep is None:
            return None
        return np.array([[((str(ch).upper()), float(pos)) in keep for pos in columns]
                         for ch in index], dtype=bool)


def render(attribution_table: pd.DataFrame, figures_dir: Path,
           significance_table: Optional[pd.DataFrame] = None,
           style: Optional[AttributionPlotStyle] = None) -> list:
    """把 `attribution_table` 按 method 画成 position × channel 热图。

    `figures_dir` 是 **figures 根目录**；实际写入 `figures_dir/plots/`。

    `significance_table` 给出显示门所需的 FDR / SNR
    （通常传 `feature_importance/key_regulatory_biomarkers.csv`）；
    为 None 时不做显示门，所有格子都画。
    """
    style = style or DEFAULT_STYLE
    gate = SignificanceGate(style).fit(significance_table)
    if style.require_significance:
        LOGGER.info("    [plots] %s", gate.reason)
    else:
        LOGGER.info("    [plots] require_significance=False，本次不做显示门")

    out_dir = Path(figures_dir) / OUTPUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list = []
    if not {"method", "channel", "position", "importance"}.issubset(attribution_table.columns):
        return paths
    work = attribution_table.dropna(subset=["channel", "position", "importance"]).copy()
    if work.empty:
        return paths
    work["channel"] = work["channel"].astype(str).str.strip().str.upper()
    work = work[work["channel"].isin(CHANNEL_ORDER)]
    if work.empty:
        return paths
    for method, sub in work.groupby("method", dropna=False):
        pivot = sub.pivot_table(index="channel", columns="position",
                                values="importance", aggfunc="mean")
        # 只保留本方法实际存在的通道，并按 CHANNEL_ORDER 排序
        pivot = pivot.reindex([c for c in CHANNEL_ORDER if c in pivot.index])
        cols = sorted(pivot.columns)
        pivot = pivot[cols]
        if pivot.empty or pivot.notna().sum().sum() == 0:
            continue
        pivot.index = [str(c) for c in pivot.index]

        # ---- 显示门：不达标的格子置 NaN（留白但保留网格位置）----
        n_hidden = 0
        if style.require_significance and gate.available:
            mask = gate.keep_mask(method, pivot.index, pivot.columns)
            if mask is not None:
                before = int(pivot.notna().sum().sum())
                pivot = pivot.where(mask)
                n_hidden = before - int(pivot.notna().sum().sum())
                if pivot.notna().sum().sum() == 0:
                    LOGGER.info("    [plots] %s: 全部格子都被显示门隐藏，跳过该图", method)
                    continue

        # ---- 色域：先隔离异常值（涂黑），再用非异常格子的稳健分位定色域 ----
        values = pivot.to_numpy(dtype=float)
        finite = np.isfinite(values)
        hard = np.zeros_like(values, dtype=bool)
        thr = style.outlier_abs_threshold
        if thr is not None:
            hard = finite & (np.abs(values) >= float(thr))
        normal = values[finite & ~hard]
        if normal.size == 0:                  # 全被闸门判为异常：退回用全部有限值定色域
            normal = values[finite]
        if normal.size == 0:
            continue
        lo = float(np.quantile(normal, style.robust_low_quantile))
        hi = float(np.quantile(normal, style.robust_high_quantile))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(normal.min()), float(normal.max())
            if hi <= lo:                      # 全等值：给一个最小可视区间
                hi = lo + (abs(lo) * 1e-6 + 1e-12)
        n_tail = int(((normal < lo) | (normal > hi)).sum())
        n_hard = int(hard.sum())

        cmap = plt.get_cmap(style.cmap).copy()
        cmap.set_over(style.outlier_color)    # 分位之外的尾巴 -> 黑
        cmap.set_under(style.outlier_color)

        # 轴标签用可读写法（DNASE -> Dnase），不影响匹配
        display = pivot.copy()
        display.index = [CHANNEL_DISPLAY.get(str(c), str(c)) for c in pivot.index]
        shown = display.where(~hard)          # 异常格子先挖空，下面再叠黑色
        fig, ax = plt.subplots(figsize=(11, max(3, 0.55 * display.shape[0] + 1)))
        sns.heatmap(shown, cmap=cmap, vmin=lo, vmax=hi, ax=ax, linewidths=0.4)
        if n_hard:
            # 异常格子叠一层纯黑（沿用同一套网格线与刻度标签，避免错位/丢标签）
            overlay = pd.DataFrame(np.where(hard, 1.0, np.nan),
                                   index=display.index, columns=display.columns)
            sns.heatmap(overlay, cmap=ListedColormap([style.outlier_color]),
                        vmin=0.0, vmax=1.0, cbar=False, ax=ax, linewidths=0.4,
                        xticklabels=list(display.columns),
                        yticklabels=list(display.index))
        title = (f"Position attribution magnitude — {method}"
                 f"  ({display.shape[0]} channels × {display.shape[1]} positions)")
        if style.annotate_range:
            extra = [f"color range {lo:.3g}~{hi:.3g} "
                     f"(q{style.robust_low_quantile:g}~q{style.robust_high_quantile:g})"]
            if thr is not None:
                extra.append(f"|value| >= {float(thr):g} = {style.outlier_color} ({n_hard})")
            if n_tail:
                extra.append(f"tail = {style.outlier_color} ({n_tail})")
            if style.require_significance and gate.available and n_hidden:
                extra.append(f"blank = not significant ({n_hidden})")
            title += "\n" + " | ".join(extra)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("position (1-based)")
        ax.set_ylabel("channel")
        paths.append(save_figure(fig, out_dir / f"position_attribution_{method}.png"))
    return paths
