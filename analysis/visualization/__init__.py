"""analysis.visualization — 证据引擎图表 (分主题模块; 300dpi PNG)。

原则: 图只从统一 tables/DataFrame 重建, 不自行扫描实验目录。
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import sys
from pathlib import Path

# --- 项目根引导: 保证从任意工作目录直接运行/被导入都能解析 analysis、core ---
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from analysis.visualization.core import save_figure, style_figure  # noqa: E402
from analysis.visualization import attribution_plots, environment_plots  # noqa: E402


def render_all(figures_dir: str | Path,
               conditional_summary: Optional[pd.DataFrame] = None,
               main_effects: Optional[pd.DataFrame] = None,
               attribution_table: Optional[pd.DataFrame] = None,
               significance_table: Optional[pd.DataFrame] = None) -> list:
    """渲染 `figures/` 下由本包负责的两个主题（缺数据则跳过）。

    交付结构（2026-09-25 定稿）：

        figures/environment/   本函数 + analysis.visualization.factorial_dag 产出
        figures/plots/         位置归因热图（position × channel，per method）
        （figures/motif/ 已随人为标签体系移出；motif 现在只产数据表）

    `attribution_table` 即 `tables/attribution_summary.csv` 的内存副本
    （同一张表也是 motif discovery 的输入，见 analysis/pipeline.py）。
    `significance_table` 即 `feature_importance/key_regulatory_biomarkers.csv`，
    供位置归因热图做**显示门**（FDR>=0.10 / SNR<0.8 的格子留白）；
    为 None 时该门自动关闭，不影响其它图。
    02_prediction / 05_cellline / 06_evidence 三个主题（performance_plots /
    cellline_plots / evidence_plots）已移出交付范围，在 `Delete/_deprecated_figures/`；
    其中 cellline/evidence 的数据表也已随人为标签体系一并移出。
    """
    out = Path(figures_dir)
    style_figure()
    rendered: list = []

    env_dir = out / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    if conditional_summary is not None and not conditional_summary.empty:
        rendered += environment_plots.render(conditional_summary, main_effects, env_dir)

    if attribution_table is not None and not attribution_table.empty:
        rendered += attribution_plots.render(                                # -> out/plots/
            attribution_table, out, significance_table=significance_table)
    return rendered
