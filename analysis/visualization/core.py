"""analysis.visualization.core — 图表样式/保存基元。"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def style_figure() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False


def save_figure(fig, path: str | Path, dpi: int = 150) -> str:
    p = Path(path)
    fig.tight_layout()
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(p)
