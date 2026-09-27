"""analysis.stats.multiple_testing — BH-FDR (Benjamini-Hochberg)。

注意: 只允许对"同一假设检验 family"校正; 线性系数、ANOVA、motif enrichment
分属不同 family, 不得混入同一个 FDR。
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np


def bh_fdr(p_values: Iterable[Optional[float]]) -> List[Optional[float]]:
    """
    Benjamini-Hochberg FDR (q 值单调非降, 数值兼容 statsmodels.multipletests)。

    返回与输入等长的 q 列表; 非有限 p 原位 NaN。
    """
    p = np.asarray([float(x) if x is not None else np.nan for x in p_values], dtype=np.float64)
    out = np.full(p.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(p)
    idx = np.arange(len(p))[finite]
    vals = p[finite]
    if vals.size == 0:
        return out.tolist()
    order = np.argsort(vals)
    ranked = vals[order]
    m = len(ranked)
    q = np.empty(m)
    min_q = 1.0
    for i in range(m - 1, -1, -1):
        min_q = min(min_q, ranked[i] * m / (i + 1))
        q[i] = min_q
    out[idx[order]] = np.minimum(q, 1.0)
    return out.tolist()
