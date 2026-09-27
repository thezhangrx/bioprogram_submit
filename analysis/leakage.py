"""analysis.leakage — canonical sample identity, overlap taxonomy 与 leakage-controlled 评估。

科学定义（本模块是唯一权威来源，其他模块不得各自实现）
----------------------------------------------------
本项目的 split 声明的是 **sequence-level generalization**：
模型要在"没见过的 sgRNA 序列"上给出预测。因此阻止跨 split 分配的 canonical
grouping key 是 **sgRNA 序列本身**（而不是 cell line、不是行号）。

为保守起见，identity class 进一步取 **反向互补同一类**：
`identity(seq) = min(seq, revcomp(seq))`。理由：一个 sgRNA 与其反向互补
靶向同一 locus 的两条链，标签来自同一靶点，属于近重复观测；
实测库内确实存在此类对（跨系 hct116 ∩ rc(hela) = 30、hela ∩ rc(hl60) = 3，
系内 hct116 12 / hek293t 130 / hela 58）。若只按精确序列分组，
`train ∩ rc(test)` 实测非零（mixed 19 条、LOCO 3 条）。

理由（可核验）：
  * 同一 sgRNA 出现在不同 cell line 时，序列通道完全相同，只有表观通道不同；
    若一行进 train、另一行进 test，模型已见过该序列 → 违反 sequence-level 声明。
  * 同一 (sgRNA,label) 跨 cell line 重复（本批 hct116↔hela 2 506 行）是最强形式；
  * 同 locus 反向互补孪生同时被 identity class 吸收，精确重复与反向互补近重复
    由运行记录里的 `audit_train_test_*_overlap` 字段分别计数。

唯一权威实现的镜像关系：
  * 分析层: `canonical_group_key()`（本文件）
  * 训练层: `core/data/cell_line_division.py` 用同一套规则做 group-aware 划分，
    并对 train/valid/test 的重叠做非零即抛错的自证。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SEQ_COL_CANDIDATES = ("sgRNA", "sgrna", "sequence", "seq", "target")
LABEL_COL_CANDIDATES = ("Normalized efficacy", "efficacy", "label", "y")


def _pick(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        low = str(c).lower()
        if any(k in low for k in ("sgrna", "sequence", "efficacy", "label")):
            return c
    return None


def sequence_key(meta: pd.DataFrame) -> pd.Series:
    """canonical sequence identity（去空格、大写）。"""
    col = _pick(meta, SEQ_COL_CANDIDATES)
    if col is None:
        raise KeyError(f"metadata 中找不到序列列: {list(meta.columns)}")
    return meta[col].astype(str).str.upper().str.strip()


def revcomp(seq: str) -> str:
    return seq.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def canonical_group_key(meta: pd.DataFrame, revcomp_canonical: bool = True) -> pd.Series:
    """跨 split 禁止共享的 canonical identity class（唯一权威定义）。

    = min(sgRNA, revcomp(sgRNA))；与训练层
    `src/input_control/cell_line_division.py::sequence_group_ids` 规则一致。
    """
    seqs = sequence_key(meta)
    if not revcomp_canonical:
        return seqs
    return seqs.map(lambda s: min(s, revcomp(s)))

