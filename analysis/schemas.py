"""analysis.schemas — 统一科学记录类型。

术语纪律（**保留**）:
    Effect              影响大小/方向 (ΔR²、系数、ISM effect、enrichment effect)
    Importance          模型依赖 (SHAP/IG/ISM/Attention/Gain/PFI/LOFO)
    StatisticalEvidence 只有明确假设检验/CI 才使用 (p/FDR/CI 等)

禁止: 对 SHAP/IG/ISM/Gain 原始值做 FDR 后称为 statistical significance;
      SNR>=2.5 只能称为 attribution/robustness strength。

已移出: Evidence Tier / EvidenceClass / EvidenceRecord 等**人为证据标签体系**
（含 `CellLineEffect`、`MotifRecord.stability`），以及 cell-line 一致性标签。
保留的是实验记录本身。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# 统一记录
# ---------------------------------------------------------------------------
@dataclass
class ExperimentRecord:
    """一条实验的元数据 + 同一 test cohort 指标。"""

    experiment_id: str
    model: str
    architecture: Optional[str]
    split_type: str
    cell_line: str
    environment: str
    seed: int
    n_train: int
    n_valid: int
    n_test: int
    r2: float
    mae: float
    rmse: float
    pearson: float
    spearman: float
    status: str = "completed"
    batch: Optional[str] = None
    source_file: Optional[str] = None
    raw: Dict = field(default_factory=dict)
