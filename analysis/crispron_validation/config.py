# -*- coding: utf-8 -*-
"""analysis.crispron_validation.config — 一次验证运行的完整配置。

所有路径都从本文件位置推导，因此从任意 CWD 运行都能找到数据与产物。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

#: analysis/crispron_validation/config.py -> 上溯 3 层到仓库根
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: CRISPRon 的 30 nt 窗口构型：prefix(4) + protospacer(20) + PAM(3) + suffix(3)
UPSTREAM_NT = 4
DOWNSTREAM_NT = 3
CRISPRON_LEN = UPSTREAM_NT + 20 + 3 + DOWNSTREAM_NT      # = 30
SPACER_LEN = 20
SEQ_LEN = 23
PAM_LEN = 3


@dataclass
class CrispronValidationConfig:
    """CNN7 × CRISPRon 的 Pos18 C→A 一致性验证配置。"""

    # ---------------- 数据集 ----------------
    dataset: str = "DeepCRISPR"
    #: 参与训练与验证的细胞系（顺序固定，保证拼接可复现）
    cell_lines: List[str] = field(
        default_factory=lambda: ["hct116", "hek293t", "hela", "hl60"])

    # ---------------- 关注位点 ----------------
    #: 1-based 位点（与 `analysis.candidates.wt_position18_selection.POS_1B` 一致）
    pos_1b: int = 18
    #: 只在原始碱基 == `target_base` 的序列上做突变（本计划 = C→A）
    target_base: str = "C"
    mutant_base: str = "A"

    # ---------------- CNN ----------------
    #: 序列核长度
    sequence_kernel: int = 7
    environment_kernel: int = 3
    #: 纯序列（不选任何表观通道）；`train()` 见到空列表会把张量切成 4 通道
    environment: str = "sequence"
    random_seed: int = 42
    epochs: int = 100
    batch_size: int = 64
    learning_rate: float = 1e-3
    patience: int = 20
    run_name: str = "cnn7_alltrain_sequence"

    # ---------------- CRISPRon ----------------
    #: 每次调用喂给 CRISPRon 的序列条数
    crispron_chunk_size: int = 2000
    crispron_timeout: int = 5400

    # ---------------- 路径 ----------------
    project_root: Path = PROJECT_ROOT
    work_dir: Path = PROJECT_ROOT / "results" / "crispron_validation"
    #: 归档的旧外部验证工作目录：**只作为可选的快速缓存来源**，
    #: 不存在时会自动从 hg19 / CRISPRon 重新生成，因此不构成运行时依赖。
    archive_work_dir: Path = PROJECT_ROOT / "Delete" / "crispron_work"

    # ---------------- 派生路径 ----------------
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data" / "processed" / self.dataset

    @property
    def summary_dir(self) -> Path:
        return self.project_root / "results" / "summary" / self.dataset

    @property
    def crispron_root(self) -> Path:
        return self.project_root / "deploy" / "crispron"

    @property
    def crispron_main(self) -> Path:
        return self.crispron_root / "software" / "crispron-main"

    @property
    def crispron_script(self) -> Path:
        return self.crispron_main / "bin" / "CRISPRon.sh"

    @property
    def crispron_venv_bin(self) -> Path:
        return self.crispron_root / "venv" / "bin"

    @property
    def crispron_wrappers(self) -> Path:
        return self.crispron_root / "software" / "wrappers"

    # ---- 中间产物 ----
    @property
    def windows_path(self) -> Path:
        return self.work_dir / f"windows_{self.dataset}.csv"

    @property
    def hg19_cache_path(self) -> Path:
        return self.work_dir / "hg19_window_cache.csv"

    @property
    def crispron_wt_path(self) -> Path:
        return self.work_dir / "crispron_wt.csv"

    @property
    def crispron_mut_path(self) -> Path:
        return self.work_dir / "crispron_c18a.csv"

    @property
    def cnn_result_dir(self) -> Path:
        return self.work_dir / "cnn_run"

    @property
    def table_path(self) -> Path:
        return self.work_dir / f"table_{self.dataset}.csv"

    # ---- 交付物 ----
    @property
    def report_md(self) -> Path:
        return self.summary_dir / "cnn7_validation.md"

    @property
    def report_csv(self) -> Path:
        return self.summary_dir / "cnn7_validation.csv"

    # ---------------- 便利方法 ----------------
    @property
    def pos_index0(self) -> int:
        """关注位点在 23 nt 里的 0-based 下标。"""
        return int(self.pos_1b) - 1

    @property
    def window_pos_index0(self) -> int:
        """关注位点在 30 nt 窗口里的 0-based 下标。"""
        return UPSTREAM_NT + self.pos_index0

    @property
    def nucleotide_channels(self) -> List[str]:
        return ["A", "C", "G", "T"]

    def ensure_dirs(self) -> None:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.summary_dir.mkdir(parents=True, exist_ok=True)

    def describe(self) -> str:
        return (f"{self.dataset} | 细胞系={'/'.join(self.cell_lines)} | "
                f"CNN sequence_kernel={self.sequence_kernel} 纯序列 全数据训练 | "
                f"关注位点 {self.pos_1b} ({self.target_base}→{self.mutant_base})")
