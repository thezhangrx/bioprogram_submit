#!/usr/bin/env python3
"""rule_discovery.py — 赛道二「规律发现」交付物生成器。

本程序**不训练模型、不重算统计量**，只做一件事：

    整合已有的分析产物（环境主效应 / 细胞系级 bootstrap CI / 归因 / 富集检验 /
    跨细胞系一致性），按「Statistical Evidence -> Effect（-> Robustness）」的
    顺序筛选出候选 microenvironment factor 与 sequence motif，
    生成赛事提交用的标准化 CSV + Excel。

输出（每个批次一个子目录）::

    results/赛道二_results/<batch>/
    ├── csv/
    │   ├── <cell_line>_microenv.csv           # 4 个
    │   ├── <cell_line>_motif.csv              # 4 个
    │   └── microenv_and_motif.csv             # 1 个（跨细胞系泛化）
    └── excel/
        └── 赛道二_results.xlsx                # 上述 9 个 CSV 各占一个 Sheet

设计约定
--------
* **不引入 Tier 等人为评价体系**。输出只表示"满足当前筛选条件的候选"，
  证据强度由原始的统计量本身（CI / FDR / 效应量 / 跨细胞系比例）表达。
* **所有阈值集中在** :class:`RuleDiscoveryConfig`，改配置即可调整评选标准，
  不需要改核心逻辑。
* **Importance 只列举、不参与筛选**（SNR / FDR 支持情况仅用于展示）。
* **不修改任何已有分析结果文件**；所有产物只写入 ``results/赛道二_results/``。

用法::

    python rule_discovery.py                          # 自动探测全部批次
    python rule_discovery.py --data-set DeepCRISPR
    python rule_discovery.py --batch-dir results/summary/DeepCRISPR
    python rule_discovery.py --print-config           # 打印生效的评选标准
    python rule_discovery.py --config-json my.json    # 覆盖部分阈值
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- 项目根引导：保证从任意工作目录运行都能解析 core / analysis ---
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

#: 交付物版本号（仅写入运行日志，便于追溯）。
#: 2026-09-26 起 ``<cell_line>_microenv.csv``、``<cell_line>_motif.csv`` 与 ``microenv_and_motif.csv``
#: 均已不再输出该列。
PIPELINE_VERSION = "1.0.0"

LOGGER = logging.getLogger("rule_discovery")


# ===========================================================================
# 0. 通用小工具（无状态，保持模块级）
# ===========================================================================
def _now_iso() -> str:
    """UTC ISO-8601 时间戳（不带微秒，便于跨表对齐）。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _rel(path: Optional[Path]) -> str:
    """把路径转成相对项目根的短字符串（写进 source_file 列）。"""
    if path is None:
        return ""
    try:
        return str(Path(path).resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _fmt_map(pairs: Dict[str, float], digits: int = 6) -> str:
    """``{'a': 0.1}`` -> ``"a=0.100000"``（按 key 排序，保证可复现）。"""
    return ";".join(f"{k}={pairs[k]:.{digits}f}" for k in sorted(pairs))


# ===========================================================================
# 1. 配置
# ===========================================================================
@dataclass
class RuleDiscoveryConfig:
    """所有筛选阈值、输入文件名与输出路径。

    用户**直接修改本类的字段**即可调整评选标准；
    也可以在命令行用 ``--config-json <文件>`` 覆盖其中任意字段，
    或写一个继承本类的子类再传给 :class:`RuleDiscoveryPipeline`。

    筛选顺序固定为 ``Statistical Evidence -> Effect``（组合表再加 Robustness），
    修改阈值只改变通过与否，不改变顺序。
    """

    # ---------------- 批次与输出 ----------------
    summary_root: str = "results/summary"
    output_root: str = "results/赛道二_results"
    batch_dirs: Tuple[str, ...] = ()          # 显式指定批次目录；空 = 自动探测
    dataset_names: Tuple[str, ...] = ()       # 按数据集名指定（相对 summary_root，大小写不敏感）
    target_cell_lines: Tuple[str, ...] = ("hct116", "hek293t", "hela", "hl60")
    cell_line_display: Dict[str, str] = field(default_factory=lambda: {
        "hct116": "HCT116", "hek293t": "HEK293T", "hela": "HeLa", "hl60": "HL60",
    })
    csv_encoding: str = "utf-8-sig"
    excel_filename: str = "赛道二_results.xlsx"

    # ---------------- 输入文件名（相对批次目录） ----------------
    tables_subdir: str = "tables"
    environment_main_effects_file: str = "environment_main_effects.csv"
    bootstrap_cellline_effects_file: str = "bootstrap_cellline_effects.csv"
    motif_candidates_file: str = "motif_candidates.csv"
    motif_instances_file: str = "motif_instances.csv"
    motif_enrichment_file: str = "motif_enrichment.csv"
    importance_file: str = "feature_importance/key_regulatory_biomarkers.csv"

    # =================================================================
    # 文件 1：<cell_line>_microenv.csv
    # =================================================================
    # --- 第一层 Statistical Evidence: 细胞系级主效应 Bootstrap CI ---
    microenv_ci_required: bool = True            # False = 跳过 CI 这一层（不推荐）
    microenv_require_ci_status_ok: bool = True   # 要求 bootstrap 行的 status == "ok"
    microenv_min_bootstrap_iterations: int = 200  # 同 analysis.config.evidence.min_bootstrap_iterations
    microenv_ci_must_exclude_zero: bool = True    # CI 不跨 0 才算通过（"不跨 0"判据）

    # --- 第二层 Effect: environment_main_effects 的环境主效应 ΔR² ---
    #   model_mean : 跨模型等权平均（统一口径：先按模型求均值, 再对模型等权平均）
    #   per_model  : 每个模型各占一行（不做任何跨模型聚合）
    microenv_effect_aggregation: str = "model_mean"
    microenv_min_abs_main_r2_delta: float = 0.01   # 绝对预测增益下限, 同 evidence.min_absolute_delta_r2
    #: True = 要求 Effect 与 CI 点估计**同号**（两者本质同一统计量，此项用于捕捉
    #: "CI 与效应来自不同来源/不同筛选"的一致性漂移；符号为 0 视为不通过）
    microenv_effect_must_match_ci_sign: bool = True
    #: 数值不稳定阈值：|ΔR²| ≥ 该值的模型行先剔除再聚合
    #: （同 analysis.config.consensus.unstable_effect_threshold）
    microenv_unstable_effect_threshold: float = 10.0

    # --- Importance（只列举, 不参与筛选） ---
    microenv_snr_threshold: float = 2.5
    microenv_fdr_threshold: float = 0.001
    #: 视为"非线性模型"的模型家族（linear 是唯一允许报告显著性的模型, 故单列）
    microenv_nonlinear_models: Tuple[str, ...] = ("cnn", "mlp", "transformer", "xgboost")
    #: 计数口径: family = 模型家族数(≤4); config = 具体配置数(cnn33/53/73 各算一个, ≤6)
    microenv_importance_model_level: str = "family"
    #: environment factor -> 归因表里的 channel 名
    microenv_factor_to_channel: Dict[str, str] = field(default_factory=lambda: {
        "ctcf": "CTCF", "dnase": "Dnase", "h3k4me3": "H3K4me3", "rrbs": "RRBS",
    })

    # =================================================================
    # 文件 2：<cell_line>_motif.csv
    # =================================================================
    # --- 第一层 Statistical Evidence: BH-FDR ---
    motif_fdr_threshold: float = 0.05
    #: "candidate"  = 用 motif_candidates.csv 的 FDR（motif 级发现 FDR）
    #: "enrichment" = 用 motif_enrichment.csv 的 FDR（Fisher 富集 FDR）
    motif_fdr_source: str = "candidate"
    motif_require_status_ok: bool = False          # True = 只保留 status == "ok" 的 motif

    # --- 第二层 Effect: |ISM effect| ---
    #: "candidate" = 用 motif_candidates.mean_effect（已有的 motif 级 ISM 效应）
    #: "instances" = 用 motif_instances.ism_effect 的样本均值（重新聚合）
    motif_ism_effect_source: str = "candidate"
    motif_abs_ism_effect_threshold: float = 0.01
    motif_min_support_count: int = 0               # 可选：最少实例支撑数

    # =================================================================
    # 文件 3：microenv_and_motif.csv
    # =================================================================
    # --- Robustness: 跨细胞系比例 ---
    cellline_ratio_threshold: float = 0.75
    #: False -> 严格大于(>)，True -> 大于等于(>=)。
    #: 注意：4 个细胞系时 ``> 0.75`` 等价于"必须 4/4 全支持"，``>= 0.75`` 允许 3/4。
    cellline_ratio_inclusive: bool = False
    #: cell line 视作"支持"的条件：any_split = 任一 split 通过; all_splits = 所有 split 都通过
    cellline_support_mode: str = "any_split"

    # =================================================================
    # 输出开关
    # =================================================================
    #: False = 每个 CSV 只保留通过筛选的候选（默认，符合交付语义）；
    #: True  = 保留全部输入行并附 selected 标志（审计用）。
    include_rejected_rows: bool = False
    #: Excel 每个 Sheet 的列宽上限（0 = 不设）
    excel_max_col_width: int = 40

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """把明显矛盾的配置在启动时就打回，而不是产出错的结果。"""
        if self.microenv_effect_aggregation not in ("model_mean", "per_model"):
            raise ValueError("microenv_effect_aggregation 只能是 'model_mean' 或 'per_model'")
        if self.microenv_importance_model_level not in ("family", "config"):
            raise ValueError("microenv_importance_model_level 只能是 'family' 或 'config'")
        if self.motif_fdr_source not in ("candidate", "enrichment"):
            raise ValueError("motif_fdr_source 只能是 'candidate' 或 'enrichment'")
        if self.motif_ism_effect_source not in ("candidate", "instances"):
            raise ValueError("motif_ism_effect_source 只能是 'candidate' 或 'instances'")
        if self.cellline_support_mode not in ("any_split", "all_splits"):
            raise ValueError("cellline_support_mode 只能是 'any_split' 或 'all_splits'")
        for name in ("microenv_fdr_threshold", "microenv_snr_threshold",
                     "microenv_min_abs_main_r2_delta", "motif_fdr_threshold",
                     "motif_abs_ism_effect_threshold", "cellline_ratio_threshold"):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} 不能为负数")
        if not 0.0 <= float(self.cellline_ratio_threshold) <= 1.0:
            raise ValueError("cellline_ratio_threshold 必须在 [0, 1] 区间")
        if int(self.microenv_min_bootstrap_iterations) < 0:
            raise ValueError("microenv_min_bootstrap_iterations 不能为负数")
        if not self.target_cell_lines:
            raise ValueError("target_cell_lines 不能为空")

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------
    def display_name(self, cell_line: str) -> str:
        """``hct116`` -> ``HCT116``（用于 Excel Sheet 名与文件名）。"""
        return self.cell_line_display.get(str(cell_line).lower(), str(cell_line))

    def factor_channel(self, factor: str) -> Optional[str]:
        """environment factor -> 归因表 channel（大小写不敏感）。"""
        key = str(factor).strip().lower()
        for factor_name, channel in self.microenv_factor_to_channel.items():
            if factor_name.lower() == key:
                return channel
        return None

    def cellline_ratio_pass(self, ratio: Optional[float]) -> bool:
        if ratio is None or not np.isfinite(ratio):
            return False
        if self.cellline_ratio_inclusive:
            return bool(ratio >= self.cellline_ratio_threshold)
        return bool(ratio > self.cellline_ratio_threshold)

    def ratio_rule_text(self) -> str:
        op = ">=" if self.cellline_ratio_inclusive else ">"
        return f"cellline_ratio {op} {self.cellline_ratio_threshold:g}"

    def as_dict(self) -> Dict:
        return asdict(self)


# ===========================================================================
# 2. 输入装载
# ===========================================================================
@dataclass
class BatchInputs:
    """一个批次已加载的全部输入表与溯源路径。"""

    batch_name: str
    batch_dir: Path
    tables_dir: Path
    environment_main_effects: pd.DataFrame
    bootstrap_cellline_effects: pd.DataFrame
    motif_candidates: pd.DataFrame
    motif_instances: pd.DataFrame
    motif_enrichment: pd.DataFrame
    biomarkers: pd.DataFrame
    sources: Dict[str, str] = field(default_factory=dict)

    def source_of(self, key: str) -> str:
        return self.sources.get(key, "")

    def combined_sources(self, *keys: str) -> str:
        seen: List[str] = []
        for key in keys:
            value = self.sources.get(key, "")
            if value and value not in seen:
                seen.append(value)
        return ";".join(seen)


class InputLoader:
    """按配置定位并加载某个批次的输入文件（缺文件立刻报错，绝不静默跳过）。"""

    #: 逻辑名 -> (配置字段名, 是否必需, 是否位于 tables/ 子目录)
    _SPEC: Tuple[Tuple[str, str, bool, bool], ...] = (
        ("environment_main_effects", "environment_main_effects_file", True, True),
        ("bootstrap_cellline_effects", "bootstrap_cellline_effects_file", True, True),
        ("motif_candidates", "motif_candidates_file", True, True),
        ("motif_instances", "motif_instances_file", True, True),
        ("motif_enrichment", "motif_enrichment_file", True, True),
        ("biomarkers", "importance_file", False, False),
    )

    def __init__(self, config: RuleDiscoveryConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def resolve_batches(self) -> List[Path]:
        """决定要处理哪些批次。

        * ``dataset_names`` -> 在 ``summary_root`` 下**大小写不敏感**地按名匹配
          （与 ``train.py --data-set`` 的约定一致）；
        * ``batch_dirs`` -> 显式路径，逐个校验；
        * 两者都空 -> 扫描 ``summary_root/*``，取**直接含 tables/ 子目录**的批次，按名字排序。
        """
        out: List[Path] = []
        for name in self.config.dataset_names:
            out.append(self._resolve_dataset_name(str(name)))
        for raw in self.config.batch_dirs:
            p = Path(raw).expanduser()
            if not p.is_dir():
                raise FileNotFoundError(f"指定的批次目录不存在：{p}")
            out.append(p.resolve())
        if out:
            return out

        root = Path(self.config.summary_root)
        if not root.is_dir():
            raise FileNotFoundError(
                f"找不到批次汇总根目录：{root}\n"
                f"  请用 --batch-dir 显式指定批次，或先运行 analysis.pipeline 生成汇总。"
            )
        found = [d for d in sorted(root.iterdir())
                 if d.is_dir() and self._tables_dir(d).is_dir()]
        if not found:
            raise FileNotFoundError(
                f"在 {root} 下没有发现任何含 {self.config.tables_subdir}/ 的批次。\n"
                f"  请用 --batch-dir 显式指定批次目录。"
            )
        return found

    def _resolve_dataset_name(self, name: str) -> Path:
        """``--data-set <名称>`` -> 批次目录（大小写不敏感，不猜默认值）。"""
        root = Path(self.config.summary_root)
        wanted = name.strip().lower()
        if root.is_dir():
            available = {d.name.lower(): d for d in sorted(root.iterdir()) if d.is_dir()}
            if wanted in available:
                return available[wanted].resolve()
            raise FileNotFoundError(
                f"在 {root} 下找不到数据集 {name!r}。\n"
                f"  可用：{', '.join(sorted(available)) or '（无）'}"
            )
        raise FileNotFoundError(f"批次汇总根目录不存在：{root}")

    # ------------------------------------------------------------------
    def _tables_dir(self, batch_dir: Path) -> Path:
        return Path(batch_dir) / self.config.tables_subdir

    def path_of(self, batch_dir: Path, rel: str, in_tables: bool) -> Path:
        """把一个配置里的相对路径解析成绝对路径。

        位于 ``tables/`` 下的输入，其配置值写**文件名**即可（如
        ``environment_main_effects.csv``）；想指向别处时直接写带目录的相对路径
        （如 ``other/environment_main_effects.csv``），此时不再加前缀。
        """
        rel = str(rel)
        candidate = Path(rel)
        if candidate.is_absolute() or candidate.parent != Path("."):
            return Path(batch_dir) / candidate
        if in_tables:
            return Path(batch_dir) / self.config.tables_subdir / candidate
        return Path(batch_dir) / candidate

    def load(self, batch_dir: Path) -> BatchInputs:
        batch_dir = Path(batch_dir).resolve()
        tables_dir = self._tables_dir(batch_dir)
        if not tables_dir.is_dir():
            raise FileNotFoundError(f"批次缺少 {self.config.tables_subdir}/ 目录：{batch_dir}")

        frames: Dict[str, pd.DataFrame] = {}
        sources: Dict[str, str] = {}
        missing: List[str] = []

        for key, attr, required, in_tables in self._SPEC:
            rel = str(getattr(self.config, attr))
            path = self.path_of(batch_dir, rel, in_tables)
            if not path.is_file():
                if required:
                    missing.append(f"  - {key:32s} -> {_rel(path)}")
                else:
                    LOGGER.warning("可选输入缺失，相关列将标记 unavailable：%s", _rel(path))
                    frames[key] = pd.DataFrame()
                    sources[key] = ""
                continue
            frames[key] = self._read(path, key)
            sources[key] = _rel(path)
            LOGGER.info("  读入 %-32s %7d 行  <- %s", key, len(frames[key]), _rel(path))

        if missing:
            raise FileNotFoundError(
                "以下**必需**输入文件不存在（请先跑完 analysis.pipeline）：\n"
                + "\n".join(missing)
                + f"\n  批次目录：{batch_dir}"
            )

        return BatchInputs(
            batch_name=batch_dir.name,
            batch_dir=batch_dir,
            tables_dir=tables_dir,
            environment_main_effects=frames["environment_main_effects"],
            bootstrap_cellline_effects=frames["bootstrap_cellline_effects"],
            motif_candidates=frames["motif_candidates"],
            motif_instances=frames["motif_instances"],
            motif_enrichment=frames["motif_enrichment"],
            biomarkers=frames["biomarkers"],
            sources=sources,
        )

    # ------------------------------------------------------------------
    def _read(self, path: Path, key: str) -> pd.DataFrame:
        """读 CSV。motif_instances 很大，只取需要的列。"""
        if key == "motif_instances":
            wanted = ["motif_id", "cell_line", "split_type", "model_variant",
                      "attribution_method", "ism_effect", "ig_effect",
                      "position_start", "position_end", "sample_id"]
            try:
                head = pd.read_csv(path, nrows=0)
                usecols = [c for c in wanted if c in head.columns]
                return pd.read_csv(path, usecols=usecols, low_memory=False)
            except Exception:                       # noqa: BLE001 - 回退为全量读
                LOGGER.warning("%s 按列读取失败，改为全量读取", _rel(path))
        return pd.read_csv(path, low_memory=False)


# ===========================================================================
# 3. 文件 1：<cell_line>_microenv.csv
# ===========================================================================
class MicroEnvDiscovery:
    """在单个细胞系背景下发现与 editing efficiency 相关的 cell environment factor。

    筛选顺序（**必须先过 Statistical Evidence，再看 Effect**）::

        第一层  Statistical Evidence : 细胞系级主效应 Bootstrap CI 不跨 0
        第二层  Effect               : 环境主效应 |ΔR²| >= 阈值（默认 0.01）
        附      Importance           : SNR>=2.5 的非线性模型数 + FDR<0.001 —— 只列举

    输入（均为已有产物，不重算）::

        tables/bootstrap_cellline_effects.csv   -> CI
        tables/environment_main_effects.csv     -> Effect
        feature_importance/key_regulatory_biomarkers.csv -> Importance
    """

    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "cell_line", "split_type", "environment_factor",
        "splits_available",
        # --- Statistical Evidence ---
        "bootstrap_CI_lower", "bootstrap_CI_upper", "CI_pass",
        "ci_estimate", "bootstrap_status", "bootstrap_n_iterations", "bootstrap_n_models",
        # --- Effect ---
        "main_r2_delta", "n_models", "model_effects",
        "effect_pass", "selected",
        # --- Importance（只展示） ---
        "nonlinear_model_support", "snr_support_threshold", "snr_support",
        "snr_support_configs", "fdr_lt_threshold_count", "fdr_support_threshold",
        # --- 溯源 ---
        "source_file", "generated_time",
    )

    def __init__(self, config: RuleDiscoveryConfig) -> None:
        self.config = config
        self._importance_cache: Dict[str, Dict[Tuple[str, str, str], Dict]] = {}

    # ------------------------------------------------------------------
    def discover(self, inputs: BatchInputs, cell_line: str) -> pd.DataFrame:
        """返回该细胞系的完整候选表（含未通过行；是否裁剪由 pipeline 决定）。"""
        statistical = self._statistical_layer(inputs, cell_line)
        effect = self._effect_layer(inputs, cell_line)
        importance = self._importance_layer(inputs, cell_line)

        keys = ["split_type", "environment_factor"]
        merged = statistical.merge(effect, on=keys, how="outer", validate="one_to_one")
        merged = merged.merge(importance, on=keys, how="left")

        # 该细胞系**实际拥有**的 split 集合（取自未筛选的输入），
        # 供 cellline_support_mode="all_splits" 判断"是否每个 split 都通过"。
        available = sorted(set(statistical["split_type"].astype(str))
                           | set(effect["split_type"].astype(str)))
        merged["splits_available"] = ";".join(available)

        merged["cell_line"] = cell_line
        merged["CI_pass"] = merged["CI_pass"].fillna(False).astype(bool)
        merged["effect_pass"] = merged["effect_pass"].fillna(False).astype(bool)
        # CI 与 Effect 同号一致性校验（两者应来自同一统计量；不同号说明来源漂移）
        if self.config.microenv_effect_must_match_ci_sign:
            eff = _num(merged["main_r2_delta"])
            ci = _num(merged["ci_estimate"])
            same_sign = (np.sign(eff) == np.sign(ci)) & eff.notna() & ci.notna() & (eff != 0)
            n_bad = int((merged["effect_pass"] & ~same_sign.fillna(False)).sum())
            if n_bad:
                LOGGER.warning("    [%s] %d 行的 Effect 与 CI 点估计不同号，按配置判为不通过",
                               cell_line, n_bad)
            merged["effect_pass"] = merged["effect_pass"] & same_sign.fillna(False)
        # 顺序语义（严格逐层）：Effect 只在 Statistical 已通过的行上评估，
        # 因此 effect_pass 列本身就表示"过了第二层"，而不是"满足第二层条件"。
        merged["effect_pass"] = merged["effect_pass"] & merged["CI_pass"]
        merged["selected"] = merged["effect_pass"]

        merged["nonlinear_model_support"] = (
            merged["nonlinear_model_support"].fillna(0).astype(int))
        merged["fdr_lt_threshold_count"] = (
            merged["fdr_lt_threshold_count"].fillna(0).astype(int))
        merged["snr_support"] = merged["snr_support"].fillna("")
        merged["snr_support_configs"] = merged["snr_support_configs"].fillna("")
        merged["snr_support_threshold"] = self.config.microenv_snr_threshold
        merged["fdr_support_threshold"] = self.config.microenv_fdr_threshold

        merged["source_file"] = inputs.combined_sources(
            "bootstrap_cellline_effects", "environment_main_effects", "biomarkers")
        merged["generated_time"] = _now_iso()

        merged = merged.reindex(columns=list(self.OUTPUT_COLUMNS))
        return merged.sort_values(
            ["selected", "split_type", "environment_factor"],
            ascending=[False, True, True], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    def _statistical_layer(self, inputs: BatchInputs, cell_line: str) -> pd.DataFrame:
        """第一层：细胞系级主效应 Bootstrap CI（不跨 0 才算通过）。"""
        cols = ["split_type", "environment_factor", "bootstrap_CI_lower",
                "bootstrap_CI_upper", "CI_pass", "ci_estimate", "bootstrap_status",
                "bootstrap_n_iterations", "bootstrap_n_models"]
        df = inputs.bootstrap_cellline_effects
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)

        sub = df[df["cell_line"].astype(str).str.lower() == cell_line.lower()].copy()
        sub = sub.rename(columns={"factor": "environment_factor",
                                  "ci_low": "bootstrap_CI_lower",
                                  "ci_high": "bootstrap_CI_upper",
                                  "estimate": "ci_estimate",
                                  "n_bootstrap": "bootstrap_n_iterations",
                                  "n_models": "bootstrap_n_models",
                                  "status": "bootstrap_status"})
        for col in ("bootstrap_CI_lower", "bootstrap_CI_upper", "ci_estimate"):
            sub[col] = _num(sub.get(col))
        sub["bootstrap_n_iterations"] = _num(sub.get("bootstrap_n_iterations")).fillna(0).astype(int)
        sub["bootstrap_n_models"] = _num(sub.get("bootstrap_n_models")).fillna(0).astype(int)
        sub["bootstrap_status"] = sub.get("bootstrap_status", "").astype(str)

        ok = np.ones(len(sub), dtype=bool)
        if self.config.microenv_require_ci_status_ok:
            ok &= sub["bootstrap_status"].str.lower().eq("ok").to_numpy()
        ok &= (sub["bootstrap_n_iterations"] >= int(self.config.microenv_min_bootstrap_iterations)).to_numpy()
        ok &= sub[["bootstrap_CI_lower", "bootstrap_CI_upper"]].notna().all(axis=1).to_numpy()
        if self.config.microenv_ci_must_exclude_zero:
            lo, hi = sub["bootstrap_CI_lower"].to_numpy(), sub["bootstrap_CI_upper"].to_numpy()
            ok &= ~((lo <= 0.0) & (hi >= 0.0))
        else:
            ok &= np.ones(len(sub), dtype=bool)
        if not self.config.microenv_ci_required:
            ok = np.ones(len(sub), dtype=bool)

        sub["CI_pass"] = ok
        return sub.reindex(columns=cols)

    # ------------------------------------------------------------------
    def _effect_layer(self, inputs: BatchInputs, cell_line: str) -> pd.DataFrame:
        """第二层：环境主效应 ΔR²（默认跨模型等权平均）。"""
        cols = ["split_type", "environment_factor", "main_r2_delta",
                "n_models", "model_effects", "effect_pass"]
        df = inputs.environment_main_effects
        if df is None or df.empty:
            return pd.DataFrame(columns=cols)

        sub = df[df["cell_line"].astype(str).str.lower() == cell_line.lower()].copy()
        sub = sub.rename(columns={"environment": "environment_factor"})
        sub["main_r2_delta"] = _num(sub.get("main_r2_delta"))
        sub = sub.dropna(subset=["main_r2_delta", "environment_factor"])

        # 数值不稳定隔离：与 analysis.config.consensus.unstable_effect_threshold 同口径
        thr = float(self.config.microenv_unstable_effect_threshold)
        n_before = len(sub)
        sub = sub[sub["main_r2_delta"].abs() < thr]
        n_dropped = n_before - len(sub)
        if n_dropped:
            LOGGER.info("    [%s] %d 个模型行的 |ΔR²| >= %g 已按数值不稳定剔除",
                        cell_line, n_dropped, thr)

        if self.config.microenv_effect_aggregation == "per_model":
            out = sub.rename(columns={"model": "model_effects"})[
                ["split_type", "environment_factor", "main_r2_delta", "model_effects"]].copy()
            out["n_models"] = 1
        else:
            grouped = sub.groupby(["split_type", "environment_factor"], dropna=False)
            out = grouped.agg(
                main_r2_delta=("main_r2_delta", "mean"),
                n_models=("main_r2_delta", "size"),
            ).reset_index()
            # 逐模型值仅作溯源展示（model_effects 的 "m=val;..." 写法）
            per_model = grouped.apply(
                lambda g: _fmt_map(dict(zip(g["model"].astype(str), g["main_r2_delta"]))),
                include_groups=False).rename("model_effects").reset_index()
            out = out.merge(per_model, on=["split_type", "environment_factor"], how="left")

        out["effect_pass"] = (
            out["main_r2_delta"].abs() >= float(self.config.microenv_min_abs_main_r2_delta))
        return out.reindex(columns=cols).sort_values(
            ["split_type", "environment_factor"], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    def _importance_layer(self, inputs: BatchInputs, cell_line: str) -> pd.DataFrame:
        """Importance（SNR / FDR 支持情况）——**只列举，不参与筛选**。"""
        cols = ["split_type", "environment_factor", "nonlinear_model_support",
                "snr_support", "snr_support_configs", "fdr_lt_threshold_count"]
        cache_key = inputs.batch_name
        if cache_key not in self._importance_cache:
            self._importance_cache[cache_key] = self._build_importance_index(inputs)
        index = self._importance_cache[cache_key]

        rows: List[Dict] = []
        for split in sorted({s for (s, _c, _f) in index.keys()}):
            for factor in sorted(self.config.microenv_factor_to_channel):
                info = index.get((split, cell_line.lower(), factor))
                if info is None:
                    continue
                rows.append({"split_type": split, "environment_factor": factor, **info})
        if not rows:
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows).reindex(columns=cols)

    def _build_importance_index(self, inputs: BatchInputs) -> Dict[Tuple[str, str, str], Dict]:
        """(split, cell_line, factor) -> Importance 指标（基于 key_regulatory_biomarkers.csv）。"""
        df = inputs.biomarkers
        index: Dict[Tuple[str, str, str], Dict] = {}
        if df is None or df.empty:
            LOGGER.warning("importance 输入为空，microenv 的 Importance 列将全部为 0/unavailable")
            return index

        col = {str(c).strip(): c for c in df.columns}
        c_split = col.get("split_type", "训练方式(split_type)")
        c_cell = col.get("cell_line", "细胞系(cell_line)")
        c_model = col.get("model", "模型(model)")
        c_feat = col.get("feature", "特征名(feature)")
        c_snr = self._pick(col, ("snr", "信噪比(SNR/t_stat)"))
        c_fdr = self._pick(col, ("fdr", "FDR校正q值"))
        if not all([c_split, c_cell, c_model, c_feat]):
            LOGGER.warning("importance 表缺少必需列，Importance 列留空")
            return index

        work = df[[c_split, c_cell, c_model, c_feat]].copy()
        work.columns = ["split_type", "cell_line", "model", "feature"]
        work["snr"] = _num(df[c_snr]) if c_snr else np.nan
        work["fdr"] = _num(df[c_fdr]) if c_fdr else np.nan
        work["split_type"] = work["split_type"].astype(str).str.strip().str.lower()
        work["cell_line"] = work["cell_line"].astype(str).str.strip().str.lower()
        work["model"] = work["model"].astype(str).str.strip()
        # feature 形如 pos13_H3K4me3 / pos1_A -> channel = 最后一个下划线之后
        work["channel"] = work["feature"].astype(str).str.rsplit("_", n=1).str[-1]

        snr_thr = float(self.config.microenv_snr_threshold)
        fdr_thr = float(self.config.microenv_fdr_threshold)
        nonlinear = {m.lower() for m in self.config.microenv_nonlinear_models}

        for (split, cell), group in work.groupby(["split_type", "cell_line"], dropna=False):
            for factor, channel in self.config.microenv_factor_to_channel.items():
                chan = group[group["channel"].str.lower() == str(channel).lower()]
                if chan.empty:
                    continue
                # --- SNR 支持：SNR >= 阈值的非线性模型 ---
                hot = chan[chan["snr"] >= snr_thr]
                hot_models = sorted({str(m) for m in hot["model"].unique()}) if not hot.empty else []
                hot_families = sorted({self._family(m) for m in hot_models})
                if self.config.microenv_importance_model_level == "config":
                    counted = hot_models
                else:
                    counted = [f for f in hot_families if f.lower() in nonlinear]
                # --- FDR 支持：FDR < 阈值的特征（本批只有 linear 报告 FDR） ---
                fdr_hot = chan[chan["fdr"] < fdr_thr] if "fdr" in chan else chan.iloc[0:0]
                n_fdr = int(len(fdr_hot))
                index[(split, cell, factor)] = {
                    "nonlinear_model_support": len(counted),
                    "snr_support": ";".join(counted),
                    "snr_support_configs": ";".join(hot_models),
                    "fdr_lt_threshold_count": n_fdr,
                }
        return index

    @staticmethod
    def _pick(colmap: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
        for name in candidates:
            if name in colmap:
                return colmap[name]
        return None

    @staticmethod
    def _family(model: str) -> str:
        """``cnn33`` / ``cnn53`` / ``cnn73`` -> ``cnn``。"""
        text = str(model).strip().lower()
        for family in ("cnn", "mlp", "transformer", "xgboost", "linear"):
            if text.startswith(family):
                return family
        return text

# ===========================================================================
# 4. 文件 2：<cell_line>_motif.csv
# ===========================================================================
class MotifDiscovery:
    """发现与 sgRNA editing efficiency 相关的序列 motif。

    筛选顺序::

        第一层  Statistical Evidence : BH-FDR < 阈值（默认 0.05）
        第二层  Effect               : |ISM effect| >= 阈值（默认 0.01）
        附      Attribution support  : IG / SHAP / 模型支持度 —— 只列举

    输入::

        tables/motif_candidates.csv   -> motif 级统计量（FDR / ISM 效应 / 支持度）
        tables/motif_instances.csv    -> 样本级归因（ISM / IG），用于逐实例复核
        tables/motif_enrichment.csv   -> Fisher 富集（备选 FDR 来源）

    说明：本项目的 motif 发现**只由 CNN 系列驱动**，因此没有 TreeSHAP 值
    （SHAP 是 XGBoost 专属归因），``SHAP_score`` 列按项目约定标记 unavailable。
    """

    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "cell_line", "split_type", "environment", "splits_available",
        "motif_id", "motif", "position",
        "position_start", "position_end", "position_mean",
        # --- Statistical Evidence ---
        "BH_FDR", "FDR_pass", "fdr_source", "enrichment_FDR",
        # --- Effect ---
        "ISM_effect", "ISM_effect_abs", "ISM_effect_pass",
        "ism_effect_source", "ISM_effect_instances", "n_instances",
        # --- Attribution support（只展示） ---
        "IG_score", "SHAP_score",
        "model_support", "model_consistency",
        "support_count", "enrichment", "odds_ratio", "p_value",
        # --- 结果 ---
        "selected",
        # --- 溯源 ---
        "source_file", "generated_time",
    )

    def __init__(self, config: RuleDiscoveryConfig) -> None:
        self.config = config
        self._instance_cache: Dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    def discover(self, inputs: BatchInputs, cell_line: str) -> pd.DataFrame:
        candidates = inputs.motif_candidates
        if candidates is None or candidates.empty:
            return pd.DataFrame(columns=list(self.OUTPUT_COLUMNS))

        instances = self._instance_summary(inputs)
        enrichment = self._enrichment_frame(inputs)

        work = candidates.copy()
        work["motif_id"] = work["motif_id"].astype(str)
        work = work.merge(instances, on="motif_id", how="left")
        work = work.merge(enrichment, on="motif_id", how="left", suffixes=("", "_enr"))

        work["cell_line"] = work["_instance_cell_line"].fillna(
            work.get("_motif_cell_line_from_id", pd.Series(index=work.index, dtype=object)))
        work = work[work["cell_line"].astype(str).str.lower() == cell_line.lower()]
        if work.empty:
            return pd.DataFrame(columns=list(self.OUTPUT_COLUMNS))

        # ---- 第一层：Statistical Evidence（BH-FDR） ----
        fdr_src = self.config.motif_fdr_source
        if fdr_src == "enrichment":
            work["BH_FDR"] = work["enrichment_FDR"]
        else:
            work["BH_FDR"] = _num(work.get("FDR"))
        work["fdr_source"] = fdr_src
        work["FDR_pass"] = work["BH_FDR"] < float(self.config.motif_fdr_threshold)
        if self.config.motif_require_status_ok and "status" in work.columns:
            work["FDR_pass"] &= work["status"].astype(str).str.lower().eq("ok")

        # ---- 第二层：Effect（|ISM effect|） ----
        if self.config.motif_ism_effect_source == "instances":
            work["ISM_effect"] = _num(work.get("ISM_effect_instances"))
        else:
            work["ISM_effect"] = _num(work.get("mean_effect"))
        work["ISM_effect_abs"] = work["ISM_effect"].abs()
        work["ISM_effect_pass"] = (
            work["ISM_effect_abs"] >= float(self.config.motif_abs_ism_effect_threshold))
        if int(self.config.motif_min_support_count) > 0:
            work["ISM_effect_pass"] &= (
                _num(work.get("support_count")).fillna(0)
                >= float(self.config.motif_min_support_count))
        # 顺序语义（严格逐层）：Effect 只在 Statistical 已通过的行上评估
        work["FDR_pass"] = work["FDR_pass"].fillna(False).astype(bool)
        work["ISM_effect_pass"] = work["ISM_effect_pass"].fillna(False).astype(bool) \
            & work["FDR_pass"]
        work["selected"] = work["ISM_effect_pass"]

        # ---- 输出字段 ----
        out = pd.DataFrame(index=work.index)
        out["cell_line"] = work["cell_line"]
        out["split_type"] = work.get("split_type", pd.Series("", index=work.index))
        out["environment"] = work.get("environment", pd.Series("", index=work.index))
        # 该细胞系实际拥有的 split（取自未筛选的 motif_candidates），供 all_splits 判断
        available = sorted(set(work["split_type"].astype(str))) if "split_type" in work.columns else []
        out["splits_available"] = ";".join(available)
        out["motif_id"] = work["motif_id"]
        out["motif"] = work.get("consensus", work.get("sequence", pd.Series("", index=work.index)))
        out["position"] = work.get("preferred_position", pd.Series(pd.NA, index=work.index))
        out["position_start"] = _num(work.get("position_start"))
        out["position_end"] = _num(work.get("position_end"))
        out["position_mean"] = _num(work.get("position_mean"))
        out["BH_FDR"] = work["BH_FDR"]
        out["FDR_pass"] = work["FDR_pass"].fillna(False)
        out["fdr_source"] = work["fdr_source"]
        out["enrichment_FDR"] = _num(work.get("enrichment_FDR"))
        out["ISM_effect"] = work["ISM_effect"]
        out["ISM_effect_abs"] = work["ISM_effect_abs"]
        out["ISM_effect_pass"] = work["ISM_effect_pass"].fillna(False)
        out["ism_effect_source"] = self.config.motif_ism_effect_source
        out["ISM_effect_instances"] = _num(work.get("ISM_effect_instances"))
        out["n_instances"] = _num(work.get("n_instances")).fillna(0).astype(int)
        out["IG_score"] = _num(work.get("IG_score"))
        out["SHAP_score"] = np.nan
        out["model_support"] = self._model_support(work)
        out["model_consistency"] = _num(work.get("model_consistency"))
        out["support_count"] = _num(work.get("support_count"))
        out["enrichment"] = _num(work.get("enrichment"))
        out["odds_ratio"] = _num(work.get("odds_ratio"))
        out["p_value"] = _num(work.get("p_value"))
        out["selected"] = work["selected"].fillna(False)
        out["source_file"] = inputs.combined_sources(
            "motif_candidates", "motif_instances", "motif_enrichment")
        out["generated_time"] = _now_iso()

        out = out.reindex(columns=list(self.OUTPUT_COLUMNS))
        return out.sort_values(
            ["selected", "BH_FDR", "ISM_effect_abs"],
            ascending=[False, True, False], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    def _instance_summary(self, inputs: BatchInputs) -> pd.DataFrame:
        """motif_id -> 样本级 ISM/IG 均值、实例数与所属细胞系（只做聚合，不重算统计）。"""
        empty = pd.DataFrame(columns=["motif_id", "ISM_effect_instances", "IG_score",
                                      "n_instances", "_instance_cell_line"])
        df = inputs.motif_instances
        if df is None or df.empty:
            return empty
        out = df.groupby("motif_id", dropna=False).agg(
            ISM_effect_instances=("ism_effect", "mean"),
            IG_score=("ig_effect", "mean"),
            n_instances=("motif_id", "size"),
            _instance_cell_line=("cell_line", "first"),
        ).reset_index()
        return out

    def _enrichment_frame(self, inputs: BatchInputs) -> pd.DataFrame:
        """motif_enrichment -> motif_id 级的 enrichment/odds_ratio/p_value/FDR。"""
        df = inputs.motif_enrichment
        if df is None or df.empty:
            return pd.DataFrame(columns=["motif_id", "enrichment", "odds_ratio",
                                         "p_value", "enrichment_FDR"])
        keep = ["motif_id", "enrichment", "odds_ratio", "p_value", "FDR"]
        sub = df[[c for c in keep if c in df.columns]].copy()
        sub["motif_id"] = sub["motif_id"].astype(str)
        return sub.rename(columns={"FDR": "enrichment_FDR"}).drop_duplicates("motif_id")

    def _model_support(self, work: pd.DataFrame) -> pd.Series:
        """模型支持度的可读描述：``source/variant/method`` 去重后以 ; 连接。"""
        parts = []
        for col in ("model_source", "model_variant", "attribution_method"):
            series = work[col].astype(str) if col in work.columns else pd.Series("", index=work.index)
            parts.append(series.where(series != "nan", ""))
        combined = parts[0].str.cat(parts[1:], sep="/", na_rep="")
        return combined.str.strip("/").str.replace(r"/{2,}", "/", regex=True)

# ===========================================================================
# 5. 文件 3：microenv_and_motif.csv
# ===========================================================================
class GeneralizationAnalyzer:
    """在前两个文件的基础上，用**跨细胞系比例**筛出泛化明显的微环境通道与 motif。

    这是唯一新增的判据（Robustness）::

        cellline_ratio = supporting_celllines / total_celllines
        cellline_ratio > 阈值（默认 0.75）  ->  通过

    微环境以 **environment factor** 为键；motif 以 **consensus 序列**为键
    （motif_id 里含细胞系名，逐细胞系唯一，跨细胞系对齐必须用序列）。
    """

    OUTPUT_COLUMNS: Tuple[str, ...] = (
        "feature_type", "feature_name",
        "supporting_celllines", "supporting_splits", "n_supporting", "n_total_celllines",
        "cellline_ratio", "cellline_ratio_pass",
        "mean_effect", "effect_variance", "effect_by_cellline",
        "statistical_summary", "source_files",
        "generated_time",
    )

    def __init__(self, config: RuleDiscoveryConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    def combine(self, microenv_by_cell: Dict[str, pd.DataFrame],
                motif_by_cell: Dict[str, pd.DataFrame],
                total_celllines: Optional[int] = None) -> pd.DataFrame:
        rows: List[Dict] = []
        total = int(total_celllines) if total_celllines else len(self.config.target_cell_lines)
        rows.extend(self._combine_microenv(microenv_by_cell, total))
        rows.extend(self._combine_motif(motif_by_cell, total))
        if not rows:
            return pd.DataFrame(columns=list(self.OUTPUT_COLUMNS))
        out = pd.DataFrame(rows).reindex(columns=list(self.OUTPUT_COLUMNS))
        return out.sort_values(
            ["cellline_ratio_pass", "feature_type", "cellline_ratio", "feature_name"],
            ascending=[False, True, False, True], kind="stable").reset_index(drop=True)

    # ------------------------------------------------------------------
    def _combine_microenv(self, frames: Dict[str, pd.DataFrame],
                          total_celllines: int) -> List[Dict]:
        by_factor: Dict[str, Dict[str, Dict]] = {}
        for cell, df in frames.items():
            if df is None or df.empty or "environment_factor" not in df.columns:
                continue
            available = self._available_splits(df)
            passed = df[df["selected"] == True]                     # noqa: E712
            for factor, sub in passed.groupby("environment_factor", dropna=False):
                if not self._supports(available, self._splits(sub)):
                    continue
                by_factor.setdefault(str(factor), {})[cell] = {
                    "effect": float(_num(sub["main_r2_delta"]).mean()),
                    "splits": sorted(set(sub["split_type"].astype(str))),
                    "ci": sorted({f"{lo:.6f}~{hi:.6f}"
                                  for lo, hi in zip(_num(sub["bootstrap_CI_lower"]),
                                                    _num(sub["bootstrap_CI_upper"]))
                                  if np.isfinite(lo) and np.isfinite(hi)}),
                }
        return self._rows_from("microenvironment", by_factor, total_celllines)

    def _combine_motif(self, frames: Dict[str, pd.DataFrame],
                       total_celllines: int) -> List[Dict]:
        by_sequence: Dict[str, Dict[str, Dict]] = {}
        for cell, df in frames.items():
            if df is None or df.empty or "motif" not in df.columns:
                continue
            available = self._available_splits(df)
            passed = df[df["selected"] == True]                     # noqa: E712
            for seq, sub in passed.groupby("motif", dropna=False):
                key = str(seq).strip().upper()
                if not key:
                    continue
                if not self._supports(available, self._splits(sub)):
                    continue
                by_sequence.setdefault(key, {})[cell] = {
                    "effect": float(_num(sub["ISM_effect"]).mean()),
                    "splits": sorted(set(sub["split_type"].astype(str))
                                     if "split_type" in sub.columns else []),
                    # 同一 consensus 在同一细胞系里可能有多条 motif 记录（不同 kernel/method），
                    # 这里只保留**去重后**的 FDR，避免汇总列被同一数值刷屏。
                    "stat": [f"FDR={v:.4g}" for v in
                             sorted(set(_num(sub["BH_FDR"]).dropna().tolist()))],
                }
        return self._rows_from("motif", by_sequence, total_celllines)

    # ------------------------------------------------------------------
    @staticmethod
    def _splits(df: pd.DataFrame) -> set:
        if df is None or df.empty or "split_type" not in df.columns:
            return set()
        return {str(s) for s in df["split_type"].dropna().unique()}

    @staticmethod
    def _available_splits(df: pd.DataFrame) -> set:
        """该细胞系**实际拥有**的 split 集合（来自 ``splits_available`` 列）。

        这一列由发现器在**筛选前**写入，因此即使 CSV 只保留通过筛选的行，
        也仍然知道该细胞系原本有哪几条 split 线——``all_splits`` 判据依赖它。
        """
        if df is None or df.empty or "splits_available" not in df.columns:
            return GeneralizationAnalyzer._splits(df)
        raw = str(df["splits_available"].iloc[0])
        return {s for s in raw.split(";") if s}

    def _supports(self, available: set, selected: set) -> bool:
        """一个细胞系是否算"支持"该 feature。

        ``any_split``：该细胞系在**任一** split 下通过即算支持（默认，较宽松）。
        ``all_splits``：该细胞系在**它拥有的每一个** split 下都通过才算支持
        （例如某细胞系同时有 single 与 all 两条线，就必须两条都通过）。
        """
        if not selected:
            return False
        if self.config.cellline_support_mode == "all_splits":
            return bool(available) and selected >= available
        return True

    # ------------------------------------------------------------------
    def _rows_from(self, feature_type: str, grouped: Dict[str, Dict[str, Dict]],
                   total_celllines: Optional[int] = None) -> List[Dict]:
        total = int(total_celllines) if total_celllines else len(self.config.target_cell_lines)
        rows: List[Dict] = []
        for name, per_cell in grouped.items():
            supporting = sorted(per_cell)
            ratio = (len(supporting) / total) if total else None
            effects = [per_cell[c]["effect"] for c in supporting
                       if np.isfinite(per_cell[c]["effect"])]
            rows.append({
                "feature_type": feature_type,
                "feature_name": name,
                "supporting_celllines": ";".join(supporting),
                "supporting_splits": ";".join(
                    f"{c}:{','.join(per_cell[c]['splits'])}" for c in supporting),
                "n_supporting": len(supporting),
                "n_total_celllines": total,
                "cellline_ratio": ratio,
                "cellline_ratio_pass": self.config.cellline_ratio_pass(ratio),
                "mean_effect": float(np.mean(effects)) if effects else None,
                "effect_variance": float(np.var(effects, ddof=0)) if len(effects) > 1 else None,
                "effect_by_cellline": _fmt_map(
                    {c: per_cell[c]["effect"] for c in supporting
                     if np.isfinite(per_cell[c]["effect"])}),
                "statistical_summary": " | ".join(
                    f"{c}: {';'.join(per_cell[c].get('ci') or per_cell[c].get('stat') or [])}"
                    for c in supporting),
                "source_files": "csv/<cell_line>_microenv.csv;csv/<cell_line>_motif.csv",
                "generated_time": _now_iso(),
            })
        return rows


# ===========================================================================
# 6. Excel 导出
# ===========================================================================
class ExcelExporter:
    """把若干 DataFrame 写进同一个工作簿的不同 Sheet（写前做成 Excel 安全类型）。"""

    #: xlsxwriter 优先（更省内存），缺失时回退 openpyxl
    _ENGINE_PREFERENCE: Tuple[str, ...] = ("xlsxwriter", "openpyxl")

    def __init__(self, config: RuleDiscoveryConfig) -> None:
        self.config = config

    def _engine(self) -> str:
        import importlib.util
        for engine in self._ENGINE_PREFERENCE:
            if importlib.util.find_spec(engine) is not None:
                return engine
        raise RuntimeError(
            "写出 Excel 需要 xlsxwriter 或 openpyxl，两者都没装。\n"
            "  可执行：pip install xlsxwriter    （或 pip install openpyxl）\n"
            "  CSV 原始结果已单独保存，Excel 只是汇总视图。"
        )

    def export(self, sheets: Dict[str, pd.DataFrame], path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        engine = self._engine()
        used: Dict[str, int] = {}
        with pd.ExcelWriter(path, engine=engine) as writer:
            for raw_name, df in sheets.items():
                name = self._safe_sheet_name(raw_name, used)
                frame = self._excel_safe(df)
                frame.to_excel(writer, sheet_name=name, index=False)
                if self.config.excel_max_col_width and engine == "xlsxwriter":
                    self._autosize(writer.sheets[name], frame)
                LOGGER.info("    Sheet %-34s %6d 行 x %2d 列", name, len(frame), frame.shape[1])
        return path

    # ------------------------------------------------------------------
    def _safe_sheet_name(self, raw: str, used: Dict[str, int]) -> str:
        """Excel Sheet 名：去非法字符、截断 31 字符、重名加后缀。"""
        name = "".join(ch for ch in str(raw) if ch not in set(r"[]:*?/\\"))[:31] or "Sheet"
        if name in used:
            used[name] += 1
            suffix = f"_{used[name]}"
            name = name[:31 - len(suffix)] + suffix
        else:
            used[name] = 0
        return name

    @staticmethod
    def _excel_safe(df: pd.DataFrame) -> pd.DataFrame:
        """把 pandas 3 的 nullable/arrow 类型转成 xlsxwriter 能写的普通 object。"""
        if df is None:
            return pd.DataFrame()
        out = df.copy()
        for col in out.columns:
            series = out[col]
            if isinstance(series.dtype, pd.CategoricalDtype):
                out[col] = series.astype(object)
            elif str(series.dtype).startswith(("string", "boolean", "Int", "Float")):
                out[col] = series.astype(object)
        out = out.astype(object)
        return out.where(pd.notna(out), None)

    def _autosize(self, worksheet, df: pd.DataFrame) -> None:
        limit = int(self.config.excel_max_col_width)
        for idx, col in enumerate(df.columns):
            try:
                sample = df[col].astype(str).head(400)
                width = max([len(str(col))] + [len(s) for s in sample]) + 2
            except Exception:                       # noqa: BLE001 - 宽度只是美观
                width = limit
            worksheet.set_column(idx, idx, min(width, limit))


# ===========================================================================
# 7. 主流程
# ===========================================================================
class RuleDiscoveryPipeline:
    """编排：定位批次 -> 装载输入 -> 三个发现器 -> 写 CSV -> 汇总 Excel。"""

    def __init__(self, config: Optional[RuleDiscoveryConfig] = None) -> None:
        self.config = config or RuleDiscoveryConfig()
        self.loader = InputLoader(self.config)
        self.microenv = MicroEnvDiscovery(self.config)
        self.motif = MotifDiscovery(self.config)
        self.generalization = GeneralizationAnalyzer(self.config)
        self.exporter = ExcelExporter(self.config)

    # ------------------------------------------------------------------
    def run(self) -> Dict[str, Path]:
        batches = self.loader.resolve_batches()
        LOGGER.info("待处理批次 %d 个：%s", len(batches), ", ".join(b.name for b in batches))
        outputs: Dict[str, Path] = {}
        for batch in batches:
            LOGGER.info("=" * 78)
            LOGGER.info("批次 %s", batch)
            outputs[batch.name] = self._process_batch(batch)
        return outputs

    # ------------------------------------------------------------------
    def _process_batch(self, batch_dir: Path) -> Path:
        inputs = self.loader.load(batch_dir)
        out_dir = Path(self.config.output_root) / inputs.batch_name
        csv_dir = out_dir / "csv"
        excel_dir = out_dir / "excel"
        csv_dir.mkdir(parents=True, exist_ok=True)
        excel_dir.mkdir(parents=True, exist_ok=True)

        microenv_frames: Dict[str, pd.DataFrame] = {}
        motif_frames: Dict[str, pd.DataFrame] = {}
        sheets: Dict[str, pd.DataFrame] = {}

        # 目标细胞系默认对齐批次实际细胞系：配置里的 4 个 DeepCRISPR 细胞系在别的
        # 数据集上一个都不存在时，退化为该批次实际的细胞系，避免产出 9 张以
        # DeepCRISPR 细胞系命名却全空的 Sheet。
        cells = list(self.config.target_cell_lines)
        actual = set()
        for frame in (inputs.motif_instances, inputs.bootstrap_cellline_effects):
            if frame is not None and not frame.empty and "cell_line" in frame.columns:
                actual |= {str(c).strip().lower() for c in frame["cell_line"].dropna()}
        if actual and not (actual & {str(c).strip().lower() for c in cells}):
            cells = sorted(actual)
            LOGGER.info("  配置的目标细胞系 %s 在本批次不存在，改用批次实际细胞系 %s",
                        list(self.config.target_cell_lines), cells)

        for cell in cells:
            display = self.config.display_name(cell)
            LOGGER.info("  --- 细胞系 %s ---", display)

            env = self.microenv.discover(inputs, cell)
            self._report_stage(
                f"{display} microenv", env,
                stages=(("第一层 Statistical Evidence (CI 不跨 0)", "CI_pass"),
                        (f"第二层 Effect (|ΔR²| >= {self.config.microenv_min_abs_main_r2_delta:g})",
                         "effect_pass")))
            env_out = self._finalize(env)
            microenv_frames[cell] = env_out
            self._write_csv(env_out, csv_dir / f"{cell}_microenv.csv")
            sheets[f"{display}_microenv"] = env_out

            mtf = self.motif.discover(inputs, cell)
            self._report_stage(
                f"{display} motif", mtf,
                stages=((f"第一层 Statistical Evidence (BH-FDR < {self.config.motif_fdr_threshold:g})",
                         "FDR_pass"),
                        (f"第二层 Effect (|ISM| >= {self.config.motif_abs_ism_effect_threshold:g})",
                         "ISM_effect_pass")))
            mtf_out = self._finalize(mtf)
            motif_frames[cell] = mtf_out
            self._write_csv(mtf_out, csv_dir / f"{cell}_motif.csv")
            sheets[f"{display}_motif"] = mtf_out

        combined_full = self.generalization.combine(microenv_frames, motif_frames,
                                                    total_celllines=len(cells))
        LOGGER.info("  --- 跨细胞系泛化 (%s) ---", self.config.ratio_rule_text())
        LOGGER.info("    输入 %d 个候选 -> 通过 %d 个（删除 %d）",
                    len(combined_full),
                    int(combined_full["cellline_ratio_pass"].sum()) if not combined_full.empty else 0,
                    len(combined_full) - int(combined_full["cellline_ratio_pass"].sum())
                    if not combined_full.empty else 0)
        combined = combined_full
        if not self.config.include_rejected_rows and not combined.empty:
            combined = combined[combined["cellline_ratio_pass"] == True]  # noqa: E712
            combined = combined.reset_index(drop=True)
        self._write_csv(combined, csv_dir / "microenv_and_motif.csv")
        sheets["microenv_and_motif"] = combined

        xlsx = self.exporter.export(sheets, excel_dir / self.config.excel_filename)
        LOGGER.info("  CSV  -> %s", _rel(csv_dir))
        LOGGER.info("  Excel-> %s", _rel(xlsx))
        return xlsx

    # ------------------------------------------------------------------
    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if self.config.include_rejected_rows or "selected" not in df.columns:
            return df.reset_index(drop=True)
        return df[df["selected"] == True].reset_index(drop=True)     # noqa: E712

    def _report_stage(self, label: str, df: pd.DataFrame,
                      stages: Sequence[Tuple[str, str]]) -> None:
        """按层汇报：输入 -> 每层保留/删除 -> 最终候选。

        统计是**递进**的：第二层的"保留"只数在第一层已通过的行里又通过第二层的行数，
        这样"删除"列才始终 >= 0，且能看出每一层各自砍掉了多少。
        """
        n_in = 0 if df is None else len(df)
        LOGGER.info("    [%s] 输入 %d 行", label, n_in)
        alive = pd.Series(True, index=df.index) if (df is not None and not df.empty) \
            else pd.Series(dtype=bool)
        prev = n_in
        for name, column in stages:
            if df is None or df.empty or column not in df.columns:
                LOGGER.info("      - %-46s 保留 %5d, 删除 %5d", name, 0, prev)
                prev = 0
                continue
            hit = pd.Series(df[column].fillna(False).astype(bool), index=df.index)
            alive = alive & hit
            keep = int(alive.sum())
            LOGGER.info("      - %-46s 保留 %5d, 删除 %5d", name, keep, prev - keep)
            prev = keep
        LOGGER.info("      => 最终候选 %d 行", prev)

    def _write_csv(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = df if df is not None else pd.DataFrame()
        frame.to_csv(path, index=False, encoding=self.config.csv_encoding)
        LOGGER.info("    写出 %-52s %6d 行", _rel(path), len(frame))


# ===========================================================================
# 8. CLI
# ===========================================================================
def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=("赛道二规律发现交付物生成器：整合已有分析产物，"
                     "按 Statistical Evidence -> Effect (-> Robustness) 筛选候选 "
                     "microenvironment factor 与 motif，输出 CSV + Excel。"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python rule_discovery.py                                    # 自动探测全部批次\n"
            "  python rule_discovery.py --data-set DeepCRISPR\n"
            "  python rule_discovery.py --batch-dir results/summary/DeepCRISPR\n"
            "  python rule_discovery.py --print-config                     # 打印生效的评选标准\n"
            "  python rule_discovery.py --config-json my_thresholds.json   # 覆盖部分阈值\n"
        ),
    )
    ap.add_argument("--batch-dir", action="append", default=[], metavar="PATH",
                    help="批次目录（含 tables/）；可重复。默认为空 = 自动扫描 --summary-root")
    ap.add_argument("--data-set", "--data_set", "--dataset", dest="data_set",
                    action="append", default=[],
                    help="按数据集名指定批次（在 --summary-root 下大小写不敏感匹配）；可重复")
    ap.add_argument("--summary-root", default="results/summary",
                    help="批次汇总根目录（默认 results/summary）")
    ap.add_argument("--output", default="", help="输出根目录（默认 results/赛道二_results）")
    ap.add_argument("--config-json", default="",
                    help="JSON 文件，覆盖 RuleDiscoveryConfig 里的任意字段")
    ap.add_argument("--include-rejected", action="store_true",
                    help="CSV 保留全部输入行并附 selected 标志（默认只保留通过筛选的行）")
    ap.add_argument("--print-config", action="store_true", help="打印生效配置后退出")
    return ap


def _config_from_args(args) -> RuleDiscoveryConfig:
    config = RuleDiscoveryConfig()
    if args.summary_root:
        config.summary_root = args.summary_root
    if args.output:
        config.output_root = args.output
    if args.include_rejected:
        config.include_rejected_rows = True

    if args.batch_dir:
        config.batch_dirs = tuple(args.batch_dir)
    if args.data_set:
        config.dataset_names = tuple(args.data_set)

    if args.config_json:
        path = Path(args.config_json)
        if not path.is_file():
            raise FileNotFoundError(f"--config-json 指定的文件不存在：{path}")
        overrides = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("--config-json 必须是 JSON object（字段名 -> 新值）")
        valid = {f.name for f in fields(RuleDiscoveryConfig)}
        unknown = sorted(set(overrides) - valid)
        if unknown:
            raise ValueError(
                f"--config-json 含未知字段：{unknown}\n  可用字段：{sorted(valid)}")
        for key, value in overrides.items():
            if key in ("target_cell_lines", "microenv_nonlinear_models",
                       "batch_dirs", "dataset_names"):
                value = tuple(value)
            setattr(config, key, value)
        LOGGER.info("已应用 --config-json 覆盖 %d 个字段：%s",
                    len(overrides), ", ".join(sorted(overrides)))
    config.__post_init__()
    return config


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)

    if args.print_config:
        print(json.dumps(config.as_dict(), ensure_ascii=False, indent=2, default=str))
        return 0

    LOGGER.info("rule_discovery v%s | 输出根目录 %s", PIPELINE_VERSION, config.output_root)
    LOGGER.info("筛选顺序: Statistical Evidence -> Effect -> (跨细胞系 Robustness)")
    try:
        outputs = RuleDiscoveryPipeline(config).run()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        LOGGER.error("%s", exc)
        return 2
    LOGGER.info("=" * 78)
    LOGGER.info("完成：%d 个批次", len(outputs))
    for name, path in outputs.items():
        LOGGER.info("  %-24s -> %s", name, _rel(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
