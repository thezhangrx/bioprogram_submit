# -*- coding: utf-8 -*-
"""core.features.engineering.label_normalization — 编辑效率（y）归一化程序。

把「原始标签列 → 落盘 [0,1] 目标值」这一步抽出来，做成**配置驱动、可审计**的独立程序。

为什么要独立出来
----------------
在此之前，归一化是**散落且隐式**的：

* ``adapt_hiranniramol`` 里硬编码 ``eff / 100.0``；
* ``adapt_deepcrispr`` / ``adapt_labuhn`` 依赖「原始列本来就是 [0,1]」的假设，不做任何变换；
* ``data/config/*.json`` 里的 ``label_column`` / ``label_scale`` **只是注释**，没有任何代码读它们。

结果是「某个数据集到底有没有被归一化、原始列叫什么」只能靠读代码猜。本模块把它变成
一条显式、可校验、能写进报告与配置的步骤。

支持的方法
----------
============  ==================================================================
``none``      原始列已是 [0,1]；**只校验，不缩放**
``divide``    除以常数（如百分制 ``Edit Efficiency`` / 100）
``minmax``    ``(y - lo) / (hi - lo)``；``lo`` / ``hi`` 必须由**训练集**统计量给出
============  ==================================================================

``minmax`` 的 ``lo`` / ``hi`` **不能在全量数据上拟合** —— 那会把验证/测试集的极值泄漏进
训练输入。因此本模块不自己算它们，只接受调用方显式传入；配置里没写就报错，绝不静默回退。

用法::

    from core.features.engineering.label_normalization import LabelNormalizationSpec

    spec = LabelNormalizationSpec.from_config(config)     # 读 data/config/<数据集>.json
    y, info = spec.apply(raw_series)

审计（列出每个数据集的原始列 / 是否归一化 / 方法 / 变换前后范围）::

    python -m core.features.engineering.label_normalization
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: 支持的归一化方法
VALID_METHODS: Tuple[str, ...] = ("none", "divide", "minmax")

#: 落盘目标值的合法区间（与 dataset_adapters._finalize 的校验一致）
TARGET_LO, TARGET_HI = 0.0, 1.0
_RANGE_TOL = 1e-9

_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "data" / "config"


class LabelNormalizationError(ValueError):
    """配置或数据不满足归一化契约时抛出（不静默兜底）。"""


def _parse_yes_no(value: object, where: str) -> bool:
    s = str(value).strip().lower()
    if s in _YES:
        return True
    if s in _NO:
        return False
    raise LabelNormalizationError(
        f"{where}: 只接受 yes/no（或 y/n、true/false、1/0），实际={value!r}")


@dataclass
class LabelNormalizationResult:
    """一次归一化的审计记录。"""

    dataset: str = ""
    raw_column: str = ""
    normalized: bool = False
    method: str = "none"
    detail: str = ""
    note: str = ""
    n_in: int = 0
    n_nan: int = 0
    range_in: Tuple[float, float] = (float("nan"), float("nan"))
    range_out: Tuple[float, float] = (float("nan"), float("nan"))

    def line(self) -> str:
        flag = "yes" if self.normalized else "no"
        return (f"[{self.dataset}] y 原始列={self.raw_column!r} 归一化={flag}"
                f" 方法={self.method}{self.detail}"
                f" | {self.range_in[0]:.4g}~{self.range_in[1]:.4g}"
                f" -> {self.range_out[0]:.4g}~{self.range_out[1]:.4g}"
                f" | n={self.n_in}（NaN {self.n_nan}）")

    def as_dict(self) -> Dict[str, object]:
        return {
            "dataset": self.dataset,
            "label_raw_column": self.raw_column,
            "label_normalized": "yes" if self.normalized else "no",
            "label_normalization_method": self.method,
            "label_range_in": list(self.range_in),
            "label_range_out": list(self.range_out),
            "n": self.n_in,
            "n_nan": self.n_nan,
        }


@dataclass
class LabelNormalizationSpec:
    """一条「原始标签列 → [0,1]」的变换契约。

    字段与 ``data/config/<数据集>.json`` 的对应关系::

        label_column                -> raw_column      （y 的原始数据列标签）
        label_normalized            -> normalized      （"yes" / "no"）
        label_normalization.method  -> method          （none / divide / minmax）
        label_normalization.divisor -> divisor         （divide 用）
        label_normalization.minmax_range -> minmax_range（minmax 用，训练集统计量）
        label_normalization.note    -> note
    """

    raw_column: str
    normalized: bool
    method: str = "none"
    divisor: Optional[float] = None
    minmax_range: Optional[Tuple[float, float]] = None
    note: str = ""
    dataset: str = ""

    # ---------------- 构造 ----------------
    @classmethod
    def from_config(cls, config: Dict, dataset: Optional[str] = None) -> "LabelNormalizationSpec":
        """从数据集配置构造；字段缺失或自相矛盾时**报错**，不做猜测。"""
        if not isinstance(config, dict):
            raise LabelNormalizationError(f"config 必须是 dict，实际 {type(config).__name__}")
        ds = dataset or str(config.get("name") or config.get("__dataset_config__") or "")
        where = f"[{ds or '?'}] label_normalization"

        raw_column = config.get("label_column")
        if not raw_column or not str(raw_column).strip():
            raise LabelNormalizationError(
                f"{where}: 配置缺少 label_column（y 的原始数据列标签）")

        block = config.get("label_normalization") or {}
        if not isinstance(block, dict):
            raise LabelNormalizationError(f"{where}: 必须是对象，实际 {type(block).__name__}")

        method = str(block.get("method", "none")).strip().lower()
        if method not in VALID_METHODS:
            raise LabelNormalizationError(
                f"{where}: 未知 method={method!r}，可选 {list(VALID_METHODS)}")

        # label_normalized 显式给出时以它为准，并与 method 交叉校验；
        # 没给就按 method 推断（向后兼容旧配置）。
        if "label_normalized" in config:
            normalized = _parse_yes_no(config["label_normalized"],
                                       f"{where}.label_normalized")
            if normalized != (method != "none"):
                raise LabelNormalizationError(
                    f"{where}: label_normalized={'yes' if normalized else 'no'} 与 "
                    f"method={method!r} 矛盾 —— "
                    f"{'method=none 表示不归一化' if normalized else '非 none 方法表示会归一化'}")
        else:
            normalized = method != "none"

        divisor = block.get("divisor")
        if method == "divide":
            if divisor is None:
                raise LabelNormalizationError(f"{where}: method='divide' 必须给 divisor")
            divisor = float(divisor)
            if not np.isfinite(divisor) or divisor <= 0:
                raise LabelNormalizationError(f"{where}: divisor 必须是正有限数，实际={divisor}")
        else:
            divisor = None

        mm = block.get("minmax_range")
        if method == "minmax":
            if not (isinstance(mm, (list, tuple)) and len(mm) == 2):
                raise LabelNormalizationError(
                    f"{where}: method='minmax' 必须给 minmax_range=[lo, hi]（**训练集**统计量）")
            lo, hi = float(mm[0]), float(mm[1])
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
                raise LabelNormalizationError(
                    f"{where}: minmax_range 需满足 hi > lo，实际={mm}")
            minmax_range = (lo, hi)
        else:
            minmax_range = None

        return cls(raw_column=str(raw_column).strip(), normalized=bool(normalized),
                   method=method, divisor=divisor, minmax_range=minmax_range,
                   note=str(block.get("note", "") or ""), dataset=ds)

    # ---------------- 应用 ----------------
    def detail(self) -> str:
        if self.method == "divide":
            return f"(÷{self.divisor:g})"
        if self.method == "minmax":
            lo, hi = self.minmax_range or (float("nan"), float("nan"))
            return f"([{lo:g}, {hi:g}] 线性映射)"
        return "(仅校验)"

    def input_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        """**原始**标签列的合法区间（用于剔除越界行）。

        ``none`` 与 ``divide`` 能推出上界，越界即数据有问题；``minmax`` 无从推断，返回
        ``(None, None)`` 表示只要求有限值。
        """
        if self.method == "none":
            return (TARGET_LO, TARGET_HI)
        if self.method == "divide":
            return (0.0, float(self.divisor))          # type: ignore[arg-type]
        return (None, None)

    def describe(self) -> str:
        """一行式口径描述，写进 AdaptReport.target_scale。"""
        flag = "yes" if self.normalized else "no"
        return (f"y<-{self.raw_column!r} 归一化={flag} "
                f"method={self.method}{self.detail()}")

    def apply(self, values, check_range: bool = True) -> Tuple[np.ndarray, LabelNormalizationResult]:
        """把原始标签值映射到 [0,1]，并返回审计记录。

        NaN 原样保留（由调用方决定是否丢弃），不参与范围校验。
        """
        arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=np.float64)
        n_in = int(arr.size)
        finite = np.isfinite(arr)
        n_nan = int(n_in - finite.sum())

        if self.method == "none":
            out = arr.copy()
        elif self.method == "divide":
            out = arr / float(self.divisor)          # type: ignore[arg-type]
        elif self.method == "minmax":
            lo, hi = self.minmax_range               # type: ignore[misc]
            out = (arr - float(lo)) / (float(hi) - float(lo))
        else:                                        # pragma: no cover - from_config 已挡
            raise LabelNormalizationError(f"未知 method={self.method!r}")

        info = LabelNormalizationResult(
            dataset=self.dataset, raw_column=self.raw_column, normalized=self.normalized,
            method=self.method, detail=self.detail(), n_in=n_in, n_nan=n_nan,
            range_in=(float(np.nanmin(arr)) if finite.any() else float("nan"),
                      float(np.nanmax(arr)) if finite.any() else float("nan")),
            range_out=(float(np.nanmin(out)) if finite.any() else float("nan"),
                       float(np.nanmax(out)) if finite.any() else float("nan")),
        )

        if check_range and self.normalized and finite.any():
            lo_o, hi_o = info.range_out
            if not (TARGET_LO - _RANGE_TOL <= lo_o and hi_o <= TARGET_HI + _RANGE_TOL):
                raise LabelNormalizationError(
                    f"[{self.dataset or '?'}] 归一化后仍在 [0,1] 之外：{lo_o:.6g} ~ {hi_o:.6g}"
                    f"（方法 {self.method}{self.detail()}）。请检查 label_normalization 配置。")
        return out, info


# --------------------------------------------------------------------------- #
# 批量审计
# --------------------------------------------------------------------------- #
def load_dataset_config(path: Path) -> Dict:
    with Path(path).open(encoding="utf-8") as fh:
        return json.load(fh)


def load_specs(config_dir: Optional[Path] = None) -> List[LabelNormalizationSpec]:
    """读取目录下所有 ``*.json`` 配置，构造各自的归一化契约。"""
    d = Path(config_dir or CONFIG_DIR)
    specs: List[LabelNormalizationSpec] = []
    for f in sorted(d.glob("*.json")):
        cfg = load_dataset_config(f)
        specs.append(LabelNormalizationSpec.from_config(cfg, dataset=cfg.get("name") or f.stem))
    return specs


def audit(config_dir: Optional[Path] = None) -> List[LabelNormalizationResult]:
    """**静态**报告：只读配置、不碰数据，因此 range 为空。"""
    out: List[LabelNormalizationResult] = []
    for spec in load_specs(config_dir):
        out.append(LabelNormalizationResult(
            dataset=spec.dataset, raw_column=spec.raw_column, normalized=spec.normalized,
            method=spec.method, detail=spec.detail(), note=spec.note))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m core.features.engineering.label_normalization",
        description="编辑效率（y）归一化：审计配置，或对某个原始列实做变换并报告前后范围")
    ap.add_argument("--config-dir", default=str(CONFIG_DIR),
                    help="数据集配置目录（默认 data/config）")
    ap.add_argument("--raw-file", default="",
                    help="给一个原始 CSV，实际执行一次变换并报告范围")
    ap.add_argument("--column", default="",
                    help="配合 --raw-file：覆盖配置里的标签列名")
    args = ap.parse_args(argv)

    specs = load_specs(Path(args.config_dir))
    print(f"数据集配置归一化口径（{args.config_dir}）：")
    for sp in specs:
        flag = "yes" if sp.normalized else "no"
        print(f"  {sp.dataset or '?':<16} y 原始列={sp.raw_column!r:<24} 归一化={flag:<4} "
              f"方法={sp.method}{sp.detail()}")
        if sp.note:
            print(f"  {'':<16} {sp.note}")

    if args.raw_file:
        print()
        raw = pd.read_csv(args.raw_file, low_memory=False)
        lower = {str(c).strip().lower(): c for c in raw.columns}
        for sp in specs:
            col = args.column or sp.raw_column
            match = lower.get(str(col).strip().lower())
            if match is None:
                print(f"  [{sp.dataset}] 原始文件里没有列 {col!r}，跳过")
                continue
            _, info = sp.apply(raw[match], check_range=True)
            print(f"  {info.line()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
