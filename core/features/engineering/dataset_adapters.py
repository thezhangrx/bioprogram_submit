"""外部数据集的输入适配层（raw → 工程规范格式）。

背景
----
工程的 raw 输入约定（``DeepCRISPR`` 格式）是::

    Chromosome, Start, End, Strand, sgRNA, [4 个表观通道], Normalized efficacy

其中 ``sgRNA`` 必须是 **23 nt**（20 nt spacer + 3 nt PAM，PAM 以 GG 结尾），
``Normalized efficacy`` 必须落在 [0, 1]。

Hiranniramol 与 Labuhn 是两篇论文的补充材料表，列名与序列切片方式都不同，
且**不含表观遗传通道**。本模块只负责"把不同来源的原始表变成上述规范列"，
不负责特征工程本身（那是 ``feature_engineering.py`` 的职责）。

设计原则
--------
* **不静默丢数据**：每一行被丢弃都必须有原因并计入 ``report``；丢弃比例过高直接报错。
* **不猜**：spacer 在 extended 序列中的位置必须唯一命中；PAM 不是 GG 就报错，
  除非显式允许（``allow_non_gg_pam``）。
* **量纲显式**：每个数据集声明自己的效率量纲（0-100 还是 0-1），转换写在适配器里。

命令行自检::

    python core/features/engineering/dataset_adapters.py --list
    python core/features/engineering/dataset_adapters.py \
        --raw-data data/raw/Labuhn/Labuhn.CSV --format labuhn --dry-run
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.features.engineering.label_normalization import (      # noqa: E402
    LabelNormalizationError,
    LabelNormalizationSpec,
)

#: 规范列名
CANONICAL_TARGET = "Normalized efficacy"

#: 各数据集「原始标签列 -> [0,1]」的**缺省**契约。
#: 调用方（feature_engineering）通常会从 ``data/config/<数据集>.json`` 构造一份传进来；
#: 这几份只是不传参时的回退，且与配置里的取值保持一致，保证行为不变。
DEFAULT_LABEL_SPECS: Dict[str, "LabelNormalizationSpec"] = {
    "deepcrispr": LabelNormalizationSpec(
        raw_column="Normalized efficacy", normalized=False, method="none",
        dataset="deepcrispr"),
    "labuhn": LabelNormalizationSpec(
        raw_column="KO_reporter_assay", normalized=False, method="none",
        dataset="labuhn"),
    "hiranniramol": LabelNormalizationSpec(
        raw_column="Edit Efficiency", normalized=True, method="divide", divisor=100.0,
        dataset="hiranniramol"),
}
CANONICAL_SEQUENCE = "sgRNA"

#: 表观通道列（仅 DeepCRISPR 提供；其余数据集没有）
EPI_COLUMNS = ["CTCF", "Dnase", "H3K4me3", "RRBS"]

#: 坐标列（仅 DeepCRISPR 提供）
COORD_COLUMNS = ["Chromosome", "Start", "End", "Strand"]

#: 引导序列（protospacer）长度：原始文件里 spacer 列的字符数，用于在
#: extended 序列中定位。注意它是**输入侧**属性，与张量长度无关。
SPACER_LENGTH = 20
#: PAM 长度（NGG = 3 nt）。
PAM_LENGTH = 3
#: **默认**张量序列长度；实际取值由数据集配置的 ``sequence_length`` 给出，
#: 并作为参数传进各适配器（``load_and_adapt(..., sequence_length=...)``）。
SEQUENCE_LENGTH = SPACER_LENGTH + PAM_LENGTH

#: 允许多大比例的行被丢弃；超过即视为原始文件结构不符，直接失败
MAX_DROP_FRACTION = 0.20


@dataclass
class AdaptReport:
    """适配过程的完整账目（用于打印与写进 summary）。"""

    dataset: str
    source: str = ""
    rows_in: int = 0
    rows_out: int = 0
    n_dropped: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    target_range_in: Tuple[float, float] = (float("nan"), float("nan"))
    target_range_out: Tuple[float, float] = (float("nan"), float("nan"))
    target_scale: str = ""
    has_epigenetics: bool = False
    pam_gg_fraction: float = float("nan")
    #: 同一 sgRNA 被重复测量、已按均值合并的序列数（Labuhn 有 5 条）
    n_replicates_merged: int = 0

    @property
    def drop_fraction(self) -> float:
        return (self.n_dropped / self.rows_in) if self.rows_in else 0.0

    def line(self) -> str:
        detail = ", ".join(f"{k}={v}" for k, v in sorted(self.drop_reasons.items())) or "-"
        return (f"[{self.dataset}] {self.rows_in} 行 -> {self.rows_out} 行"
                f"（丢弃 {self.n_dropped}，{self.drop_fraction:.2%}；原因：{detail}）"
                f" | 重复测量合并 {self.n_replicates_merged} 条序列"
                f" | 效率 {self.target_range_in[0]:.4g}~{self.target_range_in[1]:.4g}"
                f" {self.target_scale} -> {self.target_range_out[0]:.4g}~{self.target_range_out[1]:.4g}"
                f" | 表观通道={'有' if self.has_epigenetics else '无'}"
                f" | PAM-GG={self.pam_gg_fraction:.2%}")


# --------------------------------------------------------------------------- #
# 公共工具
# --------------------------------------------------------------------------- #
def _norm(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def _build_full_sequence(spacer: str, extended: str, *,
                         allow_non_gg_pam: bool = False,
                         sequence_length: int = SEQUENCE_LENGTH) -> Tuple[Optional[str], str]:
    """从 (20nt spacer, extended 序列) 切出全长序列 = spacer + PAM。

    长度由 ``sequence_length`` 给出（默认 23 = 20 nt spacer + 3 nt PAM），
    PAM 长度 = ``sequence_length - SPACER_LENGTH``。返回 ``(序列或 None, 失败原因)``。
    """
    length = int(sequence_length)
    pam_length = length - SPACER_LENGTH
    if pam_length < 0:
        raise ValueError(
            f"sequence_length={length} 小于 SPACER_LENGTH({SPACER_LENGTH})，"
            "无法从 spacer + extended 切出合法序列。"
        )

    spacer = str(spacer).strip().upper()
    extended = str(extended).strip().upper()

    if len(spacer) != SPACER_LENGTH:
        return None, "spacer_length"
    if not set(spacer) <= set("ACGT"):
        return None, "spacer_non_acgt"

    first = extended.find(spacer)
    if first < 0:
        return None, "spacer_not_in_extended"
    if extended.find(spacer, first + 1) >= 0:
        return None, "spacer_ambiguous"

    seq = extended[first:first + length]
    if len(seq) != length:
        return None, "extended_too_short"
    # pam_length < 2 时序列里没有 PAM 区（纯 protospacer），跳过 PAM 校验。
    # 注意不能写成 seq[-0:]——那等于整条序列。
    if pam_length >= 2 and not seq[-pam_length:].endswith("GG") and not allow_non_gg_pam:
        return None, "pam_not_gg"
    return seq, ""


#: 向后兼容旧名（本文件私有函数，旧调用方仍可用）。
_build_23nt = _build_full_sequence


def _finalize(df: pd.DataFrame, report: AdaptReport, *, allow_non_gg_pam: bool,
              duplicate_policy: str = "mean",
              pam_length: int = PAM_LENGTH) -> pd.DataFrame:
    """统一收尾：校验丢弃比例、去重、生成 report。"""
    if report.rows_in and report.drop_fraction > MAX_DROP_FRACTION:
        raise ValueError(
            f"[{report.dataset}] 丢弃比例过高（{report.drop_fraction:.1%} > "
            f"{MAX_DROP_FRACTION:.0%}），原始文件结构与适配器假设不符。\n"
            f"丢弃原因统计：{report.drop_reasons}\n"
            f"请核对列名与序列切片方式，不要忽略该错误。"
        )

    # 先去掉"序列+标签"完全相同的行
    before = len(df)
    df = df.drop_duplicates(subset=[CANONICAL_SEQUENCE, CANONICAL_TARGET], keep="first")
    if len(df) < before:
        report.drop_reasons["exact_duplicate_rows"] = before - len(df)

    # 再处理"同一 sgRNA、不同标签"的重复测量：取均值合并。
    # Labuhn 有 5 条这样的序列（同一 guide 测了两次，KO_reporter_assay 不同），
    # 若保留两行会给模型同一个输入两个不同标签，也会让"每行唯一序列"的约定失效。
    if duplicate_policy == "mean" and df[CANONICAL_SEQUENCE].duplicated().any():
        n_dup_seqs = int(df[CANONICAL_SEQUENCE].duplicated().sum())
        agg = {CANONICAL_TARGET: "mean"}
        for col in df.columns:
            if col not in (CANONICAL_SEQUENCE, CANONICAL_TARGET):
                agg[col] = "first"
        df = (df.groupby(CANONICAL_SEQUENCE, as_index=False)
                .agg(agg)
                .sort_values(CANONICAL_SEQUENCE, kind="stable")
                .reset_index(drop=True))
        report.n_replicates_merged = n_dup_seqs
    elif duplicate_policy == "error" and df[CANONICAL_SEQUENCE].duplicated().any():
        dup = df.loc[df[CANONICAL_SEQUENCE].duplicated(), CANONICAL_SEQUENCE].unique()[:5]
        raise ValueError(
            f"[{report.dataset}] 存在同一 sgRNA 的重复测量（duplicate_policy='error'）：{list(dup)}"
        )

    df = df.reset_index(drop=True)
    report.rows_out = len(df)

    vals = df[CANONICAL_TARGET].astype(float)
    report.target_range_out = (float(vals.min()), float(vals.max()))
    if int(pam_length) >= 2:
        report.pam_gg_fraction = float(
            df[CANONICAL_SEQUENCE].str[-int(pam_length):].str.endswith("GG").mean())
    else:
        # 张量里没有 PAM 区（sequence_length == protospacer_length），比例无定义
        report.pam_gg_fraction = float("nan")

    if not (0.0 <= vals.min() and vals.max() <= 1.0 + 1e-9):
        raise ValueError(
            f"[{report.dataset}] 规范化后效率仍在 [0,1] 之外："
            f"{vals.min()} ~ {vals.max()}，请检查 target_scale。"
        )
    return df


# --------------------------------------------------------------------------- #
# 各数据集适配器
# --------------------------------------------------------------------------- #
def adapt_deepcrispr(df: pd.DataFrame, *, allow_non_gg_pam: bool = False,
                    duplicate_policy: str = "mean",
                    sequence_length: int = SEQUENCE_LENGTH,
                    label_spec: Optional[LabelNormalizationSpec] = None,
                    ) -> Tuple[pd.DataFrame, AdaptReport]:
    """DeepCRISPR：本身已是规范格式，只做校验与列归一。

    ``sequence_length`` 由数据集配置给出；本数据集的序列列已是全长
    （含 PAM），因此直接按该长度校验。
    """
    report = AdaptReport(dataset="deepcrispr", has_epigenetics=True)   # target_scale 由 spec.describe() 填
    report.rows_in = len(df)

    need = COORD_COLUMNS + [CANONICAL_SEQUENCE, CANONICAL_TARGET]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[deepcrispr] 缺少规范列：{missing}；实际列={list(df.columns)}")

    out = df.copy()
    out[CANONICAL_SEQUENCE] = _norm(out[CANONICAL_SEQUENCE])
    if "Strand" in out.columns:
        out["Strand"] = out["Strand"].astype(str).str.strip()

    length = int(sequence_length)
    pam_length = length - SPACER_LENGTH
    bad_len = out[CANONICAL_SEQUENCE].str.len() != length
    bad_char = ~out[CANONICAL_SEQUENCE].str.fullmatch(r"[ACGT]+").fillna(False)
    if pam_length >= 2:
        bad_pam = ~out[CANONICAL_SEQUENCE].str[-pam_length:].str.endswith("GG")
    else:
        bad_pam = pd.Series(False, index=out.index)
    keep = ~(bad_len | bad_char | (bad_pam & (not allow_non_gg_pam)))
    for mask, name in ((bad_len, "sequence_length"),
                       (bad_char, "sequence_non_acgt"),
                       (bad_pam, "pam_not_gg")):
        n = int(mask.sum())
        if n and not (name == "pam_not_gg" and allow_non_gg_pam):
            report.drop_reasons[name] = n
    report.n_dropped = int((~keep).sum())
    out = out[keep]

    spec = label_spec or DEFAULT_LABEL_SPECS["deepcrispr"]
    report.target_range_in = (float(out[CANONICAL_TARGET].min()), float(out[CANONICAL_TARGET].max()))
    out = out.copy()
    out[CANONICAL_TARGET], _ = spec.apply(out[CANONICAL_TARGET])
    report.target_scale = spec.describe()
    for col in EPI_COLUMNS:
        if col not in out.columns:
            report.has_epigenetics = False
    return _finalize(out[need + [c for c in EPI_COLUMNS if c in out.columns]],
                     report, allow_non_gg_pam=allow_non_gg_pam,
                     duplicate_policy=duplicate_policy,
                     pam_length=pam_length), report


def adapt_hiranniramol(df: pd.DataFrame, *, allow_non_gg_pam: bool = False,
                       duplicate_policy: str = "mean",
                       sequence_length: int = SEQUENCE_LENGTH,
                       label_spec: Optional[LabelNormalizationSpec] = None,
                       ) -> Tuple[pd.DataFrame, AdaptReport]:
    """Hiranniramol 补充表 2：``Edit Efficiency``(0-100) + ``gRNA``(20nt) + ``Extended Target``。

    效率为 0--100 百分制，这里统一除以 100 变成 [0,1]。
    """
    report = AdaptReport(dataset="hiranniramol", has_epigenetics=False)   # target_scale 由 spec.describe() 填
    report.rows_in = len(df)

    colmap = {str(c).strip().lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n.lower() in colmap:
                return colmap[n.lower()]
        raise ValueError(f"[hiranniramol] 找不到列 {names}；实际列={list(df.columns)}")

    c_eff = pick("Edit Efficiency", "EditEfficiency")
    c_grna = pick("gRNA")
    c_ext = pick("Extended Target", "ExtendedTarget")

    spec = label_spec or DEFAULT_LABEL_SPECS["hiranniramol"]
    eff = pd.to_numeric(df[c_eff], errors="coerce")
    report.target_range_in = (float(eff.min()), float(eff.max()))
    # 合法区间由归一化契约推出：none -> [0,1]，divide -> [0, divisor]，minmax -> 不限
    lo_in, hi_in = spec.input_bounds()
    bad_eff = eff.isna()
    if lo_in is not None:
        bad_eff = bad_eff | (eff < lo_in)
    if hi_in is not None:
        bad_eff = bad_eff | (eff > hi_in)
    if int(bad_eff.sum()):
        report.drop_reasons["target_missing_or_out_of_range"] = int(bad_eff.sum())

    grna = _norm(df[c_grna])
    ext = _norm(df[c_ext])

    seqs, reasons = [], []
    for g, e in zip(grna, ext):
        s, why = _build_full_sequence(g, e, allow_non_gg_pam=allow_non_gg_pam,
                                      sequence_length=sequence_length)
        seqs.append(s)
        reasons.append(why)
    seq_ser = pd.Series(seqs, index=df.index)
    reason_ser = pd.Series(reasons, index=df.index)

    bad_seq = seq_ser.isna()
    for why, n in reason_ser[bad_seq].value_counts().items():
        report.drop_reasons[why] = report.drop_reasons.get(why, 0) + int(n)

    keep = ~(bad_eff | bad_seq)
    report.n_dropped = int((~keep).sum())

    y, _ = spec.apply(eff[keep])
    report.target_scale = spec.describe()
    out = pd.DataFrame({
        CANONICAL_SEQUENCE: seq_ser[keep],
        CANONICAL_TARGET: y,
    })
    return _finalize(out, report, allow_non_gg_pam=allow_non_gg_pam,
                     duplicate_policy=duplicate_policy,
                     pam_length=int(sequence_length) - SPACER_LENGTH), report


def adapt_labuhn(df: pd.DataFrame, *, allow_non_gg_pam: bool = False,
                duplicate_policy: str = "mean",
                sequence_length: int = SEQUENCE_LENGTH,
                label_spec: Optional[LabelNormalizationSpec] = None,
                ) -> Tuple[pd.DataFrame, AdaptReport]:
    """Labuhn 补充表 1：真实效率列为 ``KO_reporter_assay``（已是 [0,1]）。

    序列列为 ``sgRNA_sequence``(20nt) 与 ``extended_spacer``；``strand`` 映射为 +/-。
    注意：该表的 CRISPRater/SSC/CRISPRscan 等列是**其它预测器打分**，
    属于替代方法而非输入特征，不进入特征矩阵。
    """
    report = AdaptReport(dataset="labuhn", has_epigenetics=False)   # target_scale 由 spec.describe() 填
    report.rows_in = len(df)

    need = ["KO_reporter_assay", "sgRNA_sequence", "extended_spacer"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"[labuhn] 缺少列 {missing}；实际列={list(df.columns)}")

    eff = pd.to_numeric(df["KO_reporter_assay"], errors="coerce")
    report.target_range_in = (float(eff.min()), float(eff.max()))
    bad_eff = eff.isna() | (eff < 0) | (eff > 1)
    if int(bad_eff.sum()):
        report.drop_reasons["target_missing_or_out_of_range"] = int(bad_eff.sum())

    spacer = _norm(df["sgRNA_sequence"])
    ext = _norm(df["extended_spacer"])

    seqs, reasons = [], []
    for g, e in zip(spacer, ext):
        s, why = _build_full_sequence(g, e, allow_non_gg_pam=allow_non_gg_pam,
                                      sequence_length=sequence_length)
        seqs.append(s)
        reasons.append(why)
    seq_ser = pd.Series(seqs, index=df.index)
    reason_ser = pd.Series(reasons, index=df.index)

    bad_seq = seq_ser.isna()
    for why, n in reason_ser[bad_seq].value_counts().items():
        report.drop_reasons[why] = report.drop_reasons.get(why, 0) + int(n)

    keep = ~(bad_eff | bad_seq)
    report.n_dropped = int((~keep).sum())

    strand_raw = (df["strand"].astype(str).str.strip().str.lower()
                  if "strand" in df.columns else pd.Series([""] * len(df), index=df.index))
    strand = strand_raw.map({"sense": "+", "antisense": "-", "+": "+", "-": "-"}).fillna("+")

    spec = label_spec or DEFAULT_LABEL_SPECS["labuhn"]
    y, _ = spec.apply(eff[keep])
    report.target_scale = spec.describe()
    out = pd.DataFrame({
        CANONICAL_SEQUENCE: seq_ser[keep],
        "Strand": strand[keep],
        CANONICAL_TARGET: y,
    })
    return _finalize(out, report, allow_non_gg_pam=allow_non_gg_pam,
                     duplicate_policy=duplicate_policy,
                     pam_length=int(sequence_length) - SPACER_LENGTH), report


ADAPTERS = {
    "deepcrispr": adapt_deepcrispr,
    "hiranniramol": adapt_hiranniramol,
    "labuhn": adapt_labuhn,
}


def detect_format(df: pd.DataFrame) -> str:
    """按列名自动识别原始文件格式（不做任何猜测性解析）。"""
    cols = {str(c).strip().lower() for c in df.columns}
    if {"ko_reporter_assay", "extended_spacer"} <= cols or "ko_reporter_assay" in cols:
        return "labuhn"
    if "extended target" in cols or ({"edit efficiency", "grna"} <= cols):
        return "hiranniramol"
    if {"chromosome", "normalized efficacy"} <= cols:
        return "deepcrispr"
    raise ValueError(
        "无法识别原始文件格式。\n"
        f"  实际列：{list(df.columns)[:20]}\n"
        "  请用 --format 显式指定，可选：" + ", ".join(sorted(ADAPTERS))
    )


def load_and_adapt(path: str | Path, fmt: Optional[str] = None, *,
                   allow_non_gg_pam: bool = False,
                   duplicate_policy: str = "mean",
                   sequence_length: int = SEQUENCE_LENGTH,
                   label_spec: Optional[LabelNormalizationSpec] = None,
                   ) -> Tuple[pd.DataFrame, AdaptReport]:
    """读原始文件 → 规范 DataFrame。``fmt`` 为空时自动识别。

    ``label_spec`` 给出「原始标签列 -> [0,1]」的归一化契约；不给则用该格式的缺省契约
    （见 :data:`DEFAULT_LABEL_SPECS`），两者数值一致。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"原始数据文件不存在：{path}")
    raw = pd.read_csv(path, low_memory=False)
    if raw.empty:
        raise ValueError(f"原始文件为空：{path}")

    fmt = (fmt or detect_format(raw)).strip().lower()
    if fmt not in ADAPTERS:
        raise ValueError(f"未知格式 {fmt!r}，可选：{', '.join(sorted(ADAPTERS))}")

    out, report = ADAPTERS[fmt](raw, allow_non_gg_pam=allow_non_gg_pam,
                                duplicate_policy=duplicate_policy,
                                sequence_length=sequence_length,
                                label_spec=label_spec)
    report.source = str(path)
    return out, report


# --------------------------------------------------------------------------- #
# CLI（自检用）
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="原始数据集 → 工程规范列（只做输入规范化，不做特征工程）")
    ap.add_argument("--list", action="store_true", help="列出支持的格式")
    ap.add_argument("--raw-data", type=str, default="", help="原始 CSV 文件")
    ap.add_argument("--format", type=str, default="", help="格式键；留空则自动识别")
    ap.add_argument("--allow-non-gg-pam", action="store_true", help="允许 PAM 非 GG")
    ap.add_argument("--dry-run", action="store_true", help="只报告，不写文件")
    ap.add_argument("--out", type=str, default="", help="把规范 CSV 写到该路径")
    args = ap.parse_args(argv)

    if args.list or not args.raw_data:
        print("支持的原始格式：")
        for k, fn in sorted(ADAPTERS.items()):
            print(f"  {k:<14} {fn.__doc__.strip().splitlines()[0]}")
        return 0

    df, report = load_and_adapt(args.raw_data, args.format or None,
                                allow_non_gg_pam=args.allow_non_gg_pam)
    print(report.line())
    print(f"  输出列：{list(df.columns)}")
    if not args.dry_run and args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False, encoding="utf-8")
        print(f"  已写出：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
