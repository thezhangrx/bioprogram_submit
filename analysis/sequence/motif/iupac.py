"""analysis.sequence.motif.iupac — motif 表示的三种形式 (严格分开保存)。

    consensus      : 单位点最常见碱基 (ACGT)
    iupac          : 标准 IUPAC 退化码 (R/Y/S/W/K/M/B/D/H/V/N)
    human_pattern  : 人类可读形式, 例如 (A/G)TC / A(T/C)G
    regex          : 正则形式, 例如 [AG]TC / A[TC]G

规范 (与任务书 §14 一致):
    单一碱基 -> "A"; 两个候选 -> "(A/G)"; 三个 -> "(A/C/T)"; 四个 -> "(A/C/G/T)";
    完全不确定 -> "(N)"; 绝不用 "[A|G]TC" 作为唯一格式。
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

BASES = ("A", "C", "G", "T")

IUPAC_CODES: Dict[str, str] = {
    "A": "A", "C": "C", "G": "G", "T": "T",
    "AG": "R", "CT": "Y", "CG": "S", "AT": "W", "GT": "K", "AC": "M",
    "CGT": "B", "AGT": "D", "ACT": "H", "ACG": "V",
    "ACGT": "N",
}
IUPAC_TO_BASES: Dict[str, Tuple[str, ...]] = {
    "A": ("A",), "C": ("C",), "G": ("G",), "T": ("T",),
    "R": ("A", "G"), "Y": ("C", "T"), "S": ("C", "G"), "W": ("A", "T"),
    "K": ("G", "T"), "M": ("A", "C"),
    "B": ("C", "G", "T"), "D": ("A", "G", "T"), "H": ("A", "C", "T"), "V": ("A", "C", "G"),
    "N": ("A", "C", "G", "T"),
}


def normalize_bases(bases: Iterable[str], preserve_order: bool = False) -> Tuple[str, ...]:
    """碱基 -> 元组; 未知字符忽略。

    preserve_order=True 且输入为有序序列 (list/tuple) 时保留输入顺序
    (spec 例: A(T/C)G -> regex A[TC]G); 集合/无序输入用规范序 (A,C,G,T)。
    """
    ordered = preserve_order and isinstance(bases, (list, tuple))
    if ordered:
        seen: List[str] = []
        for b in bases:
            b = str(b).strip().upper()
            if b in BASES and b not in seen:
                seen.append(b)
        return tuple(seen)
    s = {str(b).strip().upper() for b in bases}
    return tuple(b for b in BASES if b in s)


def iupac_code(bases: Iterable[str]) -> str:
    """碱基集合 -> IUPAC 码; 空集合/未知 -> "N" (完全不确定, 不假装是 A)。"""
    key = "".join(normalize_bases(bases))
    return IUPAC_CODES.get(key, "N")


def human_pattern(bases: Iterable[str]) -> str:
    """碱基 -> 人类可读形式: A / (A/G) / (A/C/T) / (A/C/G/T) (有序输入保留顺序)。"""
    norm = normalize_bases(bases, preserve_order=True)
    if len(norm) == 1:
        return norm[0]
    if not norm:
        return "(N)"
    return "(" + "/".join(norm) + ")"


def regex_pattern(bases: Iterable[str]) -> str:
    """碱基 -> 正则字符类: [AG] / [ACT] / [ACGT] (有序输入保留顺序)。"""
    norm = normalize_bases(bases, preserve_order=True)
    if not norm:
        return "[ACGT]"
    if len(norm) == 1:
        return norm[0]
    return "[" + "".join(norm) + "]"


def codes_to_regex(codes: Sequence[str]) -> str:
    """IUPAC 码序列 -> 正则 (退化码展开为字符类, 顺序按 IUPAC 表)。"""
    out: List[str] = []
    for code in codes:
        bases = IUPAC_TO_BASES.get(str(code).upper())
        out.append(regex_pattern(list(bases)) if bases else "[ACGT]")
    return "".join(out)


def codes_to_human(codes: Sequence[str]) -> str:
    """IUPAC 码序列 -> 人类可读 pattern (A/G -> (A/G), N -> (N))。"""
    parts: List[str] = []
    for code in codes:
        bases = IUPAC_TO_BASES.get(str(code).upper())
        parts.append(human_pattern(list(bases)) if bases else "(N)")
    return "".join(parts)


def positions_from_frequencies(freqs: Sequence[Dict[str, float]],
                               degenerate_fraction: float = 0.25
                               ) -> List[List[str]]:
    """每个位置的碱基频率 -> 该位置的候选碱基集合 (argmax + 频率达阈值的退化碱基)。"""
    out: List[Tuple[str, ...]] = []
    for dist in freqs:
        if not dist:
            out.append(())
            continue
        ordered = sorted(((b, float(dist.get(b, 0.0))) for b in BASES),
                         key=lambda kv: kv[1], reverse=True)
        top_base, top_freq = ordered[0]
        if top_freq <= 0:
            out.append(())
            continue
        keep: List[str] = [top_base]          # 频率降序 -> human/regex 顺序有信息量
        for base, freq in ordered[1:]:
            if top_freq > 0 and freq / top_freq >= degenerate_fraction and freq > 0:
                keep.append(base)
        out.append(keep)
    return out


def consensus_from_frequencies(freqs: Sequence[Dict[str, float]],
                              degenerate_fraction: float = 0.25) -> Dict[str, object]:
    """位置频率 -> {consensus, iupac, human_pattern, regex, per_position}。"""
    per_pos = positions_from_frequencies(freqs, degenerate_fraction)
    codes = [iupac_code(b) for b in per_pos]
    consensus = "".join(b[0] if b else "N" for b in per_pos)
    return {
        "consensus": consensus,
        "iupac": "".join(codes),
        "human_pattern": codes_to_human(codes),
        "regex": codes_to_regex(codes),
        "per_position_bases": ["".join(b) for b in per_pos],
    }


def matches_iupac(sequence: str, codes: Sequence[str]) -> bool:
    """序列是否匹配 IUPAC 码模式 (长度必须一致)。"""
    if len(sequence) != len(codes):
        return False
    for base, code in zip(str(sequence).upper(), codes):
        allowed = IUPAC_TO_BASES.get(str(code).upper())
        if allowed is None or base not in allowed:
            return False
    return True


def count_occurrences(sequence: str, codes: Sequence[str]) -> int:
    """motif (IUPAC 码) 在序列中出现的次数 (滑窗, 长度需 <= 序列长度)。"""
    seq = str(sequence).upper()
    n, m = len(seq), len(codes)
    if m == 0 or m > n:
        return 0
    return sum(1 for i in range(n - m + 1) if matches_iupac(seq[i:i + m], codes))


def contains_iupac(sequence: str, codes: Sequence[str]) -> bool:
    """motif 是否出现在序列中 (滑窗)。"""
    return count_occurrences(sequence, codes) > 0


def pfm_from_instances(sequences: Sequence[str], length: Optional[int] = None) -> np.ndarray:
    """motif instances -> PFM (4 x L) 计数矩阵。"""
    seqs = [str(s).upper() for s in sequences if isinstance(s, str)]
    if length is None:
        length = len(seqs[0]) if seqs else 0
    pfm = np.zeros((4, int(length)), dtype=float)
    idx = {b: i for i, b in enumerate(BASES)}
    for s in seqs:
        if len(s) != length:
            continue
        for j, base in enumerate(s):
            if base in idx:
                pfm[idx[base], j] += 1.0
    return pfm


def information_content(pfm: np.ndarray) -> np.ndarray:
    """每个位置的信息量 (bits, R 公式), 用于 sequence logo 高度。"""
    arr = np.asarray(pfm, dtype=float)
    total = arr.sum(axis=0, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        freq = np.where(total > 0, arr / total, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(freq > 0, freq * np.log2(np.maximum(freq, 1e-12)), 0.0)
    return 2.0 + terms.sum(axis=0)      # 4 碱基 -> log2(4) = 2 bits
