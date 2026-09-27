#!/usr/bin/env python3
"""为"WT → 只改 Position 18 → CRISPRon 重预测"演示挑选**有代表性的 WT sgRNA**。

设计原则（这是本脚本存在的理由，请勿在挑选阶段引入任何突变侧信息）
------------------------------------------------------------------
挑选**只使用 WT 侧信息**：

    * 原始实验标签 ``Normalized efficacy``
    * 序列本身及其特征（GC、组成、PAM-proximal 上下文、位置模式）
    * 元数据（cell line、数据 split、坐标）
    * 数据质量（重复 / 反向互补重复 / 标签极端值 / 30-mer 可构造性）

**绝不使用**任何 C18A（或其它突变）的预测/实验结果。挑选规则在脚本里预先写死
（见 ``SELECTION_RULE``），并且 ``assert_no_mutant_information_used()`` 会在挑选
结束时断言"用于挑选的列"里不含任何突变字段。构造 C18A 只发生在挑选**完成之后**，
因此"下降多少"不可能影响选了谁。

规则摘要
--------
1. 候选池 = ``single`` 划分的 **test 集**（group-aware，seed 42，4 个细胞系），
   不从训练集挑。
2. 过滤：Position 18 == C；PAM == NGG；按 canonical identity
   ``min(seq, revcomp(seq))`` 去重；标签有限且 ∈ [0,1]；剔除各细胞系标签
   [P1, P99] 之外的极端值；坐标必须能取出合法 30-mer。
3. 分层：在**每个细胞系内部**按标签三分位切成 low / medium / high。
4. 代表性：每个 (细胞系 × 区间) 取**最接近该区间中位数**的 1 条进入候选短名单
   （不取极值）。
5. 最终 8 条 = low 2 + medium 2 + high 2 + 2 条"GC 明显不同但正常"的序列，
   约束：每个细胞系 ≤ 2 条、两两序列错配 ≥ 4 nt。

用法
----
    python analysis/candidates/wt_position18_selection.py --dataset DeepCRISPR

输出（默认 ``results/tables/candidates/``）
------------------------------------------
    wt_position18_candidates.csv          最终 8 条的全部字段
    wt_position18_selection_audit.csv     过滤漏斗 + 分层依据 + 每次取舍的理由
    wt_position18_crispron_input.fa       每条 WT / C18A 的 30-mer（CRISPRon 直接可用）
    wt_position18_cellline_reference.csv  CRISPRon 可选的基因组区间输入
    wt_position18_report.md               完整报告（含最终检查清单）
    wt_position18_for_mentor.md           展示用简表
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from analysis.leakage import canonical_group_key, revcomp                      # noqa: E402
from core.common.paths import RESULTS_TABLES, resolve_dataset                  # noqa: E402
from core.data.cell_line_division import (                          # noqa: E402
    divide_data, load_feature_schema,
)

# --------------------------------------------------------------------------- #
# 预先登记的挑选规则（写死在代码里，避免事后调整）
# --------------------------------------------------------------------------- #
SELECTION_RULE_VERSION = "wt-pos18-representative-v1"

SELECTION_RULE = f"""
{SELECTION_RULE_VERSION}
1. 候选池: DeepCRISPR `single` 划分的 test 集 (group-aware 无泄漏, seed=42), 全部细胞系。
2. 过滤: Position18==C; PAM==NGG; canonical identity min(seq,revcomp) 去重;
   标签有限且∈[0,1]; 剔除各细胞系标签 [P1,P99] 之外的极端值; 坐标可取出合法 30-mer。
3. 分层: 在每个细胞系内部, 按标签三分位切 low / medium / high (避免细胞系混淆)。
4. 代表性: 每个 (细胞系 × 区间) 取最接近该区间中位数的 1 条进入短名单 (不取极值)。
5. 终选 8 条 = low 2 + medium 2 + high 2 + 2 条 GC 偏离中位数最多但仍在
   [P5,P95] 内的"特殊但仍正常"序列; 约束 每细胞系<=2 条、两两序列错配>=4 nt。
6. 全程不使用任何突变侧结果。
"""

#: 挑选阶段允许读取的列（用于"无突变信息"自证）
WT_ONLY_COLUMNS = (
    "cell_line", "split_type", "sgRNA", "Normalized efficacy", "Chromosome",
    "Start", "End", "Strand", "identity", "label_bin", "label_bin_median",
    "gc_spacer", "gc_23nt", "pam", "pos18", "seed_region", "dist_to_center",
)

#: 项目对 PAM / protospacer 的权威约定
PAM_MOTIF = "GG"          # PAM 第 2-3 位；来源: analysis/panorama.py:524 "PAM (NGG)"
PAM_LEN = 3               # NGG 为 3 nt
POS_1B = 18               # 关注位点（1-based）
SEED_REGION_1B = (17, 20)  # PAM-proximal seed

#: 23 nt 内的四个序列区域（1-based 闭区间）—— **本模块自持**（单一来源）。
#: 历史上此处从 analysis/reporting/paper/make_assets.py 的 REGIONS 导入做交叉校验，
#: 但论文脚本已于 2026-09-20 移入 Delete/_moved_paper/，该导入必然 ImportError。
#: 现在区域定义就在本模块，cross_check_with_project() 改为自洽校验。
REGIONS = {
    "PAM-distal (1-8)": (1, 8),
    "Seed core (9-16)": (9, 16),
    "PAM-proximal seed (17-20)": (17, 20),
    "PAM (21-23)": (21, 23),
}

#: 终选配额
N_LOW = N_MEDIUM = N_HIGH = 2
N_SPECIAL = 2
MAX_PER_CELL_LINE = 2
MIN_PAIRWISE_MISMATCH = 4          # 23 nt 内至少 4 个位置不同
EXTREME_PCTL = (1.0, 99.0)         # 标签极端值剔除
SPECIAL_GC_PCTL = (5.0, 95.0)      # "特殊但仍正常" 的 GC 允许区间

HG19_CHROM_SIZES_URL = "https://api.genome.ucsc.edu/list/chromosomes?genome=hg19"
HG19_SEQ_URL = ("https://api.genome.ucsc.edu/getData/sequence"
                "?genome=hg19;chrom={chrom};start={start0};end={end}")
UPSTREAM_NT = 4            # CRISPRon: 4 nt + 20 nt spacer + 3 nt PAM + 3 nt
DOWNSTREAM_NT = 3
CRISPRON_LEN = UPSTREAM_NT + 23 + DOWNSTREAM_NT   # = 30


# --------------------------------------------------------------------------- #
# 通用工具
# --------------------------------------------------------------------------- #
def gc_content(seq: str) -> float:
    seq = str(seq).upper()
    if not seq:
        return float("nan")
    return 100.0 * sum(seq.count(b) for b in "GC") / len(seq)


def mismatches(a: str, b: str) -> int:
    return sum(1 for x, y in zip(a, b) if x != y)


def _http_json(url: str, timeout: int = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return json.load(fh)


def fetch_chrom_sizes() -> Dict[str, int]:
    """UCSC 返回 ``{"chromosomes": {"chr1": 249250621, ...}}``（名字 -> 长度）。"""
    data = _http_json(HG19_CHROM_SIZES_URL)
    raw = data.get("chromosomes", {})
    if isinstance(raw, dict):
        sizes = {str(k): int(v) for k, v in raw.items()}
    else:                                   # 兼容 list-of-dict 的旧格式
        sizes = {str(it.get("name") or it.get("chrom")): int(it["size"])
                 for it in raw if it.get("size")}
    if not sizes:
        raise RuntimeError("无法获取 hg19 染色体长度表（UCSC API 无返回）")
    return sizes


def fetch_hg19_window(chrom: str, start0: int, end: int) -> str:
    dna = _http_json(HG19_SEQ_URL.format(chrom=chrom, start0=start0, end=end)).get("dna", "")
    return str(dna).upper()


# --------------------------------------------------------------------------- #
# 位点 / PAM 约定（从 schema 派生 + 与项目权威常量交叉校验）
# --------------------------------------------------------------------------- #
class LocusConvention:
    """把"23 nt 里哪几段是 protospacer / PAM / 关注位点"集中定义一次。"""

    def __init__(self, schema: dict) -> None:
        self.seq_len = int(schema["sequence_length"])
        self.spacer_len = self.seq_len - PAM_LEN
        self.pam_start_1b = self.spacer_len + 1
        self.pam_end_1b = self.seq_len
        self.pos_index0 = POS_1B - 1
        if not (1 <= POS_1B <= self.seq_len):
            raise ValueError(f"POS_1B={POS_1B} 超出序列长度 {self.seq_len}")
        if not (SEED_REGION_1B[0] <= POS_1B <= SEED_REGION_1B[1]):
            raise ValueError(f"POS_1B={POS_1B} 不在 PAM-proximal seed {SEED_REGION_1B} 内")

    def pam_of(self, seq: str) -> str:
        return str(seq).upper()[self.spacer_len:self.seq_len]

    def pos18_of(self, seq: str) -> str:
        return str(seq).upper()[self.pos_index0]

    def valid_pam(self, seq: str) -> bool:
        pam = self.pam_of(seq)
        return len(pam) == PAM_LEN and pam.endswith(PAM_MOTIF)

    def describe(self) -> str:
        return (f"序列长度 {self.seq_len} nt = protospacer 1-{self.spacer_len} "
                f"+ PAM {self.pam_start_1b}-{self.pam_end_1b} "
                f"(NGG, 判据 '{PAM_MOTIF}' 结尾) | 关注位点 {POS_1B} "
                f"(0-based 下标 {self.pos_index0}, 含 PAM 的 23nt 内)")


def cross_check_with_project(conv: LocusConvention) -> str:
    """校验位点约定与本模块自持的 REGIONS 一致（单一来源，无跨模块导入）。"""
    pam_range = REGIONS["PAM (21-23)"]
    seed_range = REGIONS["PAM-proximal seed (17-20)"]
    ok_pam = pam_range == (conv.pam_start_1b, conv.pam_end_1b)
    ok_seed = seed_range == SEED_REGION_1B
    if not (ok_pam and ok_seed):
        raise ValueError(
            f"位点约定与本模块 REGIONS 不一致: 本脚本 PAM={conv.pam_start_1b}-{conv.pam_end_1b}, "
            f"REGIONS={pam_range}; seed 本脚本={SEED_REGION_1B}, REGIONS={seed_range}"
        )
    return f"与本模块 REGIONS 自洽 (PAM {pam_range}, seed {seed_range})"


# --------------------------------------------------------------------------- #
# 候选池
# --------------------------------------------------------------------------- #
def build_pool(dataset_dir: Path, seed: int = 42) -> Tuple[pd.DataFrame, List[dict]]:
    """从 `single` 划分的 test 集构建候选池。"""
    schema = load_feature_schema(str(dataset_dir))
    conv = LocusConvention(schema)

    from core.data.cell_line_division import discover_available_cell_lines
    cells = discover_available_cell_lines(str(dataset_dir))

    frames, funnel = [], []
    for cl in cells:
        split = divide_data(data_dir=str(dataset_dir), split_type="single",
                            cell_line=cl, random_seed=seed)
        meta = split["test_data"]["metadata"].copy()
        meta["Normalized efficacy"] = np.asarray(split["test_data"]["y"], dtype=float)
        meta["cell_line"] = cl
        meta["split_type"] = "single:test"
        frames.append(meta)
        funnel.append({"stage": "pool", "cell_line": cl, "n": int(len(meta))})

    pool = pd.concat(frames, ignore_index=True)
    return pool, funnel


def apply_filters(pool: pd.DataFrame, conv: LocusConvention,
                  chrom_sizes: Dict[str, int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """逐步过滤，并记录漏斗。"""
    fun = []

    def step(df, name, note=""):
        fun.append({"stage": name, "n": int(len(df)), "note": note})
        return df

    df = pool.copy()
    step(df, "0_pool", "single 划分 test 集, 全部细胞系")

    df["sgRNA"] = df["sgRNA"].astype(str).str.upper().str.strip()
    df = df[df["sgRNA"].str.fullmatch(r"[ACGT]+").fillna(False)]
    step(df, "1_seq_valid", "仅 ACGT")

    df = df[df["sgRNA"].str.len() == conv.seq_len]
    step(df, "2_seq_length", f"长度 == {conv.seq_len}")

    df = df[df["Normalized efficacy"].notna()]
    df = df[(df["Normalized efficacy"] >= 0) & (df["Normalized efficacy"] <= 1)]
    step(df, "3_label_in_0_1", "标签有限且 ∈ [0,1]")

    lo = df.groupby("cell_line")["Normalized efficacy"].transform(
        lambda s: s.quantile(EXTREME_PCTL[0] / 100.0))
    hi = df.groupby("cell_line")["Normalized efficacy"].transform(
        lambda s: s.quantile(EXTREME_PCTL[1] / 100.0))
    df = df[(df["Normalized efficacy"] >= lo) & (df["Normalized efficacy"] <= hi)]
    step(df, "4_no_extreme_label",
         f"剔除各细胞系标签 P{EXTREME_PCTL[0]:g}~P{EXTREME_PCTL[1]:g} 之外")

    df["identity"] = canonical_group_key(df)
    before = len(df)
    df = df.sort_values(["cell_line", "Normalized efficacy", "identity"], kind="stable")
    df = df.drop_duplicates(subset=["identity"], keep="first").reset_index(drop=True)
    step(df, "5_dedup_identity",
         f"canonical identity min(seq,revcomp) 去重, 移除 {before - len(df)} 条"
         "（同时覆盖精确重复与反向互补重复）")

    df["pos18"] = [conv.pos18_of(s) for s in df["sgRNA"]]
    n_all = len(df)
    df = df[df["pos18"] == "C"]
    step(df, "6_pos18_is_C", f"Position 18 == C（{n_all} -> {len(df)}）")

    df["pam"] = [conv.pam_of(s) for s in df["sgRNA"]]
    df["valid_pam"] = [conv.valid_pam(s) for s in df["sgRNA"]]
    df = df[df["valid_pam"]]
    step(df, "7_pam_ngg", f"PAM == N{PAM_MOTIF}（{PAM_LEN} nt）")

    # 侧翼可构造性：坐标必须能容下 4 nt 上游 + 3 nt 下游
    def flanks_ok(row):
        chrom = str(row["Chromosome"])
        if chrom not in chrom_sizes:
            return False
        start0, end = int(row["Start"]) - 1, int(row["End"])
        strand = str(row["Strand"]).strip()
        # 负链的侧翼方向相反（见 build_30mer）
        left = start0 - (DOWNSTREAM_NT if strand == "-" else UPSTREAM_NT)
        right = end + (UPSTREAM_NT if strand == "-" else DOWNSTREAM_NT)
        return left >= 0 and right <= chrom_sizes[chrom]

    df["flanks_ok"] = df.apply(flanks_ok, axis=1)
    df = df[df["flanks_ok"]]
    step(df, "8_crispron_window_available",
         f"chr 长度足够容纳 {UPSTREAM_NT}+{conv.seq_len}+{DOWNSTREAM_NT} nt（hg19）")

    return df.reset_index(drop=True), pd.DataFrame(fun)


# --------------------------------------------------------------------------- #
# 30-mer 逐条核对（带缓存 + 有限并发）
# --------------------------------------------------------------------------- #
def _crispron_window_bounds(row: pd.Series) -> Tuple[int, int]:
    """返回正链上的 30-mer 取窗区间 [lo, hi)（负链侧翼方向相反）。"""
    start0 = int(row["Start"]) - 1
    end = int(row["End"])
    if str(row["Strand"]).strip() == "-":
        return start0 - DOWNSTREAM_NT, end + UPSTREAM_NT
    return start0 - UPSTREAM_NT, end + DOWNSTREAM_NT


def verify_all_windows(df: pd.DataFrame, conv: LocusConvention, cache_path: Path,
                       workers: int = 6, retries: int = 2) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """逐条从 hg19 取 30-mer 并核对中间 23 nt 是否与数据完全一致。

    这同时验证了：基因组版本(hg19)、坐标语义(1-based 闭区间)、链方向。
    结果写进缓存，重跑不再走网络。
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    cache: Dict[Tuple[str, int, int, str], str] = {}
    if cache_path.exists():
        c = pd.read_csv(cache_path)
        for _, r in c.iterrows():
            cache[(str(r["chrom"]), int(r["start"]), int(r["end"]), str(r["strand"]))] = str(r["window"])

    keys = [(str(r["Chromosome"]), int(r["Start"]), int(r["End"]), str(r["Strand"]).strip())
            for _, r in df.iterrows()]
    todo = [k for k in dict.fromkeys(keys) if k not in cache]
    print(f"        30-mer 核对：缓存命中 {len(set(keys)) - len(todo)}，需联网 {len(todo)}")

    def worker(key):
        chrom, start, end, strand = key
        row = pd.Series({"Chromosome": chrom, "Start": start, "End": end, "Strand": strand})
        lo, hi = _crispron_window_bounds(row)
        last = ""
        for attempt in range(retries + 1):
            try:
                return key, fetch_hg19_window(chrom, lo, hi)
            except Exception as exc:                      # 网络抖动：退避重试
                last = str(exc)
                time.sleep(0.5 * (attempt + 1))
        return key, f"__ERROR__{last}"

    if todo:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for key, win in pool.map(worker, todo):
                cache[key] = win

    rows = [{"chrom": k[0], "start": k[1], "end": k[2], "strand": k[3], "window": v}
            for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(cache_path, index=False, encoding="utf-8-sig")

    matched, notes = [], []
    for _, r in df.iterrows():
        key = (str(r["Chromosome"]), int(r["Start"]), int(r["End"]), str(r["Strand"]).strip())
        win = cache.get(key, "")
        if win.startswith("__ERROR__"):
            notes.append("network_error"); matched.append(False); continue
        if len(win) != CRISPRON_LEN:
            notes.append("window_length"); matched.append(False); continue
        oriented = revcomp(win) if key[3] == "-" else win
        if oriented[UPSTREAM_NT:UPSTREAM_NT + conv.seq_len] == r["sgRNA"]:
            notes.append("ok"); matched.append(True)
        else:
            notes.append("hg19_mismatch"); matched.append(False)
    out = df.copy()
    out["hg19_window_note"] = notes
    out["hg19_match"] = matched
    return out, dict(pd.Series(notes).value_counts())


# --------------------------------------------------------------------------- #
# 分层 + 代表性挑选
# --------------------------------------------------------------------------- #
def annotate_features(df: pd.DataFrame, conv: LocusConvention) -> pd.DataFrame:
    df = df.copy()
    df["gc_spacer"] = [gc_content(s[:conv.spacer_len]) for s in df["sgRNA"]]
    df["gc_23nt"] = [gc_content(s) for s in df["sgRNA"]]
    df["seed_region"] = [s[SEED_REGION_1B[0] - 1:SEED_REGION_1B[1]] for s in df["sgRNA"]]
    return df


def stratify(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    """按细胞系内部三分位切 low/medium/high。"""
    rows, bins = [], []
    out = []
    for cl, grp in df.groupby("cell_line"):
        q1, q2 = grp["Normalized efficacy"].quantile([1 / 3, 2 / 3])
        g = grp.copy()
        g["label_bin"] = np.where(g["Normalized efficacy"] <= q1, "low",
                                  np.where(g["Normalized efficacy"] <= q2, "medium", "high"))
        med = g.groupby("label_bin")["Normalized efficacy"].median().to_dict()
        g["label_bin_median"] = g["label_bin"].map(med)
        g["dist_to_center"] = (g["Normalized efficacy"] - g["label_bin_median"]).abs()
        out.append(g)
        bins.append({"cell_line": cl, "bin": "low", "cut": float(q1), "median": float(med.get("low", np.nan)),
                     "n": int((g["label_bin"] == "low").sum())})
        bins.append({"cell_line": cl, "bin": "medium", "cut": float(q2), "median": float(med.get("medium", np.nan)),
                     "n": int((g["label_bin"] == "medium").sum())})
        bins.append({"cell_line": cl, "bin": "high", "cut": float("nan"), "median": float(med.get("high", np.nan)),
                     "n": int((g["label_bin"] == "high").sum())})
    rows = pd.concat(out, ignore_index=True)
    return rows, bins


def _accepts(cand: pd.Series, chosen: List[pd.Series]) -> Tuple[bool, str]:
    if sum(c["cell_line"] == cand["cell_line"] for c in chosen) >= MAX_PER_CELL_LINE:
        return False, f"细胞系 {cand['cell_line']} 已达上限 {MAX_PER_CELL_LINE}"
    for c in chosen:
        mm = mismatches(cand["sgRNA"], c["sgRNA"])
        if mm < MIN_PAIRWISE_MISMATCH:
            return False, f"与已选 {c['identity'][:12]} 仅差 {mm} nt (<{MIN_PAIRWISE_MISMATCH})"
    return True, "通过多样性约束"


def select(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    """按预先登记的规则挑 8 条；返回 (中选记录, 取舍日志)。"""
    log: List[dict] = []
    chosen: List[pd.Series] = []

    # 短名单：每个 (细胞系 × 区间) 取最接近区间中位数的 1 条
    short: List[pd.Series] = []
    for (cl, b), grp in df.groupby(["cell_line", "label_bin"]):
        g = grp.sort_values(["dist_to_center", "identity"], kind="stable")
        pick = g.iloc[0]
        short.append(pick)
        log.append({"stage": "shortlist", "cell_line": cl, "bin": b,
                    "identity": pick["identity"], "why": f"最接近 {b} 区间中位数"
                    f"（|Δ|={pick['dist_to_center']:.4f}）"})

    def take(cand: pd.Series, reason: str) -> None:
        rec = cand.copy()
        rec["selection_reason"] = reason      # 落到记录上，便于写进最终表
        chosen.append(rec)
        log.append({"stage": "select", "cell_line": cand["cell_line"], "bin": cand["label_bin"],
                    "identity": cand["identity"], "efficiency": float(cand["Normalized efficacy"]),
                    "why": reason})

    # 配额：low / medium / high 各 2 条
    for b, quota in (("low", N_LOW), ("medium", N_MEDIUM), ("high", N_HIGH)):
        pool_b = [c for c in short if c["label_bin"] == b]
        pool_b.sort(key=lambda c: (c["dist_to_center"], c["cell_line"], c["identity"]))
        got = 0
        for cand in pool_b:
            if got >= quota:
                break
            ok, why = _accepts(cand, chosen)
            log.append({"stage": "consider", "cell_line": cand["cell_line"], "bin": b,
                        "identity": cand["identity"], "why": why})
            if ok:
                take(cand, f"{b} 区间代表（最接近中位数 {cand['label_bin_median']:.4f}）")
                got += 1
        if got < quota:
            # 短名单不够时从该区间全量候选按同一准则补
            rest = df[df["label_bin"] == b].sort_values(["dist_to_center", "identity"], kind="stable")
            for _, cand in rest.iterrows():
                if got >= quota:
                    break
                if any(cand["identity"] == c["identity"] for c in chosen):
                    continue
                ok, why = _accepts(cand, chosen)
                log.append({"stage": "consider_fallback", "cell_line": cand["cell_line"], "bin": b,
                            "identity": cand["identity"], "why": why})
                if ok:
                    take(cand, f"{b} 区间补充代表（短名单受多样性约束后不足）")
                    got += 1

    # 2 条"特殊但仍正常"：GC 偏离中位数最多，但仍在 [P5,P95]
    gc_med = df["gc_spacer"].median()
    gc_lo, gc_hi = df["gc_spacer"].quantile([SPECIAL_GC_PCTL[0] / 100, SPECIAL_GC_PCTL[1] / 100])
    cand_special = df[(df["gc_spacer"] >= gc_lo) & (df["gc_spacer"] <= gc_hi)].copy()
    cand_special["gc_dev"] = (cand_special["gc_spacer"] - gc_med).abs()
    cand_special = cand_special.sort_values(["gc_dev", "identity"], ascending=[False, True],
                                            kind="stable")
    got = 0
    for _, cand in cand_special.iterrows():
        if got >= N_SPECIAL:
            break
        if any(cand["identity"] == c["identity"] for c in chosen):
            continue
        ok, why = _accepts(cand, chosen)
        log.append({"stage": "consider_special", "cell_line": cand["cell_line"],
                    "bin": cand["label_bin"], "identity": cand["identity"],
                    "why": f"GC={cand['gc_spacer']:.1f}% (中位 {gc_med:.1f}%) -> {why}"})
        if ok:
            take(cand, f"序列组成特殊但仍正常：GC {cand['gc_spacer']:.1f}% "
                       f"(全池中位 {gc_med:.1f}%)")
            got += 1

    return [dict(c) for c in chosen], log


# --------------------------------------------------------------------------- #
# 突变构造 + CRISPRon 30-mer
# --------------------------------------------------------------------------- #
def build_mutant(wt: str, conv: LocusConvention) -> str:
    """只改 Position 18 的 C -> A。"""
    s = list(str(wt).upper())
    s[conv.pos_index0] = "A"
    return "".join(s)


def verify_mutation(wt: str, mut: str, conv: LocusConvention) -> dict:
    """验证"只改了关注位点"。

    比较的是 ``wt[:i] == mut[:i]`` 与 ``wt[i+1:] == mut[i+1:]``（i = 关注位点的
    0-based 下标，由项目约定推导）；在 POS_1B=18 时等价于题目要求的
    ``wt[:17] == mut[:17]`` 与 ``wt[18:] == mut[18:]``。
    """
    i = conv.pos_index0
    return {
        "wt_pos18": wt[i],
        "mut_pos18": mut[i],
        "prefix_equal": wt[:i] == mut[:i],
        "suffix_equal": wt[i + 1:] == mut[i + 1:],
        "prefix_len": i,
        "suffix_from_index": i + 1,
        "only_one_position_changed": mismatches(wt, mut) == 1,
        "length_equal": len(wt) == len(mut) == conv.seq_len,
    }


def build_30mer(row: pd.Series, conv: LocusConvention) -> Tuple[str, str]:
    """取 hg19 上 4 nt 上游 + 23 nt + 3 nt 下游，并按链定向。

    返回 ``(wt_30mer, note)``。内部断言：中间 23 nt 必须与数据里的 sgRNA 完全一致
    —— 这一步同时验证了基因组版本(hg19)、坐标语义(1-based 闭区间)与链方向。
    """
    chrom = str(row["Chromosome"])
    start0 = int(row["Start"]) - 1          # 0-based, 含 PAM 的 23 nt 区间 = [start0, end)
    end = int(row["End"])
    strand = str(row["Strand"]).strip()
    if strand == "-":
        # 负链上 guide 的 5' 端落在坐标的**高**端：
        #   4 nt 上游 = 正链 [end, end+4)，3 nt 下游 = 正链 [start0-3, start0)
        lo, hi = start0 - DOWNSTREAM_NT, end + UPSTREAM_NT
    else:
        # 正链：5' 端在低端，4 nt 上游 = [start0-4, start0)，3 nt 下游 = [end, end+3)
        lo, hi = start0 - UPSTREAM_NT, end + DOWNSTREAM_NT
    window = fetch_hg19_window(chrom, lo, hi)
    if len(window) != CRISPRON_LEN:
        raise RuntimeError(f"{chrom}:{lo}-{hi} 取回长度 {len(window)} != {CRISPRON_LEN}")
    if strand == "-":
        window = revcomp(window)
    core = window[UPSTREAM_NT:UPSTREAM_NT + conv.seq_len]
    if core != row["sgRNA"]:
        raise RuntimeError(
            f"hg19 参考序列与数据不一致（{chrom}:{row['Start']}-{row['End']} {row['Strand']}）:\n"
            f"  ref ={core}\n  data={row['sgRNA']}"
        )
    return window, f"{chrom}:{lo}-{hi}({row['Strand']})"


# --------------------------------------------------------------------------- #
# 自证：挑选阶段未使用突变侧信息
# --------------------------------------------------------------------------- #
def assert_no_mutant_information_used(selected: List[dict]) -> dict:
    """断言选择过程只读取了 WT 侧字段。"""
    forbidden = []
    for rec in selected:
        for k in rec:
            low = str(k).lower()
            if any(tok in low for tok in ("mut", "c18a", "delta", "crispron_eff", "pred")):
                forbidden.append(k)
    if forbidden:
        raise AssertionError(f"挑选结果中含有突变侧字段：{sorted(set(forbidden))}")
    return {
        "rule_version": SELECTION_RULE_VERSION,
        "wt_only_columns": list(WT_ONLY_COLUMNS),
        "mutant_fields_present_in_selection": [],
        "note": "C18A 序列仅在挑选完成后由 build_mutant() 生成；"
                "脚本内没有任何 CRISPRon 调用或突变侧预测。",
    }


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
def write_outputs(out_dir: Path, conv: LocusConvention, dataset_dir: Path,
                  funnel: pd.DataFrame, bins: List[dict], selected: List[dict],
                  log: List[dict], selfcheck: dict, conv_note: str) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    picks = pd.DataFrame(selected)
    keep = ["ID", "cell_line", "sgRNA", "pos18", "mutant", "pam", "Normalized efficacy",
            "gc_spacer", "gc_23nt", "label_bin", "label_bin_median", "identity",
            "Chromosome", "Start", "End", "Strand", "split_type", "seed_region",
            "crispron_30mer_wt", "crispron_30mer_c18a", "crispron_window",
            "verify_prefix_equal", "verify_suffix_equal", "verify_only_pos18_changed",
            "verify_n_mismatch", "hg19_window_note",
            "selection_reason"]
    picks = picks[[c for c in keep if c in picks.columns]]
    paths["candidates"] = out_dir / "wt_position18_candidates.csv"
    picks.to_csv(paths["candidates"], index=False, encoding="utf-8-sig")

    audit = pd.concat([funnel, pd.DataFrame(log)], ignore_index=True)
    paths["audit"] = out_dir / "wt_position18_selection_audit.csv"
    audit.to_csv(paths["audit"], index=False, encoding="utf-8-sig")

    paths["bins"] = out_dir / "wt_position18_label_bins.csv"
    pd.DataFrame(bins).to_csv(paths["bins"], index=False, encoding="utf-8-sig")

    fa = out_dir / "wt_position18_crispron_input.fa"
    with fa.open("w", encoding="utf-8") as fh:
        for _, r in picks.iterrows():
            fh.write(f">{r['ID']}_{r['cell_line']}_WT eff={r['Normalized efficacy']:.4f} "
                     f"GC={r['gc_spacer']:.1f}\n{r['crispron_30mer_wt']}\n")
            fh.write(f">{r['ID']}_{r['cell_line']}_C18A eff={r['Normalized efficacy']:.4f} "
                     f"GC={r['gc_spacer']:.1f}\n{r['crispron_30mer_c18a']}\n")
    paths["fasta"] = fa

    ref = out_dir / "wt_position18_cellline_reference.csv"
    picks[["ID", "cell_line", "Chromosome", "Start", "End", "Strand",
           "Normalized efficacy"]].to_csv(ref, index=False, encoding="utf-8-sig")
    paths["reference"] = ref

    (out_dir / "wt_position18_selfcheck.json").write_text(
        json.dumps(selfcheck, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["selfcheck"] = out_dir / "wt_position18_selfcheck.json"

    return paths


def write_report(out_dir: Path, picks: pd.DataFrame, funnel: pd.DataFrame,
                 log: pd.DataFrame, bins: pd.DataFrame, selfcheck: dict,
                 conv: LocusConvention, conv_note: str,
                 dataset: str, seed: int) -> Path:
    """生成完整报告（含最终检查清单）。"""
    L = ["# WT → Position 18 → CRISPRon 代表性序列集", ""]
    L += [f"- 数据集：`{dataset}`（候选池 = `single` 划分的 **test 集**，group-aware 无泄漏，seed={seed}）",
          f"- 位点约定：{conv.describe()}",
          f"- 约定交叉校验：{conv_note}",
          f"- 挑选规则版本：`{SELECTION_RULE_VERSION}`（规则预先写死在脚本内）", ""]
    L += ["## 0. 挑选规则（预先登记）", "```", SELECTION_RULE.strip(), "```", ""]

    L += ["## 1. 最终入选序列（完整信息）", ""]
    for _, r in picks.iterrows():
        L += [f"### {r['ID']} — {r['cell_line']}",
              f"- WT（23 nt）：`{r['sgRNA']}`",
              f"- Position 18：`{r['pos18']}` → C18A 后 `A`",
              f"- C18A（23 nt）：`{r['mutant']}`",
              f"- PAM：`{r['pam']}`（NGG）｜ protospacer：`{r['sgRNA'][:conv.spacer_len]}`",
              f"- PAM-proximal seed (17–20)：`{r['seed_region']}`",
              f"- 原始实验效率：**{r['Normalized efficacy']:.4f}**（区间 `{r['label_bin']}`，"
              f"该区间中位数 {r['label_bin_median']:.4f}）",
              f"- GC：protospacer {r['gc_spacer']:.1f}% ｜ 23 nt {r['gc_23nt']:.1f}%",
              f"- 坐标：`{r['Chromosome']}:{r['Start']}-{r['End']}`（{r['Strand']} 链，hg19）",
              f"- CRISPRon 30-mer（WT）：`{r['crispron_30mer_wt']}`",
              f"- CRISPRon 30-mer（C18A）：`{r['crispron_30mer_c18a']}`",
              f"- 代表性理由：{r['selection_reason']}",
              ""]

    L += ["## 2. 为什么这些序列有代表性", ""]
    L += [f"- **分层依据**：在**每个细胞系内部**按原始实验效率的三分位切 low/medium/high，"
          "避免「某个细胞系整体偏高」把区间混淆。各切点与中位数：", ""]
    L += [bins.to_markdown(index=False), ""]
    L += ["- **取区间中心而非极值**：每个 (细胞系 × 区间) 先取最接近该区间中位数的序列进入短名单，",
          "  终选在这些中心代表里进行，因此不会偏向分布两端。",
          "- **多样性约束**：每个细胞系至多 2 条；两两序列错配 ≥ "
          f"{MIN_PAIRWISE_MISMATCH} nt；另含 2 条 GC 明显偏离中位数但仍在 "
          f"P{SPECIAL_GC_PCTL[0]:g}–P{SPECIAL_GC_PCTL[1]:g} 内的「特殊但仍正常」序列。",
          "- **不选最好看的**：挑选阶段完全未接触任何 C18A / CRISPRon 结果（见 §5）。", ""]

    L += ["## 3. WT → C18A 序列修改验证", "",
          "逐条断言：除 Position 18 外其余 22 个位置完全一致（`wt[:17]==mut[:17]` 且 "
          "`wt[18:]==mut[18:]`）。", ""]
    vt = picks[["ID", "sgRNA", "mutant", "verify_prefix_equal", "verify_suffix_equal",
                "verify_only_pos18_changed", "verify_n_mismatch"]]
    L += [vt.to_markdown(index=False), ""]

    L += ["## 4. CRISPRon 所需输入序列", "",
          "CRISPRon 的输入判据（官方 Help）："
          "**30 nt = 4 nt + target(20 nt) + PAM(NGG) + 3 nt**。"
          "下表的 30-mer 由 hg19 取窗并按链定向得到，且已核对中间 23 nt 与数据完全一致。", ""]
    ct = picks[["ID", "cell_line", "crispron_30mer_wt", "crispron_30mer_c18a", "crispron_window"]]
    L += [ct.to_markdown(index=False), ""]
    L += ["对应的 FASTA（可直接提交 CRISPRon）与本表同目录：`wt_position18_crispron_input.fa`。",
          "若改用 CRISPRon 的「基因组区间」输入模式，可用同目录的 "
          "`wt_position18_cellline_reference.csv`。", ""]
    L += ["## 5. 展示用简表", "", mentor_table(picks), ""]

    # 最终检查
    def fnum(stage):
        m = funnel.loc[funnel["stage"] == stage, "n"]
        return int(m.iloc[0]) if len(m) else 0

    n_pairs = len(picks) * (len(picks) - 1) // 2
    mms = [mismatches(a, b) for i, a in enumerate(picks["sgRNA"])
           for b in picks["sgRNA"].iloc[i + 1:]]
    log_txt = log.to_string(index=False)
    specials = picks[picks["selection_reason"].str.contains("特殊", na=False)]

    L += ["## 6. 最终检查清单", "",
          "| 项目 | 结果 |", "| --- | --- |",
          f"| 总候选数（test 集样本） | {fnum('0_pool')} |",
          f"| Position18=C 候选数 | {fnum('6_pos18_is_C')} |",
          f"| 合法 PAM(NGG) 数 | {fnum('7_pam_ngg')} |",
          f"| 可构造 30-mer（坐标可容）数 | {fnum('8_crispron_window_available')} |",
          f"| 与 hg19 逐条核对通过数 | {fnum('9_hg19_30mer_verified')} |",
          f"| 最终入选数 | {len(picks)} |",
          f"| low / medium / high | "
          f"{int((picks['label_bin']=='low').sum())} / "
          f"{int((picks['label_bin']=='medium').sum())} / "
          f"{int((picks['label_bin']=='high').sum())} |",
          f"| 各 cell line 数量 | "
          + "、".join(f"{k}: {v}" for k, v in picks['cell_line'].value_counts().sort_index().items())
          + " |",
          f"| 是否有重复序列 | 无（已按 canonical identity `min(seq,revcomp)` 去重） |",
          f"| 是否有高度相似序列 | 最小两两错配 **{min(mms)} nt**（阈值 ≥ {MIN_PAIRWISE_MISMATCH}）；"
          f"共 {n_pairs} 对 |",
          f"| 是否存在 C18A 反向筛选 | **不存在**：挑选只用 WT 侧字段；"
          f"C18A 序列在挑选完成后才生成，脚本内无任何 CRISPRon 调用 |",
          ""]

    L += ["### 无突变信息自证（selfcheck.json）", "```json",
          json.dumps(selfcheck, ensure_ascii=False, indent=2), "```", ""]
    L += ["### 「特殊但仍正常」的 2 条", ""]
    if len(specials):
        L += [specials[["ID", "cell_line", "gc_spacer", "label_bin",
                        "Normalized efficacy"]].to_markdown(index=False), ""]
    L += ["### 取舍日志（节选：被多样性约束拒绝的候选）", "```",
          "\n".join(log_txt.splitlines()[:40]), "```", ""]
    L += ["---", "",
          "复现：`python analysis/candidates/wt_position18_selection.py --dataset "
          f"{dataset} --seed {seed}`",
          "（hg19 取窗结果有本地缓存 `_hg19_window_cache.csv`，重跑不再联网）", ""]

    path = out_dir / "wt_position18_report.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def mentor_table(picks: pd.DataFrame) -> str:
    lines = ["| ID | Cell line | WT sequence (23 nt) | Pos18 | C18A sequence (23 nt) | "
             "Experimental efficiency | GC% | Selection reason |",
             "| -- | --------- | ------------------- | ----- | --------------------- | "
             "-----------------------: | --: | ---------------- |"]
    for _, r in picks.iterrows():
        lines.append(
            f"| {r['ID']} | {r['cell_line']} | `{r['sgRNA']}` | {r['pos18']} | "
            f"`{r['mutant']}` | {r['Normalized efficacy']:.3f} | {r['gc_spacer']:.1f} | "
            f"{r['selection_reason']} |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="挑选代表性 WT sgRNA（Position 18 = C）")
    ap.add_argument("--dataset", default="DeepCRISPR", help="数据集名称（默认 DeepCRISPR）")
    ap.add_argument("--out-dir", default=str(RESULTS_TABLES / "candidates"))
    ap.add_argument("--seed", type=int, default=42, help="划分种子（与论文一致，默认 42）")
    ap.add_argument("--http-workers", type=int, default=6,
                    help="并行取 hg19 序列的线程数（默认 6，对 UCSC 保持克制）")
    args = ap.parse_args()

    dataset_dir = resolve_dataset(args.dataset)
    out_dir = Path(args.out_dir)
    schema = load_feature_schema(str(dataset_dir))
    conv = LocusConvention(schema)
    conv_note = cross_check_with_project(conv)

    print(f"[数据集]   {args.dataset} -> {dataset_dir}")
    print(f"[位点约定] {conv.describe()}")
    print(f"[交叉校验] {conv_note}")

    print("\n[1/6] 构建候选池（single 划分 test 集）...")
    pool, pool_funnel = build_pool(dataset_dir, seed=args.seed)
    print(f"        test 集样本 {len(pool)}")

    print("[2/6] 过滤 ...")
    chrom_sizes = fetch_chrom_sizes()
    df, funnel = apply_filters(pool, conv, chrom_sizes)
    print(f"        通过过滤 {len(df)} 条")
    print(funnel.to_string(index=False))

    print("\n[2b/6] 逐条核对 hg19 30-mer（含链方向）...")
    cache_path = Path(args.out_dir) / "_hg19_window_cache.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df, win_stats = verify_all_windows(df, conv, cache_path, workers=args.http_workers)
    print(f"        核对结果: {win_stats}")
    n_before = len(df)
    df = df[df["hg19_match"]].reset_index(drop=True)
    funnel = pd.concat([funnel, pd.DataFrame([{
        "stage": "9_hg19_30mer_verified", "n": int(len(df)),
        "note": f"中间 23nt 与 hg19 完全一致（移除 {n_before - len(df)} 条: "
                f"{ {k: v for k, v in win_stats.items() if k != 'ok'} }）"}])], ignore_index=True)

    print("\n[3/6] 特征与分层 ...")
    df = annotate_features(df, conv)
    df, bins = stratify(df)

    print("[4/6] 按预先登记规则挑选（不使用任何突变侧信息）...")
    selected, log = select(df)
    if len(selected) < N_LOW + N_MEDIUM + N_HIGH + N_SPECIAL:
        print(f"        [WARN] 只选出 {len(selected)} 条（目标 "
              f"{N_LOW + N_MEDIUM + N_HIGH + N_SPECIAL} 条）——多样性约束下候选不足")

    # 自证必须在"构造突变之前"做：此时 selected 只含 WT 侧字段。
    # （构造之后必然出现 mutant 字段，那时再断言就失去意义了。）
    selfcheck = assert_no_mutant_information_used(selected)
    print(f"        自证通过：挑选阶段仅使用 {len(selfcheck['wt_only_columns'])} 个 WT 侧字段，"
          f"未出现任何突变侧字段")

    print("[5/6] 构造 C18A 并验证（仅在挑选完成之后）...")
    final = []
    for i, rec in enumerate(selected, start=1):
        row = pd.Series(rec)
        wt = row["sgRNA"]
        mut = build_mutant(wt, conv)
        v = verify_mutation(wt, mut, conv)
        if not (v["prefix_equal"] and v["suffix_equal"] and v["only_one_position_changed"]):
            raise AssertionError(f"{wt[:12]} 突变验证失败: {v}")
        m30, win = build_30mer(row, conv)
        m30_mut = list(m30)
        m30_mut[UPSTREAM_NT + conv.pos_index0] = "A"
        m30_mut = "".join(m30_mut)
        core_wt = m30[UPSTREAM_NT:UPSTREAM_NT + conv.seq_len]
        core_mut = m30_mut[UPSTREAM_NT:UPSTREAM_NT + conv.seq_len]
        assert core_wt == wt and core_mut == mut, "30-mer 中的 23nt 核心与序列不一致"
        final.append({**rec, "ID": f"WT{i:02d}", "mutant": mut, "pam": conv.pam_of(wt),
                      "crispron_30mer_wt": m30, "crispron_30mer_c18a": m30_mut,
                      "crispron_window": win,
                      "verify_prefix_equal": v["prefix_equal"],
                      "verify_suffix_equal": v["suffix_equal"],
                      "verify_only_pos18_changed": v["only_one_position_changed"],
                      "verify_n_mismatch": mismatches(wt, mut),
                      "selection_reason": row.get("selection_reason", "")})

    print("[6/6] 生成输出 ...")
    # 说明：mutant / crispron_30mer_c18a 是在上面挑选**完成之后**才生成的；
    # 挑选阶段的无突变信息自证已在 [4/6] 处完成（见 selfcheck）。
    paths = write_outputs(out_dir, conv, dataset_dir, funnel, bins, final, log,
                          selfcheck, conv_note)
    picks = pd.read_csv(paths["candidates"])

    # 最终检查清单
    checks = {
        "总候选数(test 集样本)": int(len(pool)),
        "Position18=C 候选数": int(funnel.loc[funnel["stage"] == "6_pos18_is_C", "n"].iloc[0]),
        "合法 PAM(NGG) 数": int(funnel.loc[funnel["stage"] == "7_pam_ngg", "n"].iloc[0]),
        "可构造 30-mer 数": int(len(df)),
        "最终入选数": int(len(picks)),
    }
    for b in ("low", "medium", "high"):
        checks[f"{b} 入选数"] = int((picks["label_bin"] == b).sum())
    for cl in sorted(picks["cell_line"].unique()):
        checks[f"cell_line {cl} 入选数"] = int((picks["cell_line"] == cl).sum())

    print("\n" + "=" * 72)
    for k, v in checks.items():
        print(f"  {k:<28} {v}")
    print("=" * 72)

    (out_dir / "wt_position18_for_mentor.md").write_text(
        "# WT → C18A 演示用代表性序列\n\n"
        f"候选池：`{args.dataset}` 的 `single` 划分 test 集（无泄漏，seed={args.seed}）；"
        "挑选规则预先登记，未使用任何 C18A 结果。\n\n"
        + mentor_table(picks) + "\n",
        encoding="utf-8")

    audit_all = pd.read_csv(paths["audit"])
    report_path = write_report(out_dir, picks, audit_all,
                               audit_all[audit_all["stage"].notna() & audit_all["n"].isna()],
                               pd.read_csv(paths["bins"]), selfcheck, conv, conv_note,
                               args.dataset, args.seed)
    print(f"    report       {report_path}")
    print(f"\n[✓] 输出目录: {out_dir}")
    for k, p in paths.items():
        print(f"    {k:<12} {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
