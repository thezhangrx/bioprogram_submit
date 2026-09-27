"""analysis.sequence.motif.core — seqlet 提取 / 聚类 / consensus / enrichment / stability。

数据来源 (只读):
    <data_root>/<cell_line>_metadata.csv   -> sgRNA + Normalized efficacy (行序 = 数据集序)
    tables/attribution_summary.csv         -> 统一 attribution (position × channel × method)

科学边界:
  * 当前批次 attribution 为**无符号 magnitude** (CNN_IG / CNN_ISM 均 > 0) -> 不按 attribution
    符号分正负; seqlet.direction 记 "unsigned", motif 的 effect_direction 由
    carrier vs background 的 measured efficacy 对比给出 (方向来源见 README 的 `R2` 一节);
    若未来出现有符号 attribution 列, 自动切换为按符号分离 (见 `signed_value_column`)。
  * 高归因位置**不**直接当 motif: 必须经 局部连续窗口 -> 聚类 -> consensus -> support 门槛。
  * attention 只作 supporting evidence, 绝不作为主要 extractor。
  * SNR / attention / magnitude 都不产生 p-value; 只有 enrichment 的 Fisher exact 产生 p。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.config import AnalysisConfig, MotifDiscoveryConfig
from analysis.sequence.motif.iupac import (IUPAC_TO_BASES, consensus_from_frequencies,
                                          contains_iupac, information_content,
                                          matches_iupac, pfm_from_instances)
from analysis.stats.multiple_testing import bh_fdr

SEQ_CHANNELS = ("A", "C", "G", "T")
MOTIF_ID_PREFIX = "motif"

INSTANCE_COLUMNS = ["motif_id", "sample_id", "cell_line", "split_type", "environment",
                    "model", "model_variant", "attribution_method", "position_start",
                    "position_end", "sequence", "attribution_score", "ism_effect",
                    "ig_effect", "direction", "efficacy"]

#: motif 候选表列。
#: **已移除的人为标签**：`evidence_strength`（"Strong/Moderate/…"分级）、
#: `stability`（原实现恒为 `unavailable`，零信息）。
#: **已移除的死列**：`region`（恒 `Other`，§4.6 缺陷）、`seed_support`（恒 None）。
#: **已移除的无意义数值列**：`attribution_snr`（见 README §14.6 —— 分母 std 恒为 0，
#: 结果是 `mean/1e-12`，无统计含义）。
#: **保留的是数据本身**：序列/位置/支持度/ISM 效应量/富集统计量。
CANDIDATE_COLUMNS = ["motif_id", "model_source", "model_variant", "attribution_method",
                     "split_type", "environment", "sequence", "consensus", "iupac",
                     "human_pattern", "regex", "length", "position_start", "position_end",
                     "position_mean", "position_std", "position_distribution",
                     "preferred_position", "support_count", "sample_support",
                     "mean_effect", "effect_direction",
                     "enrichment", "odds_ratio", "p_value", "FDR",
                     "status"]

ENRICHMENT_COLUMNS = ["motif_id", "foreground_count", "foreground_total",
                     "background_count", "background_total", "effect", "enrichment",
                     "odds_ratio", "p_value", "FDR", "status", "reason"]


# ---------------------------------------------------------------------------
# 1) 数据加载 (只读)
# ---------------------------------------------------------------------------
def load_sequences(data_root: str | Path, cell_line: str) -> pd.DataFrame:
    """<cell_line>_metadata.csv -> DataFrame(row_index, cell_line, sgRNA, efficacy)。"""
    path = Path(data_root) / f"{str(cell_line).lower()}_metadata.csv"
    if not path.exists():
        return pd.DataFrame(columns=["row_index", "cell_line", "sgRNA", "efficacy"])
    df = pd.read_csv(path)
    col = "sgRNA" if "sgRNA" in df.columns else None
    if col is None:
        return pd.DataFrame(columns=["row_index", "cell_line", "sgRNA", "efficacy"])
    eff_col = next((c for c in df.columns if "efficacy" in str(c).lower()), None)
    out = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "cell_line": str(cell_line).lower(),
        "sgRNA": df[col].astype(str).str.upper().str.strip(),
        "efficacy": pd.to_numeric(df[eff_col], errors="coerce") if eff_col else np.nan,
    })
    valid = out["sgRNA"].str.fullmatch(r"[ACGT]+")
    return out[valid].reset_index(drop=True)


def load_attribution(attribution_table: pd.DataFrame, cfg: MotifDiscoveryConfig,
                     methods: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """过滤出 motif 可用的 attribution 行 (model / method / environment 上下文)。"""
    if attribution_table is None or attribution_table.empty:
        return pd.DataFrame()
    df = attribution_table.copy()
    for col in ("model", "method", "environment", "split_type", "cell_line", "channel",
                "architecture"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    use_methods = set(methods or (list(cfg.primary_methods) + list(cfg.supporting_methods)))
    df = df[df["method"].isin(use_methods)]
    df = df[df["model"].isin(cfg.model_families) |
            df["method"].isin(cfg.supporting_methods)]
    df = df[df["environment"].isin(cfg.contexts)]
    df = df[df["channel"].isin(SEQ_CHANNELS)]
    return df.dropna(subset=["importance", "position"]).reset_index(drop=True)


def signed_value_column(attribution_table: pd.DataFrame) -> Optional[str]:
    """检测给定行上是否存在**有符号** attribution 列 (无 -> None, 不伪造)。

    注意: 必须在**已过滤**的行上判断 (linear coefficient 等其它模型的符号列不属于
    CNN/Transformer 序列 attribution)。
    """
    if attribution_table is None or attribution_table.empty:
        return None
    for col in ("effect", "signed_effect", "importance_signed"):
        if col in attribution_table.columns:
            vals = pd.to_numeric(attribution_table[col], errors="coerce").dropna()
            if len(vals) >= 3 and (vals < 0).any() and (vals > 0).any():
                return col
    return None


def signed_profiles(attr: pd.DataFrame, value_column: str) -> Dict[Tuple, np.ndarray]:
    """有符号 attribution 的 (23, 4) 矩阵 (与 position_profiles 同 key)。"""
    if attr is None or attr.empty or value_column not in attr.columns:
        return {}
    work = attr.copy()
    work[value_column] = pd.to_numeric(work[value_column], errors="coerce")
    work = work.dropna(subset=[value_column])
    out: Dict[Tuple, np.ndarray] = {}
    for key, sub in work.groupby(["model", "architecture", "split_type", "cell_line",
                                  "environment", "method"], dropna=False):
        pivot = sub.groupby(["position", "channel"])[value_column].mean().unstack("channel")
        mat = np.zeros((23, len(SEQ_CHANNELS)), dtype=float)
        for pos in pivot.index:
            for j, ch in enumerate(SEQ_CHANNELS):
                if ch in pivot.columns:
                    v = pivot.loc[pos, ch]
                    mat[int(pos) - 1, j] = 0.0 if pd.isna(v) else float(v)
        out[key] = mat
    return out


def position_profiles(attr: pd.DataFrame) -> Dict[Tuple, np.ndarray]:
    """(model, architecture, split_type, cell_line, environment, method) -> (23, 4) 矩阵。

    多个 seed/重复实验的同一 key 取均值 (attribution_summary 不含 seed, 见模块文档)。
    """
    profiles: Dict[Tuple, np.ndarray] = {}
    if attr.empty:
        return profiles
    for key, sub in attr.groupby(["model", "architecture", "split_type", "cell_line",
                                  "environment", "method"], dropna=False):
        pivot = sub.groupby(["position", "channel"])["importance"].mean().unstack("channel")
        n_pos = int(pivot.index.max()) if len(pivot.index) else 0
        mat = np.zeros((max(n_pos, 23), len(SEQ_CHANNELS)), dtype=float)
        for i, pos in enumerate(pivot.index):
            for j, ch in enumerate(SEQ_CHANNELS):
                if ch in pivot.columns:
                    v = pivot.loc[pos, ch]
                    mat[int(pos) - 1, j] = 0.0 if pd.isna(v) else float(v)
        profiles[key] = mat[:23, :]
    return profiles


# ---------------------------------------------------------------------------
# 2) Seqlet extraction (高归因 + 局部连续窗口; 不是 top-N 单点)
# ---------------------------------------------------------------------------
def _contrast_matrix(profile: np.ndarray) -> np.ndarray:
    """位置×碱基 的 attribution specificity: 该碱基值 − 同位置其它碱基均值。"""
    arr = np.asarray(profile, dtype=float)
    total = arr.sum(axis=1, keepdims=True)
    others = (total - arr) / max(arr.shape[1] - 1, 1)
    return arr - others


def _run_windows(seq: str, scores: np.ndarray, cfg: MotifDiscoveryConfig,
                 q_cont: float) -> List[Tuple[int, int, float]]:
    """返回 [(start0, end0_exclusive, window_score)] —— 每个长度取 run 峰值附近的窗口。"""
    hits = scores >= q_cont
    windows: List[Tuple[int, int, float]] = []
    i, n = 0, len(scores)
    while i < n:
        if not hits[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and hits[j + 1]:
            j += 1
        run_len = j - i + 1
        if run_len >= int(cfg.continuity_min_positions):
            peak = i + int(np.argmax(scores[i:j + 1]))
            for length in range(int(cfg.min_length), int(cfg.max_length) + 1):
                half = length // 2
                start = int(np.clip(peak - half, 0, max(0, n - length)))
                end = start + length
                if end > n:
                    continue
                windows.append((start, end, float(np.mean(scores[start:end]))))
        i = j + 1
    # 去重 (同一 start/end 保留最高分)
    best: Dict[Tuple[int, int], float] = {}
    for start, end, score in windows:
        key = (start, end)
        best[key] = max(best.get(key, -np.inf), score)
    ordered = sorted(((s, e, v) for (s, e), v in best.items()), key=lambda t: t[2], reverse=True)
    return ordered[:int(cfg.max_seqlets_per_sample)]


def extract_seqlets(sequences: pd.DataFrame, profile: np.ndarray,
                    cfg: MotifDiscoveryConfig, context: Dict[str, str],
                    signed: bool = False,
                    signed_profile: Optional[np.ndarray] = None) -> List[Dict]:
    """对某 context 的 attribution profile, 逐序列提取 seqlet。

    打分用 attribution **specificity** (当前碱基 attribution − 该位置其它碱基均值),
    因为 raw magnitude 对所有序列都很高, 无法区分是哪个碱基在驱动。
    """
    contrast = _contrast_matrix(profile)
    flat = contrast[np.isfinite(contrast)]
    flat = flat[flat > 0]
    if flat.size == 0:
        return []
    q_hi = float(np.quantile(flat, cfg.attribution_quantile))
    q_cont = float(np.quantile(flat, cfg.continuity_quantile))
    q_cont = max(q_cont, cfg.min_seqlet_score)
    out: List[Dict] = []
    for _, row in sequences.iterrows():
        seq = str(row["sgRNA"])
        if len(seq) != profile.shape[0]:
            continue
        scores = np.array([contrast[i, SEQ_CHANNELS.index(b)] if b in SEQ_CHANNELS else 0.0
                           for i, b in enumerate(seq)], dtype=float)
        if scores.max() < q_hi:
            continue
        for start, end, score in _run_windows(seq, scores, cfg, q_cont):
            # 阈值作用在**位置级** (窗口峰值), 窗口均值只用于排序/限流
            if float(np.max(scores[start:end])) < q_hi:
                continue
            direction = "unsigned"
            if signed and signed_profile is not None:
                sub = signed_profile[start:end, :]
                signed_mean = float(np.nanmean(sub)) if sub.size else 0.0
                if signed_mean > 0:
                    direction = "signed_positive"
                elif signed_mean < 0:
                    direction = "signed_negative"
            out.append({
                "sample_id": int(row["row_index"]),
                "cell_line": row["cell_line"],
                "efficacy": float(row["efficacy"]) if pd.notna(row["efficacy"]) else np.nan,
                "position_start": start + 1,
                "position_end": end,
                "sequence": seq[start:end],
                "attribution_score": score,
                "direction": direction,
                **context,
            })
    return out


def add_dual_method_effects(seqlets: List[Dict],
                            profiles: Dict[Tuple, np.ndarray]) -> None:
    """为每个 seqlet 记录 ISM / IG 两种方法的窗口均值 (缺失保持 NaN, 不填 0)。"""
    for s in seqlets:
        for method_key, field in (("cnn_ism", "ism_effect"), ("cnn_ig", "ig_effect")):
            key = (s["model"], s.get("model_variant"), s["split_type"], s["cell_line"],
                   s["environment"], method_key)
            prof = profiles.get(key)
            if prof is None:
                s[field] = np.nan
                continue
            i0, i1 = int(s["position_start"]) - 1, int(s["position_end"])
            sub = prof[i0:i1, :]
            s[field] = float(np.nanmean(sub)) if sub.size else np.nan


# ---------------------------------------------------------------------------
# 3) Clustering + consensus
# ---------------------------------------------------------------------------
def _similarity_to_codes(sequence: str, codes: Sequence[str]) -> float:
    """序列与 cluster consensus (IUPAC) 的一致率 —— 退化位点按可接受碱基集合判断。"""
    if not codes or len(sequence) != len(codes):
        return 0.0
    hits = 0
    for base, code in zip(str(sequence).upper(), codes):
        allowed = IUPAC_TO_BASES.get(str(code).upper(), ())
        if allowed and base in allowed:
            hits += 1
    return hits / len(codes)


def _empty_counts(length: int) -> List[Dict[str, float]]:
    return [dict() for _ in range(int(length))]


def _add_to_counts(counts: List[Dict[str, float]], sequence: str) -> None:
    for j, base in enumerate(str(sequence)[:len(counts)]):
        counts[j][base] = counts[j].get(base, 0.0) + 1.0


def _codes_from_counts(counts: List[Dict[str, float]], cfg: MotifDiscoveryConfig) -> List[str]:
    freqs = []
    for dist in counts:
        total = sum(dist.values()) or 1.0
        freqs.append({b: v / total for b, v in dist.items()})
    return list(consensus_from_frequencies(freqs, cfg.degenerate_fraction)["iupac"])


def cluster_seqlets(seqlets: List[Dict], cfg: MotifDiscoveryConfig
                    ) -> List[Dict[str, object]]:
    """贪心相似度聚类: 与已有 cluster consensus 的一致率 >= 阈值则并入, 否则新建。

    similarity 同时考虑 attribution: 优先按 attribution_score 降序处理, 使高分 seqlet
    先形成 cluster (attribution similarity 通过排序 + 窗口分数体现)。
    """
    clusters: List[Dict[str, object]] = []
    for s in sorted(seqlets, key=lambda d: d["attribution_score"], reverse=True):
        placed = False
        for cl in clusters:
            if cl["length"] != len(s["sequence"]):
                continue
            if _similarity_to_codes(s["sequence"], cl["codes"]) >= cfg.similarity_threshold:
                cl["instances"].append(s)
                # 增量维护碱基计数 (O(length)), 避免每次并入都 O(cluster size) 重算
                _add_to_counts(cl["counts"], s["sequence"])
                cl["codes"] = _codes_from_counts(cl["counts"], cfg)
                placed = True
                break
        if not placed:
            # 新 cluster 必须立即有 consensus, 否则后续相似度比较会因 codes=None 恒为 0,
            # 导致每个 seqlet 自成一簇 (历史 bug)。
            counts = _empty_counts(len(s["sequence"]))
            _add_to_counts(counts, s["sequence"])
            clusters.append({"length": len(s["sequence"]), "instances": [s],
                             "counts": counts, "codes": _codes_from_counts(counts, cfg)})
    return merge_similar_clusters(clusters, cfg)


def merge_similar_clusters(clusters: List[Dict[str, object]],
                           cfg: MotifDiscoveryConfig) -> List[Dict[str, object]]:
    """聚类后处理: consensus 相似的 cluster 合并 (避免同一 motif 出现多个 id)。"""
    merged: List[Dict[str, object]] = []
    for cl in sorted(clusters, key=lambda c: len(c["instances"]), reverse=True):
        target = None
        for m in merged:
            if m["length"] != cl["length"]:
                continue
            if _similarity_to_codes("".join(IUPAC_TO_BASES.get(c, ("N",))[0]
                                            for c in cl["codes"]), m["codes"]) \
                    >= cfg.merge_similarity:
                target = m
                break
        if target is None:
            merged.append(cl)
        else:
            target["instances"].extend(cl["instances"])
            target["counts"] = _empty_counts(target["length"])
            _add_to_counts(target["counts"], target["instances"][0]["sequence"])
            for inst in target["instances"][1:]:
                _add_to_counts(target["counts"], inst["sequence"])
            target["codes"] = _codes_from_counts(target["counts"], cfg)
    return merged


def finalize_clusters(clusters: List[Dict], cfg: MotifDiscoveryConfig,
                      sequences_by_cell: Dict[str, pd.DataFrame]) -> List[Dict]:
    """过滤 support 门槛并补齐 consensus / 位置 / 方向 / support 字段。"""
    out: List[Dict] = []
    for cl in clusters:
        inst = cl["instances"]
        sample_support = len({(i["cell_line"], i["sample_id"]) for i in inst})
        if len(inst) < int(cfg.min_seqlet_support) or sample_support < int(cfg.min_sample_support):
            continue
        length = cl["length"]
        freqs = []
        for j in range(length):
            dist: Dict[str, float] = {}
            for i in inst:
                dist[i["sequence"][j]] = dist.get(i["sequence"][j], 0.0) + 1.0
            total = sum(dist.values()) or 1.0
            freqs.append({b: v / total for b, v in dist.items()})
        cons = consensus_from_frequencies(freqs, cfg.degenerate_fraction)
        starts = np.array([i["position_start"] for i in inst], dtype=float)
        cells = sorted({i["cell_line"] for i in inst})
        # 方向: carrier vs background 的 measured efficacy 对比 (attribution 无符号)
        eff = np.array([i["efficacy"] for i in inst if np.isfinite(i.get("efficacy", np.nan))],
                       dtype=float)
        bg_eff: List[float] = []
        for cell in cells:
            seqs = sequences_by_cell.get(cell)
            if seqs is None or seqs.empty:
                continue
            mask = ~seqs["row_index"].isin([i["sample_id"] for i in inst if i["cell_line"] == cell])
            bg_eff.extend(seqs.loc[mask, "efficacy"].dropna().tolist())
        direction = None
        mean_effect = float("nan")
        if eff.size and bg_eff:
            # 方向只能来自**实测效率对比**（attribution 无符号），来源说明见 README 的 R2 一节
            mean_effect = float(np.mean(eff) - np.mean(bg_eff))
            direction = "+" if mean_effect > 0 else ("-" if mean_effect < 0 else "0")
        codes = list(cons["iupac"])
        n_ambiguous = sum(1 for c in codes if c not in ("A", "C", "G", "T"))
        ambiguous_fraction = n_ambiguous / max(length, 1)
        status = "ok" if ambiguous_fraction <= cfg.max_ambiguous_fraction else "exploratory"
        out.append({
            **cons,
            "status": status,
            "ambiguous_fraction": ambiguous_fraction,
            "length": length,
            "sequence": inst[0]["sequence"],
            "instances": inst,
            "model_source": inst[0]["model"],
            "model_variant": inst[0].get("model_variant"),
            "attribution_method": inst[0]["attribution_method"],
            "split_type": inst[0]["split_type"],
            "environment": inst[0]["environment"],
            "direction": inst[0]["direction"],
            "cell_lines": cells,
            "support_count": len(inst),
            "sample_support": sample_support,
            # 注意：**不输出 cellline_support 列**（2026-09-20 删除）—— motif 按
            # (cell_line, environment, …) 上下文聚类，该值恒为 1，会误导读者以为
            # "仅出现在 1 个细胞系"。跨细胞系支持计数（原 cellline_consistency）
            # 随 motif_consistency.csv 一并移出，如需可按 (model_variant, environment,
            # method) 重新聚合。
            "position_start": int(np.min(starts)),
            "position_end": int(np.max(np.array([i["position_end"] for i in inst]))),
            "position_mean": float(np.mean(starts)),
            "position_std": float(np.std(starts, ddof=1)) if starts.size > 1 else 0.0,
            "position_distribution": ";".join(
                f"{int(p)}:{int(c)}" for p, c in
                zip(*np.unique(starts.astype(int), return_counts=True))),
            "mean_effect": mean_effect,
            "effect_direction": direction,
            "pfm": pfm_from_instances([i["sequence"] for i in inst], length),
        })
    return out


def enrichment_for_motif(motif: Dict, sequences_by_cell: Dict[str, pd.DataFrame],
                         cfg: MotifDiscoveryConfig, fdr: float) -> Dict:
    """carrier vs background 的 motif 频率 + Fisher exact p (scipy 可用时)。"""
    bg_kind = str(cfg.enrichment_background)
    if bg_kind != "complement_of_foreground":
        raise ValueError(
            f"MotifDiscoveryConfig.enrichment_background 只支持 'complement_of_foreground'，"
            f"实际={bg_kind!r}。Fisher 精确检验要求两组互斥；用包含前景的"
            f"'all_eligible_sequences' 会让 2x2 表两行重叠、p 值系统性偏向 1。"
        )
    row = {"motif_id": motif["motif_id"],
           "foreground_count": 0, "foreground_total": 0, "background_count": 0,
           "background_total": 0, "effect": None, "enrichment": None, "odds_ratio": None,
           "p_value": None, "FDR": None, "status": "unavailable", "reason": ""}
    codes = list(motif["iupac"])
    fg_hit = fg_tot = bg_hit = bg_tot = 0
    for cell, seqs in sequences_by_cell.items():
        if seqs.empty or "efficacy" not in seqs.columns:
            continue
        eff = pd.to_numeric(seqs["efficacy"], errors="coerce")
        if eff.notna().sum() < 3:
            continue
        thr = float(eff.quantile(cfg.enrichment_foreground_quantile))
        # 前景 = 上分位；背景 = **其补集**（efficacy < thr）。
        # 两者互斥且穷尽所有非空样本（NaN 在两边都被自动排除），满足 Fisher 精确检验前提。
        fg = seqs[eff >= thr]
        bg = seqs[eff < thr]
        fg_hit += int(sum(1 for s in fg["sgRNA"] if contains_iupac(s, codes)))
        fg_tot += int(len(fg))
        bg_hit += int(sum(1 for s in bg["sgRNA"] if contains_iupac(s, codes)))
        bg_tot += int(len(bg))
    row.update({"foreground_count": fg_hit, "foreground_total": fg_tot,
                "background_count": bg_hit, "background_total": bg_tot})
    if fg_tot == 0 or bg_tot == 0 or fg_hit < int(cfg.enrichment_min_carriers):
        row["reason"] = "insufficient carriers/eligible sequences"
        return row
    fg_rate = fg_hit / fg_tot
    bg_rate = bg_hit / bg_tot
    row["effect"] = fg_rate - bg_rate
    row["enrichment"] = (fg_rate / bg_rate) if bg_rate > 0 else None
    a, b = fg_hit, fg_tot - fg_hit
    c, d = bg_hit, bg_tot - bg_hit
    row["odds_ratio"] = ((a * d) / (b * c)) if (b > 0 and c > 0 and a > 0) else None
    try:
        from scipy.stats import fisher_exact
        _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        row["p_value"] = float(p)
        row["status"] = "ok"
    except Exception:  # noqa: BLE001 - 无 scipy / 退化表格
        row["status"] = "unavailable"
        row["reason"] = "Fisher exact test unavailable (scipy missing or degenerate table)"
    return row


def apply_enrichment_fdr(rows: List[Dict]) -> List[Dict]:
    """motif_enrichment family 内 BH-FDR (与其它问题不混用)。"""
    idx = [i for i, r in enumerate(rows) if r.get("p_value") is not None]
    if idx:
        q = bh_fdr([rows[i]["p_value"] for i in idx])
        for i, qv in zip(idx, q):
            rows[i]["FDR"] = qv
    return rows

