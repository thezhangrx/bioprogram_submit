"""analysis.sequence.motif.pipeline — motif discovery 编排（**只产数据表**）。

数据流 (只读已有 artifact):
    tables/attribution_summary.csv  +  <data_root>/<cell>_metadata.csv
        -> seqlets -> clustering -> motif candidates
        -> enrichment (可选, 走 §5.4 的 motif_enrichment family)
        -> tables/motif_candidates.csv / motif_instances.csv / motif_enrichment.csv

已移出（人为标签体系）:
  * `motif_consistency.csv`（§4.3 跨模型/跨细胞系一致性）
  * `motif_evidence.csv`（§2 Evidence Tier 行）
  * `figures/motif/`（motif_consistency.png + motif_logos/）及其绘制函数
  * motif 候选表里的 `stability` / `model_consistency` / `cellline_consistency` /
    `evidence_strength` / `region` / `seed_support` 列

保留的是**数据本身**（序列、位置、支持度、ISM 效应量）与 §1 的 `attribution_snr`、
§5.4 的富集 FDR —— 这些是 `rule_discovery.py` 的输入。
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analysis.config import AnalysisConfig
from analysis.sequence.motif import core


def _safe(value) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_")


def run_motif_discovery(attribution_table: pd.DataFrame,
                        batch_dir: str | Path,
                        data_root: Optional[str | Path] = None,
                        config: Optional[AnalysisConfig] = None,
                        discovery: bool = True,
                        enrichment: bool = False) -> Dict[str, object]:
    """完整 motif discovery; 返回 {tables, figures, motifs, summary}。"""
    cfg = (config or AnalysisConfig()).motif
    root = Path(data_root) if data_root else Path(cfg.data_root)
    attr = core.load_attribution(attribution_table, cfg)
    signed_col = core.signed_value_column(attr)      # 只在序列 attribution 行上判断
    signed_profiles = core.signed_profiles(attr, signed_col) if signed_col else {}
    result: Dict[str, object] = {
        "motifs": [], "instances": [], "enrichment": [],
        "tables": {}, "figures": [], "summary": {}, "signed_column": signed_col,
        "unavailable_reasons": [],
        #: 富集是否**真的执行过**（区别于 "计划里关掉了"）。落盘方据此决定
        #: 要不要覆盖 motif_enrichment.csv —— 关掉时保留既有产物，不写占位表。
        "enrichment_performed": False,
    }
    if not discovery:
        result["unavailable_reasons"].append("user_disabled (sequence.motif_discovery=false)")
        return result
    if attr.empty:
        result["unavailable_reasons"].append(
            "no compatible sequence attribution artifact (CNN ISM/IG or Transformer IG)")
        return result

    profiles = core.position_profiles(attr)
    sequences_by_cell: Dict[str, pd.DataFrame] = {}
    for cell in sorted(set(attr["cell_line"]) - {"none", "nan", ""}):
        seqs = core.load_sequences(root, cell)
        if not seqs.empty:
            sequences_by_cell[cell] = seqs
    if not sequences_by_cell:
        result["unavailable_reasons"].append(
            f"no sequence metadata under {root} (<cell_line>_metadata.csv)")
        return result

    all_seqlets: List[Dict] = []
    motif_seqlets: Dict[Tuple, List[Dict]] = {}
    for key, profile in profiles.items():
        model, arch, split, cell, environment, method = key
        seqs = sequences_by_cell.get(cell)
        if seqs is None:
            continue
        if method not in cfg.primary_methods:
            # Transformer IG 若存在则可用; attention 只作 supporting, 不生成 motif
            if method != cfg.transformer_ig_method:
                continue
        context = {"model": model, "model_variant": arch, "split_type": split,
                   "environment": environment, "attribution_method": method}
        seqlets = core.extract_seqlets(
            seqs, profile, cfg, context, signed=bool(signed_col),
            signed_profile=signed_profiles.get(key) if signed_col else None)
        if not seqlets:
            continue
        core.add_dual_method_effects(seqlets, profiles)
        all_seqlets.extend(seqlets)
        motif_seqlets[key] = seqlets

    # 正负方向分开 (有符号 attribution 时); 当前批次为 unsigned -> 单一分组, 方向在 motif 级给出
    clusters: List[Dict] = []
    for key, seqlets in motif_seqlets.items():
        def _bucket(sl: Dict) -> str:
            d = str(sl.get("direction", "unsigned"))
            return d if d in ("signed_positive", "signed_negative") else "unsigned"

        directions: Dict[str, List[Dict]] = {}
        for sl in seqlets:
            directions.setdefault(_bucket(sl), []).append(sl)
        for tag, group in directions.items():
            if not group:
                continue
            got = core.cluster_seqlets(group, cfg)
            clusters.extend(core.finalize_clusters(got, cfg, sequences_by_cell))

    # per-context 截断 (按 support 排序), 避免同一 context 产出上百个近重复 motif
    by_context: Dict[Tuple, List[Dict]] = {}
    for cl in clusters:
        key = (cl["model_source"], cl.get("model_variant"), cl["split_type"],
               cl["cell_lines"][0], cl["environment"], cl["attribution_method"],
               cl.get("direction", "unsigned"))
        by_context.setdefault(key, []).append(cl)
    capped: List[Dict] = []
    for key, group in by_context.items():
        group.sort(key=lambda c: (c["support_count"], -c["ambiguous_fraction"]), reverse=True)
        ok = [c for c in group if c.get("ambiguous_fraction", 1.0) <= cfg.max_ambiguous_fraction]
        vague = [c for c in group if c.get("ambiguous_fraction", 1.0) > cfg.max_ambiguous_fraction]
        capped.extend(ok[:int(cfg.max_motifs_per_context)])
        capped.extend(vague[:int(cfg.max_exploratory_per_context)])
    clusters = capped

    # motif id + region
    motifs: List[Dict] = []
    for i, cl in enumerate(sorted(clusters, key=lambda c: c["support_count"], reverse=True), 1):
        cl["motif_id"] = (f"motif_{_safe(cl['model_variant'] or cl['model_source'])}_"
                          f"{_safe(cl['cell_lines'][0])}_{_safe(cl['environment'])}_"
                          f"{_safe(cl['attribution_method'])}_{i:03d}")
        motifs.append(cl)

    # enrichment (独立可选步骤)
    enrichment_rows: List[Dict] = []
    if enrichment and motifs:
        for m in motifs:
            enrichment_rows.append(core.enrichment_for_motif(m, sequences_by_cell, cfg,
                                                            cfg.enrichment_fdr))
        enrichment_rows = core.apply_enrichment_fdr(enrichment_rows)
    elif not enrichment:
        for m in motifs:
            enrichment_rows.append({"motif_id": m["motif_id"], "fdr_family": "motif_enrichment",
                                    "status": "not_performed",
                                    "reason": "motif_enrichment=false in AnalysisPlan",
                                    "foreground_count": None, "foreground_total": None,
                                    "background_count": None, "background_total": None,
                                    "effect": None, "enrichment": None, "odds_ratio": None,
                                    "p_value": None, "FDR": None})
    enrich_by_id = {r["motif_id"]: r for r in enrichment_rows}

    # 富集统计量回填（**不派生** motif 证据强度分级 / 一致性计数）
    for m in motifs:
        e = enrich_by_id.get(m["motif_id"], {})
        fdr = e.get("FDR") if e.get("status") == "ok" else None
        m["enrichment"] = e.get("enrichment")
        m["odds_ratio"] = e.get("odds_ratio")
        m["p_value"] = e.get("p_value")
        m["FDR"] = fdr
        m["status"] = m.get("status", "ok")
        if m.get("ambiguous_fraction", 0) > cfg.max_ambiguous_fraction:
            m["status"] = "exploratory"
        m["direction"] = m.get("direction", "unsigned")

    result.update({"motifs": motifs, "enrichment": enrichment_rows,
                   "instances": all_seqlets, "sequences": sequences_by_cell,
                   "enrichment_performed": bool(enrichment)})
    result["summary"] = {
        "n_seqlets": len(all_seqlets),
        "n_motifs": len(motifs),
        "n_cell_lines": len(sequences_by_cell),
        "n_contexts": len(motif_seqlets),
        "signed_attribution": bool(signed_col),
        "enrichment_performed": bool(enrichment),
    }
    return result


# ---------------------------------------------------------------------------
# 表格 / 图 / 报告
# ---------------------------------------------------------------------------
def write_motif_tables(result: Dict, tables_dir: str | Path) -> Dict[str, str]:
    tables = Path(tables_dir)
    tables.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    motifs = result.get("motifs", [])

    cand_rows = []
    for m in motifs:
        row = {c: m.get(c) for c in core.CANDIDATE_COLUMNS}
        row["sequence"] = m.get("sequence")
        row["position_distribution"] = m.get("position_distribution")
        row["preferred_position"] = _preferred_position(m)
        row["seed_support"] = None
        cand_rows.append(row)
    cand = pd.DataFrame(cand_rows, columns=core.CANDIDATE_COLUMNS)
    cand.to_csv(tables / "motif_candidates.csv", index=False)
    paths["candidates"] = str(tables / "motif_candidates.csv")

    inst_rows = []
    for m in motifs:
        for inst in m["instances"]:
            inst_rows.append({"motif_id": m["motif_id"], **{k: inst.get(k) for k in
                             core.INSTANCE_COLUMNS if k != "motif_id"}})
    inst = pd.DataFrame(inst_rows, columns=core.INSTANCE_COLUMNS)
    inst.to_csv(tables / "motif_instances.csv", index=False)
    paths["instances"] = str(tables / "motif_instances.csv")

    # 富集是**独立可选**步骤：只有真的跑了才写文件。
    # 原实现无条件写，于是 enrichment=False 时会把已有的 motif_enrichment.csv
    # 覆盖成 656 行 status="not_performed" 的占位表——而 pipeline 同时把该任务
    # 汇报为 skipped("user_disabled")，属于**静默销毁交付产物**。
    if result.get("enrichment_performed"):
        enr = pd.DataFrame(result.get("enrichment", []), columns=core.ENRICHMENT_COLUMNS)
        enr.to_csv(tables / "motif_enrichment.csv", index=False)
        paths["enrichment"] = str(tables / "motif_enrichment.csv")

    # `motif_consistency.csv`（跨变体/跨细胞系 support 计数）与
    # `motif_evidence.csv`（§2 Tier 投影）均已移出，不再产出。
    return paths


def _preferred_position(m: Dict) -> str:
    dist = {}
    for part in str(m.get("position_distribution") or "").split(";"):
        if ":" in part:
            pos, cnt = part.split(":", 1)
            try:
                dist[int(pos)] = int(cnt)
            except ValueError:
                continue
    if not dist:
        return ""
    top = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)[:3]
    return ";".join(str(p) for p, _ in sorted(top))


def run_and_write(attribution_table: pd.DataFrame, batch_dir: str | Path,
                  tables_dir: str | Path, data_root: Optional[str | Path] = None,
                  config: Optional[AnalysisConfig] = None,
                  discovery: bool = True, enrichment: bool = False) -> Dict[str, object]:
    """编排 + 落盘 (只写 tables)，返回结果字典。

    出图已随 `figures/motif/` 一并移出：本模块现在只负责**数据**。
    """
    result = run_motif_discovery(attribution_table, batch_dir, data_root=data_root,
                                 config=config, discovery=discovery, enrichment=enrichment)
    paths = write_motif_tables(result, tables_dir) if result.get("motifs") else {}
    result["table_paths"] = paths
    return result
