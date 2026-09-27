"""analysis.attribution.extractors — 从训练产物抽取统一 attribution 表。

只读; 识别实验目录内重要性/权重 CSV (adapter 处理旧/新列名差异)。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from analysis.attribution.columns import (FEATURE_ID_CANDIDATES, family_from_model,
                                          pick_column, _ROW_SPECS)

SKIP_NAME_TOKENS = ("pred", "metric", "history", "summary", "biomarker", "training")
CANONICAL_COLUMNS = ["feature", "channel", "position", "model", "architecture",
                     "split_type", "cell_line", "environment", "method",
                     "importance", "snr", "effect", "attention_entropy", "source_file"]


def _parse_position_channel(feature: str) -> tuple:
    fn = str(feature).strip()
    m = re.search(r"pos_?(\d+)", fn, re.IGNORECASE)
    if not m:
        return None, None
    pos0 = int(m.group(1))
    # 'pos20_G' 风格: 数字即 1-based 位点; 'G_pos_20' 风格: 数字为 0-based 索引
    one_based = m.start() == 0 and fn.lower().startswith("pos")
    tail = fn[: m.start()] + fn[m.end():]
    ch = None
    for token in re.split(r"[_\-.]", tail):
        if token and not token.isdigit():
            ch = token
            break
    return (pos0 if one_based else pos0 + 1), ch


def _read_info(exp_dir: Path) -> Dict[str, str]:
    files = list(exp_dir.glob("*info*.txt"))
    if not files:
        return {}
    info: Dict[str, str] = {}
    for line in files[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip().lower()] = v.strip()
    return info


def extract_attribution_table(batch_dir: str | Path) -> pd.DataFrame:
    batch = Path(batch_dir)
    records: List[Dict[str, Any]] = []
    for csv_file in sorted(batch.glob("**/*.csv")):
        name = csv_file.name.lower()
        if any(tok in name for tok in SKIP_NAME_TOKENS):
            continue
        exp_dir = csv_file.parent
        if exp_dir.name == "summary":
            continue
        info = _read_info(exp_dir)
        model_raw = str(info.get("model", "")).lower()
        family = family_from_model(model_raw or exp_dir.name)
        if family is None:
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception:  # noqa: BLE001 - 单文件失败跳过并继续
            continue
        if df.empty:
            continue
        feat_col = pick_column(list(df.columns), FEATURE_ID_CANDIDATES)
        if feat_col is None:
            continue

        split_type = str(info.get("split_type", "single")).lower()
        cell_line = str(info.get("cell_line", info.get("held_out_cell_line", "none"))).lower()
        if split_type == "mixed":
            cell_line = "none"
        environment = str(info.get("environment", info.get("combination", "all"))).lower()

        architecture: Optional[str] = None
        if family == "cnn":
            try:
                k = int(float(info.get("sequence_kernel", "3") or "3"))
            except (TypeError, ValueError):
                k = 3
            architecture = f"cnn{k}3"

        for method, imp_cands, snr_cands, eff_cands, ent_cands in _ROW_SPECS[family]:
            imp_col = pick_column(list(df.columns), imp_cands)
            snr_col = pick_column(list(df.columns), snr_cands) or None
            eff_col = pick_column(list(df.columns), eff_cands) or None
            ent_col = pick_column(list(df.columns), ent_cands) or None
            if imp_col is None and eff_col is None:
                continue
            for _, row in df.iterrows():
                feature = str(row[feat_col])
                if feature.strip().lower() in ("bias", "intercept") or not feature.strip():
                    continue
                importance = None
                if imp_col is not None and pd.notna(row.get(imp_col)):
                    importance = float(row[imp_col])
                effect = None
                if eff_col is not None and pd.notna(row.get(eff_col)):
                    effect = float(row[eff_col])
                if family == "linear" and effect is not None:
                    importance = abs(effect) if importance is None else importance
                if importance is None and effect is None:
                    continue
                snr = float(row[snr_col]) if snr_col and pd.notna(row.get(snr_col)) else None
                entropy = (float(row[ent_col])
                           if ent_col and pd.notna(row.get(ent_col)) else None)
                position, channel = _parse_position_channel(feature)
                records.append({
                    "feature": feature,
                    "channel": channel,
                    "position": position,
                    "model": family,
                    "architecture": architecture,
                    "split_type": split_type,
                    "cell_line": cell_line,
                    "environment": environment,
                    "method": method,
                    "importance": importance,
                    "snr": snr,
                    "effect": effect,
                    "attention_entropy": entropy,
                    "source_file": csv_file.name,
                })
    if not records:
        return pd.DataFrame(columns=CANONICAL_COLUMNS)
    out = pd.DataFrame(records, columns=CANONICAL_COLUMNS)
    for col in ("importance", "snr", "effect", "attention_entropy"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
