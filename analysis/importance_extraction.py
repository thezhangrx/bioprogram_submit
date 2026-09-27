# Submit/analysis/importance_extraction.py
"""
CRISPR-Cas9 Multi-Model Feature Importance & Bioinformatics Robustness Extraction Engine
=======================================================================================
输出重定向：保存在 results/[batch_name]/summary/feature_importance/ 下
新增：key_regulatory_biomarkers.csv 生成功能已从 results.py 移植至此，
     与 5 大模型的生信稳健性 .md 报告生成在同一目录。
"""

from __future__ import annotations


# --- 项目根引导: 保证从任意工作目录运行/被导入都能解析 core、analysis、workflows ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.common.paths import RESULTS_BATCHES  # 唯一路径权威


SCHEMA_CHANNELS = ["a", "c", "g", "t", "ctcf", "dnase", "h3k4me3", "rrbs"]
EPI_CHANNELS = ["ctcf", "dnase", "h3k4me3", "rrbs"]

CHANNEL_ORDER = {
    "a": 0, "c": 1, "g": 2, "t": 3,
    "ctcf": 4, "dnase": 5, "h3k4me3": 6, "rrbs": 7,
    "unknown": 99
}


def parse_info_file(info_path: str) -> Dict:
    info = {}
    if not os.path.exists(info_path):
        return info
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                info[key.strip().lower()] = val.strip().lower()
    return info


def normalize_cell_line(split_type: str, raw_cell_line: str) -> str:
    st = str(split_type).strip().lower()
    if st == 'mixed':
        return 'none'
    if pd.isna(raw_cell_line) or not str(raw_cell_line).strip() or str(raw_cell_line).strip().lower() in ['none', 'unknown']:
        return 'unknown'
    return str(raw_cell_line).strip().lower()


def get_active_channels(environment_str: str) -> set:
    env = str(environment_str).strip().lower()
    # 序列通道 = schema 中的碱基通道 (当前 8 通道: A/C/G/T); 旧版硬编码 {a,g,c} 丢弃了 T
    active = {ch for ch in SCHEMA_CHANNELS if ch in ("a", "c", "g", "t")}
    if env in ['all', 'all_features', 'full']:
        return set(SCHEMA_CHANNELS)
    for epi in EPI_CHANNELS:
        if epi in env:
            active.add(epi)
    return active


def identify_feature_channel(feature_name: str) -> str:
    fn = str(feature_name).strip().lower()
    if 'h3k4me3' in fn or 'h3k4' in fn: return 'h3k4me3'
    if 'ctcf' in fn: return 'ctcf'
    if 'dnase' in fn: return 'dnase'
    if 'rrbs' in fn: return 'rrbs'
    if re.search(r'(^|[^a-z0-9])a([^a-z0-9]|$)', fn) or re.search(r'pos\d*_?a', fn) or re.search(r'a_?pos\d*', fn): return 'a'
    if re.search(r'(^|[^a-z0-9])g([^a-z0-9]|$)', fn) or re.search(r'pos\d*_?g', fn) or re.search(r'g_?pos\d*', fn): return 'g'
    if re.search(r'(^|[^a-z0-9])c([^a-z0-9]|$)', fn) or re.search(r'pos\d*_?c', fn) or re.search(r'c_?pos\d*', fn): return 'c'
    if re.search(r'(^|[^a-z0-9])t([^a-z0-9]|$)', fn) or re.search(r'pos\d*_?t', fn) or re.search(r't_?pos\d*', fn): return 't'
    if fn.startswith('feat_') or fn.isdigit():
        try:
            idx = int(fn.replace('feat_', ''))
            return SCHEMA_CHANNELS[idx % len(SCHEMA_CHANNELS)]
        except Exception: pass
    return 'unknown'


def extract_feature_position(feature_name: str) -> int:
    fn = str(feature_name).strip().lower()
    m = re.search(r'pos_?(\d+)', fn)
    if m: return int(m.group(1))
    if fn.startswith('feat_') or fn.isdigit():
        try:
            idx = int(fn.replace('feat_', ''))
            return idx // len(SCHEMA_CHANNELS)
        except Exception: pass
    m2 = re.search(r'_(\d+)$', fn)
    if m2: return int(m2.group(1))
    return 999


def is_feature_valid_for_env(feature_name: str, environment_str: str) -> bool:
    active_channels = get_active_channels(environment_str)
    ch = identify_feature_channel(feature_name)
    if ch == 'unknown': return True
    return ch in active_channels


def get_snr_significance_code(snr_val: float) -> str:
    if pd.isna(snr_val) or snr_val is None:
        return ""
    if snr_val >= 2.5:
        return "***"
    elif snr_val >= 1.8:
        return "**"
    elif snr_val >= 1.2:
        return "*"
    elif snr_val >= 0.8:
        return "."
    return ""


# ============================================================================
# XAI 学术红线：各模型特征重要性导出白名单 + 防御性清洗
# ============================================================================
#   仅 Linear Regression 具备经典参数检验前提 (t / p / BH-FDR)。
#   非线性模型 (XGBoost / MLP / CNN / Transformer) 一律使用 SNR(mean/std) 与
#   熵/离散度指标, 严禁出现 t_stat / p_value / FDR(p_adj) 等统计列。
# ----------------------------------------------------------------------------

XAI_WHITELIST = {
    "linear": ["Linear_Coefficient", "SE", "t_stat", "p_value", "FDR"],
    "xgboost": ["XGB_Gain", "XGB_Weight", "XGB_Cover", "TreeSHAP", "SHAP_SNR"],
    "mlp": ["MLP_IG", "IG_SNR"],
    "cnn": ["CNN_IG", "CNN_ISM", "ISM_SNR"],
    "transformer": ["Transformer_Attention", "Attention_Entropy", "Attention_SNR"],
}

# 旧批次命名 (大小写不敏感) -> 白名单规范名 (向后兼容读取)
XAI_ALIAS = {
    "linear": {
        "weight": "Linear_Coefficient", "coefficient": "Linear_Coefficient",
        "coef": "Linear_Coefficient", "linear_coefficient": "Linear_Coefficient",
        "std_error": "SE", "se": "SE", "std_err": "SE",
        "t_stat": "t_stat", "t_value": "t_stat",
        "p_value": "p_value", "p_val": "p_value",
        "p_adj_fdr": "FDR", "p_adj": "FDR", "fdr": "FDR", "q_value": "FDR",
    },
    "xgboost": {
        "importance_gain": "XGB_Gain", "gain": "XGB_Gain", "xgb_gain": "XGB_Gain",
        "importance_weight": "XGB_Weight", "xgb_weight": "XGB_Weight",
        "importance_cover": "XGB_Cover", "xgb_cover": "XGB_Cover",
        "shap_mean": "TreeSHAP", "treeshap": "TreeSHAP", "mean_abs_shap": "TreeSHAP",
        "shap_snr": "SHAP_SNR",
    },
    "mlp": {
        "ig_mean": "MLP_IG", "mlp_ig": "MLP_IG", "mean_ig": "MLP_IG",
        "ig_snr": "IG_SNR",
    },
    "cnn": {
        "ism_mean_delta": "CNN_ISM", "ism_mean": "CNN_ISM", "cnn_ism": "CNN_ISM",
        "ism_snr": "ISM_SNR",
        "ig_mean": "CNN_IG", "cnn_ig": "CNN_IG",
    },
    "transformer": {
        "attn_weight": "Transformer_Attention", "attention_weight": "Transformer_Attention",
        "transformer_attention": "Transformer_Attention",
        "attn_entropy": "Attention_Entropy", "attention_entropy": "Attention_Entropy",
        "attn_snr": "Attention_SNR", "attention_snr": "Attention_SNR",
    },
}

# 经典参数统计字段 —— 非线性模型一旦检出立即剔除并告警
FORBIDDEN_PARAMETRIC_COLS = {
    "t_stat", "t_value", "p_value", "p_val", "p_adj", "p_adj_fdr", "fdr",
    "q_value", "permutation_p_val", "permutation_p_adj_fdr", "permutation_delta_r2",
}

_IDENTIFIER_COLS = {"Feature", "Position", "Channel", "feature", "position", "channel"}

XAI_SIG_NOTE = {
    "linear": "Significance codes based on BH-FDR: `***` p<0.001, `**` p<0.01, `*` p<0.05, `.` p<0.10",
    "xgboost": "Robustness codes (based on SHAP_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8",
    "mlp": "Robustness codes (based on IG_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8",
    "cnn": "Robustness codes (based on ISM_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8",
    "transformer": "Robustness codes (based on Attention_SNR): `***` SNR>=2.5, `**` >=1.8, `*` >=1.2, `.` >=0.8",
}

XAI_SNR_COL = {
    "xgboost": "SHAP_SNR",
    "mlp": "IG_SNR",
    "cnn": "ISM_SNR",
    "transformer": "Attention_SNR",
}

_MODEL_TOKEN = {
    "linear": ["linear"],
    "xgboost": ["xgb", "xgboost"],
    "mlp": ["mlp"],
    "cnn": ["cnn"],
    "transformer": ["trans", "transformer"],
}


def _sanitize_importance_table(model_key: str, df: pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    字段合规清洗 (防御性断言 + Warning):
      1. 将 legacy/旧列名映射为白名单规范名;
      2. 仅保留 白名单指标列 + 标识列(Feature/Position/Channel);
      3. 非线性模型检出 t_stat/p_value/FDR/p_adj 等 → 自动 drop + 控制台 Warning。
    """
    if df is None or df.empty:
        return df
    work = df.copy()
    alias = XAI_ALIAS.get(model_key, {})
    rename = {col: alias[str(col).lower().strip()] for col in work.columns
              if str(col).lower().strip() in alias}
    if rename:
        work = work.rename(columns=rename)

    whitelist = set(XAI_WHITELIST.get(model_key, []))
    is_linear = model_key in ("linear", "linear_regression")

    keep = [c for c in work.columns if c in _IDENTIFIER_COLS or c in whitelist]
    if not is_linear:
        forbidden_found = [c for c in work.columns if str(c).lower() in FORBIDDEN_PARAMETRIC_COLS]
        if forbidden_found:
            print(f"[Warning][xai-whitelist] 非线性模型 {model_key} (来源: {source or 'unknown'}) "
                  f"检出非法经典统计列 {forbidden_found}，已自动剔除 (学术红线)。")
        keep = [c for c in keep if str(c).lower() not in FORBIDDEN_PARAMETRIC_COLS]

    out = work.loc[:, keep] if keep else work
    return out


def _sig_for_fdr(value: float) -> str:
    if pd.isna(value):
        return ""
    if value < 0.001: return "***"
    if value < 0.01: return "**"
    if value < 0.05: return "*"
    if value < 0.10: return "."
    return ""


def _derive_sig(model_key: str, row: pd.Series) -> str:
    """非线性 -> SNR 星级; 线性 -> FDR 星级。"""
    if model_key in ("linear", "linear_regression"):
        try:
            return _sig_for_fdr(float(row.get("FDR", np.nan)))
        except (TypeError, ValueError):
            return ""
    snr_col = XAI_SNR_COL.get(model_key)
    if not snr_col or snr_col not in row.index:
        return ""
    try:
        return get_snr_significance_code(float(row[snr_col]))
    except (TypeError, ValueError):
        return ""


def _write_importance_md(model_key: str, records_df: pd.DataFrame, out_md: str,
                         title: str, metric_columns: List[str]) -> bool:
    """把已带上下文列 (split_type/environment/cell_line/feature) 的规范表写为 MD。"""
    if records_df is None or records_df.empty:
        return False

    df = records_df.copy()
    feat_col = "Feature" if "Feature" in df.columns else "feature"
    if feat_col not in df.columns:
        return False

    # 追加派生显著性 (显示用; 不写入模型白名单之外内容)
    if "sig" not in df.columns:
        df["sig"] = df.apply(lambda r: _derive_sig(model_key, r), axis=1)

    metrics = [c for c in metric_columns if c in df.columns]
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(f"> {XAI_SIG_NOTE.get(model_key, '')}\n\n")
        f.write("| split_type | environment | cell_line | feature | "
                + " | ".join(metrics) + " | sig |\n")
        f.write("| :--- | :--- | :--- | :--- | " + " | ".join([":---"] * len(metrics)) + " | :--- |\n")
        for _, r in df.iterrows():
            cells = [str(r["split_type"]), str(r["environment"]), str(r["cell_line"]),
                     str(r[feat_col])]
            for m in metrics:
                v = r.get(m, np.nan)
                cells.append(f"{float(v):.4f}" if pd.notna(v) else "N/A")
            cells.append(str(r.get("sig", "")))
            f.write("| " + " | ".join(cells) + " |\n")
    return True


def extract_linear_coefficients(target_batch_dir: str, feature_importance_dir: str):
    """Linear Regression —— 经典参数检验 (白名单: Linear_Coefficient/SE/t_stat/p_value/FDR)。"""
    records: List[Dict] = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue
        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}
        model = str(info.get('model', '')).lower()
        if not model and 'linear' not in exp_dir.lower():
            continue
        if 'linear' not in model and 'linear' not in exp_dir.lower():
            continue

        split_type = info.get('split_type', 'unknown').lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', 'unknown'))
        environment = info.get('environment', info.get('combination', 'unknown'))

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table("linear", df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns or "Linear_Coefficient" not in canon.columns:
                continue

            for _, row in canon.iterrows():
                feat = str(row[feat_col])
                if not is_feature_valid_for_env(feat, environment):
                    continue
                records.append({
                    "split_type": split_type,
                    "environment": environment,
                    "cell_line": cell_line,
                    "Feature": feat,
                    "position": extract_feature_position(feat),
                    "channel_rank": CHANNEL_ORDER.get(identify_feature_channel(feat), 99),
                    **{m: (row.get(m) if m in canon.columns else np.nan)
                       for m in XAI_WHITELIST["linear"]},
                })
        except Exception:
            continue

    if not records:
        return
    coef_df = pd.DataFrame(records).sort_values(
        by=["split_type", "environment", "cell_line", "position", "channel_rank"]
    )
    ok = _write_importance_md(
        "linear", coef_df, os.path.join(feature_importance_dir, "linear_coefficiency.md"),
        "Linear Regression Coefficients & Parameter Inference (Whitelist)",
        XAI_WHITELIST["linear"],
    )
    if ok:
        print(f"  [+] Saved Linear Coefficiency (whitelist) -> {os.path.join(feature_importance_dir, 'linear_coefficiency.md')}")


def extract_xgboost_importance(target_batch_dir: str, feature_importance_dir: str):
    """XGBoost —— XAI 白名单 (XGB_Gain/XGB_Weight/XGB_Cover/TreeSHAP/SHAP_SNR), 无 p/FDR。"""
    records: List[Dict] = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue
        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}
        model = str(info.get('model', '')).lower()
        if not model and 'xgb' not in exp_dir.lower():
            continue
        if 'xgb' not in model and 'xgboost' not in exp_dir.lower():
            continue

        split_type = info.get('split_type', 'unknown').lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', 'unknown'))
        environment = info.get('environment', info.get('combination', 'unknown'))

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table("xgboost", df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns or "XGB_Gain" not in canon.columns:
                continue

            for _, row in canon.iterrows():
                feat = str(row[feat_col])
                if not is_feature_valid_for_env(feat, environment):
                    continue
                records.append({
                    "split_type": split_type,
                    "environment": environment,
                    "cell_line": cell_line,
                    "Feature": feat,
                    "position": extract_feature_position(feat),
                    "channel_rank": CHANNEL_ORDER.get(identify_feature_channel(feat), 99),
                    **{m: (row.get(m) if m in canon.columns else np.nan)
                       for m in XAI_WHITELIST["xgboost"]},
                })
        except Exception:
            continue

    if not records:
        return
    xgb_df = pd.DataFrame(records).sort_values(
        by=["split_type", "environment", "cell_line", "position", "channel_rank"]
    )
    ok = _write_importance_md(
        "xgboost", xgb_df, os.path.join(feature_importance_dir, "xgboost_importance.md"),
        "XGBoost Feature Importance & TreeSHAP Robustness (Whitelist)",
        XAI_WHITELIST["xgboost"],
    )
    if ok:
        print(f"  [+] Saved XGBoost Importance (whitelist) -> {os.path.join(feature_importance_dir, 'xgboost_importance.md')}")


def extract_mlp_importance(target_batch_dir: str, feature_importance_dir: str):
    """MLP —— XAI 白名单 (MLP_IG/IG_SNR), 无 p/FDR。"""
    records: List[Dict] = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue
        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}
        model = str(info.get('model', '')).lower()
        if not model and 'mlp' not in exp_dir.lower():
            continue
        if 'mlp' not in model and 'mlp' not in exp_dir.lower():
            continue

        split_type = info.get('split_type', 'unknown').lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', 'unknown'))
        environment = info.get('environment', info.get('combination', 'unknown'))

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table("mlp", df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns or "MLP_IG" not in canon.columns:
                continue

            for _, row in canon.iterrows():
                feat = str(row[feat_col])
                if not is_feature_valid_for_env(feat, environment):
                    continue
                records.append({
                    "split_type": split_type,
                    "environment": environment,
                    "cell_line": cell_line,
                    "Feature": feat,
                    "position": extract_feature_position(feat),
                    "channel_rank": CHANNEL_ORDER.get(identify_feature_channel(feat), 99),
                    **{m: (row.get(m) if m in canon.columns else np.nan)
                       for m in XAI_WHITELIST["mlp"]},
                })
        except Exception:
            continue

    if not records:
        return
    mlp_df = pd.DataFrame(records).sort_values(
        by=["split_type", "environment", "cell_line", "position", "channel_rank"]
    )
    ok = _write_importance_md(
        "mlp", mlp_df, os.path.join(feature_importance_dir, "mlp_importance.md"),
        "MLP Integrated Gradients Robustness (Whitelist)",
        XAI_WHITELIST["mlp"],
    )
    if ok:
        print(f"  [+] Saved MLP Importance (whitelist) -> {os.path.join(feature_importance_dir, 'mlp_importance.md')}")


def extract_cnn_importance(target_batch_dir: str, feature_importance_dir: str):
    """
    CNN —— XAI 白名单 (CNN_IG/CNN_ISM/ISM_SNR), 无 p/FDR。
    按序列卷积核拆分输出 cnn33/cnn53/cnn73_importance.md。
    """
    records: List[Dict] = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue
        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}
        model = str(info.get('model', '')).lower()
        if not model and 'cnn' not in exp_dir.lower():
            continue
        if 'cnn' not in model and 'cnn' not in exp_dir.lower():
            continue

        try:
            seq_k = int(float(info.get('sequence_kernel', info.get('sequence_kernel_size', '3'))))
        except Exception:
            seq_k = 3
        if seq_k not in (3, 5, 7):
            seq_k = 3

        split_type = info.get('split_type', 'unknown').lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', 'unknown'))
        environment = info.get('environment', info.get('combination', 'unknown'))

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table("cnn", df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns or "CNN_ISM" not in canon.columns:
                continue

            for _, row in canon.iterrows():
                feat = str(row[feat_col])
                if not is_feature_valid_for_env(feat, environment):
                    continue
                records.append({
                    "kernel": seq_k,
                    "split_type": split_type,
                    "environment": environment,
                    "cell_line": cell_line,
                    "Feature": feat,
                    "position": extract_feature_position(feat),
                    "channel_rank": CHANNEL_ORDER.get(identify_feature_channel(feat), 99),
                    **{m: (row.get(m) if m in canon.columns else np.nan)
                       for m in XAI_WHITELIST["cnn"]},
                })
        except Exception:
            continue

    if not records:
        return
    cnn_df = pd.DataFrame(records)
    for kernel in sorted(cnn_df["kernel"].unique()):
        k = int(kernel)
        k_df = cnn_df[cnn_df["kernel"] == k].copy()
        tag = f"cnn{k}3"
        ok = _write_importance_md(
            "cnn", k_df, os.path.join(feature_importance_dir, f"{tag}_importance.md"),
            f"CNN({k}|3) In-Silico Mutagenesis & IG Attribution (Whitelist)",
            XAI_WHITELIST["cnn"],
        )
        if ok:
            print(f"  [+] Saved {tag.upper()} Importance (whitelist) -> "
                  f"{os.path.join(feature_importance_dir, f'{tag}_importance.md')}")


def extract_transformer_importance(target_batch_dir: str, feature_importance_dir: str):
    """Transformer —— XAI 白名单 (Transformer_Attention/Attention_Entropy/Attention_SNR), 无 p/FDR。"""
    records: List[Dict] = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue
        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}
        model = str(info.get('model', '')).lower()
        if not model and 'trans' not in exp_dir.lower():
            continue
        if 'trans' not in model and 'transformer' not in exp_dir.lower():
            continue

        split_type = info.get('split_type', 'unknown').lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', 'unknown'))
        environment = info.get('environment', info.get('combination', 'unknown'))

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table("transformer", df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns or "Transformer_Attention" not in canon.columns:
                continue

            for _, row in canon.iterrows():
                feat = str(row[feat_col])
                if not is_feature_valid_for_env(feat, environment):
                    continue
                records.append({
                    "split_type": split_type,
                    "environment": environment,
                    "cell_line": cell_line,
                    "Feature": feat,
                    "position": extract_feature_position(feat),
                    "channel_rank": CHANNEL_ORDER.get(identify_feature_channel(feat), 99),
                    **{m: (row.get(m) if m in canon.columns else np.nan)
                       for m in XAI_WHITELIST["transformer"]},
                })
        except Exception:
            continue

    if not records:
        return
    trans_df = pd.DataFrame(records).sort_values(
        by=["split_type", "environment", "cell_line", "position", "channel_rank"]
    )
    ok = _write_importance_md(
        "transformer", trans_df, os.path.join(feature_importance_dir, "transformer_importance.md"),
        "Transformer Attention Weights & Shannon Entropy (Whitelist)",
        XAI_WHITELIST["transformer"],
    )
    if ok:
        print(f"  [+] Saved Transformer Importance (whitelist) -> "
              f"{os.path.join(feature_importance_dir, 'transformer_importance.md')}")


SIG_ORDER_MAP = {
    "***": 3,
    "**": 2,
    "*": 1,
    ".": 0.5,
    "": 0,
    "none": 0,
    "nan": 0
}


def get_canonical_feature_id(feat_name: str) -> Tuple[int, str, str]:
    """
    【核心归一化函数】将各种格式的特征名统一归一化为标准的 (1..23 位点, 通道名, 标准显示名)
      - pos18_C   (1-based) -> (18, 'C', 'pos18_C')
      - C_pos_17  (0-based) -> (18, 'C', 'pos18_C')
      - pos_17_c  (0-based) -> (18, 'C', 'pos18_C')
      - Dnase_pos_13        -> (14, 'Dnase', 'pos14_Dnase')
      - pos14_Dnase         -> (14, 'Dnase', 'pos14_Dnase')
    """
    fn = str(feat_name).strip()
    fn_lower = fn.lower()

    if 'bias' in fn_lower or 'intercept' in fn_lower:
        return -999, 'Bias', 'Bias'

    # 1. 识别通道名称
    if 'h3k4me3' in fn_lower or 'h3k4' in fn_lower: ch = 'H3K4me3'
    elif 'ctcf' in fn_lower: ch = 'CTCF'
    elif 'dnase' in fn_lower: ch = 'Dnase'
    elif 'rrbs' in fn_lower: ch = 'RRBS'
    elif 'a' in fn_lower: ch = 'A'
    elif 'g' in fn_lower: ch = 'G'
    elif 'c' in fn_lower: ch = 'C'
    elif 't' in fn_lower: ch = 'T'
    else: ch = 'Unknown'

    # 2. 识别位点索引并做 0/1-based 归一化
    # 模式 A: pos18_C (1-based, 明确从 1 到 23)
    m1 = re.match(r'^pos(\d+)_', fn_lower)
    if m1:
        pos_1_based = int(m1.group(1))
    else:
        # 模式 B: C_pos_17 或 pos_17_C (0-based, 从 0 到 22)
        m2 = re.search(r'pos_(\d+)', fn_lower)
        if m2:
            raw_idx = int(m2.group(1))
            pos_1_based = raw_idx + 1 if raw_idx <= 22 else raw_idx
        else:
            # 模式 C: 尾部数字
            m3 = re.search(r'_(\d+)$', fn_lower)
            if m3:
                raw_idx = int(m3.group(1))
                pos_1_based = raw_idx + 1 if raw_idx <= 22 else raw_idx
            else:
                pos_1_based = -1

    if pos_1_based < 1 or pos_1_based > 23:
        canonical_name = fn
    else:
        canonical_name = f"pos{pos_1_based}_{ch}"

    return pos_1_based, ch, canonical_name


def collect_all_model_features(target_batch_dir: str) -> pd.DataFrame:
    """
    汇总 5 大模型全部实验的特征重要性 CSV -> 关键调控特征库数据源。
    读取时强制白名单清洗: 仅采信各模型白名单指标 (Linear 允许 t/p/FDR; 非线性仅 SNR 系指标)。
    """
    records = []
    csv_candidates = glob.glob(os.path.join(target_batch_dir, "**", "*.csv"), recursive=True)

    for c_file in csv_candidates:
        fname = os.path.basename(c_file).lower()
        if 'pred' in fname or 'metric' in fname or 'history' in fname or 'summary' in fname:
            continue

        exp_dir = os.path.dirname(c_file)
        info_files = glob.glob(os.path.join(exp_dir, "*info*.txt"))
        info = parse_info_file(info_files[0]) if info_files else {}

        model = str(info.get('model', '')).lower()
        if not model:
            if 'linear' in exp_dir.lower(): model = 'linear'
            elif 'xgb' in exp_dir.lower(): model = 'xgboost'
            elif 'mlp' in exp_dir.lower(): model = 'mlp'
            elif 'cnn' in exp_dir.lower(): model = 'cnn'
            elif 'trans' in exp_dir.lower(): model = 'transformer'
            else: continue

        # 归一化模型关键字 (CNN 按序列卷积核拆分为 cnn33 / cnn53 / cnn73)
        if "xgb" in model:
            base_key, model_key = "xgboost", "xgboost"
        elif "mlp" in model:
            base_key, model_key = "mlp", "mlp"
        elif "cnn" in model:
            try:
                seq_k = int(float(info.get('sequence_kernel', info.get('sequence_kernel_size', '3'))))
            except Exception:
                seq_k = 3
            if seq_k not in (3, 5, 7):
                seq_k = 3
            base_key, model_key = "cnn", f"cnn{seq_k}3"
        elif "trans" in model:
            base_key, model_key = "transformer", "transformer"
        else:
            base_key, model_key = "linear", "linear"

        split_type = str(info.get('split_type', 'single')).lower()
        cell_line = normalize_cell_line(split_type, info.get('cell_line', info.get('held_out_cell_line', 'none')))
        environment = str(info.get('environment', info.get('combination', 'all'))).lower()

        try:
            df = pd.read_csv(c_file)
            canon = _sanitize_importance_table(base_key, df, source=os.path.basename(c_file))
            feat_col = "Feature" if "Feature" in canon.columns else "feature"
            if feat_col not in canon.columns:
                continue

            # 各模型白名单: 核心贡献指标 + 信噪比
            metric_map = {
                "linear": ("Linear_Coefficient", "t_stat"),
                "xgboost": ("XGB_Gain", "SHAP_SNR"),
                "mlp": ("MLP_IG", "IG_SNR"),
                "cnn": ("CNN_ISM", "ISM_SNR"),
                "transformer": ("Transformer_Attention", "Attention_SNR"),
            }
            contrib_col, snr_col = metric_map[base_key]

            for _, row in canon.iterrows():
                raw_feat = str(row[feat_col])
                pos_1_based, ch, canonical_name = get_canonical_feature_id(raw_feat)

                if ch == 'Bias' or pos_1_based < 1 or pos_1_based > 23:
                    continue

                contrib_val = float(row[contrib_col]) if contrib_col in canon.columns and pd.notna(row[contrib_col]) else 0.0
                snr_val = float(row[snr_col]) if snr_col in canon.columns and pd.notna(row[snr_col]) else np.nan

                # 显著性: 线性 -> FDR 星级; 非线性 -> 专属 SNR 星级 (学术红线)
                if base_key == "linear":
                    fdr_val = float(row["FDR"]) if "FDR" in canon.columns and pd.notna(row["FDR"]) else np.nan
                    sig = _sig_for_fdr(fdr_val)
                    p_val = float(row["p_value"]) if "p_value" in canon.columns and pd.notna(row["p_value"]) else np.nan
                else:
                    sig = get_snr_significance_code(snr_val)
                    p_val, fdr_val = np.nan, np.nan   # 非线性: 无 p / FDR

                sig_score = SIG_ORDER_MAP.get(sig, 0)
                records.append({
                    'split_type': split_type,
                    'cell_line': cell_line,
                    'environment': environment,
                    'model_key': model_key,
                    'model_raw': model_key,   # CNN 显示为 cnn33/cnn53/cnn73
                    'canonical_feature': canonical_name,  # 归一化后的特征名 (如 pos18_C)
                    'position': pos_1_based,
                    'channel': ch,
                    'significance': sig,
                    'sig_score': sig_score,
                    'contribution_val': contrib_val,
                    'abs_contrib': abs(contrib_val),
                    'snr': snr_val,
                    'p_val': p_val,
                    'fdr': fdr_val
                })
        except Exception:
            continue

    return pd.DataFrame(records)


def generate_key_regulatory_biomarkers(df_all_feats: pd.DataFrame, output_path: str):
    """
    汇总所有实验划分下 5 大模型识别出的关键特征 (原 results.py 功能)。
    严格按显著性星级 (*** > ** > * > .) 从高到低排序。
    """
    if df_all_feats.empty:
        # 这里曾写入一份写死的 2 行"演示"生物标志物表
        # （pos18_G/pos20_G，***，贡献值 0.2840/0.2410，SNR 4.12/3.95，FDR 1.2e-4/2.5e-4），
        # 它会作为 key_regulatory_biomarkers.csv 落盘并被下游当真。
        # 已移除，改为直接报错。
        raise ValueError(
            "没有可用的特征重要性记录，拒绝生成「关键调控特征库」。\n"
            "  df_all_feats 为空说明上游 importance_extraction 没有产出任何特征行：\n"
            "    请确认已跑完训练、且 *_feature_importance.csv 存在于 <batch>/<run>/ 下。\n"
            "  本函数不再写入写死的演示行。"
        )
    else:
        df_sig = df_all_feats[df_all_feats['sig_score'] > 0].copy()

        non_mixed = df_sig[df_sig['split_type'] != 'mixed'].copy()
        mixed = df_sig[df_sig['split_type'] == 'mixed'].copy()

        if not mixed.empty:
            group_cols = ['split_type', 'cell_line', 'environment', 'model_key', 'canonical_feature']
            mixed_avg = mixed.groupby(group_cols, as_index=False).agg({
                'sig_score': 'mean',
                'contribution_val': 'mean',
                'abs_contrib': 'mean',
                'snr': 'mean',
                'p_val': 'mean',
                'fdr': 'mean'
            })
            def score_to_sym(score):
                if score >= 2.5: return '***'
                elif score >= 1.8: return '**'
                elif score >= 1.0: return '*'
                elif score >= 0.5: return '.'
                return ''
            mixed_avg['significance'] = mixed_avg['sig_score'].apply(score_to_sym)
            mixed_avg['model_raw'] = mixed_avg['model_key']
            combined = pd.concat([non_mixed, mixed_avg], ignore_index=True)
        else:
            combined = non_mixed

        combined.sort_values(
            by=['split_type', 'cell_line', 'environment', 'model_key', 'sig_score', 'abs_contrib'],
            ascending=[True, True, True, True, False, False],
            inplace=True
        )

        # 输出表头一律用英文（2026-09-26 由中文改为英文）：
        #   * 内部列名 split_type / cell_line / environment / significance / snr / fdr 已经是英文，
        #     只需把这三个内部名改成对外的英文名；
        #   * 下游 `attribution_plots.SignificanceGate` 与 `rule_discovery` 的列解析
        #     第一候选就是 model / feature / snr / fdr，改成英文后**无需改它们**；
        #   * 旧的中文表头（`模型(model)` 等）仍保留在两侧的别名表里作为向后兼容回退。
        # 列语义见 README 的 R2「train_data/ 与 feature_importance/」一节。
        combined.rename(columns={
            'model_raw': 'model',
            'canonical_feature': 'feature',
            'contribution_val': 'contribution'
        }, inplace=True)

        cols_export = [
            'split_type', 'cell_line', 'environment', 'model',
            'feature', 'significance', 'contribution', 'snr', 'fdr'
        ]
        df_sig = combined[cols_export]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_sig.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [+] 关键调控特征库 (key_regulatory_biomarkers.csv) 已生成 -> {output_path} (共保留 {len(df_sig)} 条显著特征)")


def process_batch(batch_dir: str):
    # 交付结构把 runs 与 summary 拆成同级子树：扫描用 runs 根，落盘用 summary 根。
    from analysis.data.loaders import resolve_runs_root, resolve_summary_root

    runs_root = str(resolve_runs_root(batch_dir))
    feature_importance_dir = os.path.join(str(resolve_summary_root(batch_dir)), "feature_importance")
    os.makedirs(feature_importance_dir, exist_ok=True)
    print(f"\n[*] Processing Feature Importances to: {feature_importance_dir}")

    extract_linear_coefficients(runs_root, feature_importance_dir)
    extract_xgboost_importance(runs_root, feature_importance_dir)
    extract_mlp_importance(runs_root, feature_importance_dir)
    extract_cnn_importance(runs_root, feature_importance_dir)
    extract_transformer_importance(runs_root, feature_importance_dir)

    # 关键调控特征库 (原 results.py 功能，已移植至此并与 5 大模型 .md 报告同目录输出)
    df_all_feats = collect_all_model_features(runs_root)
    biomarkers_path = os.path.join(feature_importance_dir, "key_regulatory_biomarkers.csv")
    generate_key_regulatory_biomarkers(df_all_feats, biomarkers_path)

    print(f"[✓] All 5 models' bioinformatics importance + key_regulatory_biomarkers.csv -> {feature_importance_dir}")


def _looks_like_batch(d: Path) -> bool:
    """批次目录 = 其子项中有 run 目录（single_/all_/mixed_ 前缀）。"""
    if not d.is_dir():
        return False
    try:
        return any(c.is_dir() and c.name.startswith(("single_", "all_", "mixed_"))
                   for c in d.iterdir())
    except OSError:
        return False


def _resolve_batch_paths(results_dir: str, batch_name: str, batch_dir: str,
                         latest: bool, all_batches: bool) -> List[Path]:
    """把 CLI 参数解析为**实际存在的**批次目录列表。

    批次目录的权威位置是 ``results/batches/<batch_name>``（见 core/common/paths.py）。
    为兼容旧命令行，传 ``--results_dir results`` 会被自动补成 ``results/batches``。
    """
    if batch_dir:
        p = Path(batch_dir).expanduser().resolve()
        if not p.is_dir():
            raise SystemExit(f"[Error] --batch_dir 不存在: {p}")
        return [p]

    root = Path(results_dir).expanduser().resolve()
    # 兼容旧约定：批次实际在 results/batches 下，而用户传了 results
    if not _looks_like_batch(root) and (root / "batches").is_dir():
        root = (root / "batches").resolve()

    if not root.is_dir():
        raise SystemExit(f"[Error] 批次根目录不存在: {root}")

    if batch_name:
        cand = root / batch_name
        if not cand.is_dir():
            available = sorted(d.name for d in root.iterdir() if d.is_dir())
            raise SystemExit(
                f"[Error] 批次目录不存在: {cand}\n"
                f"        批次根目录: {root}\n"
                f"        可用批次: {', '.join(available) if available else '（无）'}"
            )
        return [cand]

    # 未指定批次：results_dir 本身就是单个批次时直接用，否则枚举其下所有批次
    if _looks_like_batch(root):
        batches = [root]
    else:
        batches = sorted((d for d in root.iterdir() if _looks_like_batch(d)),
                         key=os.path.getmtime)
    if not batches:
        raise SystemExit(f"[Error] 在 {root} 下未找到任何批次目录")

    if all_batches:
        return batches
    return [batches[-1]]


def main():
    parser = argparse.ArgumentParser(description="Extract feature importances across all 5 models.")
    parser.add_argument('--results_dir', type=str, default=str(RESULTS_BATCHES),
                        help=f'批次根目录（默认 {RESULTS_BATCHES}）')
    parser.add_argument('--batch', '--batch_name', dest='batch_name', type=str, default='',
                        help='results/batches/ 下的批次名，例如 ultimate_run')
    parser.add_argument('--batch_dir', type=str, default='', help='批次目录的直接路径（覆盖 --batch）')
    parser.add_argument('--latest', action='store_true', help='处理最新的批次（缺省行为）')
    parser.add_argument('--all_batches', action='store_true', help='处理所有批次')
    args = parser.parse_args()

    batch_paths = _resolve_batch_paths(args.results_dir, args.batch_name, args.batch_dir,
                                       args.latest, args.all_batches)
    for batch_path in batch_paths:
        print(f"\n[*] 正在提取生信特征重要性，目标目录: {batch_path}")
        process_batch(str(batch_path))


if __name__ == '__main__':
    main()