# core/xai/xai_importance.py
"""
XAI 特征重要性输出【白名单】与导出清洗 (学术红线)
=================================================
1. 仅 Linear Regression 具备经典参数假设检验前提 (t / p / BH-FDR)。
2. XGBoost / MLP / CNN / Transformer 一律采用信噪比 SNR (mean/std) 与离散度/熵指标，
   严禁输出或伪造 t_stat / p_value / FDR(p_adj) / Significance 等统计检验字段。
3. 各模型训练导出前的特征重要性表必须严格仅含以下白名单列 (标识列除外)。

线性模型保存 CSV 列的规范 (1-based 内容见下)：
    Linear Regression : Linear_Coefficient, SE, t_stat, p_value, FDR
    XGBoost           : XGB_Gain, XGB_Weight, XGB_Cover, TreeSHAP, SHAP_SNR
    MLP               : MLP_IG, IG_SNR
    CNN               : CNN_IG, CNN_ISM, ISM_SNR
    Transformer       : Transformer_Attention, Attention_Entropy, Attention_SNR
标识列: Feature (+ Position/Channel 仅对 3D 空间模型内部保留可选, 亦非指标字段)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 各模型白名单 (指标字段, 标识列不在此列)
# ---------------------------------------------------------------------------
WHITELIST_COLUMNS: Dict[str, List[str]] = {
    "linear": ["Linear_Coefficient", "SE", "t_stat", "p_value", "FDR"],
    "linear_regression": ["Linear_Coefficient", "SE", "t_stat", "p_value", "FDR"],
    "xgboost": ["XGB_Gain", "XGB_Weight", "XGB_Cover", "TreeSHAP", "SHAP_SNR"],
    "mlp": ["MLP_IG", "IG_SNR"],
    "cnn": ["CNN_IG", "CNN_ISM", "ISM_SNR"],
    "transformer": ["Transformer_Attention", "Attention_Entropy", "Attention_SNR"],
}

# 行/位置标识列 (不属于指标)
_ID_COLUMNS = {"Feature", "Position", "Channel", "feature", "position", "channel"}

# 仅线性回归允许的经典统计检验字段; 非线性出现即触发剔除
_FORBIDDEN_STATS = {
    "t_stat", "t_value", "p_value", "p_val", "p_adj", "p_adj_fdr", "fdr",
    "permutation_p_val", "permutation_p_adj_fdr", "permutation_delta_r2", "q_value",
}

# 旧命名 -> 白名单规范名 (向后兼容读取旧批次结果)
LEGACY_RENAME_MAP: Dict[str, Dict[str, str]] = {
    "linear": {
        "weight": "Linear_Coefficient", "coefficient": "Linear_Coefficient",
        "coef": "Linear_Coefficient",
        "std_error": "SE", "std_error": "SE", "se": "SE",
        "p_adj_fdr": "FDR", "p_adj": "FDR", "q": "FDR",
    },
    "xgboost": {
        "importance_gain": "XGB_Gain", "gain": "XGB_Gain",
        "importance_weight": "XGB_Weight",
        "importance_cover": "XGB_Cover",
        "shap_mean": "TreeSHAP", "mean_abs_shap": "TreeSHAP",
    },
    "mlp": {
        "ig_mean": "MLP_IG", "mean_ig": "MLP_IG",
    },
    "cnn": {
        "ism_mean_delta": "CNN_ISM", "ism_mean": "CNN_ISM",
        "ig_mean": "CNN_IG", "cnn_ig": "CNN_IG",
    },
    "transformer": {
        "attn_weight": "Transformer_Attention", "attention_weight": "Transformer_Attention",
        "attn_entropy": "Attention_Entropy", "attention_entropy": "Attention_Entropy",
    },
}


def export_feature_table(
    model_key: str,
    df: pd.DataFrame,
    rename: Optional[Dict[str, str]] = None,
    drop: Optional[List[str]] = None,
    origin: str = "",
) -> pd.DataFrame:
    """
    导出前强制白名单清洗 (供 src 各模型 save 前调用):
      1. 按 rename 将内部列名映射为白名单规范名;
      2. 仅保留 白名单指标列 + 标识列 (Feature/Position/Channel);
      3. 剔除其余一切 (含残留 p/FDR/significance 或内部中间列)。

    返回列顺序为 [标识列...] + 白名单列。
    """
    model_key = str(model_key).lower()
    if model_key not in WHITELIST_COLUMNS:
        # 未知模型不做白名单强制, 原样返回 (不阻断运行)
        return df.copy()

    work = df.copy()
    rename_map: Dict[str, str] = dict(rename or {})
    rename_map.update({k: v for k, v in LEGACY_RENAME_MAP.get(model_key, {}).items()})
    # 大小写不敏感应用重命名 (列名可能为 Weight / Importance_gain / p_adj_fdr ...)
    lower_to_col = {str(c).lower(): c for c in work.columns}
    effective: Dict[str, str] = {}
    for key, val in rename_map.items():
        actual = lower_to_col.get(str(key).lower())
        if actual is not None and actual not in effective:
            effective[actual] = val
    if effective:
        work = work.rename(columns=effective)

    whitelist = WHITELIST_COLUMNS[model_key]
    keep_ids = [c for c in work.columns if c in _ID_COLUMNS]
    keep_metrics = [c for c in whitelist if c in work.columns]
    out_cols = keep_ids + keep_metrics
    cleaned = work.loc[:, out_cols]

    # 防御性断言: 非线性模型禁止残留经典统计字段
    if model_key not in ("linear", "linear_regression"):
        forbidden_left = [c for c in cleaned.columns if str(c).lower() in _FORBIDDEN_STATS]
        if forbidden_left:
            raise ValueError(
                f"[xai_importance] 非法统计字段残留 {forbidden_left} (模型 {model_key}, 来源 {origin or 'unknown'})"
            )
    return cleaned

