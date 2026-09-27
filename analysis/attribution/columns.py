"""analysis.attribution.columns — 各模型重要性 CSV 列 -> 统一字段 的映射 (adapter)。

新训练输出为白名单列 (src/xai_importance); 旧批次存在历史列名,
此处按优先级取第一个存在列, 明确 adapter 语义 (来源列记入 source)。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# 每个 (family) 的输出行定义: (method, importance候选, snr候选, effect候选, entropy候选)
_ROW_SPECS: Dict[str, List[Tuple]] = {
    "linear": [
        ("linear_coefficient", ["Linear_Coefficient", "Weight", "Coefficient", "Coef"],
         ["t_stat", "t_value"], ["Linear_Coefficient", "Weight", "Coefficient", "Coef"], []),
    ],
    "xgboost": [
        ("xgboost_treeshap", ["TreeSHAP", "SHAP_mean"], ["SHAP_SNR"], [], []),
        ("xgboost_gain", ["XGB_Gain", "Importance_gain"], [], [], []),
    ],
    "mlp": [
        ("mlp_ig", ["MLP_IG", "IG_Mean"], ["IG_SNR"], [], []),
    ],
    "cnn": [
        ("cnn_ism", ["CNN_ISM", "ISM_Mean_Delta"], ["ISM_SNR"], [], []),
        ("cnn_ig", ["CNN_IG", "IG_Mean"], [], [], []),
    ],
    "transformer": [
        ("transformer_attention", ["Transformer_Attention", "Attn_Weight"],
         ["Attention_SNR", "Attn_SNR"], [],
         ["Attention_Entropy", "Attn_Entropy"]),
    ],
}

# 特征标识列候选 (小写)
FEATURE_ID_CANDIDATES = ["feature"]
POSITION_CANDIDATES = ["position", "pos"]
CHANNEL_CANDIDATES = ["channel", "ch"]


def family_from_model(model_raw: str) -> Optional[str]:
    m = str(model_raw).lower()
    if "linear" in m:
        return "linear"
    if "xgb" in m:
        return "xgboost"
    if "mlp" in m:
        return "mlp"
    if "cnn" in m:
        return "cnn"
    if "trans" in m:
        return "transformer"
    return None


def pick_column(columns: List[str], candidates: List[str]) -> Optional[str]:
    """按列名(忽略大小写)顺序挑选候选列, 返回实际列名。"""
    lower = {str(c).lower(): c for c in columns}
    for cand in candidates:
        if str(cand).lower() in lower:
            return lower[str(cand).lower()]
    return None
