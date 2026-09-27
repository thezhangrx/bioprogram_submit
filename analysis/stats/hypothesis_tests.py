"""analysis.stats.hypothesis_tests — 归因统计检验接口。

规则: 对 attribution (SHAP/IG/ISM...) 做显著性前必须先建立 null distribution
(permutation/bootstrap), 再计算 p, 之后才允许 FDR。SNR 本身不是 p。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis.stats.multiple_testing import bh_fdr


@dataclass
class PermutationTestResult:
    observed: float
    p_value: float
    n_permutations: int
    seed: int
    null_stats_available: bool = True


def permutation_test_for_effect(
    effect_values: Sequence[float],
    null_effect: float = 0.0,
    n_permutations: int = 1000,
    seed: int = 2024,
    one_sided: str = "greater",
) -> PermutationTestResult:
    """
    Sign-flip 随机化检验 (one-sample permutation test)。

    H0: (x - null_effect) 关于 0 对称 -> 符号可交换; 统计量 = 均值。
    one_sided='greater' -> p = P(perm_mean >= observed); 'less' 对称; 'two_sided' 双侧。
    对 paired 差值 (例如 per-sample loss difference) 该检验等价于标准的配对置换检验。
    """
    arr = np.asarray([float(x) for x in effect_values if x is not None and np.isfinite(x)])
    observed = float(np.mean(arr)) if arr.size else float("nan")
    if arr.size < 3:
        return PermutationTestResult(observed, float("nan"), int(n_permutations), int(seed),
                                     null_stats_available=False)
    rng = np.random.default_rng(seed)
    # 符号翻转 (sign-flip) 随机化检验: H0 下 (x - null_effect) 关于 0 对称, 符号可交换。
    # 注意: 直接对中心化值做"重排"对均值是恒等变换 (置换不改变均值), 会得到退化的 p;
    # 因此必须翻转符号而不是重排位置。
    centered = arr - float(null_effect)
    stat_obs = float(np.mean(centered))
    count = 0
    for _ in range(int(n_permutations)):
        signs = rng.integers(0, 2, size=centered.size) * 2 - 1
        perm_stat = float(np.mean(signs * centered))
        if one_sided == "greater":
            count += 1 if perm_stat >= stat_obs else 0
        elif one_sided == "less":
            count += 1 if perm_stat <= stat_obs else 0
        else:
            count += 1 if abs(perm_stat) >= abs(stat_obs) else 0
    p = (count + 1) / (int(n_permutations) + 1)
    return PermutationTestResult(observed, float(p), int(n_permutations), int(seed))


def anova_interface(
    data,
    factors: List[str],
    response: str,
    design_info: Optional[dict] = None,
    config: Optional[dict] = None,
):
    """兼容包装: 真正的 factorial ANOVA 在 `factorial_anova`。

    旧签名 (data, factors, response, design_info, config) 保持不变;
    数据不足/缺列时返回 available=False + reason, 绝不返回伪统计量。
    """
    if data is None or len(getattr(data, "index", [])) == 0:
        return {"available": False, "reason": "empty data", "factor": None, "effect": None,
                "F_statistic": None, "p_value": None, "FDR": None, "effect_size": None,
                "ci_low": None, "ci_high": None}
    df = data if hasattr(data, "columns") else None
    if df is None:
        return {"available": False, "reason": "data must be a pandas DataFrame",
                "factor": None, "effect": None, "F_statistic": None, "p_value": None,
                "FDR": None, "effect_size": None, "ci_low": None, "ci_high": None}
    missing = [c for c in list(factors) + [response] if c not in df.columns]
    if missing:
        return {"available": False, "reason": f"missing columns: {missing}", "factor": None,
                "effect": None, "F_statistic": None, "p_value": None, "FDR": None,
                "effect_size": None, "ci_low": None, "ci_high": None}
    block = list((design_info or {}).get("blocks", []) or
                 (config or {}).get("block_factors", []))
    terms = factorial_anova(df, response=response, factors=list(factors),
                            block_factors=[b for b in block if b in df.columns],
                            min_observations=int((config or {}).get("min_observations", 32)),
                            min_residual_df=int((config or {}).get("min_residual_df", 5)),
                            include_interactions=bool((config or {}).get("include_interactions", True)),
                            n_iterations=int((config or {}).get("ci_iterations", 0)),
                            seed=int((config or {}).get("seed", 2024)))
    ok = [t for t in terms if t.status == "ok"]
    if not ok:
        reason = terms[0].reason if terms else "no term could be estimated"
        return {"available": False, "reason": reason, "factor": None, "effect": None,
                "F_statistic": None, "p_value": None, "FDR": None,
                "effect_size": None, "ci_low": None, "ci_high": None}
    first = ok[0]
    return {"available": True, "reason": "", "factor": first.factor, "effect": first.effect,
            "F_statistic": first.F_statistic, "p_value": first.p_value, "FDR": None,
            "effect_size": first.effect_size, "ci_low": first.ci_low, "ci_high": first.ci_high,
            "terms": [t.as_dict() for t in terms]}


__all__ = ["bh_fdr", "PermutationTestResult", "permutation_test_for_effect",
           "anova_interface", "AnovaTermResult", "factorial_anova"]


# ---------------------------------------------------------------------------
# Factorial ANOVA (extra-sum-of-squares F test, 支持 block 结构)
# ---------------------------------------------------------------------------
@dataclass
class AnovaTermResult:
    """一个 ANOVA term 的结果 (数据不足时 status != ok 且统计量为 None)。"""

    factor: str
    effect: Optional[float]
    F_statistic: Optional[float]
    p_value: Optional[float]
    df_num: Optional[float]
    df_den: Optional[float]
    effect_size: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]
    n_obs: int
    model_scope: str = "blocked_factorial"
    status: str = "ok"
    reason: str = ""

    def as_dict(self) -> dict:
        return {
            "factor": self.factor, "effect": self.effect, "F_statistic": self.F_statistic,
            "p_value": self.p_value, "df_num": self.df_num, "df_den": self.df_den,
            "effect_size": self.effect_size, "ci_low": self.ci_low, "ci_high": self.ci_high,
            "n_obs": self.n_obs, "model_scope": self.model_scope,
            "status": self.status, "reason": self.reason,
        }


def _dummy_columns(df, col: str) -> tuple:
    """单因子 dummy 列 (丢弃首水平, 避免共线)。返回 (names, matrix)。"""
    levels = sorted(pd.unique(df[col].astype(str)))
    names = [f"{col}={lv}" for lv in levels[1:]]
    mat = np.column_stack([(df[col].astype(str) == lv).to_numpy(dtype=float)
                           for lv in levels[1:]]) if len(levels) > 1 else np.zeros((len(df), 0))
    return names, mat


def _design_from_terms(df, terms: List[str]) -> tuple:
    """按 term 列表构造设计矩阵 (支持 a*b 交互 = dummy 列逐元素乘积)。

    返回 (column_names, matrix); 首列为截距。
    """
    names: List[str] = ["(intercept)"]
    parts: List[np.ndarray] = [np.ones((len(df), 1))]
    for term in terms:
        factors = term.split("*")
        cols = [None] * len(factors)
        col_names = [""] * len(factors)
        for i, f in enumerate(factors):
            nms, mat = _dummy_columns(df, f)
            col_names[i], cols[i] = nms, mat
        if any(c.shape[1] == 0 for c in cols):
            continue
        block = cols[0]
        label_block = col_names[0]
        for i in range(1, len(cols)):
            block = np.einsum("ij,ik->ijk", block, cols[i]).reshape(len(df), -1)
            label_block = [f"{a}:{b}" for a in label_block for b in col_names[i]]
        parts.append(block)
        names += [f"{term}[{nm}]" for nm in label_block]
    return names, np.hstack(parts)


def _fit_rss(y: np.ndarray, X: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return float(np.sum(resid ** 2))


def _f_sf(f_value: float, df_num: float, df_den: float) -> float:
    """F 分布上尾概率 (scipy 优先; 无 scipy 时用正则不完全 Beta 连分式)。"""
    try:
        from scipy.stats import f as _f_dist
        return float(_f_dist.sf(f_value, df_num, df_den))
    except Exception:  # noqa: BLE001
        if df_num <= 0 or df_den <= 0:
            return float("nan")
        from math import lgamma
        x = df_den / (df_den + df_num * f_value) if f_value > 0 else 1.0
        a, b = df_den / 2.0, df_num / 2.0

        def betacf(aa, bb, xx, itmax=200, eps=3e-12):
            qab, qap, qam = aa + bb, aa + 1.0, aa - 1.0
            c, d = 1.0, 1.0 - qab * xx / qap
            d = 1e-30 if abs(d) < 1e-30 else d
            d = 1.0 / d
            h = d
            for m in range(1, itmax + 1):
                m2 = 2 * m
                aa_m = m * (bb - m) * xx / ((qam + m2) * (aa + m2))
                d = 1.0 + aa_m * d
                d = 1e-30 if abs(d) < 1e-30 else d
                c = 1.0 + aa_m / c
                c = 1e-30 if abs(c) < 1e-30 else c
                d = 1.0 / d
                h *= d * c
                aa_m = -(aa + m) * (qab + m) * xx / ((aa + m2) * (qap + m2))
                d = 1.0 + aa_m * d
                d = 1e-30 if abs(d) < 1e-30 else d
                c = 1.0 + aa_m / c
                c = 1e-30 if abs(c) < 1e-30 else c
                d = 1.0 / d
                delta = d * c
                h *= delta
                if abs(delta - 1.0) < eps:
                    break
            return h

        def betai(aa, bb, xx):
            if xx <= 0:
                return 0.0
            if xx >= 1:
                return 1.0
            bt = np.exp(lgamma(aa + bb) - lgamma(aa) - lgamma(bb)
                        + aa * np.log(xx) + bb * np.log(1 - xx))
            if xx < (aa + 1) / (aa + bb + 2):
                return bt * betacf(aa, bb, xx) / aa
            return 1.0 - bt * betacf(bb, aa, 1 - xx) / bb
        try:
            return float(betai(b, a, x))
        except Exception:  # noqa: BLE001
            return float("nan")


def factorial_anova(frame,
                    response: str,
                    factors: Sequence[str],
                    block_factors: Sequence[str] = (),
                    include_interactions: bool = True,
                    min_observations: int = 32,
                    min_residual_df: int = 5,
                    n_iterations: int = 0,
                    seed: int = 2024,
                    model_scope: str = "blocked_factorial",
                    alpha: float = 0.05) -> List[AnovaTermResult]:
    """真实 factorial ANOVA: Type-II 边际 extra-sum-of-squares F 检验。

    设计 (按实际数据层级, 不把每个 R² 当作独立重复):
      response  : R² (或调用方指定的连续指标)
      factors   : 环境因子 (2 水平 on/off) -> 主效应 + 成对交互
      block     : model / cell_line / split_type 等层级作为 block 因子 (消去模型与细胞系均值差异)
    只对 terms 做 F 检验; effect 为该 factor 水平 1-水平 0 的调整后差异 (模型系数),
    CI 由行级 bootstrap 重估 (n_iterations=0 -> CI 为 None, 不伪造)。
    """
    if frame is None or len(frame) == 0:
        return [AnovaTermResult("(none)", None, None, None, None, None, None, None, None, 0,
                                model_scope, "unavailable", "empty data")]
    need = [response] + [f for f in factors] + [b for b in block_factors]
    missing = [c for c in need if c not in frame.columns]
    if missing:
        return [AnovaTermResult("(none)", None, None, None, None, None, None, None, None, 0,
                                model_scope, "unavailable", f"missing columns: {missing}")]
    df = frame.dropna(subset=need).copy()
    for col in list(factors) + list(block_factors):
        df[col] = df[col].astype(str)
    if len(df) < int(min_observations):
        return [AnovaTermResult("(none)", None, None, None, None, None, None, None, None,
                                len(df), model_scope, "unavailable",
                                f"insufficient_data: n={len(df)} < {min_observations}")]

    binary = [f for f in factors if df[f].nunique() == 2]
    usable_factors = [f for f in factors if df[f].nunique() >= 2]
    terms: List[str] = list(usable_factors)
    if include_interactions and len(usable_factors) >= 2:
        terms += [f"{a}*{b}" for i, a in enumerate(usable_factors)
                  for b in usable_factors[i + 1:]]

    def term_columns(term: str) -> List[str]:
        return term.split("*")

    def contains(term: str, factor: str) -> bool:
        return factor in term_columns(term)

    model_terms = list(usable_factors) + \
        ([t for t in terms if "*" in t] if include_interactions else []) + list(block_factors)
    col_names, X_full = _design_from_terms(df, model_terms)
    y = pd.to_numeric(df[response], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(y)
    X_full, y = X_full[mask], y[mask]
    df = df.loc[mask]
    n_obs = int(len(y))
    if n_obs < int(min_observations):
        return [AnovaTermResult("(none)", None, None, None, None, None, None, None, None,
                                n_obs, model_scope, "unavailable",
                                f"insufficient_data: n={n_obs} < {min_observations}")]
    rss_full = _fit_rss(y, X_full)
    df_den = n_obs - X_full.shape[1]
    if df_den < int(min_residual_df):
        return [AnovaTermResult("(none)", None, None, None, None, None, None, None, None,
                                n_obs, model_scope, "unavailable",
                                f"insufficient_residual_df: {df_den} < {min_residual_df}")]
    # 注意：这里**不再**计算 SS_total。经典 η² 的分母 SS_total 含 block 因子方差
    # （本批占 74.9%），会把效应量系统性低估约 4 倍；改用 partial η²（见下方 eta）。
    ms_den = rss_full / df_den if df_den > 0 else float("nan")

    out: List[AnovaTermResult] = []
    for term in terms:
        cols = term_columns(term)
        # Type-II 边际: 主效应去掉自身及包含它的交互; 交互只去掉自身 (保留其主效应)
        if len(cols) == 1:
            reduced_terms = [t for t in model_terms
                             if t != term and cols[0] not in t.split("*")]
        else:
            reduced_terms = [t for t in model_terms if t != term]
        _, X_red = _design_from_terms(df, reduced_terms)
        rss_red = _fit_rss(y, X_red)
        df_num = X_full.shape[1] - X_red.shape[1]
        if df_num <= 0:
            continue
        ss_term = max(rss_red - rss_full, 0.0)
        f_val = (ss_term / df_num) / ms_den if ms_den and ms_den > 0 else float("nan")
        p_val = _f_sf(float(f_val), df_num, df_den) if np.isfinite(f_val) else float("nan")
        # partial η² = SS_term / (SS_term + SS_error)，与
        # AnovaRuleConfig.effect_size_metric = "partial_eta_squared" 一致。
        # 不能用 SS_term / SS_total：SS_total 里含 block 因子（模型/细胞系/划分）的方差，
        # 本批 block 占 74.9%，会把效应量系统性低估约 4 倍。
        denom = ss_term + rss_full
        eta = ss_term / denom if denom > 0 else float("nan")

        effect = float("nan")
        ci_low = ci_high = None
        if len(cols) == 1 and cols[0] in binary:
            levels = sorted(pd.unique(df[cols[0]]))
            beta, *_ = np.linalg.lstsq(X_full, y, rcond=None)
            diff = 0.0
            for lv in levels[1:]:
                key = f"{cols[0]}[{cols[0]}={lv}]"
                if key in col_names:
                    diff += float(beta[col_names.index(key)])
            effect = diff
            if n_iterations and n_iterations > 0:
                rng = np.random.default_rng(seed)
                boots = np.empty(int(n_iterations))
                for i in range(int(n_iterations)):
                    idx = rng.integers(0, n_obs, size=n_obs)
                    bb, *_ = np.linalg.lstsq(X_full[idx], y[idx], rcond=None)
                    d = 0.0
                    for lv in levels[1:]:
                        key = f"{cols[0]}[{cols[0]}={lv}]"
                        if key in col_names:
                            d += float(bb[col_names.index(key)])
                    boots[i] = d
                ci_low = float(np.percentile(boots, 100 * alpha / 2))
                ci_high = float(np.percentile(boots, 100 * (1 - alpha / 2)))
        out.append(AnovaTermResult(
            factor=term, effect=effect, F_statistic=float(f_val) if np.isfinite(f_val) else None,
            p_value=float(p_val) if np.isfinite(p_val) else None,
            df_num=float(df_num), df_den=float(df_den),
            effect_size=float(eta) if np.isfinite(eta) else None,
            ci_low=ci_low, ci_high=ci_high, n_obs=n_obs, model_scope=model_scope,
            status="ok" if np.isfinite(f_val) and np.isfinite(p_val) else "unavailable",
            reason="" if np.isfinite(f_val) else "F statistic not estimable"))
    if not out:
        out = [AnovaTermResult("(none)", None, None, None, None, None, None, None, None, n_obs,
                               model_scope, "unavailable", "no estimable term (design saturated)")]
    return out


def permutation_test_for_effect_fast(
    effect_values: Sequence[float],
    null_effect: float = 0.0,
    n_permutations: int = 1000,
    seed: int = 2024,
    one_sided: str = "greater",
) -> PermutationTestResult:
    """`permutation_test_for_effect` 的向量化实现 (语义一致, 用于批量接线)。

    H0: E[effect] = null_effect (效应值可交换); 通过对中心化后的值做随机重排得到 null 分布。
    """
    arr = np.asarray([float(x) for x in effect_values if x is not None and np.isfinite(x)])
    observed = float(np.mean(arr)) if arr.size else float("nan")
    if arr.size < 3:
        return PermutationTestResult(observed, float("nan"), int(n_permutations), int(seed),
                                     null_stats_available=False)
    rng = np.random.default_rng(seed)
    centered = arr - float(null_effect)
    stat_obs = float(np.mean(centered))
    signs = rng.integers(0, 2, size=(int(n_permutations), centered.size)) * 2 - 1
    perm_stats = (signs * centered).mean(axis=1)
    if one_sided == "greater":
        count = int(np.sum(perm_stats >= stat_obs))
    elif one_sided == "less":
        count = int(np.sum(perm_stats <= stat_obs))
    else:
        count = int(np.sum(np.abs(perm_stats) >= abs(stat_obs)))
    p = (count + 1) / (int(n_permutations) + 1)
    return PermutationTestResult(observed, float(p), int(n_permutations), int(seed))
