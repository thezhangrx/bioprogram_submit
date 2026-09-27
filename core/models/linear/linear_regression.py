# core/models/linear/linear_regression.py

"""
Linear Regression Model
=======================

CRISPR-Cas9 sgRNA Editing Efficiency Prediction

本模块负责：

    1. 接收已经完成 feature engineering 的 X / y
    2. 可选接收 validation set
    3. 训练 Linear Regression
    4. 使用 Moore-Penrose pseudoinverse 求解
    5. 计算权重标准误 (SE)、t-统计量 (t-stat)、双尾 p-值及 FDR (Benjamini-Hochberg) 校正
    6. prediction
    7. evaluation
    8. 保存模型
    9. 保存 prediction
    10. 保存带有统计显著性信息的 feature weights
    11. 保存线性代数诊断信息

模型：

    y_hat = Xw + b

实际实现：

    X_bias = [X, 1]

    w_bias = X_bias^+ y
"""


import json
import logging
import os
import pickle
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None


# ============================================================
# 1. Model
# ============================================================

class LinearRegressionModel:
    """
    Moore-Penrose Pseudoinverse Linear Regression.

    数学形式：

        y_hat = Xw + b

    实际：

        X_bias = [X, 1]

        w_bias = X_bias^+ y
    """

    def __init__(
        self,
        use_scaler: bool = False,
        pinv_rcond: float = 1e-15
    ):
        """
        参数
        ----

        use_scaler:
            是否使用 StandardScaler。

        pinv_rcond:
            np.linalg.pinv 的截断阈值。
        """

        self.use_scaler = use_scaler

        self.pinv_rcond = pinv_rcond

        self.scaler = None

        self.weights = None

        self.is_fitted = False

        self.feature_count = None

        # 线性代数诊断
        self.singular_values = None
        self.rank = None
        self.condition_number = None

        # 统计推断指标
        self.std_errors = None
        self.t_stats = None
        self.p_values = None
        self.p_adj_fdr = None
        self.significance = None

    # --------------------------------------------------------
    # Fit
    # --------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        训练模型并计算统计检验指标 (SE, t-stat, p-value, FDR)。
        """

        X_train = np.asarray(
            X_train,
            dtype=np.float64
        )

        y_train = np.asarray(
            y_train,
            dtype=np.float64
        ).reshape(-1)

        # ----------------------------------------------------
        # Shape & Finite check
        # ----------------------------------------------------

        if X_train.ndim != 2:
            raise ValueError("X_train 必须是二维数组：(N, features)")

        if X_train.shape[0] != len(y_train):
            raise ValueError(
                f"X_train 和 y_train 样本数不一致：X={X_train.shape[0]}, y={len(y_train)}"
            )

        if not np.isfinite(X_train).all():
            raise ValueError("X_train 中存在 NaN 或 Inf。")

        if not np.isfinite(y_train).all():
            raise ValueError("y_train 中存在 NaN 或 Inf。")

        n_samples, n_features = X_train.shape
        self.feature_count = n_features

        # ----------------------------------------------------
        # Optional scaler
        # ----------------------------------------------------

        if self.use_scaler:
            if StandardScaler is None:
                raise ImportError("use_scaler=True 需要 scikit-learn。")
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)

        # ----------------------------------------------------
        # Add bias (放置在最后一列)
        # ----------------------------------------------------

        ones = np.ones((n_samples, 1), dtype=np.float64)
        X_bias = np.hstack([X_train, ones])

        # ----------------------------------------------------
        # SVD diagnostics
        # ----------------------------------------------------

        singular_values = np.linalg.svd(X_bias, compute_uv=False)
        self.singular_values = singular_values

        sigma_max = singular_values[0] if len(singular_values) > 0 else 0.0
        tolerance = self.pinv_rcond * sigma_max

        self.rank = int(np.sum(singular_values > tolerance))
        nonzero_singular_values = singular_values[singular_values > tolerance]

        if len(nonzero_singular_values) == 0:
            self.condition_number = np.inf
        else:
            self.condition_number = float(sigma_max / nonzero_singular_values[-1])

        # ----------------------------------------------------
        # Moore-Penrose pseudoinverse 求解权重
        # ----------------------------------------------------

        X_pinv = np.linalg.pinv(X_bias, rcond=self.pinv_rcond)
        self.weights = (X_pinv @ y_train).reshape(-1)

        # ----------------------------------------------------
        # 统计推断：计算 SE, t-stat, p-value, FDR
        # ----------------------------------------------------

        y_pred = X_bias @ self.weights
        residuals = y_train - y_pred

        # 自由度 dof = N - rank
        dof = max(1, n_samples - self.rank)
        mse_resid = np.sum(residuals ** 2) / dof

        # Cov(w) = MSE * (X^T X)^+ = MSE * (X_pinv @ X_pinv^T)
        # 对角线方差: Var(w_i) = MSE * sum_j (X_pinv[i, j]^2)
        var_w = mse_resid * np.sum(X_pinv ** 2, axis=1)
        var_w = np.maximum(var_w, 1e-15)  # 避免0方差

        self.std_errors = np.sqrt(var_w)
        self.t_stats = self.weights / self.std_errors

        # 双尾 p-值
        self.p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(self.t_stats), df=dof))

        # FDR (Benjamini-Hochberg) 多重检验校正
        n_total_weights = len(self.p_values)
        sorted_idx = np.argsort(self.p_values)
        sorted_p = self.p_values[sorted_idx]
        q_values = np.empty(n_total_weights)
        min_q = 1.0
        for i in range(n_total_weights - 1, -1, -1):
            rank_i = i + 1
            q = sorted_p[i] * n_total_weights / rank_i
            min_q = min(min_q, q)
            q_values[i] = min(min_q, 1.0)

        self.p_adj_fdr = np.empty(n_total_weights)
        self.p_adj_fdr[sorted_idx] = q_values

        # 显著性标记
        def get_sig_symbol(p_adj):
            if p_adj < 0.001: return '***'
            elif p_adj < 0.01: return '**'
            elif p_adj < 0.05: return '*'
            elif p_adj < 0.1: return '.'
            return ''

        self.significance = [get_sig_symbol(p) for p in self.p_adj_fdr]
        self.is_fitted = True

        return self

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        预测。
        """

        if not self.is_fitted:
            raise RuntimeError("模型尚未训练。")

        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组：(N, features)")

        if X.shape[1] != self.feature_count:
            raise ValueError(
                f"输入特征数量与训练时不一致：expected={self.feature_count}, actual={X.shape[1]}"
            )

        if not np.isfinite(X).all():
            raise ValueError("X 中存在 NaN 或 Inf。")

        if self.use_scaler:
            if self.scaler is None:
                raise RuntimeError("模型启用了 scaler，但 scaler 不存在。")
            X = self.scaler.transform(X)

        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        X_bias = np.hstack([X, ones])

        y_pred = X_bias @ self.weights
        return y_pred.reshape(-1)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    def save(
        self,
        model_dir: str
    ):
        """
        保存模型状态与诊断。
        """

        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "linear_regression_model.pkl")
        scaler_path = os.path.join(model_dir, "linear_regression_scaler.pkl")
        diagnostics_path = os.path.join(model_dir, "linear_regression_diagnostics.json")

        model_state = {
            "weights": self.weights,
            "std_errors": self.std_errors,
            "t_stats": self.t_stats,
            "p_values": self.p_values,
            "p_adj_fdr": self.p_adj_fdr,
            "significance": self.significance,
            "feature_count": self.feature_count,
            "use_scaler": self.use_scaler,
            "pinv_rcond": self.pinv_rcond,
            "is_fitted": self.is_fitted,
            "singular_values": self.singular_values,
            "rank": self.rank,
            "condition_number": self.condition_number
        }

        with open(model_path, "wb") as f:
            pickle.dump(model_state, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        diagnostics = {
            "feature_count": self.feature_count,
            "numerical_rank": self.rank,
            "condition_number": self.condition_number,
            "pinv_rcond": self.pinv_rcond,
            "singular_values": (
                self.singular_values.tolist()
                if self.singular_values is not None
                else []
            )
        }

        with open(diagnostics_path, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=4, ensure_ascii=False)

        return {
            "model_path": model_path,
            "scaler_path": scaler_path,
            "diagnostics_path": diagnostics_path
        }

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    def load(
        self,
        model_dir: str
    ):
        """
        加载模型。
        """

        model_path = os.path.join(model_dir, "linear_regression_model.pkl")
        scaler_path = os.path.join(model_dir, "linear_regression_scaler.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型：{model_path}")

        with open(model_path, "rb") as f:
            model_state = pickle.load(f)

        self.weights = model_state["weights"]
        self.std_errors = model_state.get("std_errors")
        self.t_stats = model_state.get("t_stats")
        self.p_values = model_state.get("p_values")
        self.p_adj_fdr = model_state.get("p_adj_fdr")
        self.significance = model_state.get("significance")
        self.feature_count = model_state["feature_count"]
        self.use_scaler = model_state["use_scaler"]
        self.pinv_rcond = model_state.get("pinv_rcond", 1e-15)
        self.is_fitted = model_state["is_fitted"]
        self.singular_values = model_state.get("singular_values")
        self.rank = model_state.get("rank")
        self.condition_number = model_state.get("condition_number")

        if self.use_scaler and os.path.exists(scaler_path):
            with open(scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None

        return self


# ============================================================
# 2. Metrics
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
):
    """
    计算：MSE, RMSE, MAE, R2, Pearson, Spearman
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true 和 y_pred 长度不一致。")

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true 或 y_pred 中存在 NaN/Inf。")

    error = y_true - y_pred
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = np.nan if np.isclose(ss_tot, 0) else (1.0 - ss_res / ss_tot)

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson = np.nan
    else:
        pearson = np.corrcoef(y_true, y_pred)[0, 1]

    true_rank = pd.Series(y_true).rank(method="average").to_numpy()
    pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()

    if np.std(true_rank) == 0 or np.std(pred_rank) == 0:
        spearman = np.nan
    else:
        spearman = np.corrcoef(true_rank, pred_rank)[0, 1]

    return {
        "MSE": float(mse),
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R2": float(r2),
        "Pearson": float(pearson),
        "Spearman": float(spearman)
    }


# ============================================================
# 3. Feature name utilities
# ============================================================

def generate_default_feature_names(
    feature_count: int
):
    """
    动态生成默认 feature names (末尾带 Bias)。
    """
    return [f"Feature_{i + 1}" for i in range(feature_count)] + ["Bias"]


# ============================================================
# 4. Logger
# ============================================================

# ============================================================
# 4a. 哑变量陷阱防护: 剔除 _T 参照特征 (T 为基准对照)
# ============================================================

REFERENCE_T_SUFFIX = "_T"


def select_non_t_reference_features(
    X: np.ndarray,
    feature_names: Optional[List[str]],
) -> "tuple[np.ndarray, Optional[List[str]], List[int]]":
    """
    将 (N, 184) 四碱基 One-Hot 输入收缩为 (N, 161):
      剔除所有以 '_T' 结尾的特征 (pos1_T ~ pos23_T), 保留截距项以 T 为基准。
    旧 7 通道数据 (无 _T 列) 原样返回, keep_indices 为空列表。
    返回 (X_sub, feature_names_sub, keep_indices)。
    """
    X = np.asarray(X)
    if feature_names is None or len(feature_names) != X.shape[1]:
        return X, feature_names, []
    names = [str(n).strip() for n in feature_names]
    drop_mask = [n.endswith(REFERENCE_T_SUFFIX) for n in names]
    if not any(drop_mask):
        return X, list(feature_names), []
    keep_indices = [i for i, d in enumerate(drop_mask) if not d]
    X_sub = X[:, keep_indices]
    names_sub = [names[i] for i in keep_indices]
    return X_sub, names_sub, keep_indices


def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")

    logger = logging.getLogger(f"LinearRegression_{log_path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger, log_path


# ============================================================
# 5. Save prediction results
# ============================================================

def save_predictions(y_true, y_pred, output_file):
    predictions_df = pd.DataFrame({
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1)
    })
    predictions_df["error"] = predictions_df["y_true"] - predictions_df["y_pred"]
    predictions_df.to_csv(output_file, index=False)


# ============================================================
# 6. Save weights (含统计显著性指标)
# ============================================================

def save_weights(
    model: LinearRegressionModel,
    feature_names,
    output_file: str
):
    """
    保存模型权重及统计推断指标 (白名单: Linear_Coefficient, SE, t_stat, p_value, FDR)。
    """
    weights = model.weights.reshape(-1)

    if feature_names is None:
        feature_names = generate_default_feature_names(model.feature_count)

    # 规范化 feature_names：若未包含 Bias，在末尾添加
    if len(feature_names) == model.feature_count:
        feature_names = list(feature_names) + ["Bias"]

    if len(feature_names) != (model.feature_count + 1):
        raise ValueError(
            f"feature_names 数量必须等于 model.feature_count + 1 (Bias)。\n"
            f"features={model.feature_count}, names={len(feature_names)}"
        )

    weights_df = pd.DataFrame({
        "Feature": feature_names,
        "Weight": weights,
        "Std_Error": model.std_errors if model.std_errors is not None else np.nan,
        "t_stat": model.t_stats if model.t_stats is not None else np.nan,
        "p_value": model.p_values if model.p_values is not None else np.nan,
        "p_adj_fdr": model.p_adj_fdr if model.p_adj_fdr is not None else np.nan,
        "Significance": model.significance if model.significance is not None else ""
    })
    # 学术红线: 线性回归导出仅保留白名单字段
    #   Linear_Coefficient, SE, t_stat, p_value, FDR (Significance/其余一律剔除)
    try:
        from core.xai.xai_importance import export_feature_table
    except ImportError:
        from xai_importance import export_feature_table
    weights_df = export_feature_table(
        "linear", weights_df, origin="linear_regression.save_weights"
    )
    weights_df.to_csv(output_file, index=False)


# ============================================================
# 7. Save singular values
# ============================================================

def save_linear_diagnostics(model, output_file):
    diagnostics = {
        "feature_count": model.feature_count,
        "numerical_rank": model.rank,
        "condition_number": model.condition_number,
        "pinv_rcond": model.pinv_rcond,
        "singular_values": (
            model.singular_values.tolist()
            if model.singular_values is not None
            else []
        )
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=4, ensure_ascii=False)


# ============================================================
# 8. Save experiment artifacts
# ============================================================

def save_results(
    model,
    y_valid,
    valid_metrics,
    y_test,
    y_pred,
    test_metrics,
    feature_names,
    result_dir: str,
    run_name: str,
    config: Optional[dict] = None
):
    os.makedirs(result_dir, exist_ok=True)

    metrics_path = os.path.join(result_dir, "linear_regression_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    validation_metrics_path = os.path.join(result_dir, "linear_regression_validation_metrics.json")
    if valid_metrics is not None:
        with open(validation_metrics_path, "w", encoding="utf-8") as f:
            json.dump(valid_metrics, f, indent=4, ensure_ascii=False)

    predictions_path = os.path.join(result_dir, "linear_regression_predictions.csv")
    save_predictions(y_true=y_test, y_pred=y_pred, output_file=predictions_path)

    weights_path = os.path.join(result_dir, "linear_regression_weights.csv")
    save_weights(model=model, feature_names=feature_names, output_file=weights_path)

    diagnostics_path = os.path.join(result_dir, "linear_regression_diagnostics.json")
    save_linear_diagnostics(model, diagnostics_path)

    info_path = os.path.join(result_dir, "linear_regression_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("Linear Regression Experiment\n")
        f.write("========================================\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Feature count: {model.feature_count}\n")
        f.write(f"Numerical rank: {model.rank}\n")
        f.write(f"Condition number: {model.condition_number}\n")
        f.write(f"Scaler: {model.use_scaler}\n")
        f.write(f"pinv_rcond: {model.pinv_rcond}\n")

        if config is not None:
            f.write("\nExperiment Configuration\n")
            f.write("----------------------------------------\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")

        if valid_metrics is not None:
            f.write("\nValidation Metrics\n")
            f.write("----------------------------------------\n")
            for key, value in valid_metrics.items():
                f.write(f"{key}: {value}\n")

        f.write("\nTest Metrics\n")
        f.write("----------------------------------------\n")
        for key, value in test_metrics.items():
            f.write(f"{key}: {value}\n")

    return {
        "metrics_path": metrics_path,
        "validation_metrics_path": validation_metrics_path if valid_metrics is not None else None,
        "predictions_path": predictions_path,
        "weights_path": weights_path,
        "diagnostics_path": diagnostics_path,
        "info_path": info_path
    }


# ============================================================
# 9. Unified train interface
# ============================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,

    X_test: np.ndarray,
    y_test: np.ndarray,

    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,

    feature_names=None,

    run_name: str = "linear_regression",

    model_dir: str = "models",
    result_dir: str = "results",
    log_dir: str = "results/logs",

    config: Optional[dict] = None,

    use_scaler: bool = False,

    pinv_rcond: float = 1e-15,

    random_seed = None
):
    current_model_dir = os.path.join(model_dir, run_name)
    current_result_dir = os.path.join(result_dir, run_name)
    current_log_dir = os.path.join(log_dir, run_name)

    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_result_dir, exist_ok=True)
    os.makedirs(current_log_dir, exist_ok=True)

    logger, log_path = create_logger(current_log_dir)
    logger.info("Starting Linear Regression experiment.")

    # 格式转换
    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_test = np.asarray(y_test, dtype=np.float64).reshape(-1)

    if X_valid is not None:
        X_valid = np.asarray(X_valid, dtype=np.float64)
    if y_valid is not None:
        y_valid = np.asarray(y_valid, dtype=np.float64).reshape(-1)

    # 维度检查
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train 与 X_test 必须是二维数组。")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"X_train 和 X_test 特征数量不同：{X_train.shape[1]} != {X_test.shape[1]}")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("样本数不匹配。")

    has_validation = (X_valid is not None and y_valid is not None)
    if has_validation:
        if X_valid.ndim != 2 or X_valid.shape[1] != X_train.shape[1] or len(X_valid) != len(y_valid):
            raise ValueError("X_valid 维度或样本数不匹配。")

    # 哑变量陷阱防护: 显式剔除 _T 参照特征 (新 23x8 编码下 184 -> 161; T 为基准)
    if feature_names is not None and len(feature_names) == X_train.shape[1]:
        X_train, feature_names, keep_indices = select_non_t_reference_features(
            X_train, feature_names
        )
        if keep_indices:
            X_test = X_test[:, keep_indices]
            if has_validation and X_valid is not None:
                X_valid = X_valid[:, keep_indices]
            logger.info(
                f"已剔除 {len(keep_indices)} 个 _T 参照特征 (T 为基准); "
                f"线性模型输入 -> {X_train.shape[1]} 维"
            )

    # 规范化 feature_names
    feature_count = X_train.shape[1]
    if feature_names is None:
        feature_names = generate_default_feature_names(feature_count)
    elif len(feature_names) == feature_count:
        feature_names = list(feature_names) + ["Bias"]
    elif len(feature_names) != (feature_count + 1):
        raise ValueError(
            f"feature_names 数量必须为 feature_count + 1 (含 Bias)。"
            f"feature_count={feature_count}, names={len(feature_names)}"
        )

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logger.info(f"Feature count: {feature_count}, Scaler: {use_scaler}")

    # 训练拟合
    model = LinearRegressionModel(use_scaler=use_scaler, pinv_rcond=pinv_rcond)
    model.fit(X_train, y_train)

    logger.info("Model fitting & statistical inference completed.")
    logger.info(f"Numerical rank: {model.rank}, Condition number: {model.condition_number:.2e}")

    # 统计显著特征数量
    if model.p_adj_fdr is not None:
        n_sig_001 = np.sum(model.p_adj_fdr < 0.001)
        n_sig_05 = np.sum(model.p_adj_fdr < 0.05)
        logger.info(f"Significantly contributing features (FDR < 0.05): {n_sig_05} (FDR < 0.001: {n_sig_001})")

    # Validation evaluation
    valid_metrics = None
    valid_predictions = None
    if has_validation:
        valid_predictions = model.predict(X_valid)
        valid_metrics = calculate_metrics(y_valid, valid_predictions)
        validation_predictions_path = os.path.join(current_result_dir, "linear_regression_validation_predictions.csv")
        save_predictions(y_true=y_valid, y_pred=valid_predictions, output_file=validation_predictions_path)

    # Test evaluation
    y_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Test Evaluation -> R2: {test_metrics['R2']:.4f}, Pearson: {test_metrics['Pearson']:.4f}, MAE: {test_metrics['MAE']:.4f}")

    # 保存模型与结果
    model_paths = model.save(current_model_dir)
    result_paths = save_results(
        model=model,
        y_valid=y_valid,
        valid_metrics=valid_metrics,
        y_test=y_test,
        y_pred=y_pred,
        test_metrics=test_metrics,
        feature_names=feature_names,
        result_dir=current_result_dir,
        run_name=run_name,
        config=config
    )

    logger.info("Linear Regression experiment finished successfully.")

    return {
        "model": model,
        "valid_metrics": valid_metrics,
        "validation_predictions": valid_predictions,
        "metrics": test_metrics,
        "predictions": y_pred,
        "model_paths": model_paths,
        "result_paths": result_paths,
        "log_path": log_path
    }


# ============================================================
# 10. Standalone test
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        "\n[错误] core/models/linear/linear_regression.py 不再提供演示入口。\n"
        "\n"
        "  历史上这里会用 np.random 合成随机数据、训练一遍模型并打印指标。\n"
        "  那些数字看起来合理，但与任何真实数据无关，容易被误当成实验结果。\n"
        "  为杜绝伪造产物，该入口已改为直接报错。\n"
        "\n"
        "  请改用项目正式入口（见 README §7 训练 / §9 数据挖掘）：\n"
        "    python train.py --model linear      --split-type single --cell-line hct116 --environment sequence --data-set DeepCRISPR\n"
        "    python data_digging.py --batch-name <batch> --data-set DeepCRISPR\n"
    )
