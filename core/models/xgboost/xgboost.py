# core/models/xgboost/xgboost.py

"""
XGBoost
=======

CRISPR-Cas9 sgRNA Editing Efficiency Prediction

本模块负责：

    1. 创建 XGBRegressor
    2. Training Set 训练
    3. Validation Set 监控与 Early Stopping
    4. Validation / Test 性能评估
    5. TreeSHAP 归因与 SHAP_SNR 信噪比 (XAI 白名单: XGB_Gain/XGB_Weight/XGB_Cover/TreeSHAP/SHAP_SNR)
    ※ 学术红线: XGBoost 不产生任何 t/p/FDR 统计量 (置换检验已移除)
    7. 保存模型、预测结果、训练历史与特征稳健性统计表
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================
# 1. Default parameters
# ============================================================

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
}


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
        raise ValueError(
            f"y_true 和 y_pred 长度不一致：{len(y_true)} != {len(y_pred)}"
        )

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true 或 y_pred 中存在 NaN 或 Inf。")

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
        "Spearman": float(spearman),
    }


# ============================================================
# 3. Feature names
# ============================================================

def generate_default_feature_names(feature_count: int):
    if feature_count <= 0:
        raise ValueError("feature_count 必须 > 0。")
    return [f"Feature_{i + 1}" for i in range(feature_count)]


# ============================================================
# 4. Logger
# ============================================================

def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    logger_name = f"XGBoost_{os.path.abspath(log_path)}"
    logger = logging.getLogger(logger_name)
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
# 5. XGBoost model wrapper
# ============================================================

class XGBoostModel:
    def __init__(
        self,
        params: Optional[dict] = None,
        random_seed: int = 42,
        n_jobs: int = -1,
        verbose: bool = False,
        early_stopping_rounds: Optional[int] = 30
    ):
        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)

        self.random_seed = random_seed
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_count = None
        self.is_fitted = False
        self.best_iteration = None
        self.best_score = None
        self.evals_result = {}
        self.model = None

    def _create_model(self, use_early_stopping: bool):
        model_kwargs = dict(self.params)
        model_kwargs["random_state"] = self.random_seed
        model_kwargs["n_jobs"] = self.n_jobs
        model_kwargs["verbosity"] = 1 if self.verbose else 0

        if use_early_stopping and self.early_stopping_rounds is not None:
            model_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self.model = xgb.XGBRegressor(**model_kwargs)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_valid: Optional[np.ndarray] = None,
        y_valid: Optional[np.ndarray] = None
    ):
        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)

        if X_train.ndim != 2 or len(X_train) != len(y_train):
            raise ValueError("X_train 与 y_train 维度或样本数不匹配。")
        if not np.isfinite(X_train).all() or not np.isfinite(y_train).all():
            raise ValueError("X_train 或 y_train 中存在 NaN 或 Inf。")

        self.feature_count = X_train.shape[1]
        has_validation = (X_valid is not None and y_valid is not None)
        eval_set = None

        if has_validation:
            X_valid = np.asarray(X_valid, dtype=np.float32)
            y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1)
            eval_set = [(X_train, y_train), (X_valid, y_valid)]

        self._create_model(use_early_stopping=has_validation)

        if has_validation:
            self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        else:
            self.model.fit(X_train, y_train, verbose=False)

        self.is_fitted = True

        if has_validation:
            best_iteration = getattr(self.model, "best_iteration", None)
            self.best_iteration = int(best_iteration) if best_iteration is not None else None
            best_score = getattr(self.model, "best_score", None)
            try:
                self.best_score = float(best_score) if best_score is not None else None
            except Exception:
                self.best_score = None
        else:
            n_estimators = self.params.get("n_estimators", 0)
            self.best_iteration = int(n_estimators) - 1 if n_estimators > 0 else None
            self.best_score = None

        try:
            self.evals_result = self.model.evals_result()
        except Exception:
            self.evals_result = {}

        return self

    def predict(self, X: np.ndarray):
        if not self.is_fitted:
            raise RuntimeError("XGBoost 模型尚未训练。")
        X = np.asarray(X, dtype=np.float32)
        return self.model.predict(X)

    def save(self, model_dir: str, experiment_config: Optional[dict] = None):
        if not self.is_fitted:
            raise RuntimeError("不能保存尚未训练的模型。")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "xgboost_model.json")
        config_path = os.path.join(model_dir, "xgboost_config.json")

        self.model.save_model(model_path)

        config = {
            "model": "xgboost",
            "params": self.params,
            "random_seed": self.random_seed,
            "n_jobs": self.n_jobs,
            "feature_count": self.feature_count,
            "early_stopping_rounds": self.early_stopping_rounds,
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
        }
        if experiment_config is not None:
            config.update(experiment_config)

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

        return {"model_path": model_path, "config_path": config_path}

    def load(self, model_dir: str):
        model_path = os.path.join(model_dir, "xgboost_model.json")
        config_path = os.path.join(model_dir, "xgboost_config.json")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到 XGBoost 模型：{model_path}")

        self._create_model(use_early_stopping=False)
        self.model.load_model(model_path)

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            self.params = config.get("params", self.params)
            self.random_seed = config.get("random_seed", self.random_seed)
            self.n_jobs = config.get("n_jobs", self.n_jobs)
            self.feature_count = config.get("feature_count", None)
            self.early_stopping_rounds = config.get("early_stopping_rounds", self.early_stopping_rounds)
            self.best_iteration = config.get("best_iteration", None)
            self.best_score = config.get("best_score", None)

        self.is_fitted = True
        return self


# ============================================================
# 6. Feature importance & Robustness Statistics
# ============================================================

def get_feature_importance(
    model: XGBoostModel,
    feature_names=None,
    X_eval: Optional[np.ndarray] = None,
    y_eval: Optional[np.ndarray] = None,
    random_seed: int = 42
):
    """
    提取树模型特征重要性 (XAI 白名单输出)：
        - XGB_Gain (特征分裂平均增益), XGB_Weight (分裂频次), XGB_Cover (覆盖量)
        - TreeSHAP (全体样本 |SHAP| 均值), SHAP_SNR (mean/std 归因信噪比)

    学术红线：XGBoost 不具备经典参数检验前提 —— 一律不输出 t/p/FDR 类统计量，
    置换检验 p 值与显著性列已彻底移除。
    """

    if not model.is_fitted:
        raise RuntimeError("模型尚未训练。")

    if feature_names is None:
        feature_names = generate_default_feature_names(model.feature_count)

    feature_names = list(feature_names)
    if len(feature_names) != model.feature_count:
        raise ValueError(
            f"feature_names 数量与特征数不一致：{len(feature_names)} != {model.feature_count}"
        )

    booster = model.model.get_booster()

    # 1. 常规重要性指标 (Gain, Weight, Cover)
    importance_types = ["gain", "weight", "cover"]
    result = {"Feature": feature_names.copy()}

    for imp_type in importance_types:
        score = booster.get_score(importance_type=imp_type)
        result[f"Importance_{imp_type}"] = [
            score.get(f"f{i}", 0.0) for i in range(model.feature_count)
        ]

    # 2. 原生 TreeSHAP 与归因信噪比 SNR = mean(|SHAP|) / std(SHAP)
    shap_means = [np.nan] * model.feature_count
    shap_snrs = [np.nan] * model.feature_count

    if X_eval is not None:
        try:
            dmat = xgb.DMatrix(X_eval)
            # pred_contribs=True 计算原生 TreeSHAP，最后一列为偏置
            shap_matrix = booster.predict(dmat, pred_contribs=True)[:, :-1]
            abs_shap = np.abs(shap_matrix)

            for i in range(model.feature_count):
                col_shap = shap_matrix[:, i]
                col_abs = abs_shap[:, i]
                m_val = float(np.mean(col_abs))
                s_val = float(np.std(col_shap))
                shap_means[i] = m_val
                shap_snrs[i] = float(m_val / (s_val + 1e-12))
        except Exception:
            pass

    result["SHAP_mean"] = shap_means
    result["SHAP_SNR"] = shap_snrs

    importance_df = pd.DataFrame(result)

    # 学术红线白名单清洗: XGB_Gain, XGB_Weight, XGB_Cover, TreeSHAP, SHAP_SNR
    try:
        from core.xai.xai_importance import export_feature_table
    except ImportError:
        from xai_importance import export_feature_table
    importance_df = export_feature_table("xgboost", importance_df, origin="xgboost.get_feature_importance")

    # 默认按 XGB_Gain 降序排列
    importance_df = importance_df.sort_values("XGB_Gain", ascending=False).reset_index(drop=True)
    return importance_df


# ============================================================
# 7. Save predictions
# ============================================================

def save_predictions(y_true, y_pred, output_file):
    predictions_df = pd.DataFrame({
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1),
    })
    predictions_df["error"] = predictions_df["y_true"] - predictions_df["y_pred"]
    predictions_df.to_csv(output_file, index=False)


# ============================================================
# 8. Save training history
# ============================================================

def save_training_history(model: XGBoostModel, output_file: str):
    evals_result = model.evals_result
    if not evals_result:
        pd.DataFrame(columns=["iteration", "train_rmse", "validation_rmse"]).to_csv(output_file, index=False)
        return

    train_metric = evals_result.get("validation_0", {}).get("rmse", [])
    valid_metric = evals_result.get("validation_1", {}).get("rmse", [])
    n_iterations = max(len(train_metric), len(valid_metric))

    rows = []
    for iteration in range(n_iterations):
        rows.append({
            "iteration": iteration,
            "train_rmse": train_metric[iteration] if iteration < len(train_metric) else np.nan,
            "validation_rmse": valid_metric[iteration] if iteration < len(valid_metric) else np.nan,
        })
    pd.DataFrame(rows).to_csv(output_file, index=False)


# ============================================================
# 9. Save results
# ============================================================

def save_results(
    model: XGBoostModel,
    y_valid,
    y_valid_pred,
    valid_metrics,
    y_test,
    y_test_pred,
    test_metrics,
    feature_names,
    result_dir: str,
    run_name: str,
    config: Optional[dict] = None,
    importance_df: Optional[pd.DataFrame] = None
):
    os.makedirs(result_dir, exist_ok=True)

    metrics_path = os.path.join(result_dir, "xgboost_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    validation_metrics_path = os.path.join(result_dir, "xgboost_validation_metrics.json")
    if valid_metrics is not None:
        with open(validation_metrics_path, "w", encoding="utf-8") as f:
            json.dump(valid_metrics, f, indent=4, ensure_ascii=False)
    else:
        validation_metrics_path = None

    predictions_path = os.path.join(result_dir, "xgboost_predictions.csv")
    save_predictions(y_true=y_test, y_pred=y_test_pred, output_file=predictions_path)

    validation_predictions_path = os.path.join(result_dir, "xgboost_validation_predictions.csv")
    if y_valid is not None and y_valid_pred is not None:
        save_predictions(y_true=y_valid, y_pred=y_valid_pred, output_file=validation_predictions_path)
    else:
        validation_predictions_path = None

    # 保存特征重要性 (XAI 白名单: XGB_Gain/XGB_Weight/XGB_Cover/TreeSHAP/SHAP_SNR)
    importance_path = os.path.join(result_dir, "xgboost_feature_importance.csv")
    if importance_df is not None:
        importance_df.to_csv(importance_path, index=False)

    history_path = os.path.join(result_dir, "xgboost_training_history.csv")
    save_training_history(model=model, output_file=history_path)

    info_path = os.path.join(result_dir, "xgboost_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("XGBoost Experiment\n")
        f.write("========================================\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n")
        f.write(f"Feature count: {model.feature_count}\n")
        f.write(f"Best iteration: {model.best_iteration}\n")
        f.write(f"Best validation score: {model.best_score}\n")
        f.write(f"Early stopping rounds: {model.early_stopping_rounds}\n\n")

        f.write("Hyperparameters\n")
        f.write("----------------------------------------\n")
        for key, value in model.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"random_state: {model.random_seed}\n")
        f.write(f"n_jobs: {model.n_jobs}\n")

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
        "validation_metrics_path": validation_metrics_path,
        "predictions_path": predictions_path,
        "validation_predictions_path": validation_predictions_path,
        "importance_path": importance_path,
        "history_path": history_path,
        "info_path": info_path,
    }


# ============================================================
# 10. Unified training interface
# ============================================================

def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    feature_names=None,
    run_name: str = "xgboost",
    model_dir: str = "models",
    result_dir: str = "results",
    log_dir: str = "results/logs",
    config: Optional[dict] = None,
    use_scaler: bool = False,
    random_seed: int = 42,
    params: Optional[dict] = None,
    n_jobs: int = -1,
    verbose: bool = False,
    early_stopping_rounds: Optional[int] = 30
):
    current_model_dir = os.path.join(model_dir, run_name)
    current_result_dir = os.path.join(result_dir, run_name)
    current_log_dir = os.path.join(log_dir, run_name)

    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_result_dir, exist_ok=True)
    os.makedirs(current_log_dir, exist_ok=True)

    logger, log_path = create_logger(current_log_dir)
    logger.info("Starting XGBoost experiment.")

    # 格式转换
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.float32).reshape(-1)

    if X_valid is not None:
        X_valid = np.asarray(X_valid, dtype=np.float32)
    if y_valid is not None:
        y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1)

    # 维度检查
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError("X_train 与 X_test 必须为二维。")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"X_train 与 X_test 特征数不一致：{X_train.shape[1]} != {X_test.shape[1]}")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("样本数不匹配。")

    has_validation = (X_valid is not None and y_valid is not None)
    if has_validation:
        if X_valid.ndim != 2 or X_valid.shape[1] != X_train.shape[1] or len(X_valid) != len(y_valid):
            raise ValueError("X_valid 维度或样本数不匹配。")

    feature_count = X_train.shape[1]
    if feature_names is None:
        feature_names = generate_default_feature_names(feature_count)
    feature_names = list(feature_names)

    if len(feature_names) != feature_count:
        raise ValueError(f"feature_names 数量与特征数不一致：{len(feature_names)} != {feature_count}")

    effective_early_stopping = early_stopping_rounds if has_validation else None

    # 模型实例化与训练
    model = XGBoostModel(
        params=params,
        random_seed=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        early_stopping_rounds=effective_early_stopping
    )

    model.fit(
        X_train=X_train,
        y_train=y_train,
        X_valid=X_valid if has_validation else None,
        y_valid=y_valid if has_validation else None
    )

    logger.info(f"Model fitting completed. Best iteration: {model.best_iteration}")

    # Validation evaluation
    valid_predictions = None
    valid_metrics = None
    if has_validation:
        valid_predictions = model.predict(X_valid)
        valid_metrics = calculate_metrics(y_valid, valid_predictions)
        logger.info(f"Validation R2: {valid_metrics['R2']:.4f}")

    # Test evaluation
    y_pred = model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Test Evaluation -> R2: {test_metrics['R2']:.4f}, Pearson: {test_metrics['Pearson']:.4f}")

    # 计算特征重要性 (XAI 白名单: XGB_Gain/XGB_Weight/XGB_Cover/TreeSHAP/SHAP_SNR)
    eval_X = X_test if len(X_test) > 0 else X_train
    eval_y = y_test if len(y_test) > 0 else y_train
    logger.info("Computing TreeSHAP SNR feature importance (whitelist)...")
    importance_df = get_feature_importance(
        model=model,
        feature_names=feature_names,
        X_eval=eval_X,
        y_eval=eval_y,
        random_seed=random_seed
    )

    # 保存模型与结果
    config = dict(config or {})
    config.update({
        "model": "xgboost",
        "input_dim": feature_count,
        "feature_count": feature_count,
        "random_seed": random_seed,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
    })

    model_paths = model.save(model_dir=current_model_dir, experiment_config=config)
    result_paths = save_results(
        model=model,
        y_valid=y_valid if has_validation else None,
        y_valid_pred=valid_predictions if has_validation else None,
        valid_metrics=valid_metrics if has_validation else None,
        y_test=y_test,
        y_test_pred=y_pred,
        test_metrics=test_metrics,
        feature_names=feature_names,
        result_dir=current_result_dir,
        run_name=run_name,
        config=config,
        importance_df=importance_df
    )

    logger.info("XGBoost experiment finished.")

    return {
        "model": model,
        "metrics": test_metrics,
        "valid_metrics": valid_metrics,
        "predictions": y_pred,
        "validation_predictions": valid_predictions,
        "feature_importance": importance_df,
        "best_iteration": model.best_iteration,
        "best_score": model.best_score,
        "history": model.evals_result,
        "model_paths": model_paths,
        "result_paths": result_paths,
        "log_path": log_path,
    }


# ============================================================
# 11. Standalone test
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        "\n[错误] core/models/xgboost/xgboost.py 不再提供演示入口。\n"
        "\n"
        "  历史上这里会用 np.random 合成随机数据、训练一遍模型并打印指标。\n"
        "  那些数字看起来合理，但与任何真实数据无关，容易被误当成实验结果。\n"
        "  为杜绝伪造产物，该入口已改为直接报错。\n"
        "\n"
        "  请改用项目正式入口（见 README §7 训练 / §9 数据挖掘）：\n"
        "    python train.py --model xgboost     --split-type single --cell-line hct116 --environment sequence --data-set DeepCRISPR\n"
        "    python data_digging.py --batch-name <batch> --data-set DeepCRISPR\n"
    )
