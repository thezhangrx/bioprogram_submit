# core/models/mlp/mlp.py

"""
MLP
===

CRISPR-Cas9 sgRNA Editing Efficiency Prediction

本模块负责：

    1. 创建动态输入维度的 MLP 回归模型
    2. Training Set 训练与 Validation Set 监控 / Early Stopping
    3. 恢复最佳 Validation Checkpoint
    4. Held-out Test 最终评价
    5. 原生 Integrated Gradients (IG) 与 IG_SNR 计算
    ※ 学术红线: MLP 不具备经典检验前提, 仅输出白名单指标 (MLP_IG/IG_SNR)
    6. 保存模型、预测结果、训练历史与特征稳健性统计表
"""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import random
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from sklearn.preprocessing import StandardScaler
except ImportError:
    StandardScaler = None

# ============================================================
# 1. Random seed
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ============================================================
# 2. Logger
# ============================================================

def create_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "training.log")
    logger_name = f"MLP_{os.path.abspath(log_path)}"
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
# 2.5 Optimizer / Scheduler / Activation 构造
# ============================================================

# 默认值 (adam / none / None) 必须精确复现改造前的硬编码行为:
#   optimizer  -> Adam(params, lr=learning_rate, weight_decay=weight_decay)
#   scheduler  -> 不创建任何调度器, 训练循环不调用 .step()
#   activation -> 沿用各模型原有激活 (CNN/MLP 为 ReLU, Transformer 为 GELU)
SUPPORTED_OPTIMIZERS = ("adam", "adamw", "sgd", "rmsprop", "adagrad")
SUPPORTED_SCHEDULERS = ("none", "cosine", "step", "exponential", "plateau")
SUPPORTED_ACTIVATIONS = ("relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu", "silu")


def build_optimizer(name, parameters, learning_rate, weight_decay):
    """按名称构造优化器。

    之所以抽成函数: CNN/MLP/Transformer 三个入口共用同一份语义,
    避免三处各写一段 if-else 后默认值漂移 —— 默认必须逐字等价于
    ``Adam(params, lr=learning_rate, weight_decay=weight_decay)``,
    否则 results/batches/ultimate_run 无法用原命令复现。
    """
    key = str(name if name is not None else "adam").strip().lower()
    if key == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=learning_rate, weight_decay=weight_decay)
    if key == "adagrad":
        return torch.optim.Adagrad(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"未知 optimizer: {name!r}; 允许: {SUPPORTED_OPTIMIZERS}")


def build_scheduler(name, optimizer, epochs, patience):
    """按名称构造学习率调度器; ``none`` 时返回 None。

    返回 None 表示"完全不创建调度器", 训练循环因此不步进学习率 ——
    与改造前逐位一致。plateau 需要监控指标, 由训练循环用 isinstance 判定;
    其余调度器按 epoch 步进。
    """
    key = str(name if name is not None else "none").strip().lower()
    if key in ("none", ""):
        return None
    if key == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs)))
    if key == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, int(epochs) // 3), gamma=0.1)
    if key == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    if key == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(1, int(patience)))
    raise ValueError(f"未知 scheduler: {name!r}; 允许: {SUPPORTED_SCHEDULERS}")


def build_activation(name):
    """按名称构造激活层; ``name is None`` (或 "none"/"default") 时返回 None。

    返回 None 表示"沿用本模型原有激活", 调用方必须保留自己的默认
    (CNN/MLP 为 ReLU, Transformer 为 GELU)。不能用一个统一默认值兜底,
    否则 Transformer 在默认参数下会被静默换成 ReLU, 破坏原批次可复现性。
    """
    if name is None:
        return None
    key = str(name).strip().lower()
    if key in ("none", "default"):
        return None
    mapping = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
        "leaky_relu": nn.LeakyReLU,
        "elu": nn.ELU,
        "silu": nn.SiLU,
    }
    if key not in mapping:
        raise ValueError(f"未知 activation: {name!r}; 允许: {SUPPORTED_ACTIVATIONS}")
    return mapping[key]()


# ============================================================
# 3. MLP Model
# ============================================================

class MLPModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.2,
        activation=None
    ):
        super().__init__()
        if input_dim <= 0 or hidden_dim1 <= 0 or hidden_dim2 <= 0:
            raise ValueError("Dimensions must be > 0.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout_rate = dropout

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU() if activation is None else build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU() if activation is None else build_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"MLP input must be 2D (N, D), got shape {x.shape}")
        return self.network(x)


# ============================================================
# 4. Metrics
# ============================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred length mismatch.")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("y_true or y_pred contains NaN/Inf.")

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
# 5. Evaluation helper
# ============================================================

def evaluate_model(model: nn.Module, data_loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    predictions = []
    targets = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            prediction = model(batch_x).squeeze(-1)
            loss = criterion(prediction, batch_y)
            batch_count = len(batch_x)

            total_loss += loss.item() * batch_count
            total_samples += batch_count
            predictions.append(prediction.cpu().numpy())
            targets.append(batch_y.cpu().numpy())

    if total_samples == 0:
        raise ValueError("Evaluation dataset is empty.")

    average_loss = total_loss / total_samples
    y_pred = np.concatenate(predictions)
    y_true = np.concatenate(targets)
    return float(average_loss), y_true, y_pred


# ============================================================
# 6. Integrated Gradients Robustness (XAI whitelist)
# ============================================================

def compute_integrated_gradients(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    steps: int = 30
) -> np.ndarray:
    """
    原生 PyTorch 实现 Integrated Gradients (完备性积分梯度)
    """
    model.eval()
    N, D = X.shape
    baseline = np.zeros_like(X, dtype=np.float32)
    ig_attributions = np.zeros((N, D), dtype=np.float32)

    # 沿直线路径离散积分
    alphas = np.linspace(0.0, 1.0, steps + 1)
    diff = X - baseline

    for i in range(N):
        x_orig = X[i]
        b_orig = baseline[i]
        step_pts = np.array([b_orig + a * (x_orig - b_orig) for a in alphas], dtype=np.float32)
        step_tensor = torch.from_numpy(step_pts).to(device).requires_grad_(True)

        preds = model(step_tensor).squeeze(-1)
        # 求各步梯度
        grads = torch.autograd.grad(preds.sum(), step_tensor)[0].cpu().numpy()
        avg_grads = np.mean(grads[:-1], axis=0)
        ig_attributions[i] = diff[i] * avg_grads

    return ig_attributions


def compute_mlp_robustness_importance(
    model: nn.Module,
    X_eval: np.ndarray,
    feature_names: list[str],
    device: torch.device,
) -> pd.DataFrame:
    """
    MLP 特征归因稳健性 (XAI 白名单输出)：
        - MLP_IG    : 积分梯度归因均值 (mean |IG|)
        - IG_SNR    : mean/std 积分梯度信噪比
    学术红线：MLP 不具备经典参数检验前提 —— 不产生 t/p/FDR，置换检验与显著性列已移除。
    """
    N, D = X_eval.shape

    # 1. Integrated Gradients
    ig_matrix = compute_integrated_gradients(model, X_eval, device, steps=25)
    abs_ig = np.abs(ig_matrix)
    ig_means = np.mean(abs_ig, axis=0)
    ig_stds = np.std(ig_matrix, axis=0)
    ig_snrs = ig_means / (ig_stds + 1e-12)

    df_importance = pd.DataFrame({
        "Feature": feature_names,
        "IG_Mean": ig_means,
        "IG_SNR": ig_snrs,
    })

    # 学术红线白名单清洗: 仅保留 MLP_IG / IG_SNR
    try:
        from core.xai.xai_importance import export_feature_table
    except ImportError:
        from xai_importance import export_feature_table
    df_importance = export_feature_table(
        "mlp", df_importance,
        rename={"IG_Mean": "MLP_IG"},
        origin="mlp.compute_mlp_robustness_importance",
    )

    # 默认按 MLP_IG 降序排列
    df_importance.sort_values("MLP_IG", ascending=False, inplace=True)
    df_importance.reset_index(drop=True, inplace=True)
    return df_importance


# ============================================================
# 7. Save predictions and results
# ============================================================

def save_predictions(y_true, y_pred, output_file):
    predictions = pd.DataFrame({
        "y_true": np.asarray(y_true).reshape(-1),
        "y_pred": np.asarray(y_pred).reshape(-1)
    })
    predictions["error"] = predictions["y_true"] - predictions["y_pred"]
    predictions.to_csv(output_file, index=False)


def save_results(
    y_valid,
    y_valid_pred,
    valid_metrics,
    y_test,
    y_test_pred,
    test_metrics,
    history,
    result_dir,
    run_name,
    config,
    importance_df: Optional[pd.DataFrame] = None
):
    os.makedirs(result_dir, exist_ok=True)

    metrics_path = os.path.join(result_dir, "mlp_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=4, ensure_ascii=False)

    validation_metrics_path = os.path.join(result_dir, "mlp_validation_metrics.json")
    if valid_metrics is not None:
        with open(validation_metrics_path, "w", encoding="utf-8") as f:
            json.dump(valid_metrics, f, indent=4, ensure_ascii=False)
    else:
        validation_metrics_path = None

    predictions_path = os.path.join(result_dir, "mlp_predictions.csv")
    save_predictions(y_true=y_test, y_pred=y_test_pred, output_file=predictions_path)

    validation_predictions_path = os.path.join(result_dir, "mlp_validation_predictions.csv")
    if y_valid is not None and y_valid_pred is not None:
        save_predictions(y_true=y_valid, y_pred=y_valid_pred, output_file=validation_predictions_path)
    else:
        validation_predictions_path = None

    # 保存包含生信稳健性参数的特征重要性表
    importance_path = os.path.join(result_dir, "mlp_feature_importance.csv")
    if importance_df is not None:
        importance_df.to_csv(importance_path, index=False)

    history_path = os.path.join(result_dir, "mlp_training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index=False)

    info_path = os.path.join(result_dir, "mlp_info.txt")
    with open(info_path, "w", encoding="utf-8") as f:
        f.write("MLP Experiment\n")
        f.write("========================================\n")
        f.write(f"Run name: {run_name}\n")
        f.write(f"Time: {datetime.now().isoformat()}\n\n")

        f.write("Configuration\n")
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
        "info_path": info_path
    }


def save_model(model: nn.Module, model_path: str, config: dict):
    torch.save({"model_state_dict": model.state_dict(), "config": config}, model_path)


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
    run_name: str = "mlp",
    model_dir: str = "models",
    result_dir: str = "results",
    log_dir: str = "results/logs",
    config: Optional[dict] = None,
    random_seed: int = 42,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hidden_dim1: int = 128,
    hidden_dim2: int = 64,
    dropout: float = 0.2,
    weight_decay: float = 0.0,
    patience: int = 20,
    min_delta: float = 1e-6,
    num_workers: int = 0,
    device: Optional[str] = None,
    use_scaler: bool = False,
    optimizer: str = "adam",
    scheduler: str = "none",
    activation=None,
):
    set_seed(random_seed)

    current_model_dir = os.path.join(model_dir, run_name)
    current_result_dir = os.path.join(result_dir, run_name)
    current_log_dir = os.path.join(log_dir, run_name)

    os.makedirs(current_model_dir, exist_ok=True)
    os.makedirs(current_result_dir, exist_ok=True)
    os.makedirs(current_log_dir, exist_ok=True)

    logger, log_path = create_logger(current_log_dir)
    logger.info("Starting MLP experiment.")

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
        raise ValueError("MLP input must be 2D array.")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"Feature count mismatch: {X_train.shape[1]} != {X_test.shape[1]}")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError("Sample count mismatch.")

    has_validation = (X_valid is not None and y_valid is not None)
    if has_validation:
        if X_valid.ndim != 2 or X_valid.shape[1] != X_train.shape[1] or len(X_valid) != len(y_valid):
            raise ValueError("X_valid dimensions mismatch.")

    input_dim = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"Feature_{i + 1}" for i in range(input_dim)]
    feature_names = list(feature_names)

    # Scaler 处理
    scaler = None
    if use_scaler:
        if StandardScaler is None:
            raise ImportError("use_scaler=True requires scikit-learn.")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        if has_validation:
            X_valid = scaler.transform(X_valid).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        logger.info("StandardScaler fitted on training data.")

    # 设备配置
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_obj = torch.device(device)
    logger.info(f"Device: {device_obj}")

    # 模型与优化器
    model = MLPModel(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        dropout=dropout,
        activation=activation
    ).to(device_obj)

    criterion = nn.MSELoss()
    # 默认 optimizer="adam" 与改造前的硬编码 Adam(lr, weight_decay) 完全等价
    optimizer_obj = build_optimizer(optimizer, model.parameters(), learning_rate, weight_decay)
    # 默认 scheduler="none" 时返回 None -> 不创建调度器, 训练循环不步进学习率
    scheduler_obj = build_scheduler(scheduler, optimizer_obj, epochs, patience)

    # 数据加载器
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    valid_loader = None
    if has_validation:
        valid_dataset = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}, Input dim: {input_dim}")

    # 训练循环
    history = []
    best_val_loss = np.inf
    best_epoch = None
    best_state_dict = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device_obj)
            batch_y = batch_y.to(device_obj)

            optimizer_obj.zero_grad()
            prediction = model(batch_x).squeeze(-1)
            loss = criterion(prediction, batch_y)
            loss.backward()
            optimizer_obj.step()

            batch_count = len(batch_x)
            epoch_loss += loss.item() * batch_count
            sample_count += batch_count

        train_loss = epoch_loss / sample_count

        val_loss = None
        if has_validation:
            val_loss, _, _ = evaluate_model(model=model, data_loader=valid_loader, criterion=criterion, device=device_obj)

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "validation_loss": float(val_loss) if val_loss is not None else np.nan
        })

        if has_validation:
            improvement = best_val_loss - val_loss
            if improvement > min_delta:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        should_log = (epoch == 1 or epoch % 10 == 0 or epoch == epochs or (has_validation and epoch == best_epoch))
        if should_log:
            if val_loss is None:
                logger.info(f"Epoch {epoch:4d} | Train Loss: {train_loss:.8f}")
            else:
                logger.info(f"Epoch {epoch:4d} | Train Loss: {train_loss:.8f} | Validation Loss: {val_loss:.8f}")

        # 调度器步进 (scheduler_obj 为 None 时整个分支被跳过, 默认路径零开销且行为不变)
        if scheduler_obj is not None:
            if isinstance(scheduler_obj, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler_obj.step(val_loss if val_loss is not None else train_loss)
            else:
                scheduler_obj.step()

        if has_validation and epochs_without_improvement >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # 恢复最佳权重
    if has_validation:
        model.load_state_dict(best_state_dict)
        logger.info(f"Restored best checkpoint from epoch {best_epoch}.")
    else:
        best_epoch = epoch

    # Final Validation
    valid_predictions = None
    valid_metrics = None
    if has_validation:
        final_valid_loss, final_y_valid, valid_predictions = evaluate_model(
            model=model, data_loader=valid_loader, criterion=criterion, device=device_obj
        )
        valid_metrics = calculate_metrics(final_y_valid, valid_predictions)
        logger.info(f"Final Validation R2: {valid_metrics['R2']:.4f}")

    # Final Test
    test_tensor = torch.from_numpy(X_test).to(device_obj)
    model.eval()
    with torch.no_grad():
        y_pred = model(test_tensor).squeeze(-1).cpu().numpy()

    test_metrics = calculate_metrics(y_test, y_pred)
    logger.info(f"Final Test Evaluation -> R2: {test_metrics['R2']:.4f}, Pearson: {test_metrics['Pearson']:.4f}")

    # 计算生信稳健性重要性 (XAI 白名单: MLP_IG / IG_SNR)
    eval_X = X_test if len(X_test) > 0 else X_train
    logger.info("Computing Integrated Gradients whitelist importance...")
    importance_df = compute_mlp_robustness_importance(
        model=model,
        X_eval=eval_X,
        feature_names=feature_names,
        device=device_obj
    )
    logger.info(f"MLP XAI whitelist importance computed ({len(importance_df)} features).")

    # 保存配置与模型
    config = dict(config or {})
    config.update({
        "model": "mlp",
        "input_dim": input_dim,
        "feature_count": input_dim,
        "random_seed": random_seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "device_resolved": str(device_obj),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
    })

    model_path = os.path.join(current_model_dir, "mlp_model.pt")
    save_model(model=model, model_path=model_path, config=config)

    scaler_path = None
    if scaler is not None:
        scaler_path = os.path.join(current_model_dir, "mlp_scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    config_path = os.path.join(current_model_dir, "mlp_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    result_paths = save_results(
        y_valid=final_y_valid if has_validation else None,
        y_valid_pred=valid_predictions if has_validation else None,
        valid_metrics=valid_metrics if has_validation else None,
        y_test=y_test,
        y_test_pred=y_pred,
        test_metrics=test_metrics,
        history=history,
        result_dir=current_result_dir,
        run_name=run_name,
        config=config,
        importance_df=importance_df
    )

    logger.info("MLP experiment finished successfully.")

    return {
        "model": model,
        "metrics": test_metrics,
        "valid_metrics": valid_metrics,
        "predictions": y_pred,
        "validation_predictions": valid_predictions,
        "feature_importance": importance_df,
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "model_paths": {
            "model_path": model_path,
            "config_path": config_path,
            "scaler_path": scaler_path
        },
        "result_paths": result_paths,
        "log_path": log_path
    }


# ============================================================
# 10. Standalone test
# ============================================================

if __name__ == "__main__":
    raise SystemExit(
        "\n[错误] core/models/mlp/mlp.py 不再提供演示入口。\n"
        "\n"
        "  历史上这里会用 np.random 合成随机数据、训练一遍模型并打印指标。\n"
        "  那些数字看起来合理，但与任何真实数据无关，容易被误当成实验结果。\n"
        "  为杜绝伪造产物，该入口已改为直接报错。\n"
        "\n"
        "  请改用项目正式入口（见 README §7 训练 / §9 数据挖掘）：\n"
        "    python train.py --model mlp         --split-type single --cell-line hct116 --environment sequence --data-set DeepCRISPR\n"
        "    python data_digging.py --batch-name <batch> --data-set DeepCRISPR\n"
    )
