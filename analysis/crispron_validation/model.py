# -*- coding: utf-8 -*-
"""analysis.crispron_validation.model — 用四个细胞系的**全部数据**训练 CNN7。

关键点
------
* **train = test = 全部序列**（没有任何留出）。因此这里得到的模型预测是
  in-sample 的，本流程用它做"同一批序列上模型对突变的反应"的对照，
  **不是**泛化性证据。
* 直接复用 `core.models.cnn.cnn.train()`，不复制训练逻辑、不修改 core。
  该函数按 `config['channel_names']` 推出序列通道数（A/C/G/T → 4），
  当 `config['selected_environments'] == []` 时把输入张量切成 4 通道、
  令环境分支为 0 通道 —— 于是"纯序列"这个设定由 core 自己保证。
* 突变预测：把每条序列第 18 位（1-based）的 one-hot 由 C 改成 A，
  再用**同一个模型**预测一次。只改这一位，PAM 区不受影响。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CrispronValidationConfig


def load_all_data(cfg: CrispronValidationConfig) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """按 `cfg.cell_lines` 顺序拼接四个细胞系的全部数据。

    返回 ``(X_3d, y, meta)``；``meta`` 含 `cell_line` 与 `sgRNA`，
    行序与 `X_3d` 严格一致（这是后续把预测值接回序列的唯一依据）。
    """
    import core.data.cell_line_division as cld

    schema = cld.load_feature_schema(str(cfg.data_dir))
    Xs, ys, metas = [], [], []
    for cl in cfg.cell_lines:
        d = cld.load_cell_line(data_dir=str(cfg.data_dir), cell_line=cl, schema=schema)
        Xs.append(np.asarray(d["X_3d"], dtype=np.float32))
        ys.append(np.asarray(d["y"], dtype=np.float32).reshape(-1))
        m = d["metadata"].copy()
        m["cell_line"] = cl
        metas.append(m)
        print(f"    {cl:<9} n={len(m):<6} X={np.asarray(d['X_3d']).shape}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    meta = pd.concat(metas, ignore_index=True)
    if not {"cell_line", "sgRNA"}.issubset(meta.columns):
        raise KeyError(f"metadata 缺少 cell_line / sgRNA 列；实际={list(meta.columns)}")
    meta["sgRNA"] = meta["sgRNA"].astype(str).str.upper().str.strip()
    if len(meta) != len(X):
        raise ValueError(f"meta 行数 {len(meta)} != X 样本数 {len(X)}")
    print(f"    合计 X={X.shape}, y={y.shape}")
    return X, y, meta


class Cnn7AllDataModel:
    """在全部数据上训练 CNN7，并给出 WT / C18A 两份预测。"""

    def __init__(self, cfg: CrispronValidationConfig) -> None:
        self.cfg = cfg
        self.result: Optional[Dict] = None

    # ---------------- 训练 ----------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        from core.models.cnn.cnn import train as cnn_train

        cfg = self.cfg
        root = cfg.cnn_result_dir
        config = {
            "dataset": cfg.dataset,
            "run_name": cfg.run_name,
            "model": "cnn_dual_branch",
            "split_type": "all_train",          # 非标准划分：全部数据既训练又评估
            "cell_line": "all",
            "cell_lines": list(cfg.cell_lines),
            "environment": cfg.environment,
            "combination": cfg.environment,
            "selected_environments": [],        # 空 -> core 切成 4 通道、环境分支 0 通道
            "environment_count": 0,
            "sequence_channels": len(cfg.nucleotide_channels),
            "channel_count": int(X.shape[2]),
            "channel_names": ["A", "C", "G", "T", "CTCF", "Dnase", "H3K4me3", "RRBS"],
            "sequence_length": int(X.shape[1]),
            "sequence_kernel": cfg.sequence_kernel,
            "environment_kernel": cfg.environment_kernel,
            "random_seed": cfg.random_seed,
            "n_train": int(len(X)),
            "n_test": int(len(X)),
            "train_equals_test": True,
            "note": "全部数据既作训练又作评估（无留出），预测为 in-sample",
        }
        print(f"  训练 CNN7：n={len(X)}, sequence_kernel={cfg.sequence_kernel}, "
              f"epochs={cfg.epochs}")
        self.result = cnn_train(
            X_train=X, y_train=y,
            X_test=X, y_test=y,
            X_valid=None, y_valid=None,
            run_name=cfg.run_name,
            model_dir=str(root / "models"),
            result_dir=str(root / "results"),
            log_dir=str(root / "logs"),
            config=config,
            random_seed=cfg.random_seed,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            sequence_kernel=cfg.sequence_kernel,
            environment_kernel=cfg.environment_kernel,
            patience=cfg.patience,
        )
        m = self.result["metrics"]
        print(f"    in-sample: R2={m.get('R2'):.4f} RMSE={m.get('RMSE'):.4f} "
              f"Pearson={m.get('Pearson'):.4f}")
        return self.result

    # ---------------- 预测 ----------------
    def predict(self, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """对输入张量推理。``X`` 可以带多余通道，这里按 core 的口径只取序列通道。"""
        if self.result is None:
            raise RuntimeError("请先调用 fit()。")
        import torch

        model = self.result["model"]
        n_seq = len(self.cfg.nucleotide_channels)
        Xs = np.asarray(X, dtype=np.float32)
        if Xs.ndim != 3:
            raise ValueError(f"输入必须是 (N, L, C)，实际 {Xs.shape}")
        Xs = Xs[:, :, :n_seq]                    # core 在 train() 里做的是同一切法
        device = next(model.parameters()).device
        model.eval()
        outs: List[np.ndarray] = []
        t = torch.from_numpy(np.ascontiguousarray(Xs))
        with torch.no_grad():
            for i in range(0, len(t), int(batch_size)):
                batch = t[i:i + int(batch_size)].to(device)
                outs.append(model(batch).squeeze(-1).detach().cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0, dtype=np.float32)


def mutate_pos18(X: np.ndarray, row_mask: np.ndarray, pos_index0: int,
                 base: str) -> np.ndarray:
    """把 ``row_mask`` 命中的行、``pos_index0`` 处的 one-hot 改成 ``base``。

    只动序列通道（前 4 个），其余通道原样复制。
    """
    order = ["A", "C", "G", "T"]
    b = str(base).upper()
    if b not in order:
        raise ValueError(f"目标碱基非法: {base!r}")
    Xm = np.array(X, dtype=np.float32, copy=True)
    idx = np.where(np.asarray(row_mask, dtype=bool))[0]
    if idx.size:
        Xm[np.ix_(idx, [int(pos_index0)], range(4))] = 0.0
        Xm[idx, int(pos_index0), order.index(b)] = 1.0
    return Xm
