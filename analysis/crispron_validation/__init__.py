# -*- coding: utf-8 -*-
"""analysis.crispron_validation — CNN7 全数据训练 × CRISPRon 的 Pos18 C→A 一致性验证。

--------------------------------------------------------------------------
这一步在回答什么
--------------------------------------------------------------------------
对 DeepCRISPR 四个细胞系里 **Pos18 是 C** 的序列，问一个问题：

    「把 18 位的 C 突变成 A，会**降低**编辑效率吗？」

两套独立证据给答案，然后看它们是否**同向**：

  左列 `y_true - ism`
      数据集里的**真实**编辑效率，减去 CNN7（第 18 位改成 A 之后）的**预测**效率。
      含义：这个突变在"实测标签 vs 我们模型的预测"之间造成多大落差。

  右列 `crispron_wt - crispron_c18a`
      第三方预测器 CRISPRon 对**野生型**序列的预测，减去它对 **C18A 突变体**的预测。
      含义：同一个突变在 CRISPRon 眼里造成多大落差。

  末列 `consistency`
      两列**同号为 `+`，异号为 `-`**。占比越高，说明"这个位点突变的效应方向"
      在实测标签 / 我们的模型 / 外部模型三方之间越自洽。

--------------------------------------------------------------------------
流程
--------------------------------------------------------------------------
1. `windows`  —— 为每条序列取 CRISPRon 需要的 30 nt 窗口（4+20+3+3），
                 并生成第 18 位 C→A 的突变窗口。**回验**窗口中间 23 nt == 数据里的 sgRNA。
2. `crispron` —— 调用 `deploy/crispron/` 里已装好的 CRISPRon，得到 WT 与 C18A 两组预测值。
3. `model`    —— 用**四个细胞系的全部数据**（无留出，train = test）训练
                 `sequence_kernel=7`、纯序列的 CNN，并在同一批数据上预测；
                 再把每条序列第 18 位改成 A，用同一模型再预测一次。
4. `table`    —— 组装 `cell_line / sgRNA / y_true-ism / crispron_wt-crispron_c18a /
                 consistency` 五列，写 `results/summary/DeepCRISPR/cnn7_validation.{md,csv}`。

**注意**：第 3 步的训练集 = 测试集（`train_ratio` 概念在这里不适用），
因此 `y_true - ism` 里的模型预测是 **in-sample** 的。这不是泛化性声明，
而是一次"同一批序列上，模型与外部预测器对同一个突变的反应是否同向"的对照。

用法::

    python -m analysis.crispron_validation                # 全流程
    python -m analysis.crispron_validation --steps windows
    python -m analysis.crispron_validation --steps crispron
    python -m analysis.crispron_validation --steps model
    python -m analysis.crispron_validation --steps table
"""
from __future__ import annotations

from .config import CrispronValidationConfig

__all__ = ["CrispronValidationConfig"]
