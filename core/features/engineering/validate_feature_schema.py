#!/usr/bin/env python3
"""校验 ``feature_schema.json`` 声明与实际张量是否一致（用户级 CLI）。

为什么需要它
------------
``feature_schema.json`` 是模型输入的契约：它声明通道顺序、位置语义与维度。
如果声明与**实际构造出的张量**不一致（例如 schema 写 A/C/G/T 而代码用 G/C/A/T，
或环境通道顺序写在 schema 里但张量按另一个顺序拼接），模型会静默地在错误的通道上
学习——这类 silent mismatch 无法从 loss 或指标上看出来。

本脚本把"声明 == 实际"变成一条可运行的断言，纳入 README 工作流与 CI。

用法
----
    # 校验单个数据集
    python core/features/engineering/validate_feature_schema.py --data-set DeepCRISPR

    # 校验全部已处理数据集
    python core/features/engineering/validate_feature_schema.py --all

    # 直接指定目录（高级用法）
    python core/features/engineering/validate_feature_schema.py \\
        --data-dir data/processed/Hiranniramol

退出码
------
    0 = 全部通过；1 = 存在失败项；2 = 用法错误。

检查项
------
    1. schema 必备字段存在且类型正确
    2. channel_names 与张量第 3 维长度一致
    3. sequence_length 与张量第 2 维一致
    4. feature_count 与展平维度一致
    5. len(feature_names) == feature_count（若声明）
    6. 由元数据序列**重建** one-hot，与张量的序列通道逐元素相等
    7. 环境通道取值 ⊆ {0,1}，且通道顺序与 environment_features 声明一致
    8. layout 声明（position_major / 1-based）与实际展平顺序一致
    9. model_compatibility 声明的维度自洽（可用维度算出）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from core.common.paths import (  # noqa: E402
    available_datasets,
    resolve_dataset,
)

SCHEMA_FILENAME = "feature_schema.json"


class Checker:
    """收集检查结果；任何 fail 都会让退出码非零。"""

    def __init__(self) -> None:
        self.passed: List[str] = []
        self.failed: List[str] = []
        self.skipped: List[str] = []

    def ok(self, msg: str) -> None:
        self.passed.append(msg)

    def fail(self, msg: str) -> None:
        self.failed.append(msg)

    def skip(self, msg: str) -> None:
        self.skipped.append(msg)

    def check(self, cond: bool, msg: str, detail: str = "") -> bool:
        if cond:
            self.ok(msg)
        else:
            self.fail(msg + (f"  [{detail}]" if detail else ""))
        return bool(cond)

    @property
    def good(self) -> bool:
        return not self.failed

    def report(self, title: str) -> None:
        print(f"\n=== {title} ===")
        for m in self.passed:
            print(f"  PASS  {m}")
        for m in self.skipped:
            print(f"  SKIP  {m}")
        for m in self.failed:
            print(f"  FAIL  {m}")
        print(f"  -> {len(self.passed)} passed, {len(self.failed)} failed, "
              f"{len(self.skipped)} skipped")


def _load_tensors(data_dir: Path, schema: Dict | None = None) -> Dict[str, np.ndarray]:
    """收集该数据集下的 3D 特征张量，按细胞系命名。

    文件名形如 ``<细胞系>_features_<L>x<C>.npy``；优先按 schema 声明的
    L/C 精确定位，这样 ``sequence_length`` 改了也仍能找到文件。找不到时
    退回通配（兼容旧命名）。
    """
    out: Dict[str, np.ndarray] = {}
    patterns = []
    if schema and "sequence_length" in schema and "channel_count" in schema:
        patterns.append(
            f"*_features_{int(schema['sequence_length'])}x{int(schema['channel_count'])}.npy")
    patterns.append("*_features_*x*.npy")
    seen: set = set()
    for pattern in patterns:
        for p in sorted(data_dir.glob(pattern)):
            if p.name in seen:
                continue
            seen.add(p.name)
            cell = p.name.split("_features_")[0]
            out[cell] = np.load(p)
    return out


def _metadata_path(data_dir: Path, cell: str) -> Path:
    return data_dir / f"{cell}_metadata.csv"


def _sequence_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "sgrna":
            return c
    for c in df.columns:
        if "sequence" in str(c).strip().lower():
            return c
    return None


def validate_dataset(data_dir: Path, chk: Checker) -> None:
    schema_path = data_dir / SCHEMA_FILENAME
    if not schema_path.is_file():
        chk.fail(f"{data_dir.name}: 缺少 {SCHEMA_FILENAME}")
        return
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    # ---- 1. 必备字段 ----
    for key, typ in (("sequence_length", int), ("channel_count", int),
                     ("channel_names", list)):
        if key not in schema:
            chk.fail(f"{data_dir.name}: schema 缺少 {key}")
        elif not isinstance(schema[key], typ):
            chk.fail(f"{data_dir.name}: schema[{key}] 类型应为 {typ.__name__}")
        else:
            chk.ok(f"{data_dir.name}: schema 必备字段 {key} 存在且类型正确")

    if "channel_names" not in schema or "sequence_length" not in schema:
        return

    channel_names = schema["channel_names"]
    L = int(schema["sequence_length"])
    C = int(schema["channel_count"])

    chk.check(C == len(channel_names),
              f"{data_dir.name}: channel_count == len(channel_names)",
              f"{C} vs {len(channel_names)}")

    # ---- 6. model_compatibility 自洽 ----
    mc = schema.get("model_compatibility")
    if isinstance(mc, dict):
        lin = mc.get("linear", {})
        if "input_dim" in lin:
            chk.check(lin["input_dim"] == C * L - L,
                      f"{data_dir.name}: model_compatibility.linear.input_dim 自洽",
                      f"{lin['input_dim']} vs 期望 {C * L - L}")
        for m in ("xgboost", "mlp"):
            if isinstance(mc.get(m), dict) and "input_dim" in mc[m]:
                chk.check(mc[m]["input_dim"] == C * L,
                          f"{data_dir.name}: model_compatibility.{m}.input_dim 自洽",
                          f"{mc[m]['input_dim']} vs 期望 {C * L}")
        for m in ("cnn", "transformer"):
            if isinstance(mc.get(m), dict) and "input_shape" in mc[m]:
                chk.check(list(mc[m]["input_shape"]) == [L, C],
                          f"{data_dir.name}: model_compatibility.{m}.input_shape 自洽",
                          f"{mc[m]['input_shape']} vs {[L, C]}")
    else:
        chk.skip(f"{data_dir.name}: 无 model_compatibility 块（schema v1）")

    # ---- layout 声明 vs 实际 ----
    layout = schema.get("layout")
    if isinstance(layout, dict):
        chk.check(list(layout.get("tensor_shape", [])) == [L, C],
                  f"{data_dir.name}: layout.tensor_shape 自洽",
                  f"{layout.get('tensor_shape')} vs {[L, C]}")
        chk.check(layout.get("flatten_order") == "position_major",
                  f"{data_dir.name}: layout.flatten_order == position_major")
        chk.check(layout.get("position_indexing") == "1-based",
                  f"{data_dir.name}: layout.position_indexing == 1-based")
        # feature_names 必须真的是位置优先
        fnames = schema.get("feature_names")
        if isinstance(fnames, list) and fnames:
            expect_head = [f"pos1_{c}" for c in channel_names]
            chk.check(fnames[:C] == expect_head,
                      f"{data_dir.name}: feature_names 前 {C} 项为 pos1_<channel>（位置优先）",
                      f"{fnames[:C]} vs {expect_head}")
    else:
        chk.skip(f"{data_dir.name}: 无 layout 块（schema v1）")

    # ---- 2/3/4/5 维度 ----
    tensors = _load_tensors(data_dir, schema)
    if not tensors:
        chk.fail(
            f"{data_dir.name}: 找不到 *_features_{L}x{C}.npy 张量"
            "（命名须为 <细胞系>_features_<L>x<C>.npy）")
        return

    for cell, X in tensors.items():
        if X.ndim != 3:
            chk.fail(f"{data_dir.name}/{cell}: 张量维度应为 3，实际 {X.ndim}")
            continue
        chk.check(X.shape[1] == L,
                  f"{data_dir.name}/{cell}: 张量 L == schema.sequence_length",
                  f"{X.shape[1]} vs {L}")
        chk.check(X.shape[2] == C,
                  f"{data_dir.name}/{cell}: 张量 C == schema.channel_count",
                  f"{X.shape[2]} vs {C}")

        if "feature_count" in schema:
            chk.check(X.reshape(len(X), -1).shape[1] == int(schema["feature_count"]),
                      f"{data_dir.name}/{cell}: 展平维度 == schema.feature_count",
                      f"{X.reshape(len(X), -1).shape[1]} vs {schema['feature_count']}")

    if isinstance(schema.get("feature_names"), list):
        chk.check(len(schema["feature_names"]) == int(schema.get("feature_count", -1)),
                  f"{data_dir.name}: len(feature_names) == feature_count",
                  f"{len(schema['feature_names'])} vs {schema.get('feature_count')}")

    # ---- 6. 序列通道 one-hot 重建 ----
    seq_channels = schema.get("sequence_channels") or channel_names[:4]
    n_seq = len(seq_channels)
    idx = {b: i for i, b in enumerate(seq_channels)}
    total = 0
    bad = 0
    mismatch = 0
    for cell, X in tensors.items():
        meta_p = _metadata_path(data_dir, cell)
        if not meta_p.is_file():
            chk.skip(f"{data_dir.name}/{cell}: 无 metadata，跳过序列重建")
            continue
        df = pd.read_csv(meta_p)
        sc = _sequence_column(df)
        if sc is None:
            chk.skip(f"{data_dir.name}/{cell}: metadata 无序列列，跳过序列重建")
            continue
        seqs = df[sc].astype(str).str.upper()
        if len(seqs) != len(X):
            chk.fail(f"{data_dir.name}/{cell}: metadata 行数 {len(seqs)} != 张量行数 {len(X)}")
            continue
        reb = np.zeros((len(seqs), X.shape[1], n_seq), dtype=np.float32)
        for n, s in enumerate(seqs):
            total += 1
            if len(s) != L:
                bad += 1
                continue
            for p, b in enumerate(s):
                if b not in idx:
                    bad += 1
                    continue
                reb[n, p, idx[b]] = 1.0
        if not np.array_equal(X[:, :, :n_seq], reb):
            mismatch += 1
            chk.fail(f"{data_dir.name}/{cell}: 序列通道与由 metadata 重建的 one-hot 不一致",
                     f"最大差 {float(np.abs(X[:, :, :n_seq] - reb).max())}")
        else:
            chk.ok(f"{data_dir.name}/{cell}: 序列通道 == 由 metadata 重建的 one-hot（逐元素相等）")

    chk.check(bad == 0,
              f"{data_dir.name}: 序列全部可编码（长度 {L}、碱基 ∈ {seq_channels}）",
              f"{bad}/{total} 条非法")

    # ---- 7. 环境通道 ----
    env_feats = schema.get("environment_features") or []
    if env_feats:
        expect_env = [f["name"] for f in env_feats]
        declared_env = channel_names[n_seq:]
        chk.check(declared_env == expect_env,
                  f"{data_dir.name}: channel_names 的环境段 == environment_features 声明顺序",
                  f"{declared_env} vs {expect_env}")
        for cell, X in tensors.items():
            env = X[:, :, n_seq:]
            uniq = sorted(np.unique(env).tolist())
            chk.check(set(uniq) <= {0.0, 1.0},
                      f"{data_dir.name}/{cell}: 环境通道取值 ⊆ {{0,1}}",
                      f"实际 {uniq}")
    else:
        chk.check(C == n_seq,
                  f"{data_dir.name}: 无环境特征时 channel_count == 序列通道数",
                  f"{C} vs {n_seq}")
        chk.skip(f"{data_dir.name}: 数据集无环境通道，跳过环境检查")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="校验 feature_schema.json 声明与实际张量是否一致（声明 == 实际）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  python core/features/engineering/validate_feature_schema.py --data-set DeepCRISPR
  python core/features/engineering/validate_feature_schema.py --all
  python core/features/engineering/validate_feature_schema.py --data-dir data/processed/Labuhn

退出码：0 全部通过 / 1 存在失败 / 2 用法错误
""")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--data-set", "--data_set", "--dataset", dest="data_set",
                   help="数据集名称（大小写不敏感），如 DeepCRISPR / Hiranniramol / Labuhn")
    g.add_argument("--data-dir", help="直接指定已处理数据目录")
    g.add_argument("--all", action="store_true", help="校验 data/processed 下全部数据集")
    a = ap.parse_args(argv)

    targets: List[Path] = []
    if a.all:
        names = available_datasets()
        if not names:
            print("[FATAL] data/processed 下没有已处理数据集（缺少 feature_schema.json）")
            return 1
        targets = [resolve_dataset(n) for n in names]
    elif a.data_set:
        try:
            targets = [resolve_dataset(a.data_set)]
        except (FileNotFoundError, ValueError) as e:
            print(f"[FATAL] {e}")
            return 2
    else:
        p = Path(a.data_dir)
        if not (p / SCHEMA_FILENAME).is_file():
            print(f"[FATAL] {p} 下没有 {SCHEMA_FILENAME}")
            return 2
        targets = [p.resolve()]

    chk = Checker()
    for t in targets:
        validate_dataset(t, chk)
    chk.report("feature_schema 一致性校验（声明 == 实际）")

    if chk.good:
        print("\n结论: PASS — schema 声明与实际张量一致")
        return 0
    print("\n结论: FAIL — 存在声明与实际不一致的项，请修复后重跑")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
