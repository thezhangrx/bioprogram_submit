# core/data/cell_line_division.py
"""
Cell Line Division (动态数据发现与零硬编码版)
============================================
自动动态扫描 data_dir 目录下实际存在的细胞系，彻底去除写死的 4 大细胞系。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_SCHEMA_FILENAME = "feature_schema.json"
DEFAULT_TRAIN_FRACTION = 0.70
DEFAULT_VALIDATION_FRACTION = 0.15
DEFAULT_TEST_FRACTION = 0.15


def load_feature_schema(data_dir: str) -> Dict:
    """读取 data_dir 下的 feature_schema.json。

    该文件由特征工程写入，**必须已存在**。这里不再"缺失就自动写一份 8 通道
    DeepCRISPR schema"——那会让任何指向错误目录的调用静默按 4 细胞系 8 通道
    的假设继续跑，产生看似正常、实际维度错配的结果。
    """
    schema_path = os.path.join(data_dir, DEFAULT_SCHEMA_FILENAME)

    if not os.path.exists(schema_path) or os.path.getsize(schema_path) == 0:
        raise FileNotFoundError(
            f"找不到特征 schema：{schema_path}\n"
            "  data_dir 必须指向**已经跑过特征工程**的目录，例如：\n"
            "    data/processed/DeepCRISPR     (8 通道 / 184 维)\n"
            "    data/processed/Hiranniramol   (4 通道 /  92 维)\n"
            "    data/processed/Labuhn         (4 通道 /  92 维)\n"
            "  请先运行：\n"
            "    python core/features/engineering/feature_engineering.py \\\n"
            "        --raw-data <原始数据> --output-dir <该目录> --config <feature config>"
        )

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except Exception as error:
        raise ValueError(
            f"schema 文件损坏无法解析：{schema_path}\n  {error}\n"
            "  请重新运行特征工程生成，不要手工修补。"
        ) from error

    if not isinstance(schema, dict) or not schema.get("channel_names"):
        raise ValueError(
            f"schema 内容不完整（缺少 channel_names）：{schema_path}"
        )

    return schema


def discover_available_cell_lines(data_dir: str) -> List[str]:
    """
    动态扫描 data_dir 目录下实际拥有的细胞系。

    P0 整改 (H1): 结果排序后再返回。Path.glob 的顺序取决于文件系统的
    readdir 顺序 (本地 ext4 与超算 Lustre/GPFS 可能不同), 而该顺序决定
    mixed 划分前的行拼接顺序 -> 会改变 group-aware 划分得到的
    train/valid/test 集合, 破坏跨机器可复现性。排序保证确定性。
    """
    p = Path(data_dir)
    found = []
    if p.exists() and p.is_dir():
        for f in p.glob("*_metadata.csv"):
            cl = f.name.replace("_metadata.csv", "").lower()
            if cl not in found and cl != "feature":
                found.append(cl)
        for f in p.glob("*_features_*.npy"):
            m = re.match(r"([a-zA-Z0-9]+)_features_", f.name)
            if m:
                cl = m.group(1).lower()
                if cl not in found:
                    found.append(cl)
    found = sorted(found)

    if not found:
        raise FileNotFoundError(
            f"在 {data_dir} 中没有发现任何细胞系/数据集。\n"
            "  判定依据是存在以下任一文件：\n"
            "    <name>_metadata.csv   或   <name>_features_*.npy\n"
            "  这里**不再回退到 DeepCRISPR 的 4 个细胞系**——那会让指向错误目录的\n"
            "  调用继续往下跑，最后报出与真实原因无关的 FileNotFoundError。"
        )

    return found


def get_feature_file_paths(data_dir: str, cell_line: str, schema: Dict) -> Dict[str, str]:
    cell_line = str(cell_line).strip().lower()
    sequence_length = int(schema.get("sequence_length", 23))
    channel_count = int(schema.get("channel_count", 8))
    feature_count = int(schema.get("feature_count", sequence_length * channel_count))

    return {
        "X_3d": os.path.join(data_dir, f"{cell_line}_features_{sequence_length}x{channel_count}.npy"),
        "X_2d": os.path.join(data_dir, f"{cell_line}_features_{feature_count}.npy"),
        "y": os.path.join(data_dir, f"{cell_line}_labels.npy"),
        "metadata": os.path.join(data_dir, f"{cell_line}_metadata.csv")
    }


def validate_split_fractions(train_fraction: float, validation_fraction: float, test_fraction: float):
    fractions = [train_fraction, validation_fraction, test_fraction]
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("Train / Validation / Test 比例必须 > 0。")
    total = sum(fractions)
    if not np.isclose(total, 1.0, atol=1e-5):
        raise ValueError(f"比例之和必须为1，当前={total}")


def validate_cell_line_dataset(cell_line: str, X_3d: np.ndarray, X_2d: np.ndarray, y: np.ndarray, metadata: pd.DataFrame, schema: Dict):
    sequence_length = int(schema.get("sequence_length", 23))
    channel_count = int(schema.get("channel_count", 8))
    feature_count = int(schema.get("feature_count", sequence_length * channel_count))

    if X_3d.ndim != 3 or X_2d.ndim != 2:
        raise ValueError(f"{cell_line}: 张量维度错误。")

    y = np.asarray(y).reshape(-1)
    n_samples = len(y)

    if len(X_3d) != n_samples or len(X_2d) != n_samples or len(metadata) != n_samples:
        raise ValueError(f"{cell_line}: 样本数不一致: X_3d={len(X_3d)}, y={n_samples}, meta={len(metadata)}")


def load_cell_line(data_dir: str, cell_line: str, schema: Optional[Dict] = None) -> Dict:
    if schema is None:
        schema = load_feature_schema(data_dir)

    cell_line = str(cell_line).strip().lower()
    paths = get_feature_file_paths(data_dir=data_dir, cell_line=cell_line, schema=schema)

    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{cell_line} 缺少 {name}：{path}")

    X_3d = np.load(paths["X_3d"])
    X_2d = np.load(paths["X_2d"])
    y = np.load(paths["y"]).reshape(-1)
    metadata = pd.read_csv(paths["metadata"]).reset_index(drop=True)

    validate_cell_line_dataset(cell_line, X_3d, X_2d, y, metadata, schema)

    return {
        "cell_line": cell_line,
        "X_3d": X_3d,
        "X_2d": X_2d,
        "y": y,
        "metadata": metadata
    }


def load_all_cell_lines(data_dir: str, cell_lines: Optional[List[str]] = None, schema: Optional[Dict] = None) -> Dict[str, Dict]:
    if schema is None:
        schema = load_feature_schema(data_dir)

    if not cell_lines:
        cell_lines = discover_available_cell_lines(data_dir)

    normalized = [str(c).strip().lower() for c in cell_lines]
    datasets = {}
    for cell_line in normalized:
        datasets[cell_line] = load_cell_line(data_dir=data_dir, cell_line=cell_line, schema=schema)
    return datasets


def create_split_indices(n_samples: int, train_fraction: float = DEFAULT_TRAIN_FRACTION, validation_fraction: float = DEFAULT_VALIDATION_FRACTION, test_fraction: float = DEFAULT_TEST_FRACTION, random_seed: int = 42) -> Dict[str, np.ndarray]:
    validate_split_fractions(train_fraction, validation_fraction, test_fraction)
    if n_samples < 3:
        raise ValueError("至少需要 3 个样本才能划分数据集。")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_size = int(np.floor(n_samples * train_fraction))
    validation_size = int(np.floor(n_samples * validation_fraction))
    train_size = max(1, train_size)
    validation_size = max(1, validation_size)

    return {
        "train": indices[:train_size],
        "validation": indices[train_size:train_size + validation_size],
        "test": indices[train_size + validation_size:] if (train_size + validation_size) < n_samples else indices[train_size:]
    }


def create_train_valid_indices(n_samples: int, train_fraction: float = 0.85,
                               validation_fraction: float = 0.15,
                               random_seed: int = 42) -> Dict[str, np.ndarray]:
    """仅 train / validation 两路划分 (供 LOCO 使用: 留出细胞系整体作为 test)。

    与 create_split_indices 保持同一 RNG 语义 (`default_rng(seed).shuffle`) 与 floor 规则,
    但不走 validate_split_fractions —— 那里要求三段比例均 > 0, 而 LOCO 的 test 比例恒为 0,
    这正是原先 split_all_cell_lines 的留一分支必然抛错、从而永远无法生效的原因。
    """
    total = float(train_fraction) + float(validation_fraction)
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"train + validation 比例之和必须等于 1, 当前={total}")
    if n_samples < 2:
        raise ValueError("至少需要 2 个样本才能划分 train / validation。")

    rng = np.random.default_rng(random_seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    train_size = max(1, int(np.floor(n_samples * float(train_fraction))))
    validation_size = max(1, n_samples - train_size)
    return {
        "train": indices[:train_size],
        "validation": indices[train_size:train_size + validation_size],
    }


def _raw_sequence_ids(metadata: pd.DataFrame) -> np.ndarray:
    if "sgRNA" not in metadata.columns:
        raise KeyError("metadata 缺少 sgRNA 列，无法执行 group-aware split")
    return metadata["sgRNA"].astype(str).str.upper().str.strip().to_numpy()


def sequence_group_ids(metadata: pd.DataFrame, revcomp_canonical: bool = True) -> np.ndarray:
    """canonical sequence identity（P0 泄漏整改的 grouping key）。

    同一 sgRNA 序列在不同 cell line 出现时序列通道相同 → 若跨 split 分配，
    模型已见过该序列，违反 sequence-level generalization 声明。

    P0 整改 (L5 反向互补): revcomp_canonical=True 时以
    min(seq, revcomp(seq)) 作为身份。反向互补的两个 sgRNA 靶向同一 locus 的
    两条链, 属于近重复观测; 实测库内存在跨系 revcomp 对
    (hct116 ∩ rc(hela)=30, hela ∩ rc(hl60)=3), 只按精确序列分组会残留泄漏。
    """
    raw = _raw_sequence_ids(metadata)
    if not revcomp_canonical:
        return raw
    return np.array([min(seq, reverse_complement(seq)) for seq in raw])


def raw_sequence_ids(metadata: pd.DataFrame) -> np.ndarray:
    """原始 sgRNA 序列（仅做大小写/空白清理）。审计以此为准（诚实度量）。"""
    return _raw_sequence_ids(metadata)


_COMPLEMENT_TABLE = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(sequence: str) -> str:
    """反向互补序列。"""
    return str(sequence).upper().strip().translate(_COMPLEMENT_TABLE)[::-1]


def _identity_sets(metadata: pd.DataFrame) -> Dict[str, set]:
    """审计身份集合: 使用**原始**序列 (划分用 canonical group, 审计用 raw)。"""
    sequences = set(raw_sequence_ids(metadata)) if len(metadata) else set()
    loci = set()
    if {"Chromosome", "Start", "End", "Strand"}.issubset(set(metadata.columns)) and len(metadata):
        loci = set(zip(metadata["Chromosome"].astype(str),
                       metadata["Start"].astype(str),
                       metadata["End"].astype(str),
                       metadata["Strand"].astype(str)))
    return {"sequences": sequences, "loci": loci}


def split_identity_audit(split_data: Dict) -> Dict[str, int]:
    """在划分完成处直接审计 train/valid/test 的身份重叠。

    审计对象是**原始 sgRNA 序列**（划分使用 revcomp-canonical group）,
    因此这里的指标是诚实的泄漏度量, 而不是恒为 0 的同义反复。
    返回值全部为 int, 便于写入 run config / *_info.txt 并被解析。
    """
    tr = _identity_sets(split_data["train_data"]["metadata"])
    va = _identity_sets(split_data["valid_data"]["metadata"])
    te = _identity_sets(split_data["test_data"]["metadata"])

    train_seqs = tr["sequences"]
    audit = {
        "audit_train_test_sequence_overlap": len(train_seqs & te["sequences"]),
        "audit_train_valid_sequence_overlap": len(train_seqs & va["sequences"]),
        "audit_valid_test_sequence_overlap": len(va["sequences"] & te["sequences"]),
        "audit_unique_train_sequences": len(train_seqs),
        "audit_unique_valid_sequences": len(va["sequences"]),
        "audit_unique_test_sequences": len(te["sequences"]),
        "audit_train_test_locus_overlap": len(tr["loci"] & te["loci"]),
        "audit_train_test_revcomp_overlap": len(
            train_seqs & {reverse_complement(s) for s in te["sequences"]}
        ),
    }
    return audit


def assert_no_sequence_leakage(split_data: Dict, audit: Dict[str, int]) -> None:
    """group-aware 划分下, 序列级重叠必须为 0, 否则立即失败而不是产出泄漏结果。

    审计基于原始序列, 因此同时覆盖精确重复 (L1-L4) 与反向互补近重复 (L5)。
    """
    offenders = {k: v for k, v in audit.items()
                 if k.endswith("sequence_overlap") and v > 0}
    if offenders:
        raise RuntimeError(
            "group-aware split 仍然出现序列重叠, 拒绝继续训练: "
            + ", ".join(f"{k}={v}" for k, v in offenders.items())
        )


def group_aware_split_indices(groups: np.ndarray, train_fraction: float = 0.70,
                              validation_fraction: float = 0.15, test_fraction: float = 0.15,
                              random_seed: int = 42) -> Dict[str, np.ndarray]:
    """按 group（=sgRNA 序列）整体分配，保证同一序列不跨 train/valid/test。"""
    total = float(train_fraction) + float(validation_fraction) + float(test_fraction)
    if not np.isclose(total, 1.0, atol=1e-8):
        # 允许 LOCO 训练池的 train+valid=1, test=0（test 由留出细胞系整体充当）
        if not (float(test_fraction) == 0.0
                and np.isclose(float(train_fraction) + float(validation_fraction), 1.0, atol=1e-8)):
            raise ValueError(f"三段比例之和必须为 1，当前={total}")
    uniq = np.array(sorted(set(groups)))
    rng = np.random.default_rng(int(random_seed))
    perm = rng.permutation(len(uniq))
    n = len(uniq)
    n_tr = int(np.floor(n * float(train_fraction)))
    n_va = int(np.floor(n * float(validation_fraction)))
    g_tr = set(uniq[perm[:n_tr]])
    g_va = set(uniq[perm[n_tr:n_tr + n_va]])
    g_te = set(uniq[perm[n_tr + n_va:]])
    idx = np.arange(len(groups))
    return {"train": idx[np.isin(groups, list(g_tr))],
            "validation": idx[np.isin(groups, list(g_va))],
            "test": idx[np.isin(groups, list(g_te))]}


def slice_dataset(dataset: Dict, indices: np.ndarray) -> Dict:
    return {
        "X_3d": dataset["X_3d"][indices],
        "X_2d": dataset["X_2d"][indices],
        "y": dataset["y"][indices],
        "metadata": dataset["metadata"].iloc[indices].reset_index(drop=True)
    }


def merge_datasets(datasets: List[Dict]) -> Dict:
    if not datasets:
        raise ValueError(
            "merge_datasets 收到空列表：没有任何数据集可用于拼接。\n"
            "  常见原因：all(留一) 划分里 --cell-lines 只剩被留出的那一个，"
            "训练池为空。\n"
            "  请保证参与划分的数据集 >= 2 个，且被留出的那个也在 --cell-lines 中。"
        )
    return {
        "X_3d": np.concatenate([d["X_3d"] for d in datasets], axis=0),
        "X_2d": np.concatenate([d["X_2d"] for d in datasets], axis=0),
        "y": np.concatenate([d["y"] for d in datasets], axis=0),
        "metadata": pd.concat([d["metadata"] for d in datasets], axis=0, ignore_index=True)
    }


def split_single_cell_line(dataset: Dict, cell_line: str, train_fraction: float = DEFAULT_TRAIN_FRACTION, validation_fraction: float = DEFAULT_VALIDATION_FRACTION, test_fraction: float = DEFAULT_TEST_FRACTION, random_seed: int = 42, group_aware: bool = True) -> Dict:
    # P0 整改: 单细胞系内部同样按 sgRNA 序列分组, 使『同一序列不跨 split』
    # 成为三种 split 的统一契约 (当前数据每个细胞系内序列唯一, 数值上等价,
    # 但契约由构造保证而不是靠数据巧合)。
    if group_aware:
        groups = sequence_group_ids(dataset["metadata"])
        split_indices = group_aware_split_indices(groups, train_fraction, validation_fraction,
                                                  test_fraction, random_seed)
    else:
        split_indices = create_split_indices(len(dataset["y"]), train_fraction, validation_fraction, test_fraction, random_seed)
    return {
        "split_type": "single",
        "cell_line": cell_line,
        "random_seed": random_seed,
        "train_cell_lines": [cell_line],
        "validation_cell_lines": [cell_line],
        "test_cell_lines": [cell_line],
        "train_data": slice_dataset(dataset, split_indices["train"]),
        "valid_data": slice_dataset(dataset, split_indices["validation"]),
        "test_data": slice_dataset(dataset, split_indices["test"])
    }


def split_all_cell_lines(datasets: Dict[str, Dict], cell_lines: List[str], test_cell_line: Optional[str] = None, train_fraction: float = DEFAULT_TRAIN_FRACTION, validation_fraction: float = DEFAULT_VALIDATION_FRACTION, test_fraction: float = DEFAULT_TEST_FRACTION, random_seed: int = 42, group_aware: bool = True) -> Dict:
    """
    留一细胞系 (Leave-One-Out) 或 跨细胞系划分
    """
    if len(datasets) == 1 or len(cell_lines) <= 1:
        # 单一细胞系退化保护
        only_cl = list(datasets.keys())[0]
        return split_single_cell_line(datasets[only_cl], only_cl, train_fraction, validation_fraction, test_fraction, random_seed, group_aware=group_aware)

    if test_cell_line and test_cell_line.lower() in datasets:
        target_test = test_cell_line.lower()
        train_cls = [c for c in cell_lines if c != target_test]
        train_data = merge_datasets([datasets[c] for c in train_cls])
        test_data = datasets[target_test]

        # P0 整改 (A2): 训练池剔除 hold-out 系的全部序列，避免跨细胞系重复序列被训练见过；
        # 池内再按 sgRNA 分组做 85/15 划分（test 由留出系整体充当）。
        if group_aware:
            held_seqs = set(sequence_group_ids(test_data["metadata"]))
            pool_groups = sequence_group_ids(train_data["metadata"])
            keep = ~np.isin(pool_groups, list(held_seqs))
            n_dropped = int((~keep).sum())
            if not keep.any():
                raise ValueError(
                    f"LOCO 训练池被 hold-out（{target_test}）的序列全部掏空："
                    f"训练集剩余 0 条。\n"
                    "  说明两个数据集的 sgRNA 序列高度重叠，无法构成留一验证。\n"
                    "  请改用其它数据集组合，或不要对该组合使用 all 划分。"
                )
            train_data = slice_dataset(train_data, np.where(keep)[0])
            pool_groups = pool_groups[keep]
            idx = group_aware_split_indices(pool_groups, train_fraction=0.85,
                                            validation_fraction=0.15, test_fraction=0.0,
                                            random_seed=random_seed)
        else:
            n_dropped = 0
            idx = create_train_valid_indices(len(train_data["y"]), train_fraction=0.85,
                                             validation_fraction=0.15, random_seed=random_seed)
        return {
            "split_type": "all",
            "cell_line": target_test,
            "random_seed": random_seed,
            "train_cell_lines": train_cls,
            "validation_cell_lines": train_cls,
            "test_cell_lines": [target_test],
            "train_data": slice_dataset(train_data, idx["train"]),
            "valid_data": slice_dataset(train_data, idx["validation"]),
            "test_data": test_data,
            "heldout_sequences_excluded_from_train": int(n_dropped) if group_aware else 0
        }

    # 各自独立划分后合并
    split_results = {}
    for i, cl in enumerate(cell_lines):
        split_results[cl] = split_single_cell_line(datasets[cl], cl, train_fraction, validation_fraction, test_fraction, random_seed + i, group_aware=group_aware)

    return {
        "split_type": "all",
        "cell_line": cell_lines[0] if cell_lines else "none",
        "random_seed": random_seed,
        "train_cell_lines": list(cell_lines),
        "validation_cell_lines": list(cell_lines),
        "test_cell_lines": list(cell_lines),
        "train_data": merge_datasets([split_results[c]["train_data"] for c in cell_lines]),
        "valid_data": merge_datasets([split_results[c]["valid_data"] for c in cell_lines]),
        "test_data": merge_datasets([split_results[c]["test_data"] for c in cell_lines])
    }


def split_mixed_cell_lines(datasets: Dict[str, Dict], cell_lines: List[str], train_fraction: float = DEFAULT_TRAIN_FRACTION, validation_fraction: float = DEFAULT_VALIDATION_FRACTION, test_fraction: float = DEFAULT_TEST_FRACTION, random_seed: int = 42, group_aware: bool = True) -> Dict:
    if len(datasets) == 1:
        only_cl = list(datasets.keys())[0]
        return split_single_cell_line(datasets[only_cl], only_cl, train_fraction, validation_fraction, test_fraction, random_seed, group_aware=group_aware)

    merged = merge_datasets([datasets[c] for c in cell_lines if c in datasets])
    # P0 整改: 默认按 sgRNA 分组划分，避免跨细胞系重复序列同时进入 train 与 test
    # （旧行为 = 逐样本随机划分，实测 test∩train 21%）。group_aware=False 可复现旧口径。
    if group_aware:
        groups = sequence_group_ids(merged["metadata"])
        split_indices = group_aware_split_indices(groups, train_fraction, validation_fraction,
                                                  test_fraction, random_seed)
    else:
        split_indices = create_split_indices(len(merged["y"]), train_fraction, validation_fraction,
                                             test_fraction, random_seed)
    return {
        "split_type": "mixed",
        "cell_line": "none",
        "random_seed": random_seed,
        "train_cell_lines": list(cell_lines),
        "validation_cell_lines": list(cell_lines),
        "test_cell_lines": list(cell_lines),
        "train_data": slice_dataset(merged, split_indices["train"]),
        "valid_data": slice_dataset(merged, split_indices["validation"]),
        "test_data": slice_dataset(merged, split_indices["test"])
    }


def attach_public_fields(split_data: Dict) -> Dict:
    result = dict(split_data)
    result.update({
        "X_train_3d": split_data["train_data"]["X_3d"],
        "X_valid_3d": split_data["valid_data"]["X_3d"],
        "X_test_3d": split_data["test_data"]["X_3d"],
        "X_train_2d": split_data["train_data"]["X_2d"],
        "X_valid_2d": split_data["valid_data"]["X_2d"],
        "X_test_2d": split_data["test_data"]["X_2d"],
        "y_train": split_data["train_data"]["y"],
        "y_valid": split_data["valid_data"]["y"],
        "y_test": split_data["test_data"]["y"],
        "metadata_train": split_data["train_data"]["metadata"],
        "metadata_valid": split_data["valid_data"]["metadata"],
        "metadata_test": split_data["test_data"]["metadata"],
    })
    return result


def _split_digest(split_data: Dict) -> str:
    """train/valid/test 序列集合 + 样本数的 sha256 摘要 (16 位)。

    用于事后证明『分析所用的划分 == 训练所用的划分』, 无需保存完整索引。
    """
    payload = []
    for part in ("train", "valid", "test"):
        seqs = sorted(_identity_sets(split_data[f"{part}_data"]["metadata"])["sequences"])
        payload.append(f"{part}:{len(seqs)}:" + ",".join(seqs))
    return hashlib.sha256("|".join(payload).encode("utf-8")).hexdigest()[:16]


def divide_data(
    data_dir: str,
    split_type: str,
    cell_line: Optional[str] = None,
    cell_lines: Optional[List[str]] = None,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_seed: int = 42,
    group_aware: bool = True
) -> Dict:
    split_type = str(split_type).strip().lower()
    validate_split_fractions(train_fraction, validation_fraction, test_fraction)

    schema = load_feature_schema(data_dir)
    available_cells = discover_available_cell_lines(data_dir)

    if not cell_lines:
        cell_lines = available_cells

    requested = [str(c).strip().lower() for c in cell_lines if str(c).strip()]
    unknown = [c for c in requested if c not in available_cells]

    # 旧实现会把不存在的名字静默过滤掉；当全部被过滤掉时，all(LOCO) 会拿着空
    # cell_lines 重新 discover 全部数据集，从而**静默忽略调用方指定的 held-out**，
    # single 则直接 IndexError。两种都让错误原因难以定位，这里一律显式报错。
    if unknown:
        raise ValueError(
            f"以下数据集在 {data_dir} 中不存在：{unknown}\n"
            f"  可用：{available_cells}\n"
            "  请检查 --cell-lines / cell_lines 参数（不会静默忽略）。"
        )

    cell_lines = requested

    if not cell_lines:
        raise ValueError(
            f"cell_lines 为空且 {data_dir} 中没有可用数据集。"
        )

    if split_type == "single":
        if cell_line is None and not cell_lines:
            raise ValueError(
                "single 划分需要 cell_line 或至少一个 cell_lines 条目。"
            )
        target_cell = str(cell_line).strip().lower() if cell_line else cell_lines[0]
        if target_cell not in available_cells:
            raise ValueError(
                f"cell_line={target_cell!r} 不在 {data_dir} 中；可用：{available_cells}"
            )
        dataset = load_cell_line(data_dir=data_dir, cell_line=target_cell, schema=schema)
        result = split_single_cell_line(dataset, target_cell, train_fraction, validation_fraction, test_fraction, random_seed, group_aware=group_aware)
    else:
        datasets = load_all_cell_lines(data_dir=data_dir, cell_lines=cell_lines, schema=schema)
        if split_type == "all":
            result = split_all_cell_lines(datasets, cell_lines, test_cell_line=cell_line, train_fraction=train_fraction, validation_fraction=validation_fraction, test_fraction=test_fraction, random_seed=random_seed, group_aware=group_aware)
        else:
            result = split_mixed_cell_lines(datasets, cell_lines, train_fraction=train_fraction, validation_fraction=validation_fraction, test_fraction=test_fraction, random_seed=random_seed, group_aware=group_aware)

    result["schema"] = schema
    result["train_fraction"] = train_fraction
    result["validation_fraction"] = validation_fraction
    result["test_fraction"] = test_fraction
    result["group_aware"] = bool(group_aware)
    result["n_train"] = len(result["train_data"]["y"])
    result["n_valid"] = len(result["valid_data"]["y"])
    result["n_test"] = len(result["test_data"]["y"])

    # P0 整改: 在划分处直接审计身份重叠并写入 run metadata;
    # group_aware 下重叠必须为 0, 否则抛错 (不产出泄漏结果)。
    audit = split_identity_audit(result)
    result.update(audit)
    if group_aware:
        assert_no_sequence_leakage(result, audit)
    result["split_digest"] = _split_digest(result)

    return attach_public_fields(result)
