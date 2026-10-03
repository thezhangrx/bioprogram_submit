# data_digging.py
"""
CRISPR-Cas9 已测数据训练挖掘引擎 (Training Scope 网格实验)
================================================================
从 predict.py 拆分 (职责单一化):
    data_digging.py  已测数据集"深入挖掘"网格实验 (引导程序第4步之 2)
                     对已测数据按所选表观特征(Training Scope)组合 -> 训练全网格,
                     输出到 results/[batch]/ 与 logs/、models/。
    predict.py       只做 mixed 十折交叉验证 + 目标数据集预测
                     (引导程序第4步之 3, Target Epigenetics)。
用法:
    python data_digging.py --batch-name <b> --models linear xgboost \
        --cell-lines hct116 hela --split-types single all mixed \
        --environments sequence sequence_ctcf ...          # 显式环境组合
    python data_digging.py --training-scope-epis ctcf dnase h3k4me3 rrbs   # 由第4步选项2展开
    python data_digging.py --dry-run ...                                   # 只打印实验计划
"""

from __future__ import annotations


# --- 项目根引导: 保证从任意工作目录运行/被导入都能解析 core、analysis、workflows ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.common.paths import add_dataset_arguments, resolve_data_dir


def build_training_scope_combinations(active_epis: List[str]) -> List[str]:
    """由第4步选项2(Training Scope)勾选的表观特征展开为网格环境组合。

    与 Input/backend_runner.build_active_environment_combinations 同规则:
    sequence 基线 + 各非空子集 + 全4项时补 'all'。
    """
    active = [e.lower().strip() for e in active_epis if e and str(e).strip()]
    if not active:
        return ["sequence"]
    combos = ["sequence"]
    for r in range(1, len(active) + 1):
        for subset in itertools.combinations(active, r):
            combos.append("sequence_" + "_".join(sorted(subset)))
    if set(active) == {"ctcf", "dnase", "h3k4me3", "rrbs"} and "all" not in combos:
        combos.append("all")
    return combos


MODELS = ["linear", "xgboost", "mlp", "transformer"]
CNN_MODELS = ["cnn"]
ALL_MODELS = [*MODELS, "cnn"]
# 数据集/细胞系列表**不再硬编码**：由 --data-dir 下实际发现的文件决定
# (core.data.cell_line_division.discover_available_cell_lines)。
MIXED_SEEDS = [42, 43, 44, 45]
CNN_KERNELS = [3, 5, 7]

# 新增可调超参的默认值, 必须与 train.py 的 DEFAULT_* 一致。
# build_command 只在取值偏离这些默认值时才追加对应 flag, 因此默认路径
# 生成的命令行与改造前**逐字节相同** —— 这是 results/batches/ultimate_run
# (1344 次运行) 可复现性的前提。
DEFAULT_OPTIMIZER = "adam"          # 改造前硬编码 torch.optim.Adam
DEFAULT_SCHEDULER = "none"          # 改造前不存在任何学习率调度器
DEFAULT_ACTIVATION = None           # None = 沿用各模型原有激活 (CNN/MLP=ReLU, Transformer=GELU)
DEFAULT_NUM_WORKERS = 0             # 改造前 DataLoader 取模型默认值 0
DEFAULT_GPU_ID = None               # 不覆盖 CUDA_VISIBLE_DEVICES
OPTIMIZER_CHOICES = ["adam", "adamw", "sgd", "rmsprop", "adagrad"]
SCHEDULER_CHOICES = ["none", "cosine", "step", "exponential", "plateau"]
ACTIVATION_CHOICES = ["none", "relu", "gelu", "tanh", "sigmoid", "leaky_relu", "elu", "silu"]


def _canonicalize_combinations(combos: List[str]) -> List[str]:
    """归一化环境组合: “全部表观特征”的组合统一只保留 `all`。

    个别来源/手写列表可能同时带 `sequence_ctcf_dnase_h3k4me3_rrbs`(显式全特征)
    与 `all`(同一含义) -> 造成环境组合重复、实验翻倍(1428 = 17×84)。
    规则: 只要列表中含 `all`, 就丢弃与之等价的显式全特征名; 否则保留原样。
    """
    combos = list(combos)
    if "all" not in combos:
        return combos
    # 全集 = 所有组合里出现过的表观 token 的并集 (含单/双/三元 -> 恒为 4 个)
    universe: set = set()
    for c in combos:
        if c == "all":
            continue
        universe.update(t for t in str(c).split("_") if t != "sequence" and t)
    if not universe:
        return combos
    out = []
    removed = 0
    for c in combos:
        if c == "all":
            out.append(c)
            continue
        toks = {t for t in str(c).split("_") if t != "sequence" and t}
        if toks == universe:
            removed += 1   # 显式全特征组合 == all, 丢弃
            continue
        out.append(c)
    if removed:
        print(f"[Environments] 检测到 {removed} 个与 `all` 等价的显式全特征组合, 已并入 `all`")
    return out


def load_environment_combinations(data_dir: str) -> List[str]:
    """由该数据集的 feature_schema.json 展开环境组合。

    不再在 schema 缺失时回退到"DeepCRISPR 的 16 种组合"——那会让一个没有表观
    通道的数据集被排上 15 个根本不存在输入列的实验。schema 缺失直接报错。
    """
    schema_path = Path(data_dir) / "feature_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"找不到 {schema_path}，无法确定该数据集支持哪些环境组合。\n"
            "  请先跑特征工程生成该目录，或显式用 --environments 指定组合。"
        )

    from core.features.channels.cell_environment_combination import generate_combination_names

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    combos = list(generate_combination_names(
        schema, include_all=True, include_sequence=True, sizes=[0, 1, 2, 3]))

    if not combos:
        raise ValueError(
            f"{schema_path} 未展开出任何环境组合；channel_names={schema.get('channel_names')}"
        )

    return combos


Experiment = Tuple[str, str, str, Optional[str], Optional[int], Optional[int]]


def sanitize_batch_name(batch_name: str) -> str:
    return str(batch_name).strip().lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def generate_experiments(
    environments: List[str],
    selected_models: Optional[List[str]] = None,
    selected_cells: Optional[List[str]] = None,
    selected_splits: Optional[List[str]] = None,
    selected_kernels: Optional[List[int]] = None,
    available_cells: Optional[List[str]] = None
) -> List[Experiment]:
    """展开网格。

    ``available_cells`` 由数据目录实际内容决定（缺失时不再回退到 DeepCRISPR 的
    4 个细胞系常量，避免对其它数据集排出一堆不存在的实验）。
    """
    experiments = []
    models_to_run = [m for m in (selected_models or ALL_MODELS) if m != "cnn"]
    include_cnn = "cnn" in (selected_models or ALL_MODELS)
    cells_to_run = list(selected_cells or available_cells or [])
    splits_to_run = [s.lower() for s in (selected_splits or ["single", "all", "mixed"])]
    # 修复: --cnn-kernels 之前被解析但从未使用, 导致无论怎么传都跑 3/5/7 三个 kernel。
    kernels_to_run = [int(k) for k in (selected_kernels or CNN_KERNELS)]

    # 留一细胞系(LOCO) 至少需要 2 个细胞系; 只选 1 个时会走 cell_line_division 的
    # "单一细胞系退化保护", all 与 single 变成同一次实验。这里显式告警, 避免误跑。
    if "mixed" in splits_to_run and len(cells_to_run) < 2:
        print("[WARN] split_types 含 'mixed', 但只选中 1 个数据集 "
              f"({cells_to_run}) -> mixed 会静默退化为单数据集划分, "
              "但 run 名仍为 mixed_*（目录标签与真实划分不符）。\n"
              "       如需真正的 mixed 划分, 请用 --cell-lines 指定 >=2 个。")

    if "all" in splits_to_run and len(cells_to_run) < 2:
        print("[WARN] split_types 含 'all'(留一细胞系/数据集), 但只选中 1 个 "
              f"({cells_to_run}) -> LOCO 无法成立, all 会退化为单数据集划分。\n"
              "       请用 --cell-lines 指定 >=2 个（该数据目录下可用的有："
              f"{', '.join(available_cells or []) or '未知'}）。")

    for model in models_to_run:
        for environment in environments:
            if "single" in splits_to_run:
                for cell_line in cells_to_run:
                    experiments.append((model, environment, "single", cell_line, None, None))
            if "all" in splits_to_run:
                for cell_line in cells_to_run:
                    experiments.append((model, environment, "all", cell_line, None, None))
            if "mixed" in splits_to_run:
                for seed in MIXED_SEEDS:
                    experiments.append((model, environment, "mixed", None, seed, None))

    if include_cnn:
        for environment in environments:
            if "single" in splits_to_run:
                for cell_line in cells_to_run:
                    for kernel in kernels_to_run:
                        experiments.append(("cnn", environment, "single", cell_line, None, kernel))
            if "all" in splits_to_run:
                for cell_line in cells_to_run:
                    for kernel in kernels_to_run:
                        experiments.append(("cnn", environment, "all", cell_line, None, kernel))
            if "mixed" in splits_to_run:
                for seed in MIXED_SEEDS:
                    for kernel in kernels_to_run:
                        experiments.append(("cnn", environment, "mixed", None, seed, kernel))

    return experiments


def build_run_name(experiment: Experiment) -> str:
    (model, environment, split_type, cell_line, seed, kernel) = experiment
    if split_type == "single":
        name = f"single_{cell_line}_{model}_{environment}"
    elif split_type == "all":
        name = f"all_{model}_{environment}_heldout_{cell_line}"
    elif split_type == "mixed":
        name = f"mixed_{model}_{environment}_seed_{seed}"
    else:
        name = f"{split_type}_{model}_{environment}"

    if model == "cnn" and kernel:
        name += f"_kernel_{kernel}"
    return name


def parse_folder_info(folder: Path) -> Optional[dict]:
    info_files = list(folder.glob("*info*.txt"))
    metric_files = list(folder.glob("*metrics*.json"))
    if not info_files or not metric_files: return None

    info = {}
    with open(info_files[0], "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                k, v = line.strip().split(":", 1)
                info[k.strip().lower()] = v.strip().lower()
    return info


def build_completed_lookup(results_batch_dir: Path) -> dict:
    lookup = {}
    if not results_batch_dir.exists(): return lookup

    for run_dir in results_batch_dir.iterdir():
        if not run_dir.is_dir() or run_dir.name == "summary": continue
        info = parse_folder_info(run_dir)
        if not info: continue

        model = info.get("model", "")
        if "linear" in model: model = "linear"
        elif "xgb" in model: model = "xgboost"
        elif "mlp" in model: model = "mlp"
        elif "cnn" in model: model = "cnn"
        elif "trans" in model: model = "transformer"

        env = info.get("environment", info.get("combination", ""))
        split_type = info.get("split_type", "")
        cell_line = info.get("cell_line", info.get("held_out_cell_line", ""))
        if split_type == "mixed": cell_line = None
        elif cell_line in ["none", "", "unknown"]: cell_line = None

        seed_str = info.get("random_seed", info.get("seed", "42"))
        try: seed = int(seed_str)
        except Exception: seed = 42

        kernel_str = info.get("sequence_kernel", info.get("kernel", "3"))
        try: kernel = int(kernel_str)
        except Exception: kernel = 3

        key = (model, env, split_type, cell_line, seed if split_type == "mixed" else None, kernel if model == "cnn" else None)
        lookup[key] = run_dir
    return lookup


def classify_experiments(experiments: List[Experiment], results_batch_dir: Path):
    completed_lookup = build_completed_lookup(results_batch_dir)
    completed = []
    pending = []

    for exp in experiments:
        (model, environment, split_type, cell_line, seed, kernel) = exp
        norm_cell = str(cell_line).lower() if cell_line else None
        norm_env = str(environment).lower()
        key = (model, norm_env, split_type, norm_cell, seed if split_type == "mixed" else None, kernel if model == "cnn" else None)

        if key in completed_lookup:
            completed.append((exp, completed_lookup[key]))
        else:
            pending.append(exp)
    return completed, pending


def build_command(
    experiment: Experiment,
    batch_name: str,
    data_dir: str,
    model_dir: str,
    results_dir: str,
    logs_dir: str,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    use_scaler: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    hidden_dim1: int,
    hidden_dim2: int,
    conv_channels1: int,
    conv_channels2: int,
    device: Optional[str],
    loco_cells: Optional[List[str]] = None,
    optimizer: str = DEFAULT_OPTIMIZER,
    scheduler: str = DEFAULT_SCHEDULER,
    activation: Optional[str] = DEFAULT_ACTIVATION,
    num_workers: int = DEFAULT_NUM_WORKERS,
    gpu_id: Optional[str] = DEFAULT_GPU_ID,
) -> List[str]:
    (model, environment, split_type, cell_line, seed, kernel) = experiment
    random_seed = seed if split_type == "mixed" else 42
    run_name = build_run_name(experiment)

    # 关键：调用 train.py (原 main.py)
    train_script = str(Path(__file__).resolve().parent / "train.py")

    command = [
        sys.executable, train_script,
        "--model", model,
        "--split-type", split_type,
        "--environment", environment,
        "--batch-name", batch_name,
        "--run-name", run_name,
        "--data-dir", data_dir,
        "--model-dir", model_dir,
        "--results-dir", results_dir,
        "--logs-dir", logs_dir,
        "--train-ratio", str(train_ratio),
        "--valid-ratio", str(valid_ratio),
        "--test-ratio", str(test_ratio),
        "--seed", str(random_seed),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--learning-rate", str(learning_rate),
        "--dropout", str(dropout),
        "--weight-decay", str(weight_decay),
        "--patience", str(patience),
        "--min-delta", str(min_delta),
        "--hidden-dim1", str(hidden_dim1),
        "--hidden-dim2", str(hidden_dim2),
        "--conv-channels1", str(conv_channels1),
        "--conv-channels2", str(conv_channels2),
    ]

    if cell_line is not None:
        command.extend(["--cell-line", cell_line])

    # LOCO 修复: split_type == "all" 时必须同时给出完整的细胞系集合,
    # 否则 train.py 会把 cell_lines 构造成 [cell_line] 单个元素,
    # divide_data 只加载 1 个数据集, 触发 cell_line_division 的
    # "单一细胞系退化保护" -> all 退化成 single (见 docs/statistics_and_parameters_zh.md §6 发现 2)。
    # 这里传全量集合, 让 split_all_cell_lines 走真正的留一分支:
    #   train/valid = 其余 3 个细胞系合并后按 0.85/0.15 划分, test = 留出细胞系全部样本。
    #
    # H1 可复现性修复: mixed 也显式传 --cell-lines。此前 mixed 依赖
    # discover_available_cell_lines 的文件系统 readdir 顺序决定行拼接顺序,
    # 本地与超算可能不同 -> group-aware 划分结果漂移。显式顺序保证跨机器一致。
    if split_type in ("all", "mixed") and loco_cells:
        command.extend(["--cell-lines", *[str(c) for c in loco_cells]])

    if model == "cnn":
        command.extend(["--sequence-kernel", str(kernel), "--environment-kernel", "3"])

    if device is not None:
        command.extend(["--device", device])

    # 新增可调超参: 仅当取值偏离默认值时才追加 flag。
    # 默认路径不加任何新 flag -> 命令行与改造前逐字节相同 (原批次可复现);
    # 用户显式设置时, 值被原样透传给 train.py 对应参数。
    if optimizer is not None and str(optimizer).strip().lower() != DEFAULT_OPTIMIZER:
        command.extend(["--optimizer", str(optimizer)])
    if scheduler is not None and str(scheduler).strip().lower() != DEFAULT_SCHEDULER:
        command.extend(["--scheduler", str(scheduler)])
    # activation 的 "无指定" 有两种等价写法: None 与 "none"/"default"
    if activation is not None and str(activation).strip().lower() not in ("none", "default"):
        command.extend(["--activation", str(activation)])
    if num_workers and int(num_workers) != DEFAULT_NUM_WORKERS:
        command.extend(["--num-workers", str(int(num_workers))])
    if gpu_id is not None:
        command.extend(["--gpu-id", str(gpu_id)])

    if use_scaler:
        command.append("--use-scaler")

    return command


def run_one_experiment(experiment: Experiment, index: int, total: int, env: Optional[Dict[str, str]] = None,
                       **kwargs) -> Dict:
    (model, environment, split_type, cell_line, seed, kernel) = experiment
    command = build_command(experiment=experiment, **kwargs)
    run_name = build_run_name(experiment)

    print(f"\n[Experiment {index}/{total}] Starting: {run_name}")
    try:
        result = subprocess.run(command, check=False, env=env)
        status = "success" if result.returncode == 0 else "failed"
    except Exception as error:
        print(f"[!] Execution failed: {error}")
        return {"model": model, "run_name": run_name, "return_code": -1, "status": "failed"}

    return {"model": model, "run_name": run_name, "return_code": result.returncode, "status": status}


def run_experiment_in_process(experiment: Experiment, index: int, total: int, **kwargs) -> Dict:
    """
    进程内执行单实验: 复用当前 python 解释器调用 train.main(),
    argv 与子进程路径完全一致 (消除每实验一次 python 启动/import 开销)。
    仅限顺序执行使用 (不得与多线程并发混用, 因 sys.argv 为进程全局状态)。
    """
    (model, environment, split_type, cell_line, seed, kernel) = experiment
    command = build_command(experiment=experiment, **kwargs)
    run_name = build_run_name(experiment)
    print(f"\n[Experiment {index}/{total}] Starting (in-process): {run_name}")

    old_argv = sys.argv
    sys.argv = [str(command[1])] + [str(c) for c in command[2:]]
    try:
        import train as train_mod
        train_mod.main()
        return {"model": model, "run_name": run_name, "return_code": 0, "status": "success"}
    except Exception as error:
        print(f"[!] In-process execution failed: {error}")
        return {"model": model, "run_name": run_name, "return_code": -1, "status": "failed"}
    finally:
        sys.argv = old_argv


def detect_gpu_ids() -> List[str]:
    """探测本节点可用 GPU id 列表 (不 import torch, 避免父进程加载深度学习框架)。

    优先级: 已设置的 CUDA_VISIBLE_DEVICES > nvidia-smi > 空 (纯 CPU 节点)。
    """
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset is not None and preset.strip() != "":
        return [x.strip() for x in preset.split(",") if x.strip() != ""]
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15
        )
        if proc.returncode == 0:
            return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    except Exception:
        pass
    return []


def resolve_gpu_binding(requested: Optional[List[str]], workers: int) -> List[str]:
    """决定并发 worker 的 GPU 绑定列表。

    requested is None -> 自动探测 (nvidia-smi / 已设的 CUDA_VISIBLE_DEVICES)
    requested == []   -> 显式关闭绑定
    requested 非空    -> 使用给定 id 列表
    """
    if requested is not None:
        return [str(x) for x in requested if str(x).strip() != ""]
    detected = detect_gpu_ids()
    if int(workers) <= 1 or len(detected) <= 1:
        return []
    return detected


def build_worker_env(workers: int, per_worker_threads: int = 0,
                     gpu_ids: Optional[List[str]] = None,
                     worker_slot: int = 0) -> Optional[Dict[str, str]]:
    """
    并发执行时的子进程环境 (Phase-9 调优接口 + 多卡绑定)。

    线程: 默认 (per_worker_threads<=0): 不封顶线程, 子进程环境与串行完全一致 -> 数值结果逐位一致。
    显式指定 >0 时按『每实验线程数』封顶 OMP/MKL/BLAS, 可提升并发吞吐, 但会改变
    线性代数/归约的浮点求和顺序 -> 可能引入 ~1e-4 级数值漂移 (病态线性 'all' 实验更敏感),
    需用户显式接受后才启用。

    多卡: 传入 gpu_ids 时, 第 worker_slot 个并发槽绑定 gpu_ids[worker_slot % len]。
    没有这一步, N 个 worker 会全部挤在 0 号卡上 (8×A100 节点上会浪费 7 张卡并可能 OOM)。
    绑定只改变"用哪张卡", 不改变数值结果 (同一型号卡), 且被记录进 run 元数据
    (env_fingerprint 的 cvd= 字段)。
    """
    if int(workers) <= 1:
        return None
    env = os.environ.copy()
    changed = False
    if int(per_worker_threads) > 0:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            env[var] = str(int(per_worker_threads))
        changed = True
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[int(worker_slot) % len(gpu_ids)])
        changed = True
    return env if changed else None


# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="CRISPR 已测数据训练挖掘引擎 (Training Scope 网格实验). "
                    "由 predict.py 拆分而来: 本程序不再负责 mixed 十折/候选预测 (见 predict.py).")
    parser.add_argument("--batch-name", default="", type=str, help="批次名称，为空时直接存放在根目录")
    add_dataset_arguments(parser)
    parser.add_argument("--model-dir", type=str, default=None,
                        help="模型输出根目录（默认 models/<数据集>）")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="训练产物根目录（默认 results/train_results/<数据集>）")
    parser.add_argument("--logs-dir", type=str, default=None,
                        help="日志根目录（默认 logs/<数据集>）")

    # 第4步选项2接口: 由勾选的表观特征展开网格组合; 或显式 --environments
    parser.add_argument("--training-scope-epis", nargs="+", default=None,
                        help="向导第4步选项2 (Training Scope): 已测数据训练要深入挖掘的表观特征, "
                             "据此自动展开全部环境组合")
    parser.add_argument("--environments", nargs="+", default=None,
                        help="显式环境组合列表 (与 --training-scope-epis 二选一)")

    parser.add_argument("--models", nargs="+", default=None, choices=ALL_MODELS)
    parser.add_argument("--cell-lines", nargs="+", default=None,
                        help="要跑的数据集/细胞系；缺省=该 --data-dir 下实际发现的全部。"
                             "不再限制为 DeepCRISPR 的 4 个。")
    parser.add_argument("--split-types", nargs="+", default=["single", "all", "mixed"], choices=["single", "all", "mixed"])
    parser.add_argument("--mixed-seeds", nargs="+", type=int, default=MIXED_SEEDS)
    parser.add_argument("--cnn-kernels", nargs="+", type=int, choices=CNN_KERNELS, default=CNN_KERNELS)

    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-6)

    parser.add_argument("--hidden-dim1", type=int, default=128)
    parser.add_argument("--hidden-dim2", type=int, default=64)
    parser.add_argument("--conv-channels1", type=int, default=32)
    parser.add_argument("--conv-channels2", type=int, default=64)

    parser.add_argument("--device", type=str, default=None)

    # ---- 新增可调超参 (透传给 train.py; 默认值 = 改造前行为) ----
    parser.add_argument("--optimizer", type=str, choices=OPTIMIZER_CHOICES, default=DEFAULT_OPTIMIZER,
                        help=f"优化器 (仅 mlp/cnn/transformer 生效), 透传 train.py --optimizer; "
                             f"默认 {DEFAULT_OPTIMIZER} = 改造前的 Adam")
    parser.add_argument("--scheduler", type=str, choices=SCHEDULER_CHOICES, default=DEFAULT_SCHEDULER,
                        help=f"学习率调度器, 透传 train.py --scheduler; "
                             f"默认 {DEFAULT_SCHEDULER} = 不创建调度器 (与改造前一致)")
    parser.add_argument("--activation", type=str, choices=ACTIVATION_CHOICES, default=DEFAULT_ACTIVATION,
                        help="隐藏层激活, 透传 train.py --activation; 缺省/none = 沿用各模型原有激活")
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help=f"DataLoader num_workers, 透传 train.py --num-workers; 默认 {DEFAULT_NUM_WORKERS}")
    parser.add_argument("--gpu-id", type=str, default=DEFAULT_GPU_ID,
                        help="每个实验进程绑定的物理 GPU (设置 CUDA_VISIBLE_DEVICES), "
                             "透传 train.py --gpu-id; 缺省=不改该环境变量。"
                             "多卡并发时请注意: 它与 --gpus 的 worker 轮转绑定是两套机制")
    parser.add_argument("--use-scaler", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="实验级并发数 (workers=1 时保持原有串行子进程执行, 结果逐位一致)")
    parser.add_argument("--in-process", action="store_true",
                        help="进程内顺序执行实验 (复用解释器, 省去每实验一次 python 启动/import; "
                             "仅建议与 workers=1 同用)")
    parser.add_argument("--gpus", nargs="*", default=None,
                        help="绑定给并发 worker 的 GPU id 列表 (默认自动探测 nvidia-smi; "
                             "传 --gpus 且不给值可关闭绑定)。worker i 使用 gpus[i %% len(gpus)]。")
    parser.add_argument("--threads-per-worker", type=int, default=0,
                        help="并发时每实验 CPU 线程数封顶 (0=不封顶; 封顶可能改变浮点结果, 默认不启用)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # 输出目录默认值与交付结构一致：models/<数据集> · results/train_results/<数据集> · logs/<数据集>
    # （用 --data-dir 时取该目录名作为 <数据集>；显式传入的 --model-dir/--results-dir/--logs-dir 优先）
    dataset_tag = args.data_set or Path(args.data_dir or "").name
    args.model_dir = args.model_dir or f"models/{dataset_tag}"
    args.results_dir = args.results_dir or f"results/train_results/{dataset_tag}"
    args.logs_dir = args.logs_dir or f"logs/{dataset_tag}"
    return args


def main():
    args = parse_args()

    # 数据集解析：--data-set <名称> 或 --data-dir <路径>（二选一，必填）
    args.data_dir = str(resolve_data_dir(args.data_dir, args.data_set))
    print(f"[Dataset] {getattr(args, 'data_set', None) or '(由 --data-dir 指定)'} -> {args.data_dir}")

    # 展开/解析环境组合 (向导第4步选项2: Training Scope)
    if args.training_scope_epis is not None:
        environments = build_training_scope_combinations(args.training_scope_epis)
        print(f"[Training Scope] 由表观特征 {args.training_scope_epis} 展开环境组合...")
    else:
        environments = args.environments if args.environments is not None \
            else load_environment_combinations(args.data_dir)
    environments = _canonicalize_combinations(environments)
    print(f"[Environments] 共 {len(environments)} 种: {', '.join(environments)}")

    from core.data.cell_line_division import discover_available_cell_lines
    available_cells = discover_available_cell_lines(args.data_dir)
    print(f"[Datasets] {args.data_dir} 下发现 {len(available_cells)} 个: "
          f"{', '.join(available_cells)}")

    if args.cell_lines:
        unknown = [c for c in args.cell_lines if c.lower() not in available_cells]
        if unknown:
            raise SystemExit(
                f"[Error] --cell-lines 中有该数据目录下不存在的数据集：{unknown}\n"
                f"        可用：{available_cells}"
            )

    batch_name = sanitize_batch_name(args.batch_name) if args.batch_name else ""
    experiments = generate_experiments(environments=environments,
                                       selected_models=args.models,
                                       selected_cells=args.cell_lines,
                                       selected_splits=args.split_types,
                                       selected_kernels=args.cnn_kernels,
                                       available_cells=available_cells)

    results_batch_dir = Path(args.results_dir) / batch_name if batch_name else Path(args.results_dir)
    completed, pending = classify_experiments(experiments=experiments, results_batch_dir=results_batch_dir)

    print(f"\n{'='*70}\nExperiment Plan -> Total: {len(experiments)} | "
          f"Completed: {len(completed)} | Pending: {len(pending)}\n{'='*70}")

    if args.dry_run:
        print("[*] Dry run completed.")
        return

    common_kwargs = {
        "batch_name": batch_name,
        "data_dir": args.data_dir,
        "model_dir": args.model_dir,
        "results_dir": args.results_dir,
        "logs_dir": args.logs_dir,
        "train_ratio": args.train_ratio,
        "valid_ratio": args.valid_ratio,
        "test_ratio": args.test_ratio,
        "use_scaler": args.use_scaler,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "min_delta": args.min_delta,
        "hidden_dim1": args.hidden_dim1,
        "hidden_dim2": args.hidden_dim2,
        "conv_channels1": args.conv_channels1,
        "conv_channels2": args.conv_channels2,
        "device": args.device,
        "optimizer": args.optimizer,
        "scheduler": args.scheduler,
        "activation": args.activation,
        "num_workers": args.num_workers,
        "gpu_id": args.gpu_id,
        "loco_cells": list(args.cell_lines) if args.cell_lines else list(available_cells),
    }

    if pending:
        total_pending = len(pending)
        n_workers = int(args.workers or 1)

        if args.in_process:
            if n_workers > 1:
                print("[Note] --in-process 与 --workers>1 不同时使用; 已忽略 workers, 采用顺序执行。")
            for idx, exp in enumerate(pending, start=1):
                run_experiment_in_process(exp, idx, total_pending, **common_kwargs)
        elif n_workers > 1:
            gpu_ids = resolve_gpu_binding(args.gpus, n_workers)
            if gpu_ids:
                print(f"[GPU] 并发绑定: worker i -> GPU {gpu_ids} (i mod {len(gpu_ids)})")
            else:
                print("[GPU] 未探测到 GPU 或已关闭绑定 (纯 CPU 节点)")
            with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="exp") as pool:
                futures = []
                for slot, (idx, exp) in enumerate(enumerate(pending, start=1)):
                    worker_env = build_worker_env(
                        n_workers, int(args.threads_per_worker or 0),
                        gpu_ids=gpu_ids, worker_slot=slot % n_workers)
                    futures.append(pool.submit(run_one_experiment, exp, idx, total_pending,
                                               env=worker_env, **common_kwargs))
                results = [f.result() for f in futures]
            n_ok = sum(1 for r in results if r.get("status") == "success")
            n_fail = len(results) - n_ok
            print(f"\n[✓] Batch executed with workers={n_workers}: success={n_ok}, failed={n_fail}")
        else:
            for idx, exp in enumerate(pending, start=1):
                run_one_experiment(exp, idx, total_pending, **common_kwargs)

        print("\n[✓] All batch experiments executed.")
    else:
        print("\n[✓] All experiments already completed!")


if __name__ == "__main__":
    main()
