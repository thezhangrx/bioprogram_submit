# Submit/analysis/collect_results.py
"""
CRISPR-Cas9 sgRNA prediction result collector
============================================
输出重定向：保存在 results/[batch_name]/summary/train_data/ 下
"""

from __future__ import annotations


# --- 项目根引导: 保证从任意工作目录运行/被导入都能解析 core、analysis、workflows ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import ast
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


MODEL_ORDER = [
    "linear",
    "xgboost",
    "mlp",
    "cnn(3|3)",
    "cnn(5|3)",
    "cnn(7|3)",
    "transformer"
]

SPLIT_ORDER = {
    "single": 0,
    "all": 1,
    "mixed": 2
}


def parse_info_txt(path: Path) -> Dict:
    info = {}
    if not path.exists():
        return info

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue

            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value.startswith("[") and value.endswith("]"):
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    pass
            elif value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            else:
                try:
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)
                except Exception:
                    pass

            info[key] = value

    return info


def build_model_name(info: Dict) -> str:
    model = str(info.get("model", "")).strip().lower()

    if model in ["linear_regression", "linear"]:
        return "linear"

    if "cnn" in model:
        seq_shape = info.get("sequence_kernel_shape")
        env_shape = info.get("environment_kernel_shape")
        if seq_shape and isinstance(seq_shape, list) and len(seq_shape) > 0:
            seq_k = seq_shape[0]
        else:
            seq_k = info.get("sequence_kernel", 3)

        if env_shape and isinstance(env_shape, list) and len(env_shape) > 0:
            env_k = env_shape[0]
        else:
            env_k = info.get("environment_kernel", 3)

        return f"cnn({seq_k}|{env_k})"

    return model


def load_json(path: Optional[Path]) -> Dict:
    if path is None or not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_file(folder: Path, keyword: str, exclude: Optional[str] = None) -> Optional[Path]:
    files = list(folder.glob(f"*{keyword}*"))
    if exclude:
        files = [f for f in files if exclude not in f.name]
    return files[0] if files else None


def collect_one_experiment(folder: Path) -> Dict:
    info_file = find_file(folder, "_info.txt")
    txt_info = parse_info_txt(info_file) if info_file else {}

    data = {}
    data["run_name"] = folder.name
    data.update(txt_info)

    split_type = str(data.get("split_type", "unknown")).lower()
    data["split_type"] = split_type

    raw_cell = data.get("cell_line", "")
    if split_type == "mixed":
        data["cell_line"] = "none"
    else:
        data["cell_line"] = str(raw_cell).lower() if pd.notna(raw_cell) and raw_cell else "unknown"

    data["environment"] = str(data.get("environment", data.get("combination", "unknown"))).lower()
    model_name = build_model_name(data)
    data["model"] = model_name
    data["model_display"] = model_name

    metric_file = find_file(folder, "_metrics.json", exclude="validation")
    metrics = load_json(metric_file)
    for k, v in metrics.items():
        data[k] = float(v) if isinstance(v, (int, float, np.number)) else v

    valid_file = find_file(folder, "_validation_metrics.json")
    valid = load_json(valid_file)
    for k, v in valid.items():
        data["validation_" + k] = float(v) if isinstance(v, (int, float, np.number)) else v

    return data


def collect_batch(batch_dir: Path) -> pd.DataFrame:
    batch_dir = Path(batch_dir)
    records = []

    for folder in batch_dir.iterdir():
        if not folder.is_dir() or folder.name == "summary":
            continue

        if list(folder.glob("*info*.txt")) or list(folder.glob("*metrics*.json")):
            try:
                record = collect_one_experiment(folder)
                records.append(record)
            except Exception as e:
                print(f"[Warning] Failed to collect: {folder.name} -> {e}")

    if not records:
        raise RuntimeError(f"在 {batch_dir} 下未找到任何有效实验记录。")

    df = pd.DataFrame(records)
    df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


def format_pair(a, b) -> str:
    if pd.isna(a) or pd.isna(b):
        return ""
    return f"{a:.4f}/{b:.4f}"


def filter_valid(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        ~(df["R2"].isna() | (abs(df["R2"]) > 10.0) | (df["MAE"] > 10.0) | (df["RMSE"] > 10.0))
    ].copy()


def env_sort_key(env: str):
    env = str(env).lower()
    if env == "sequence":
        return (0, "")
    parts = env.split("_")
    if parts[0] == "sequence":
        rest = parts[1:]
        return (len(rest), "_".join(rest))
    elif env == "all":
        return (99, "all")
    return (999, env)


def get_sort_keys(row, model_col="model", env_col="environment", split_col="split_type", cell_col="cell_line"):
    split = str(row.get(split_col, "")).lower()
    env = str(row.get(env_col, "")).lower()
    model = str(row.get(model_col, "")).lower()
    cell = str(row.get(cell_col, "")).lower()

    split_rank = SPLIT_ORDER.get(split, 999)
    env_rank = env_sort_key(env)
    model_rank = MODEL_ORDER.index(model) if model in MODEL_ORDER else len(MODEL_ORDER)
    return (split_rank, env_rank[0], env_rank[1], model_rank, cell)


def sort_dataframe(df: pd.DataFrame, model_col="model", env_col="environment", split_col="split_type", cell_col="cell_line"):
    if df.empty:
        return df
    df = df.loc[:, ~df.columns.duplicated()].copy()
    keys = df.apply(lambda row: get_sort_keys(row, model_col, env_col, split_col, cell_col), axis=1)
    df["_sort_key"] = keys
    df = df.sort_values("_sort_key").drop(columns=["_sort_key"])
    return df


def _delta_baseline_key(row) -> tuple:
    """ΔR² 的配对身份 = 实验身份 (P1/D5 整改)。

    旧实现只用 (split_type, cell_line, model), 会把
      - mixed 的 4 个 seed 的 sequence 基线互相覆盖 (后写覆盖先写);
      - CNN 的 3 个 sequence-kernel 基线互相覆盖;
    导致 ΔR² 与错误的基线相减。这里与 data_digging.classify_experiments
    使用同一套身份键: 补上 mixed 的 random_seed 与 cnn 的 sequence_kernel。
    """
    split_type = str(row.get("split_type", "")).lower()
    cell_line = str(row.get("cell_line", "")).lower()
    model = str(row.get("model", "")).lower()
    seed = row.get("random_seed", None)
    kernel = row.get("sequence_kernel", None)
    return (
        split_type,
        cell_line,
        model,
        int(seed) if split_type == "mixed" and pd.notna(seed) else None,
        int(kernel) if model == "cnn" and pd.notna(kernel) else None,
    )


def calculate_delta_R2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["delta_R2"] = np.nan

    baseline_map = {}
    for _, row in df.iterrows():
        if row.get("environment") == "sequence":
            baseline_map[_delta_baseline_key(row)] = row.get("R2", np.nan)

    for idx, row in df.iterrows():
        if row.get("environment") == "sequence":
            df.loc[idx, "delta_R2"] = 0.0
            continue

        base_r2 = baseline_map.get(_delta_baseline_key(row), np.nan)
        if not pd.isna(base_r2):
            df.loc[idx, "delta_R2"] = row.get("R2", np.nan) - base_r2

    return df


def build_result_dataframe(df: pd.DataFrame, include_cell_line: bool = True) -> pd.DataFrame:
    df = calculate_delta_R2(df)
    output = pd.DataFrame()

    if include_cell_line:
        output["cell_line"] = df["cell_line"]

    output["model"] = df["model"]
    output["environment"] = df["environment"]

    output["MAE/RMSE"] = [format_pair(a, b) for a, b in zip(df.get("MAE", np.nan), df.get("RMSE", np.nan))]
    output["Pearson/Spearman"] = [format_pair(a, b) for a, b in zip(df.get("Pearson", np.nan), df.get("Spearman", np.nan))]
    output["R2"] = df.get("R2", np.nan)
    output["delta_R2"] = df.get("delta_R2", np.nan)

    output["validation_MAE/RMSE"] = [format_pair(a, b) for a, b in zip(df.get("validation_MAE", np.nan), df.get("validation_RMSE", np.nan))]
    output["validation_Pearson/Spearman"] = [format_pair(a, b) for a, b in zip(df.get("validation_Pearson", np.nan), df.get("validation_Spearman", np.nan))]
    output["validation_R2"] = df.get("validation_R2", np.nan)

    if include_cell_line:
        output = sort_dataframe(output, model_col="model", env_col="environment", cell_col="cell_line")
    else:
        output = sort_dataframe(output, model_col="model", env_col="environment")
    return output


def create_single_result(df: pd.DataFrame) -> pd.DataFrame:
    data = df[df["split_type"] == "single"].copy()
    data = filter_valid(data)
    result = build_result_dataframe(data, include_cell_line=True)
    cols = ["cell_line", "model", "environment"] + [c for c in result.columns if c not in ["cell_line", "model", "environment"]]
    return result[cols]


def create_all_result(df: pd.DataFrame) -> pd.DataFrame:
    data = df[df["split_type"] == "all"].copy()
    data = filter_valid(data)
    data = data.reset_index(drop=True)
    result = build_result_dataframe(data, include_cell_line=False)
    # 必须按 index 标签对齐：build_result_dataframe 内部已按 (model, environment) 排序，
    # 直接 .values 会把 test_cell_line 贴到排序后的行上（与 R2 错位）。
    result["test_cell_line"] = data.loc[result.index, "cell_line"].values
    cols = ["model", "environment"] + [c for c in result.columns if c not in ["model", "environment", "test_cell_line"]] + ["test_cell_line"]
    result = result[cols]
    return sort_dataframe(result, model_col="model", env_col="environment", cell_col="test_cell_line")


def create_mixed_result(df: pd.DataFrame) -> pd.DataFrame:
    data = df[df["split_type"] == "mixed"].copy()
    data = filter_valid(data)

    group_cols = ["model", "environment"]
    metric_cols = [
        "R2", "MAE", "RMSE", "Pearson", "Spearman",
        "validation_R2", "validation_MAE", "validation_RMSE",
        "validation_Pearson", "validation_Spearman"
    ]
    metric_cols = [c for c in metric_cols if c in data.columns]

    mean_df = data.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
    mean_df["split_type"] = "mixed"
    mean_df["cell_line"] = "none"

    result = build_result_dataframe(mean_df, include_cell_line=False)
    cols = ["model", "environment"] + [c for c in result.columns if c not in ["model", "environment"]]
    return result[cols]


def save_result_tables(df: pd.DataFrame, metrics_dir: Path, selected_splits: Optional[List[str]] = None):
    """
    保存评测指标 CSV。

    selected_splits: 用户/引导程序勾选的划分模式 (single / all / mixed)。
                    仅生成对应模式的结果表，未勾选的模式不产出对应 CSV 文件。
                    为 None 时默认生成全部三种模式。
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if selected_splits is None:
        selected_splits = ["single", "all", "mixed"]
    selected_splits = [str(s).strip().lower() for s in selected_splits]

    # 全量原始指标表始终生成 (汇总所有已收集实验)
    all_df = df.copy()
    if "model_display" in all_df.columns:
        all_df = all_df.drop(columns=["model_display"])
    all_df = all_df.loc[:, ~all_df.columns.duplicated()]
    all_df = sort_dataframe(all_df, model_col="model", env_col="environment", split_col="split_type", cell_col="cell_line")
    all_df.to_csv(metrics_dir / "all_experiments.csv", index=False)
    print(f"  [+] all_experiments.csv saved")

    # 按引导程序勾选的划分模式选择性生成对应结果表
    if "single" in selected_splits:
        single_df = create_single_result(df)
        single_df.to_csv(metrics_dir / "single_cell_line_result.csv", index=False)
        print(f"  [+] single_cell_line_result.csv saved")
    else:
        print(f"  [-] single_cell_line_result.csv skipped (single 模式未勾选)")

    if "all" in selected_splits:
        all_res_df = create_all_result(df)
        all_res_df.to_csv(metrics_dir / "all_cell_line_result.csv", index=False)
        print(f"  [+] all_cell_line_result.csv saved")
    else:
        print(f"  [-] all_cell_line_result.csv skipped (all 模式未勾选)")

    if "mixed" in selected_splits:
        mixed_df = create_mixed_result(df)
        mixed_df.to_csv(metrics_dir / "mixed_cell_line_result.csv", index=False)
        print(f"  [+] mixed_cell_line_result.csv saved")
    else:
        print(f"  [-] mixed_cell_line_result.csv skipped (mixed 模式未勾选)")

    print(f"  [+] Result CSV tables saved to: {metrics_dir}")


def create_baseline(df: pd.DataFrame, selected_splits: Optional[List[str]] = None) -> pd.DataFrame:
    if selected_splits:
        selected_splits = [str(s).strip().lower() for s in selected_splits]
        df = df[df["split_type"].isin(selected_splits)].copy()

    data = filter_valid(df)
    baseline = data[data["environment"] == "sequence"].copy()
    
    non_mixed = baseline[baseline["split_type"] != "mixed"].copy()
    mixed_data = baseline[baseline["split_type"] == "mixed"].copy()
    
    metric_cols = ["R2", "MAE", "RMSE", "Pearson", "Spearman"]
    metric_cols = [c for c in metric_cols if c in baseline.columns]
    
    # 关键：对 mixed 模式跨 4 个随机种子求平均
    if not mixed_data.empty:
        mixed_avg = mixed_data.groupby(["model", "split_type"], dropna=False)[metric_cols].mean().reset_index()
        mixed_avg["cell_line"] = "none"
        combined = pd.concat([non_mixed, mixed_avg], ignore_index=True)
    else:
        combined = non_mixed
        
    cols = ["model", "split_type", "cell_line"] + metric_cols
    cols = [c for c in cols if c in combined.columns]
    combined = combined[cols]
    return sort_dataframe(combined, model_col="model", split_col="split_type", cell_col="cell_line")


def save_baseline_table(df: pd.DataFrame, metrics_dir: Path, selected_splits: Optional[List[str]] = None):
    """
    仅保存 sequence-only 基线性能表 (baseline.csv)。

    注：summary.md 与 environment_effect_summary.csv 已按要求不再生成。
    """
    metrics_dir.mkdir(parents=True, exist_ok=True)
    baseline = create_baseline(df, selected_splits=selected_splits)
    baseline.to_csv(metrics_dir / "baseline.csv", index=False)
    print(f"  [+] Sequence-only baseline table saved to: {metrics_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect and analyze CRISPR-Cas9 experiment results.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""示例：
  # 默认读取 results/batches 下最近修改的批次
  python -m analysis.collect_results --latest

  # 指定批次名（--results-dir 必须与训练时用的值一致）
  python -m analysis.collect_results --results-dir results/batches --batch-name smoke

  # 直接指定批次目录
  python -m analysis.collect_results --batch-dir results/batches/smoke

  # 只汇总 single 划分
  python -m analysis.collect_results --batch-name smoke --split-types single

产物：<batch>/summary/train_data/{single,all,mixed}_cell_line_result.csv 等
退出码：0 成功 / 1 批次目录不存在 / 2 用法错误
""")
    parser.add_argument("--results-dir", type=str, default="results/batches",
                        help="结果根目录（默认 results/batches；必须与训练时的 --results-dir 一致）")
    parser.add_argument("--batch-name", type=str, default="", help="批次名，如 smoke")
    parser.add_argument("--batch-dir", type=str, default="",
                        help="直接指定批次目录（优先级高于 --batch-name）")
    parser.add_argument("--latest", action="store_true", help="处理最近修改的批次")
    # 引导程序/命令行控制：按勾选的划分模式选择性生成对应结果 CSV
    parser.add_argument("--split-types", nargs="+", choices=["single", "all", "mixed"], default=None,
                        help="仅生成所选划分模式对应的结果 CSV (如: --split-types single all)")
    args = parser.parse_args()

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
    elif args.batch_name:
        batch_dir = Path(args.results_dir) / args.batch_name
    else:
        results_root = Path(args.results_dir)
        # 智能判定：如果 results_root 下直接包含了 single_/all_/mixed_ 实验文件夹，则它本身就是目标目录
        if results_root.is_dir() and any(
                d.is_dir() and d.name.startswith(("single_", "all_", "mixed_"))
                for d in results_root.iterdir()):
            batch_dir = results_root
        elif results_root.is_dir():
            sub_dirs = [d for d in results_root.iterdir() if d.is_dir() and d.name != "summary"]
            batch_dir = max(sub_dirs, key=os.path.getmtime) if sub_dirs else results_root
        else:
            batch_dir = results_root

    if not batch_dir.exists():
        # 必须返回**非零**退出码：历史实现用裸 `return`（退出码 0），会让
        # `collect_results && next_step` 这类脚本在批次缺失时静默继续。
        print(f"[Error] Batch directory '{batch_dir}' does not exist.", file=_sys.stderr)
        print("        提示：--results-dir 必须与训练时一致；或用 --latest / --batch-dir 指定。",
              file=_sys.stderr)
        return 1

    selected_splits = [s.lower() for s in args.split_types] if args.split_types else None
    if selected_splits:
        print(f"[*] 引导程序勾选划分模式: {selected_splits} -> 仅生成对应结果 CSV")

    # 当前交付结构把 runs 与 summary 拆成同级目录；分别解析（旧布局两者相同）
    from analysis.data.loaders import resolve_runs_root, resolve_summary_root

    runs_root = resolve_runs_root(batch_dir)
    metrics_tables_dir = resolve_summary_root(batch_dir) / "train_data"
    print(f"\n[*] 正在汇总模型指标，runs 根: {runs_root}")
    df = collect_batch(runs_root)

    save_result_tables(df, metrics_tables_dir, selected_splits=selected_splits)
    save_baseline_table(df, metrics_tables_dir, selected_splits=selected_splits)
    print(f"[✓] 成功生成 train_data -> {metrics_tables_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())