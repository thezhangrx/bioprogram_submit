"""analysis.data.loaders — 从训练产物建立统一实验表 (只读, 不修改原始结果)。

输入优先顺序 (自动探测, 发现字段差异走 adapter 而不是改训练输出):
  1. <batch>/summary/train_data/all_experiments.csv      (交付结构; 2026-09-25 由
                                                          metrics_tables 改名)
  2. <batch>/summary/all_experiments.csv
  3. 直接扫描实验目录 (legacy adapter, 隔离在 legacy_fallback)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


METRIC_COLUMNS = ["R2", "MAE", "RMSE", "Pearson", "Spearman", "MSE"]
REQUIRED_ID_COLUMNS = ["model", "split_type", "cell_line", "environment"]

#: 统一实验表的规范化列（load_experiment_table 的输出契约）。
EXPERIMENT_TABLE_COLUMNS = (REQUIRED_ID_COLUMNS
                            + ["random_seed", "n_train", "n_valid", "n_test"]
                            + METRIC_COLUMNS + ["source_table", "run_name"])

#: 指标表候选路径（相对 summary 根）。与 `probe_experiment_sources` 共享。
METRICS_TABLE_CANDIDATES = (
    ("train_data", "all_experiments.csv"),      # 当前交付结构
    ("metrics_tables", "all_experiments.csv"),  # 旧批次回退
    ("", "all_experiments.csv"),
)


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def resolve_summary_root(batch_dir: str | Path) -> Path:
    """把"批次目录"解析成"summary 根目录"（两种布局都支持）。

    * **布局 A（旧）**：`<batch>/<run>/…` 与 `<batch>/summary/…` 并存
      → 返回 `<batch>/summary`
    * **布局 B（当前交付结构，2026-09-25）**：`--batch-dir` 直接指向
      `results/summary/<数据集>/`，其下直接就是 `train_data/ tables/ reports/ figures/`
      → 返回该目录本身

    判据：只要参数目录下**直接**存在 `train_data/`、`tables/`、`reports/`、`figures/`
    中任意一个，就认定它已经是 summary 根。
    """
    b = Path(batch_dir)
    if any((b / sub).is_dir() for sub in ("train_data", "tables", "reports", "figures")):
        return b
    return b / "summary"


def resolve_runs_root(batch_dir: str | Path) -> Path:
    """把"批次目录"解析成"逐 run 产物根目录"（与 `resolve_summary_root` 互逆）。

    当前交付结构（2026-09-25）把 runs 与 summary 拆成了同级的两棵子树：

        results/train_results/<数据集>/<run>/…      ← runs（metrics/predictions/info/归因）
        results/summary/<数据集>/{tables,reports,figures,train_data}/…  ← summary

    因此当 `batch_dir` 指到 `summary/<数据集>` 时，需要在同级 `train_results/<数据集>`
    去找 run 目录（`extract_attribution_table`、`collect_batch`、bootstrap/permutation 的
    per-sample 配对都需要它）。找不到时退回 `batch_dir` 本身（旧布局 A 的行为不变）。
    """
    b = Path(batch_dir)
    if not any((b / sub).is_dir() for sub in ("train_data", "tables", "reports", "figures")):
        return b                      # 布局 A：batch_dir 本身就是 runs 根
    sibling = b.parent.parent / "train_results" / b.name
    return sibling if sibling.is_dir() else b


def metrics_table_candidates(batch_dir: str | Path) -> List[Path]:
    """统一实验表的候选文件路径（不含目录扫描回退）。"""
    summary_root = resolve_summary_root(Path(batch_dir))
    return [
        (summary_root / sub / name) if sub else (summary_root / name)
        for sub, name in METRICS_TABLE_CANDIDATES
    ]


def probe_experiment_sources(batch_dir: str | Path) -> Dict[str, Any]:
    """**只探测不加载**：统一实验表的两个来源各自是否可用。

    这正是"异常/一致性检测到底有没有数据可跑"的判据：

    * ``metrics_table`` —— 三个候选指标表里第一个存在且非空的路径（否则 ``None``）；
    * ``directory_scan`` —— **仅当指标表缺失时**才真正扫 run 目录（info + metrics json），
      报告扫描是否凑得出行；指标表可用时为 ``False``（表示"没走这条回退"）；
    * ``available`` —— 两者任一可用；两者皆否即"检测无法进行"；
    * ``errors`` —— 候选表**存在但读不动**时的报错文本。此时不是"没有数据"而是
      "数据坏了"，调用方应原样抛出而不是降级成跳过。

    调用方（``analysis.pipeline``）在 ``available=False`` 且 ``errors`` 为空时
    **不再 print 跳过提示**，而是照常生成产物并把 ``[!] 未实现检测跳过``
    写进 ``reports/data_quality.md``。
    """
    batch = Path(batch_dir)

    metrics_path: Optional[Path] = None
    errors: List[str] = []
    for cand in metrics_table_candidates(batch):
        if not cand.exists():
            continue
        try:
            if not pd.read_csv(cand, nrows=1).empty:
                metrics_path = cand
                break
        except Exception as exc:  # noqa: BLE001 - 记录后换下一个候选
            errors.append(f"读取 {cand} 失败: {exc}")
            continue

    # 目录扫描是**回退路径**：指标表可用时不做（避免每次分析都白扫上千个 run 目录）。
    scan_ok = False
    scan_rows = 0
    if metrics_path is None:
        scanned = _legacy_scan_experiments(batch)
        scan_ok = scanned is not None and not scanned.empty
        scan_rows = int(len(scanned)) if scan_ok else 0

    return {
        "metrics_table": str(metrics_path) if metrics_path is not None else None,
        "directory_scan": bool(scan_ok),
        "scan_rows": scan_rows,
        "available": bool(metrics_path is not None or scan_ok),
        "errors": errors,
    }


def empty_experiment_table() -> pd.DataFrame:
    """带完整列契约的空实验表（检测不可用时用它"照常生成"下游产物）。"""
    return pd.DataFrame(columns=list(EXPERIMENT_TABLE_COLUMNS))


def load_experiment_table(batch_dir: str | Path) -> pd.DataFrame:
    """建立统一 experiment table (DataFrame), 列: run/model/.../metrics。

    `batch_dir` 可以是「同时含 runs 与 summary 的批次目录」（布局 A），
    也可以是「summary/<数据集> 本身」（布局 B，当前交付结构）。见 resolve_summary_root。

    来源顺序：候选指标表 → 目录扫描（``_legacy_scan_experiments``）。
    两者都不可用时抛 ``RuntimeError``；调用方若想优雅降级，先用
    :func:`probe_experiment_sources` 判断再决定（``analysis.pipeline`` 即如此）。
    """
    batch = Path(batch_dir)
    probe = probe_experiment_sources(batch)
    table: Optional[pd.DataFrame] = None
    source = ""
    if probe["metrics_table"]:
        table = pd.read_csv(probe["metrics_table"])
        source = str(probe["metrics_table"])
    if table is None or table.empty:
        table = _legacy_scan_experiments(batch)
        source = "legacy-directory-scan"
    if table is None or table.empty:
        # 候选表存在但读不动 ≠ 没有数据：这种情况要原样报错，不能降级成"跳过"。
        if probe["errors"]:
            raise ValueError(probe["errors"][0])
        raise RuntimeError(f"在 {batch} 下未找到可用的实验指标表")

    # 字段适配: id 列大小写归一, 指标列保留 canonical 大小写 (R2/MAE/RMSE/...)
    lower2col = {str(c).strip().lower(): c for c in table.columns}
    work = table.copy()
    keep_original = set(table.columns)
    for col in REQUIRED_ID_COLUMNS + ["random_seed", "n_train", "n_valid", "n_test"]:
        actual = lower2col.get(col)
        if actual is None:
            if col != "environment":
                work[col] = pd.NA
            else:
                work[col] = "unknown"
        elif actual != col:
            work[col] = work[actual]
    for col in METRIC_COLUMNS:
        actual = lower2col.get(col.lower())
        if actual is None:
            work[col] = float("nan")
        else:
            work[col] = _coerce_float(work[actual])
    work["source_table"] = source
    # 输出统一 schema 列 (原始多余列不再暴露给分析层)
    work = work[[c for c in EXPERIMENT_TABLE_COLUMNS if c in work.columns]].copy()
    return work


def _legacy_scan_experiments(batch: Path) -> Optional[pd.DataFrame]:
    """legacy adapter: 无 summary 表时直接扫实验目录 info+metrics json。"""
    rows: List[Dict[str, Any]] = []
    for exp_dir in batch.iterdir():
        if not exp_dir.is_dir() or exp_dir.name == "summary":
            continue
        info = _parse_info(exp_dir)
        if not info:
            continue
        metrics = _load_metrics_json(exp_dir)
        if not metrics:
            continue
        row: Dict[str, Any] = {
            "model": str(info.get("model", exp_dir.name)).lower(),
            "split_type": str(info.get("split_type", "single")).lower(),
            "cell_line": str(info.get("cell_line", "none")).lower(),
            "environment": str(info.get("environment", info.get("combination", "all"))).lower(),
            "random_seed": int(float(info.get("random_seed", 42) or 42)),
        }
        for key in METRIC_COLUMNS:
            row[key] = metrics.get(key, float("nan"))
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def _parse_info(exp_dir: Path) -> Dict[str, str]:
    files = list(exp_dir.glob("*info*.txt"))
    if not files:
        return {}
    info: Dict[str, str] = {}
    for line in files[0].read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            info[k.strip().lower()] = v.strip()
    return info


def _load_metrics_json(exp_dir: Path) -> Dict[str, Any]:
    files = [p for p in exp_dir.glob("*_metrics.json") if "validation" not in p.name]
    if not files:
        return {}
    return json.loads(files[0].read_text(encoding="utf-8", errors="ignore"))


def coverage_summary(table: pd.DataFrame) -> Dict[str, Any]:
    """overview.md 所需的 model/cell-line/environment 覆盖。"""
    return {
        "experiment_count": int(len(table)),
        "valid_count": int(
            table["R2"].notna().sum()
            if "R2" in table.columns else len(table)
        ),
        "models": sorted(table["model"].dropna().unique().tolist()),
        "cell_lines": sorted(table["cell_line"].dropna().unique().tolist()),
        "environments": sorted(table["environment"].dropna().unique().tolist()),
        "splits": sorted(table["split_type"].dropna().unique().tolist()),
    }
