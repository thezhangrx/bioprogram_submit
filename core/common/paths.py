"""项目路径解析（唯一权威定义）。

设计原则
--------
* 所有默认路径由**项目根**推导，不写死绝对路径；
* 项目根由本文件位置推导（``core/common/paths.py`` → parents[2]），
  因此从任意工作目录运行、或被其它进程 import 都能正确定位；
* 各入口脚本（train / data_digging / predict / design / screen / analysis CLI）
  只引用本模块，不再各自拼路径。

目录约定
--------
    data/processed/<Dataset>   已处理特征与标签（模型输入），按数据集分层，与 data/raw 对齐
    data/raw/<Dataset>         原始数据（同样按数据集分层）
    data/candidate             候选/待测序列表
    data/config                **每个数据集一个自包含 JSON**（规格 + 特征映射 + 序列长度）
    results/batches            每次实验批次的完整产物（<batch>/summary/...）
    results/logs               训练与运行日志
    results/tables             跨批次汇总表
    models/weights             训练产出的模型权重（按批次归档）

数据集寻址
----------
训练/预测入口用 ``--data-set <名称>`` 指定数据集，例如 ``--data-set Hiranniramol``，
由本模块的 :func:`resolve_dataset` 解析成 ``data/processed/Hiranniramol``。
名称大小写不敏感；找不到时列出可用数据集，而不是回退到任何默认值。

数据集**配置**（原始/处理后目录、特征映射、序列长度）统一放在
``data/config/<数据集>.json``，由 :func:`datasets_registry` 扫描成注册表；
特征工程 CLI 直接把该文件当 ``--config`` 用。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

#: 本文件位于 ``<root>/core/common/paths.py``，故 parents[2] 即项目根。
#: 修改此表达式前请先跑 tests/test_project_paths.py（会断言全部默认路径存在）。
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
#: 已处理数据的**根**目录；每个数据集是它的一个子目录（data/processed/<Dataset>）
DATA_PROCESSED: Path = DATA_DIR / "processed"
DATA_RAW: Path = DATA_DIR / "raw"
DATA_CANDIDATE: Path = DATA_DIR / "candidate"
#: 数据集配置目录：``data/config/<数据集>.json``，每个文件自包含
#: （数据集规格 + feature 映射 + 序列长度）。取代了旧的 ``data/metadata/``。
DATA_CONFIG: Path = DATA_DIR / "config"

RESULTS_DIR: Path = PROJECT_ROOT / "results"
RESULTS_BATCHES: Path = RESULTS_DIR / "batches"
RESULTS_TABLES: Path = RESULTS_DIR / "tables"
LOGS_DIR: Path = RESULTS_DIR / "logs"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODELS_WEIGHTS: Path = MODELS_DIR / "weights"

#: 供 argparse default 使用的字符串形式
STR_DATA_PROCESSED = str(DATA_PROCESSED)
STR_RESULTS_BATCHES = str(RESULTS_BATCHES)
STR_MODELS_WEIGHTS = str(MODELS_WEIGHTS)
STR_LOGS = str(LOGS_DIR)

#: 判定"这个目录是一个已处理数据集"的标志文件
DATASET_SCHEMA_FILENAME = "feature_schema.json"

#: 数据集配置（单一数据源）：``data/config/<数据集>.json``，一文件一数据集。
#: orchestrator / README / 测试都读它。新增数据集只需放一个 JSON，
#: 不需要改任何 Python 代码。
DATASET_CONFIG_SUFFIX = ".json"


def dataset_config_dir(root: Path | str | None = None) -> Path:
    """数据集配置目录。``root`` 语义与其它函数一致：**项目根**。"""
    if root:
        return Path(root) / "data" / "config"
    return DATA_CONFIG


def available_datasets(root: Path | str | None = None) -> List[str]:
    """列出 ``data/processed`` 下所有**已完成特征工程**的数据集名。

    只认含 ``feature_schema.json`` 的子目录；排序保证跨机器确定性。
    """
    base = Path(root) if root else DATA_PROCESSED
    if not base.is_dir():
        return []
    return sorted(
        d.name for d in base.iterdir()
        if d.is_dir() and (d / DATASET_SCHEMA_FILENAME).is_file()
    )


def dataset_dirs(root: Path | str | None = None) -> Dict[str, Path]:
    """``{小写数据集名: 目录}``，便于大小写不敏感查找。"""
    base = Path(root) if root else DATA_PROCESSED
    return {name.lower(): base / name for name in available_datasets(base)}


def resolve_dataset(name: str, root: Path | str | None = None) -> Path:
    """把 ``--data-set`` 的名称解析成已处理数据集目录。

    大小写不敏感；也接受传入一个已经存在的目录路径（方便脚本直连）。
    解析失败时抛出带"可用数据集"列表的错误——**不会回退到任何默认数据集**。
    """
    raw = str(name).strip()
    if not raw:
        raise ValueError("--data-set 不能为空")

    as_path = Path(raw).expanduser()
    if as_path.is_dir() and (as_path / DATASET_SCHEMA_FILENAME).is_file():
        return as_path.resolve()

    dirs = dataset_dirs(root)
    if raw.lower() in dirs:
        return dirs[raw.lower()].resolve()

    base = Path(root) if root else DATA_PROCESSED
    raise FileNotFoundError(
        f"找不到数据集 {raw!r}。\n"
        f"  已处理数据根目录：{base}\n"
        f"  可用数据集：{', '.join(available_datasets(base)) or '（无）'}\n"
        f"  每个数据集必须是该目录下含 {DATASET_SCHEMA_FILENAME} 的子目录；"
        f"请先运行特征工程生成。"
    )


def resolve_data_dir(data_dir: str | None = None, data_set: str | None = None) -> Path:
    """统一解析入口的"要跑哪个数据集"。

    * ``--data-set NAME``：解析成 ``data/processed/NAME``（大小写不敏感）；
    * ``--data-dir PATH``：直接使用该路径（高级用法：预处理目录、组合目录）；
    * 两者都不给 / 同时给：报错并说明用法。

    **不提供任何默认数据集**——避免"忘了指定就跑错数据集"。
    """
    if data_set and data_dir:
        raise ValueError(
            "--data-set 与 --data-dir 只能给一个：\n"
            "  --data-set DeepCRISPR            按数据集名（推荐）\n"
            "  --data-dir data/processed/xxx    直接指定目录"
        )
    if data_set:
        return resolve_dataset(data_set)
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    raise ValueError(
        "必须指定要跑的数据集：\n"
        "  --data-set <名称>      例如 --data-set DeepCRISPR / Hiranniramol / Labuhn\n"
        f"  当前可用：{', '.join(available_datasets()) or '（无，请先跑特征工程）'}\n"
        "  （也可用 --data-dir <路径> 直接指定目录）"
    )


def add_dataset_arguments(parser) -> None:
    """给 argparse 加上统一的 ``--data-set`` / ``--data-dir`` 选项组。

    出现在 ``--help`` 里的说明就是可用数据集清单，避免用户去翻代码。
    """
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--data-set", "--data_set", "--dataset", dest="data_set", default=None,
        help="要跑的数据集名称（大小写不敏感），例如 DeepCRISPR / Hiranniramol / Labuhn。"
             f" 当前可用：{', '.join(available_datasets()) or '（无）'}"
    )
    group.add_argument(
        "--data-dir", dest="data_dir", default=None,
        help="直接指定已处理数据目录（高级用法；与 --data-set 二选一）"
    )
