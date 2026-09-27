# core/features/channels/cell_environment_combination.py

"""
Cell Environment Combination
============================

CRISPR-Cas9 sgRNA Editing Efficiency Prediction

本模块负责：

    1. 读取 feature schema
    2. 自动识别 sequence / environment channels
    3. 生成 environment combinations
    4. 控制输入中哪些 environment 被启用
    5. 为不同模型生成统一输入
    6. 为实验记录生成 combination metadata

============================================================
当前实验设计
============================================================

Sequence channels：

    A
    C
    G
    T

Environment channels：

    CTCF
    Dnase
    H3K4me3
    RRBS

因此标准实验组合为：

    1. sequence-only

    2. sequence + 1 environment

    3. sequence + 2 environments

    4. sequence + 3 environments

    5. sequence + all environments

============================================================
例如
============================================================

sequence

sequence_ctcf
sequence_dnase
sequence_h3k4me3
sequence_rrbs

sequence_ctcf_dnase
sequence_ctcf_h3k4me3
sequence_ctcf_rrbs
sequence_dnase_h3k4me3
...

sequence_ctcf_dnase_h3k4me3
sequence_ctcf_dnase_rrbs
...

all

============================================================
输入
============================================================

X_3d:

    (N, L, C)

例如：

    (N, 23, 8)

schema：

    feature_schema.json

============================================================
控制方式
============================================================

没有选择的 environment：

    → 全部置零

Sequence channels：

    → 永远保留

例如：

    sequence

    mask:

        [1,1,1,0,0,0,0]

    sequence_ctcf

        [1,1,1,1,0,0,0]

============================================================
模型输入
============================================================

Linear / XGBoost / MLP：

    (N,L,C)
        ↓
    (N,L*C)

CNN / Transformer：

    保持：

    (N,L,C)

============================================================
重要
============================================================

本模块不负责：

    - Train / Validation / Test 划分
    - cell line 划分
    - model training
    - hyperparameter optimization

数据划分由：

    cell_line_division.py

负责。

"""

from __future__ import annotations

import itertools
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================
# 1. Constants
# ============================================================

DEFAULT_SCHEMA_FILENAME = (
    "feature_schema.json"
)

# 完整 4 碱基 One-Hot (线性回归以 T 为基准剔除 _T 列, 张量本身保留 T)
DEFAULT_SEQUENCE_CHANNELS = [
    "A",
    "C",
    "G",
    "T",
]

SUPPORTED_MODEL_2D = {
    "linear",
    "linear_regression",
    "xgboost",
    "mlp",
}

SUPPORTED_MODEL_3D = {
    "cnn",
    "transformer",
}


# ============================================================
# 2. Schema
# ============================================================

def load_feature_schema(
    data_dir: str
) -> Dict:
    """
    读取 feature_schema.json。
    """

    schema_path = os.path.join(
        data_dir,
        DEFAULT_SCHEMA_FILENAME
    )

    if not os.path.exists(
        schema_path
    ):
        raise FileNotFoundError(
            f"feature_schema.json 不存在："
            f"{schema_path}"
        )

    with open(
        schema_path,
        "r",
        encoding="utf-8"
    ) as f:

        schema = json.load(
            f
        )

    required_keys = [
        "sequence_length",
        "channel_count",
        "channel_names",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in schema
    ]

    if missing_keys:

        raise ValueError(
            "feature_schema.json 缺少字段："
            + ", ".join(
                missing_keys
            )
        )

    channel_names = schema[
        "channel_names"
    ]

    if not isinstance(
        channel_names,
        list
    ):
        raise ValueError(
            "schema['channel_names'] 必须是 list。"
        )

    channel_count = int(
        schema["channel_count"]
    )

    if len(channel_names) != channel_count:

        raise ValueError(
            "channel_names 数量与 "
            "channel_count 不一致："
            f"{len(channel_names)} != "
            f"{channel_count}"
        )

    return schema


# ============================================================
# 3. Channel information
# ============================================================

def get_channel_names(
    schema: Dict
) -> List[str]:
    """
    获取所有 channel 名称。
    """

    channels = list(
        schema[
            "channel_names"
        ]
    )

    if len(channels) != int(
        schema["channel_count"]
    ):
        raise ValueError(
            "channel_names 与 channel_count 不一致。"
        )

    # 检查重复
    if len(channels) != len(set(channels)):

        raise ValueError(
            "channel_names 中存在重复 channel。"
        )

    return channels


def get_sequence_channels(
    schema: Dict
) -> List[str]:
    """
    获取 sequence channels。

    优先使用 schema：

        sequence_channels

    没有时：

        A / G / C
    """

    sequence_channels = schema.get(
        "sequence_channels",
        DEFAULT_SEQUENCE_CHANNELS
    )

    if not isinstance(
        sequence_channels,
        list
    ):
        raise ValueError(
            "sequence_channels 必须是 list。"
        )

    all_channels = set(
        get_channel_names(
            schema
        )
    )

    for channel in sequence_channels:

        if channel not in all_channels:

            raise ValueError(
                f"Sequence channel "
                f"{channel} 不存在于 schema。"
            )

    return list(
        sequence_channels
    )


def get_environment_channels(
    schema: Dict
) -> List[str]:
    """
    自动得到：

        所有 channel
        -
        sequence channels
    """

    all_channels = (
        get_channel_names(
            schema
        )
    )

    sequence_channels = (
        get_sequence_channels(
            schema
        )
    )

    sequence_set = set(
        sequence_channels
    )

    return [
        channel
        for channel in all_channels
        if channel not in sequence_set
    ]


def build_channel_index(
    schema: Dict
) -> Dict[str, int]:
    """
    channel name → channel index
    """

    channels = (
        get_channel_names(
            schema
        )
    )

    return {
        channel: index
        for index, channel
        in enumerate(channels)
    }


# ============================================================
# 4. Environment selection validation
# ============================================================

def validate_selected_environments(
    schema: Dict,
    selected_environments: Sequence[str]
) -> List[str]:
    """
    验证 environment 是否存在。

    同时：

        1. 去除重复
        2. 保留用户输入顺序
    """

    if selected_environments is None:

        raise ValueError(
            "selected_environments 不能为 None。"
        )

    if isinstance(
        selected_environments,
        str
    ):
        selected_environments = [
            selected_environments
        ]

    available = (
        get_environment_channels(
            schema
        )
    )

    available_set = set(
        available
    )

    result = []

    for environment in (
        selected_environments
    ):

        if not isinstance(
            environment,
            str
        ):
            raise TypeError(
                "environment 名称必须是字符串。"
            )

        environment = (
            environment.strip()
        )

        if not environment:

            raise ValueError(
                "environment 名称不能为空。"
            )

        if environment not in available_set:

            raise ValueError(
                f"未知 environment："
                f"{environment}\n"
                f"当前可用："
                f"{available}"
            )

        if environment not in result:

            result.append(
                environment
            )

    return result


# ============================================================
# 5. Combination naming
# ============================================================

def make_combination_name_from_environments(
    selected_environments: Sequence[str]
) -> str:
    """
    根据 environment 列表生成稳定名称。

    []：

        sequence

    ["CTCF"]：

        sequence_ctcf

    ["CTCF","Dnase"]：

        sequence_ctcf_dnase

    全部：

        all
    """

    selected = list(
        selected_environments
    )

    if not selected:

        return "sequence"

    sorted_environments = sorted(
        selected,
        key=lambda x: x.lower()
    )

    return (
        "sequence_"
        +
        "_".join(
            environment.lower()
            for environment
            in sorted_environments
        )
    )


def make_combination_name(
    schema: Dict,
    selected_environments: Optional[
        Sequence[str]
    ] = None,
    combination: Optional[str] = None
) -> str:
    """
    为当前组合生成稳定名称。
    """

    selected = (
        resolve_environment_selection(
            schema=schema,
            combination=combination,
            selected_environments=
                selected_environments
        )
    )

    all_environment = (
        get_environment_channels(
            schema
        )
    )

    # 0 个表观通道（纯序列数据集）时"选中全部"就是什么都不加，名字是 "sequence"。
    # 旧实现会让 `len([])==len([])` 命中下面的分支返回 "all"，把纯序列 run 的
    # combination 元数据记成 4 因子全组合，下游 analysis 的 2^4 逻辑会被误导。
    if len(all_environment) == 0:

        return make_combination_name_from_environments(
            selected
        )

    if (
        len(selected)
        == len(all_environment)
        and set(selected)
        == set(all_environment)
    ):

        return "all"

    return make_combination_name_from_environments(
        selected
    )


# ============================================================
# 6. Generate combinations
# ============================================================

def get_combinations_by_size(
    schema: Dict,
    size: int
) -> List[
    Tuple[str, List[str]]
]:
    """
    返回指定 environment 数量的全部组合。

    size=0：

        [("sequence", [])]

    size=1：

        sequence + 单环境

    size=2：

        sequence + 双环境

    size=3：

        sequence + 三环境

    size=environment_count：

        对应全部环境。
    """

    environments = (
        get_environment_channels(
            schema
        )
    )

    size = int(
        size
    )

    if size < 0:

        raise ValueError(
            "environment size 不能 < 0。"
        )

    if size > len(
        environments
    ):

        raise ValueError(
            f"environment size={size} "
            f"超过当前环境数量="
            f"{len(environments)}"
        )

    combinations = []

    for selected in itertools.combinations(
        environments,
        size
    ):

        selected = list(
            selected
        )

        name = (
            make_combination_name_from_environments(
                selected
            )
        )

        # size == all environment 时
        # 仍然返回组合，但标准名称使用 all
        #
        # 例外：0 个表观通道（纯序列数据集）时 size=0 也满足 size==len(environments)，
        # 但那不是 "all"（没有任何表观通道可选），名字只能是 "sequence"——
        # 否则纯序列 run 的 combination 元数据会被记成 "all"。
        if size == len(environments) and len(environments) > 0:
            name = "all"

        combinations.append(
            (
                name,
                selected
            )
        )

    return combinations


def generate_combinations_by_size(
    schema: Dict,
    size: int
) -> Dict[str, List[str]]:
    """
    指定环境数量生成组合字典。
    """

    combinations = (
        get_combinations_by_size(
            schema,
            size
        )
    )

    return {
        name: environments
        for name, environments
        in combinations
    }


def generate_environment_combinations(
    schema: Dict,
    sizes: Optional[
        Iterable[int]
    ] = None
) -> Dict[str, List[str]]:
    """
    生成指定规模的 environment combinations。

    默认：

        size=0
        size=1
        size=2
        size=3
        all

    对当前4个 environment：

        1
        + 4
        + 6
        + 4
        + 1

    共：

        16

    个组合。
    """

    environments = (
        get_environment_channels(
            schema
        )
    )

    environment_count = len(
        environments
    )

    # 纯序列数据集（0 个表观通道）：唯一有意义的组合就是 "sequence"。
    # 旧实现会在 sizes=[0,1,2,3] 里跳过 size=0（因为 0 == environment_count），
    # 然后对 size=1 调用 get_combinations_by_size 并抛出
    # "environment size=1 超过当前环境数量=0"，导致 4 通道数据根本无法训练。
    if environment_count == 0:
        return {make_combination_name_from_environments([]): []}

    if sizes is None:

        sizes = [
            0,
            1,
            2,
            3,
        ]

    sizes = list(
        sizes
    )

    result = {}

    for size in sizes:

        size = int(
            size
        )

        if size == environment_count:

            continue

        combinations = (
            generate_combinations_by_size(
                schema,
                size
            )
        )

        result.update(
            combinations
        )

    # --------------------------------------------------------
    # Always add all
    # --------------------------------------------------------

    result["all"] = list(
        environments
    )

    return result


def generate_combination_names(
    schema: Dict,
    include_all: bool = True,
    include_sequence: bool = True,
    sizes: Optional[
        Iterable[int]
    ] = None
) -> List[str]:
    """
    只生成组合名称。

    当前4个环境默认：

        sequence
        4 × single environment
        6 × double environment
        4 × triple environment
        all

    共：

        16 combinations
    """

    if sizes is None:

        sizes = [
            0,
            1,
            2,
            3
        ]

    sizes = [
        int(size)
        for size in sizes
    ]

    combinations = []

    environment_count = len(
        get_environment_channels(
            schema
        )
    )

    # 纯序列数据集（0 个表观通道）：唯一有意义的环境组合就是 "sequence"。
    # 旧实现会让 size==0 命中下面的 `size == environment_count` 分支而额外产出
    # "all"，但 0 个表观通道时 "all" 与 "sequence" 完全等价 → 会跑出重复实验。
    if environment_count == 0:
        return ["sequence"] if include_sequence else []

    for size in sizes:

        if size == 0 and not include_sequence:
            continue

        if (
            size == environment_count
            and not include_all
        ):
            continue

        if size == environment_count:

            if include_all:

                combinations.append(
                    "all"
                )

            continue

        size_combinations = (
            get_combinations_by_size(
                schema,
                size
            )
        )

        for name, _ in (
            size_combinations
        ):

            if (
                name == "sequence"
                and not include_sequence
            ):
                continue

            combinations.append(
                name
            )

    # --------------------------------------------------------
    # Add all
    # --------------------------------------------------------

    if (
        include_sequence 
        and "sequence" not in combinations
    ):

        combinations.insert(
            0, "sequence"
        ) 

    if (
        include_all
        and "all" not in combinations
    ):

        combinations.append(
            "all"
        )

    
    return combinations


# ============================================================
# 7. Resolve environment selection
# ============================================================

def resolve_environment_selection(
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None,
    environment_size: Optional[int] = None
) -> List[str]:
    """
    解析最终使用的 environment。

    三种方式：

    1. combination

    2. selected_environments

    3. environment_size

    三者最多使用一种。
    """

    provided_count = sum(
        item is not None
        for item in [
            combination,
            selected_environments,
            environment_size
        ]
    )

    if provided_count > 1:

        raise ValueError(
            "combination、"
            "selected_environments、"
            "environment_size "
            "只能提供其中一种。"
        )

    all_environments = (
        get_environment_channels(
            schema
        )
    )

    # ========================================================
    # No selection
    # ========================================================

    if provided_count == 0:

        # 默认全部 environment
        return list(
            all_environments
        )

    # ========================================================
    # Explicit list
    # ========================================================

    if selected_environments is not None:

        return (
            validate_selected_environments(
                schema,
                selected_environments
            )
        )

    # ========================================================
    # Environment size
    # ========================================================

    if environment_size is not None:

        size = int(
            environment_size
        )

        combinations = (
            get_combinations_by_size(
                schema,
                size
            )
        )

        if not combinations:

            raise ValueError(
                f"没有找到 environment_size="
                f"{size} 的组合。"
            )

        # 这个接口如果只传 size，
        # 但一个 size 对应多个组合，
        # 无法唯一确定具体组合。
        if len(combinations) != 1:

            raise ValueError(
                f"environment_size={size} "
                "对应多个 environment combination。"
                "\n请使用 combination "
                "或 selected_environments "
                "明确指定具体组合。"
            )

        return list(
            combinations[0][1]
        )

    # ========================================================
    # Combination name
    # ========================================================

    combination = (
        str(
            combination
        )
        .strip()
        .lower()
    )

    combinations = (
        generate_environment_combinations(
            schema
        )
    )

    combination_lookup = {
        name.lower(): environments
        for name, environments
        in combinations.items()
    }

    if combination not in (
        combination_lookup
    ):

        raise ValueError(
            f"未知 environment combination："
            f"{combination}\n\n"
            f"当前支持：\n"
            + "\n".join(
                f"    {name}"
                for name
                in combinations.keys()
            )
        )

    return list(
        combination_lookup[
            combination
        ]
    )


# ============================================================
# 8. Channel mask
# ============================================================

def create_channel_mask(
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None
) -> np.ndarray:
    """
    创建 channel mask。

    例如：

        sequence

        [1,1,1,0,0,0,0]

    sequence_ctcf

        [1,1,1,1,0,0,0]
    """

    channels = (
        get_channel_names(
            schema
        )
    )

    channel_index = (
        build_channel_index(
            schema
        )
    )

    sequence_channels = (
        get_sequence_channels(
            schema
        )
    )

    selected_environment = (
        resolve_environment_selection(
            schema=schema,
            combination=combination,
            selected_environments=
                selected_environments
        )
    )

    mask = np.zeros(
        len(channels),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Sequence always enabled
    # --------------------------------------------------------

    for sequence_channel in (
        sequence_channels
    ):

        mask[
            channel_index[
                sequence_channel
            ]
        ] = 1.0

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    for environment in (
        selected_environment
    ):

        mask[
            channel_index[
                environment
            ]
        ] = 1.0

    return mask


# ============================================================
# 9. Apply environment combination
# ============================================================

def apply_environment_combination(
    X_3d: np.ndarray,
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None
) -> np.ndarray:
    """
    对 X_3d 应用 environment mask。

    输入：

        (N,L,C)

    输出：

        (N,L,C)

    原数组不会被修改。
    """

    if not isinstance(
        X_3d,
        np.ndarray
    ):
        X_3d = np.asarray(
            X_3d,
            dtype=np.float32
        )

    if X_3d.ndim != 3:

        raise ValueError(
            "X_3d 必须是三维："
            "(N,L,C)"
        )

    sequence_length = int(
        schema["sequence_length"]
    )

    channel_count = int(
        schema["channel_count"]
    )

    if X_3d.shape[1:] != (
        sequence_length,
        channel_count
    ):

        raise ValueError(
            "X_3d shape 与 schema 不一致："
            f"实际={X_3d.shape}，"
            f"预期=(N,"
            f"{sequence_length},"
            f"{channel_count})"
        )

    if not np.isfinite(
        X_3d
    ).all():

        raise ValueError(
            "X_3d 中存在 NaN 或 Inf。"
        )

    mask = create_channel_mask(

        schema=schema,

        combination=combination,

        selected_environments=
            selected_environments
    )

    return (
        X_3d.copy()
        *
        mask.reshape(
            1,
            1,
            channel_count
        )
    )


# ============================================================
# 10. Flatten
# ============================================================

def flatten_features(
    X_3d: np.ndarray
) -> np.ndarray:
    """
    (N,L,C)
        ↓
    (N,L*C)
    """

    if X_3d.ndim != 3:

        raise ValueError(
            "X_3d 必须是三维。"
        )

    return X_3d.reshape(
        X_3d.shape[0],
        -1
    )


# ============================================================
# 11. Model input type
# ============================================================

def normalize_model_type(
    model_type: str
) -> str:

    model_type = (
        str(model_type)
        .strip()
        .lower()
    )

    aliases = {

        "linear_regression":
            "linear",

    }

    return aliases.get(
        model_type,
        model_type
    )


# ============================================================
# 12. Prepare one model input
# ============================================================

def prepare_model_input(
    X_3d: np.ndarray,
    schema: Dict,
    model_type: str,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None
) -> np.ndarray:
    """
    将原始 3D feature 转换成模型输入。
    """

    model_type = (
        normalize_model_type(
            model_type
        )
    )

    if (
        model_type not in
        SUPPORTED_MODEL_2D
        and
        model_type not in
        SUPPORTED_MODEL_3D
    ):

        raise ValueError(
            f"未知 model_type："
            f"{model_type}"
        )

    X_controlled = (
        apply_environment_combination(

            X_3d=X_3d,

            schema=schema,

            combination=combination,

            selected_environments=
                selected_environments
        )
    )

    if (
        model_type
        in SUPPORTED_MODEL_2D
    ):

        return flatten_features(
            X_controlled
        )

    return X_controlled


# ============================================================
# 13. Prepare Train / Validation / Test
# ============================================================

def prepare_train_valid_test(
    split_data: Dict,
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None,
    model_type: str = "xgboost"
) -> Dict:
    """
    将新的 cell_line_division.py
    输出转换成模型输入。

    split_data 支持：

        single
        all
        mixed
    """

    # --------------------------------------------------------
    # Resolve once
    # --------------------------------------------------------

    selected = (
        resolve_environment_selection(

            schema=schema,

            combination=combination,

            selected_environments=
                selected_environments
        )
    )

    combination_name = (
        make_combination_name(

            schema=schema,

            selected_environments=selected
            if combination is None
            else None,

            combination=combination
        )
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X_train = prepare_model_input(

        X_3d=
            split_data[
                "X_train_3d"
            ],

        schema=schema,

        model_type=model_type,

        selected_environments=selected
    )

    X_valid = prepare_model_input(

        X_3d=
            split_data[
                "X_valid_3d"
            ],

        schema=schema,

        model_type=model_type,

        selected_environments=selected
    )

    X_test = prepare_model_input(

        X_3d=
            split_data[
                "X_test_3d"
            ],

        schema=schema,

        model_type=model_type,

        selected_environments=selected
    )

    return {

        "X_train":
            X_train,

        "y_train":
            split_data[
                "y_train"
            ],

        "X_valid":
            X_valid,

        "y_valid":
            split_data[
                "y_valid"
            ],

        "X_test":
            X_test,

        "y_test":
            split_data[
                "y_test"
            ],

        "selected_environments":
            selected,

        "environment_count":
            len(selected),

        "combination":
            combination_name,

        "model_type":
            normalize_model_type(
                model_type
            ),

        "feature_count":
            (
                X_train.shape[1]
                if X_train.ndim == 2
                else X_train.shape[1:]
            ),

        "input_shape_train":
            list(
                X_train.shape
            ),

        "input_shape_valid":
            list(
                X_valid.shape
            ),

        "input_shape_test":
            list(
                X_test.shape
            )
    }


# ============================================================
# 14. Generate all combinations for data
# ============================================================


# ============================================================
# 15. Custom combination
# ============================================================


# ============================================================
# 16. Combination metadata
# ============================================================

def get_combination_metadata(
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None
) -> Dict:
    """
    获取实验组合结构信息。

    适合写入 config.json。
    """

    all_channels = (
        get_channel_names(
            schema
        )
    )

    sequence_channels = (
        get_sequence_channels(
            schema
        )
    )

    environment_channels = (
        get_environment_channels(
            schema
        )
    )

    selected = (
        resolve_environment_selection(

            schema=schema,

            combination=combination,

            selected_environments=
                selected_environments
        )
    )

    mask = create_channel_mask(

        schema=schema,

        combination=combination,

        selected_environments=
            selected_environments
    )

    name = make_combination_name(

        schema=schema,

        combination=combination,

        selected_environments=
            selected_environments
    )

    return {

        "combination":
            name,

        "sequence_channels":
            sequence_channels,

        "available_environment_channels":
            environment_channels,

        "selected_environments":
            selected,

        "environment_count":
            len(selected),

        "all_channels":
            all_channels,

        "channel_count":
            len(all_channels),

        "channel_mask":
            mask.astype(
                int
            ).tolist()
    }


# ============================================================
# 17. Print combination
# ============================================================

def print_combination_info(
    schema: Dict,
    combination: Optional[str] = None,
    selected_environments: Optional[
        Sequence[str]
    ] = None
):
    """
    打印当前 combination 信息。
    """

    metadata = (
        get_combination_metadata(

            schema=schema,

            combination=combination,

            selected_environments=
                selected_environments
        )
    )

    print()
    print("=" * 70)

    print(
        f"Combination: "
        f"{metadata['combination']}"
    )

    print(
        "Sequence channels:",
        metadata[
            "sequence_channels"
        ]
    )

    print(
        "Environment channels:",
        metadata[
            "available_environment_channels"
        ]
    )

    print(
        "Selected environments:",
        metadata[
            "selected_environments"
        ]
    )

    print(
        "Environment count:",
        metadata[
            "environment_count"
        ]
    )

    print(
        "Channel mask:",
        metadata[
            "channel_mask"
        ]
    )

    print("=" * 70)


# ============================================================
# 18. Standalone test
# ============================================================

if __name__ == "__main__":

    DATA_DIR = (
        str(Path(__file__).resolve().parents[3] / "data" / "processed" / "DeepCRISPR")
    )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    schema = load_feature_schema(
        DATA_DIR
    )

    print(
        "\nFeature Schema"
    )

    print(
        "Sequence length:",
        schema[
            "sequence_length"
        ]
    )

    print(
        "Channel count:",
        schema[
            "channel_count"
        ]
    )

    print(
        "Channels:",
        schema[
            "channel_names"
        ]
    )

    print(
        "\nSequence channels:",
        get_sequence_channels(
            schema
        )
    )

    print(
        "Environment channels:",
        get_environment_channels(
            schema
        )
    )

    # --------------------------------------------------------
    # Combinations
    # --------------------------------------------------------

    combinations = (
        generate_environment_combinations(
            schema
        )
    )

    print(
        "\nNumber of combinations:",
        len(combinations)
    )

    for name, environments in (
        combinations.items()
    ):

        print(
            f"{name:45s}",
            environments
        )

    # --------------------------------------------------------
    # Simulated X
    # --------------------------------------------------------

    sequence_length = int(
        schema[
            "sequence_length"
        ]
    )

    channel_count = int(
        schema[
            "channel_count"
        ]
    )

    X = np.ones(
        (
            5,
            sequence_length,
            channel_count
        ),
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Check every combination
    # --------------------------------------------------------

    for combination in combinations:

        print_combination_info(
            schema=schema,
            combination=combination
        )

        X_cnn = prepare_model_input(

            X_3d=X,

            schema=schema,

            model_type="cnn",

            combination=combination
        )

        X_xgb = prepare_model_input(

            X_3d=X,

            schema=schema,

            model_type="xgboost",

            combination=combination
        )

        print(
            "CNN input:",
            X_cnn.shape
        )

        print(
            "XGBoost input:",
            X_xgb.shape
        )
