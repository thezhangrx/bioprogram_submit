# `data/config/` — 手把手写一个数据集的 JSON

一个数据集 = `data/config/` 下的**一个自包含 JSON**，文件名（去掉 `.json`）就是数据集名。

现在已经有三份可以直接照抄的样例，本文件把它们**逐字段拆开讲**：

| 文件 | 形态 | 序列长度 | 通道数 | 标签列 | y 是否归一化 |
|---|---|---|---|---|---|
| `DeepCRISPR.json` | 4 个细胞系各一个 CSV | 23 | **8**（4 序列 + 4 表观） | `Normalized efficacy` | no |
| `Hiranniramol.json` | 单文件 | 23 | 4（纯序列） | `Edit Efficiency` | **yes**（÷100） |
| `Labuhn.json` | 单文件 | 23 | 4（纯序列） | `KO_reporter_assay` | no |

---

## 0. 先看结论：哪些键是"承重"的

写之前先知道**哪些键改了真的有用**，免得花时间填了一堆没人读的字段。

### 真正被代码读取 / 校验的键（**必须认真填**）

| 键 | 谁在读 | 填错会怎样 |
|---|---|---|
| `name` | `load_feature_config` 提升进工作配置 | 只影响标识；但建议与文件名一致 |
| `sequence_length` | `get_sequence_length()` → 张量第 2 维 | 张量形状全错 |
| `protospacer_length` | `get_protospacer_length()` → 推 PAM 位置 | PAM/protospacer 区间划分错 |
| `channel_count` | **声明了就会校验**，与由映射算出的通道数比对 | 报错 `算出的通道数 N 不一致，请同步修改` |
| `label_column` | 标签归一化程序：y 的**原始列名** | 找不到该列 → 取不到 y |
| `label_normalized` | `yes`/`no`，与 `method` 交叉校验 | 两者矛盾会直接报错 |
| `label_normalization` | `{method, divisor, minmax_range, note}` | 方法非法/缺参数会报错 |
| `feature_config` | **特征工程真正读的主体** | 缺这个块就退回"整个 blob 当配置" |
| `feature_config.environment_features` | **必须存在这个键** | 缺了直接 `ValueError` |
| `feature_config.sequence_channels` | one-hot 通道顺序 | 顺序错 → 通道含义整体错位 |
| `feature_config.metadata_columns` | 写进 `*_metadata.csv` 的列 | 数据里没有的列会被自动跳过，不报错 |

### 只是"给人看"的键（代码**零引用**，写不写都不影响运行）

`schema_version` · `description` · `raw_dir` · `processed_dir` · `has_environment` ·
`sequence_column` · `label_scale` · `input_sequence_nt` · `pam` · `cell_lines` ·
`raw_layout` · `source` · `notes`

> 这些字段**强烈建议照填**：它们是这个数据集的口径档案（下一节会说明各自该写什么），
> 只是**不要指望它们能改变程序行为**。特别地：`raw_dir` / `processed_dir` 不会被自动读取，
> 特征工程命令行要**显式**传 `--raw-data` / `--output-dir`。

---

## 1. 四步接入新数据集

```bash
# ① 放原始数据
data/raw/<你的数据集名>/xxx.csv          # 单文件或按细胞系多文件都行

# ② 写配置（就是本文件教的东西）
data/config/<你的数据集名>.json

# ③ 跑特征工程：原始 CSV → 张量 + feature_schema.json
python core/features/engineering/feature_engineering.py \
    --raw-data   data/raw/<你的数据集名> \
    --output-dir data/processed/<你的数据集名> \
    --config     data/config/<你的数据集名>.json \
    --format     <适配器名>              # 见 §6，自动识别不了时必须显式给

# ④ 校验「配置声明 == 实际张量」，通过后再训练
python core/features/engineering/validate_feature_schema.py --data-set <你的数据集名>
python train.py --model cnn --split-type single --cell-lines <细胞系> \
    --environment sequence --data-dir data/processed/<你的数据集名>
```

---

## 2. 逐字段讲解（按 JSON 里的实际顺序）

> 📌 本节代码块里的 `//` 是**讲解性标注**，JSON 本身不支持注释 —— 照抄进你自己的文件时**请删掉**。
> 需要整段复制的完整样例见 **§5**（那份是严格合法的 JSON）。每个片段下面都有一张表
> 逐个说明"这个键在设置什么"，所以删掉注释不影响阅读。

下面每一小节都给出**三个样例各自怎么填**，你照着改就行。

### 2.1 `schema_version` / `name` / `description`

```json
"schema_version": 2,
"name": "DeepCRISPR",
"description": "每个数据集一个自包含配置文件：数据集规格 + 特征映射 + 序列长度。人可直接编辑。",
```

| 键 | 在设置什么 |
|---|---|
| `schema_version` | 配置格式版本，三份样例都是 `2`。**代码不读它**，改它不会改变行为 |
| `name` | 数据集名。三份分别是 `DeepCRISPR` / `Hiranniramol` / `Labuhn`，**与文件名一致** |
| `description` | 一句话说明，随便写 |

### 2.2 `raw_dir` / `processed_dir`

```json
"raw_dir": "data/raw/DeepCRISPR",
"processed_dir": "data/processed/DeepCRISPR",
```

| 键 | 在设置什么 |
|---|---|
| `raw_dir` | 原始数据目录（相对仓库根）。**仅文档** —— 命令行请传 `--raw-data` |
| `processed_dir` | 处理后产物的落点。**仅文档** —— 命令行请传 `--output-dir` |

> 三份样例都把这两个路径写成与数据集名一致，方便你复制进命令。

### 2.3 `sequence_length` / `protospacer_length` —— 最重要的两个数

```json
"sequence_length": 23,
"protospacer_length": 20,
```

| 键 | 在设置什么 |
|---|---|
| `sequence_length` | **张量第 2 维**（位置数）= 一条输入序列有几个 nt。三份样例都是 `23` |
| `protospacer_length` | 其中**引导序列（spacer）**占多少 nt。三份样例都是 `20` |

PAM 长度与位置区间**不是单独填的**，而是由上面两个数算出来：

```
pam_length       = sequence_length - protospacer_length   # 23 - 20 = 3
protospacer 位置 = 1 .. protospacer_length                # 1–20
PAM 位置         = protospacer_length+1 .. sequence_length # 21–23
```

**若你的数据没有 PAM**（例如只给 20 nt spacer），就把两个数填成相等：

```json
"sequence_length": 20,
"protospacer_length": 20,
```

此时 PAM 位置区间为空，`feature_schema.json` 会声明「张量里不含 PAM」。

### 2.4 `has_environment` / `channel_count`

```json
"has_environment": true,      // DeepCRISPR
"channel_count": 8,
```
```json
"has_environment": false,     // Hiranniramol / Labuhn
"channel_count": 4,
```

| 键 | 在设置什么 |
|---|---|
| `has_environment` | 这个数据集**有没有表观遗传通道**。**仅文档**，真正起作用的是 `environment_features` 列了几项 |
| `channel_count` | 通道总数 = `len(sequence_channels)` + 启用的环境特征数。**声明了就会被校验** |

`channel_count` 是**防手滑**用的：你改了 `environment_features` 却忘了改这个数，特征工程会直接报错
（`声明的 channel_count=8 与算出的通道数 7 不一致，请同步修改`），不会静默产出错张量。
算不准就**先删掉这一行**，跑一次看报错里算出的值，再写回来。

### 2.5 `sequence_column` / `label_column` / `label_*` —— 原始文件里的两列

```json
"sequence_column": "sgRNA",
"label_column": "Normalized efficacy",
"label_normalized": "no",
"label_normalization": { "method": "none", "divisor": null, "note": "上游已归一化到 [0,1]；本仓库只校验、不缩放。" },
"label_scale": "[0,1]（上游已归一化，适配器不再缩放）",
"input_sequence_nt": 23,
```

| 键 | 在设置什么 |
|---|---|
| `sequence_column` | 原始 CSV 里放**序列**的列名（DeepCRISPR 是 `sgRNA`，Labuhn 是 `sgRNA_sequence`，Hiranniramol 是 `gRNA`）。**仅文档** |
| `label_column` | 原始 CSV 里放**编辑效率 y** 的列名。**承重**：标签归一化程序按它取列 |
| `label_normalized` | `"yes"` / `"no"`：y 是否需要被归一化。**承重** |
| `label_normalization` | 归一化方法，见下。**承重** |
| `label_scale` | 人类可读说明。仅文档（保留是为了让读者一眼看懂量纲） |
| `input_sequence_nt` | 原始序列的 nt 数。**仅文档** |

#### y 归一化：三种数据集、三种填法

程序只认 `method` 的三种取值：

| `method` | 做什么 | 什么时候用 | 样例 |
|---|---|---|---|
| `none` | **只校验** y ∈ [0,1]，不缩放 | 原始列本来就是 [0,1] | `DeepCRISPR.json`、`Labuhn.json` |
| `divide` | `y / divisor` | 原始列是百分制等 | `Hiranniramol.json`（`divisor: 100.0`） |
| `minmax` | `(y - lo) / (hi - lo)` | 原始列是任意量纲 | 本仓库暂无样例 |

**只校验不缩放**（`DeepCRISPR.json` / `Labuhn.json`）：

```json
"label_column": "Normalized efficacy",
"label_normalized": "no",
"label_normalization": {
  "method": "none",
  "divisor": null,
  "note": "上游（Chuai et al. 2018）已把效率列归一化到 [0,1]；本仓库只校验、不缩放。"
},
```

**除以 100**（`Hiranniramol.json`）：

```json
"label_column": "Edit Efficiency",
"label_normalized": "yes",
"label_normalization": {
  "method": "divide",
  "divisor": 100.0,
  "note": "原始为 0-100 百分制，除以 100 映射到 [0,1]。"
},
```

**线性映射到 [0,1]**（暂无样例，供需要时用）：

```json
"label_normalized": "yes",
"label_normalization": {
  "method": "minmax",
  "minmax_range": [0.0, 8.0],
  "note": "原始打分 0–8，线性映射到 [0,1]。"
},
```

> ⚠️ `method` 必须与 `label_normalized` 一致：填 `"no"` 却给 `method: "divide"`（或反过来）
> 会**直接报错**，不会被静默忽略。
>
> ⚠️ `minmax` 的 `minmax_range` **必须来自训练集统计量**。用全量数据（含验证/测试）的
> 极值去拟合，等于把测试集信息泄漏进训练输入。程序不替你算这两个数。
>
> 改完可以先干跑一次审计，确认每个数据集的 y 口径：
>
> ```bash
> python -m core.features.engineering.label_normalization
> ```

### 2.6 `pam`

```json
"pam": "NGG（第 21-23 位；PAM 已包含在 23 nt 输入中）",
```
```json
"pam": "由 extended_spacer 定位后取后 3 nt（要求 GG）",
```

| 键 | 在设置什么 |
|---|---|
| `pam` | 这个数据集的 PAM 是什么、在序列里怎么定位。**仅文档**，但**很值得写清楚**：PAM 的取向直接决定「第 21–23 位」这类说法成不成立 |

### 2.7 `cell_lines` / `raw_layout` / `source` / `notes`

```json
"cell_lines": ["hct116", "hek293t", "hela", "hl60"],
"raw_layout": "每细胞系一个 CSV：hct116.csv / hek293t.csv / hela.csv / hl60.csv",
"source": "Chuai et al. 2018, Genome Biology (DeepCRISPR)",
"notes": "唯一带表观遗传通道的数据集；four-cell-line 与 LOCO 划分只在此数据集上有意义。",
```

| 键 | 在设置什么 |
|---|---|
| `cell_lines` | 该数据集包含哪些细胞系。**仅文档**（细胞系是靠扫描 `data/processed/<数据集>/*_features_*.npy` 反推的，不读这里） |
| `raw_layout` | 原始文件长什么样：单文件还是每系一个。**仅文档** |
| `source` | 数据出处，写论文引用。**仅文档**，但**必须写**（复用别人的数据要能溯源） |
| `notes` | 使用这个数据集的注意事项。**仅文档**，但很有价值：三份样例的 `notes` 都在提醒"标签语义与别的数据集不同、不能直接比数值" |

> 纯序列数据集（`Hiranniramol` / `Labuhn`）的 `cell_lines` 只有一项：`["labuhn"]` / `["hiranniramol"]`。

### 2.8 `feature_config` 块 —— 特征工程真正读的部分

这个块才是特征工程 CLI `--config` 实际读取的内容；顶层那些键会被"提升"进来一起用。

```json
"feature_config": {
  "sequence_channels": ["A", "C", "G", "T"],
  "environment_features": [ ... ],
  "metadata_columns": [ ... ]
}
```

#### `sequence_channels` —— one-hot 通道顺序

```json
"sequence_channels": ["A", "C", "G", "T"],
```

三份样例**都是这四个、都是这个顺序**。通道下标由此确定：`A(0) C(1) G(2) T(3)`，
环境通道接在它们后面。**顺序不能随意改** —— 改了通道含义会整体错位。

#### `environment_features` —— 表观/环境轨道列表

**这个键必须存在**（纯序列数据集就写空列表 `[]`，见 `Labuhn.json`）。
每一项的五个字段：

```json
{
  "name": "CTCF",                  // ① 通道名
  "column": "CTCF",                // ② 原始 CSV 里的列名
  "type": "per_position_binary",   // ③ 特征类型
  "encoding": { "A": 1, "N": 0 },  // ④ 原始字符串 -> 数值的映射
  "enabled": true                  // ⑤ 是否启用
}
```

| 字段 | 在设置什么 |
|---|---|
| `name` | 通道名。会进 `channel_names`，也进逐特征名（如 `pos5_CTCF`） |
| `column` | 去原始 CSV 的哪一列取值。**可以和 `name` 不同**（例如表头带空格/单位） |
| `type` | 三种之一，见下表 |
| `encoding` | **原始字符串 → 数值**的映射表。⚠️ 这是"`A` 表示可及"这种**语义**映射，**不是**通道下标 |
| `enabled` | 布尔。`false` = 该特征不生成通道（方便临时关掉某条轨道做消融） |

`type` 的三种取值：

| `type` | 原始数据形态 | 输出形状 | 适用 |
|---|---|---|---|
| `per_position_binary` | 每个样本一个 L 字符串，如 `AAAAAANNNAA…` | `L × 1` | 逐位点"可及/不可及"（CTCF、Dnase、H3K4me3、RRBS 都是这种） |
| `per_position_numeric` | 每个样本 L 个连续值（`[0.1,0.2,…]` 或 `"0.1,0.2,…"`） | `L × 1` | 逐位点连续信号 |
| `global_numeric` | 每个样本**一个**标量 | `1 × 1` → **广播到 L 个位置** | 全局打分（NHEJScore、CellCycleScore 等） |

> `global_numeric` 之所以要广播，是为了让 CNN / Transformer 能拿到与序列等长的张量
> —— 三段（序列、逐位点环境、全局环境）长度一致，不需要额外对齐。

#### `metadata_columns` —— 写进 metadata 的列

```json
"metadata_columns": ["Cell line", "Chromosome", "Start", "End", "Strand", "sgRNA"],
```
```json
"metadata_columns": ["Cell line", "Strand", "sgRNA"],
```

| 键 | 在设置什么 |
|---|---|
| `metadata_columns` | 要把原始 CSV 的**哪些列**带进 `*_metadata.csv`。**数据里不存在的列会被自动跳过**，不会报错 |

这些列是下游分析（motif、外部验证）回溯"这条序列来自哪个位点/哪条链"的依据。
纯序列数据集没有坐标列，就少写几个（如 `Hiranniramol` / `Labuhn` 只写 `Cell line` / `Strand` / `sgRNA`）。

---

## 3. 抄哪一份？按你的数据形态选

| 你的数据 | 抄这份 | 要改什么 |
|---|---|---|
| 有 4 个（或 N 个）细胞系、每个一个 CSV、带表观通道 | `DeepCRISPR.json` | 细胞系列表、`raw_layout`、`environment_features` 的列名 |
| 单文件、纯序列、已归一化到 [0,1] | `Labuhn.json` | `sequence_column` / `label_column` / `metadata_columns` |
| 单文件、纯序列、**百分制** | `Hiranniramol.json` | 同上 + `divisor` |

三份的 `sequence_length` / `protospacer_length` / `sequence_channels` / `feature_config` 结构
**都一样**，差别只在：细胞系数、`environment_features` 有几项、标签列叫什么、y 要不要缩放。

---

## 4. 常见错误与排查

| 症状 | 原因 | 怎么修 |
|---|---|---|
| `feature config 缺少 'environment_features'` | `feature_config` 块里没这个键 | 纯序列就写 `"environment_features": []`，别省 |
| `声明的 channel_count=N 与算出的通道数 M 不一致` | 改了 `environment_features` 但没同步 `channel_count` | 改成 M（或先删掉 `channel_count` 这行） |
| `找不到列 [...]；实际列=[...]` | 适配器要求的列名对不上 | 对着报错里的"实际列"改原始 CSV 表头，或用 `--format` 指定正确适配器 |
| `无法识别原始文件格式` | 列名不匹配任何已知格式 | 用 `--format` 显式指定，见 §6 |
| `[...] 规范化后效率仍在 [0,1] 之外` | y 没归一化，或归一化方法填错 | 检查 `label_normalization`；百分制就写 `divide` + `divisor: 100` |
| `label_normalized='yes' 与 method='none' 矛盾` | 两个字段打架 | 让它们一致：要归一化就换非 `none` 的 method |
| 训练报通道数不对 | `sequence_length` / `channel_count` 与实际张量不符 | 跑 `validate_feature_schema.py --data-set <名>` 定位 |

排查顺序建议：**先跑特征工程**（它会打印每个细胞系的 `rows_in -> rows_out` 与效率区间），
**再跑 schema 校验**（它会断言"声明 == 实际"），两步都过再训练。

---

## 5. 最小可运行示例（纯序列、无 PAM、y 已是 [0,1]）

假设你的数据叫 `MyData`，单文件 `data/raw/MyData/MyData.csv`，列名：
`cell_line, sequence, efficiency`，序列是 20 nt spacer + 3 nt PAM。

```json
{
  "schema_version": 2,
  "name": "MyData",
  "description": "自建数据集：20 nt spacer + NGG。",

  "raw_dir": "data/raw/MyData",
  "processed_dir": "data/processed/MyData",

  "sequence_length": 23,
  "protospacer_length": 20,

  "has_environment": false,
  "channel_count": 4,
  "sequence_column": "sequence",
  "label_column": "efficiency",
  "label_normalized": "no",
  "label_normalization": {
    "method": "none",
    "divisor": null,
    "note": "上游已归一化到 [0,1]。"
  },
  "label_scale": "[0,1]",
  "input_sequence_nt": 23,
  "pam": "NGG（第 21-23 位）",
  "cell_lines": ["mydata"],
  "raw_layout": "单文件：MyData.csv",
  "source": "（填你的出处）",
  "notes": "（填注意事项，例如标签语义与其它数据集是否可比）",

  "feature_config": {
    "sequence_channels": ["A", "C", "G", "T"],
    "environment_features": [],
    "metadata_columns": ["cell_line", "sequence"]
  }
}
```

---

## 6. ⚠️ 关于"不用改 Python"这句话的边界

本文件开头那种"加个 JSON 就能接新数据集"的说法，**只在你的原始 CSV 能套用现有三种适配器时成立**。

特征工程先要判断"这个 CSV 是什么格式"，可选值**只有三个**：

```python
ADAPTERS = {"deepcrispr": ..., "hiranniramol": ..., "labuhn": ...}   # dataset_adapters.py
```

| 格式 | 自动识别依据（看列名） | 它对原始文件的假设 |
|---|---|---|
| `deepcrispr` | 有 `Chromosome` + `Normalized efficacy` | 序列列已是全长（含 PAM）；可选 4 个表观列 |
| `hiranniramol` | 有 `Extended Target`，或有 `Edit Efficiency` + `gRNA` | 20 nt gRNA + 53 nt 延长序列，靠定位切全长 |
| `labuhn` | 有 `KO_reporter_assay` | 20 nt spacer + 30 nt `extended_spacer`，靠定位切全长 |

所以：

- **你的 CSV 列名能凑上其中一种**（哪怕列名不同，只要结构一致），就能**只写 JSON**，
  用 `--format` 显式指过去即可；
- **结构也不同**（比如给的是 `基因名 + 20nt spacer + 效率`，没有任何延长序列），
  就必须在 `core/features/engineering/dataset_adapters.py` 里**新增一个适配器**
  并注册进 `ADAPTERS`，然后 `--format <新名字>`。这一步**需要改 Python**。

写新适配器时按现有三个的模板来：返回 `(规范 DataFrame, AdaptReport)`，列名用
`dataset_adapters.py` 顶部那几个常量 —— 序列列 `CANONICAL_SEQUENCE`、标签列
`CANONICAL_TARGET`、坐标组 `COORD_COLUMNS`；并把 y 交给 `label_spec.apply(...)` 处理
—— **不要**在适配器里硬编码 `/100` 之类的缩放（那正是本仓库刚统一掉的做法，
见 `core/features/engineering/label_normalization.py`）。

---

## 7. 这份配置被用到的两条路径

```
特征工程  feature_engineering.py --config data/config/<名>.json
   │         load_feature_config()：展开 feature_config 块 + 提升
   │         sequence_length / protospacer_length / channel_count /
   │         name / metadata_columns / label_* 到工作配置
   ├─> 适配器      raw CSV → 规范列（序列 + y），y 经 LabelNormalizationSpec 归一化
   └─> 特征工程    规范列 → features_*.npy + *_labels.npy + *_metadata.csv
                              + feature_schema.json（声明通道/长度/PAM/展平公式）
                              + feature_engineering_summary.csv（逐细胞系的收编统计）

训练      train.py --data-dir data/processed/<名>
   └─> 读 feature_schema.json（不是本文件）决定输入形状
```

> 注意最后一行：**训练读的是 `feature_schema.json`，不是这份配置**。
> 本文件是"怎么从原始 CSV 造出张量"的说明书；`feature_schema.json` 是"造出来的张量长什么样"
> 的声明。两者由 `validate_feature_schema.py` 对账。
