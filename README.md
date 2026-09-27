***面向多类基因编辑系统的细胞环境感知型核酸工具智能设计与调控机理挖掘平台***

> **参赛赛道**：赛道二 · AI 基因编辑与核酸工具设计
> **公开数据库**：DeepCRISPR，Labuhn，Hiranniramol


目录

|-项目一览
|----1. 项目简介
|----2. 仓库结构
|----3. 运行环境
|----4. 数据集
|-项目流程
|----5. Feature schema（编写映射）
|----6. 预处理（Preprocessing）
|----7. 单 run 训练
|----8. 数据挖掘（Data Digging）
|----9. 评估（Evaluation）
|----10. 结果汇总（Result Collection）
|----11. 报告生成与表格收录
|----12. 可视化（Visualization）
|----13. 输出结构
|----14. 完整流程代码
|----15. 可复现性
|-项目底层
|----四维度：一个结论凭什么成立
|----16. 模型性能指标（Performance metrics）
|------16.1 各指标的原理与目标
|------16.2 使用时必须注意的三个约定
|----17. 可解释性方法（XAI methods）
|------17.0 白名单（规定了"允许存在什么"）
|------17.1 线性回归：回归系数 + 经典参数推断
|------17.2 XGBoost：三种分裂重要度 + TreeSHAP
|------17.3 MLP：积分梯度（IG）
|------17.4 CNN：双分支 IG + 无符号 ISM
|------17.5 Transformer：注意力强度 + 注意力熵
|------17.6 归因稳健性：SNR 的 4 档分档
|----18. 统计分析方法（Statistical methods）
|------18.1 ANOVA：析因方差分析（Type-II 边际 / 额外平方和）
|------18.2 Permutation test：符号翻转随机化检验
|------18.3 Fisher 精确检验（Motif 富集）
|------18.4 Bootstrap（6 种 CI 类型）
|----19. 统计量
|------19.1 p-value
|------19.2 FDR（Benjamini–Hochberg q 值）
|------19.3 CI（置信区间）—— 共 6 种类型
|------19.4 η²（方差解释比）
|----附：底座的其它支撑机制
|-R1. 赛道二规律发现交付物（rule_discovery.py）
|----这个程序做什么
|----运行
|----输出结构
|----筛选顺序（三个文件各自的判据）
|------<cell_line>_microenv.csv / <cell_line>_motif.csv / microenv_and_motif.csv
|----可配置的评选标准
|----科学声明
|-R2. results/summary/<batch> 文件说明
|----目录一览
|----JSON（analysis_plan / analysis_status / execution_log）
|----figures（environment / plots）
|----report（overview / data_quality / anomaly_report）
|----train_data/ 与 feature_importance/（含 cnn7_validation）
|----tables（20 张表逐表说明）
|----术语表
|----科学声明
|-R3. 平台外部验证说明
|----R3.0 为什么要做外部验证
|----R3.1 数据集层面的外部复现（Hiranniramol / Labuhn）
|----R3.2 模型层面的外部对照（第三方预测平台 CRISPRon）
|----R3.3 结论汇总：哪些 DeepCRISPR 挖掘结果被外部照应
|----R3.4 局限与边界
|-R4. 平台建设说明

---
# 项目一览

## 1. 项目简介

**核心挖掘目标**：微环境，序列motif，跨细胞泛化。
**平台核心目的**: 项目在**预测 → 模型归因 → 统计检验 → 跨细胞系比较 → 证据整合 → 生物学假设** 过程中，整合不同特征输入方式所得训练结果，从而**对微环境重要性进行评估**,**提取可靠序列motif以设计更好gRNA**,**总结细胞微环境和序列motif特异性**，为科研人员进行进一步实验提供方向与依据。


**输入 / 输出**

| | 内容 |
|---|---|
| **输入** | `data/raw/<数据集>/*.csv`、`data/config/<数据集>.json`（编码配置 + 序列长度）|
| **处理** | `core/`（特征工程 → 划分 → 模型 → 归因）、`workflows/`（训练 / 挖掘 / 预测 / 编排）、`analysis/`（汇总 → 统计 → 证据 → 报告 → 图） |
| **输出** | `results/<batch>/<run>/`（逐 run 指标与预测）、`results/summary/<batch>/`（批次汇总、图、特征库）、`models/<batch>/<run>/`（权重）、`results/logs/<batch>/`（日志） |
| **模型** | 线性回归 / XGBoost / MLP / 双分支 CNN（卷积核 3/5/7）/ Transformer，共 7 个配置 |
| **分析目标** | 环境通道与序列motif的增量预测价值及跨细胞系泛化|

**规模**：3 个数据集、5 类模型（7 配置）× 16 环境组合 × 3 种划分 = DeepCRISPR 上 1344 次受控实验。

---

## 2. 仓库结构

```text
★ 用户入口（工作目录）
train.py                         单次实验（§7）
data_digging.py                  网格批量挖掘，参数最完整的用户级 CLI（§8）
rule_discovery.py                赛道二规律发现交付物生成（R1）                       （主运行入口）

data/                            数据（不写代码，只放数据与配置）
  raw/<数据集>/                  原始 CSV（输入）
  processed/<数据集>/            处理后张量 + feature_schema.json（模型输入）
  config/                        ← 用户配置区（见 §5）；**一数据集一文件**
    DeepCRISPR.json                        数据集规格 + 特征映射 + 序列长度（8 通道）
    Hiranniramol.json                      同上（4 通道，纯序列）
    Labuhn.json                            同上（4 通道，纯序列）
    README.md                              

core/                            科学引擎（二次开发区）
  common/paths.py                唯一权威路径解析（--data-set 解析在这里）
  features/engineering/          特征工程 + schema 生成 + schema 校验器（feature_engineering.py /
                                 dataset_adapters.py / validate_feature_schema.py）
  features/channels/             通道组合与环境 lattice（cell_environment_combination.py）
  data/cell_line_division.py     身份类划分 + 泄漏审计
  models/<5 类模型>/             线性回归 / XGBoost / MLP / 双分支 CNN / Transformer
  xai/xai_importance.py          归因白名单（每模型允许导出的列）

analysis/                        分析层（只读训练产物，不训练）
  入口与编排
    pipeline.py                  分析引擎（统计 / 证据 / 报告 / 图表）
    collect_results.py           指标汇总 → summary/<数据集>/train_data/
    importance_extraction.py     关键调控特征库 → summary/<数据集>/feature_importance/
    config.py                    全部分析阈值与参数的唯一出处
    plans.py / schemas.py        分析计划与统一记录类型
    data/loaders.py              统一实验表加载 + runs/summary 两棵子树的路径解析
    data/validation.py           指标自相矛盾校验（→ anomaly_report）
  分析主题
    prediction.py                预测/泛化性能（model×split、LOCO）
    attribution/                 归因抽取（extractors / columns）
    environment/                 环境效应（incremental_effect / factorial_dag）
    sequence/motif/              motif 提取 / 聚类 / 富集（core / iupac / pipeline，**只产数据表**）
    stats/                       bootstrap / hypothesis_tests / multiple_testing / tasks
  出图与报告
    visualization/               证据图渲染（environment_plots / attribution_plots /
                                 factorial_dag / core）
    reports/markdown_report.py   三份交付报告的正文构造函数
  外部模型验证链（自持）
    crispron_validation/         CNN7 × CRISPRon 的 Pos18 C→A 一致性验证
                                 （windows / crispron / model / table / pipeline / __main__）
    candidates/                  WT 位点候选挑选（同属序列–效率方向性检验）
    leakage.py                   身份类与泄漏工具（被 candidates 与 crispron_validation 引用）

results/                         输出（全部由程序生成）
  train_results/<数据集>/<run>/  逐 run 产物：指标 JSON / 预测 CSV / 归因 CSV / info.txt
                                 ※ 本次提交随附的 1344 次实验落在这里（无 batch 层）
  summary/<数据集>/              批次级汇总（结构见 §13）
    analysis_plan.json / analysis_status.json / execution_log.json
    reports/  tables/  figures/  feature_importance/  train_data/
    cnn7_validation.md / .csv    CNN7×CRISPRon 一致性验证交付物
  crispron_validation/           外部验证的中间产物（窗口表 / CRISPRon 原始输出 / CNN 权重与预测）

models/<数据集>/<run>/           模型权重（.pkl / .pt / .json）
logs/<数据集>/<run>/             训练日志

deploy/
  requirements.txt               依赖清单（兼容区间，见 §14 阶段 0 注）
  crispron/                      第三方模型 CRISPRon 运行时（外部验证用，约 2.1 GB）
```

---

## 3. 运行环境

### 项目调试环境

| 项 | 审计栈（开发 / 分析） | 超算目标栈（训练） |
|---|---|---|
| 用途 | 跑 `analysis/`、`rule_discovery.py`、出报告与图 | 跑 `train.py` / `data_digging.py` 生成 `results/train_results/` |
| OS | Linux（WSL2，glibc 2.39） | CentOS 7（glibc 2.17） |
| Python | 3.12.3 | 3.10.21 (conda-forge) |
| PyTorch | 2.13.0+cu130 | 2.6.0+cu124 |
| CUDA / GPU | CUDA 13.0 / **无可用 GPU**（纯 CPU 运行） | CUDA 12.4 / 8 × A100-SXM4-80GB |
| NumPy | 2.5.2 | 2.2.6 |
| pandas | 3.0.5 | 2.3.3 |
| SciPy | 1.18.0 | 1.15.2 |
| scikit-learn | 1.9.0 | 1.7.2 |
| XGBoost | 3.4.1 | 2.0.3 |
| joblib | 1.5.3 | 1.6.0 |

> 「项目调试环境」表里的审计栈（numpy 2.5.2 / pandas 3.0.5 / torch 2.13）版本更新，同样能跑通全链路
> （本仓库的分析产物就是在它上面生成的），但**不是**交付训练产物的来源。对外披露复现环境时请以本表为准。

### 核心依赖

| 类别 | 包 | 说明 |
|---|---|---|
| 硬依赖（代码顶层 `import`） | `numpy` `pandas` `scipy` `scikit-learn` `xgboost` `torch` `matplotlib` `seaborn` | 前 6 个跑模型，后 2 个出图 |
| 可选（两个装一个即可） | `xlsxwriter` **或** `openpyxl` | 只有 `rule_discovery.py` 写 Excel 汇总视图时需要。两者都没装时它会明确报错并提示安装命令，CSV 主结果不受影响 |
| 本项目主环境**不需要** | `shap` `statsmodels` `tabulate` `biopython` | 全仓库**零引用**（`biopython` 只被 CRISPRon 自己的代码使用，见下） |

### 第三方 CRISPRon 的隔离环境

| 项 | CRISPRon venv |
|---|---|
| Python | 3.10.20 |
| TensorFlow / Keras | 2.14.0 / 2.14.0 |
| NumPy / pandas | 1.26.4 / 2.2.2 |
| scikit-learn | 1.4.2 |
| biopython | 1.83 |
| 另需 | ViennaRNA `RNAfold` 可执行文件（仓库内由 `deploy/crispron/software/wrappers/RNAfold` 提供） |

> 安装步骤与自检记录见 `deploy/crispron/INSTALL_NOTES.md` 与
> `deploy/crispron/logs/selftest_result.md`（官方 `bin/test.sh` 结果 `TEST ok`）。

### 依赖库安装

```bash
pip install -r deploy/requirements.txt
```

---

## 4. 数据集

**平台以支持用户挖掘自己的数据集为目标,导入数据集应以csv格式为准, 格式参考data/raw中的csv。**

平台附有三个数据集,各由一个**自包含**配置文件登记：`data/config/DeepCRISPR.json` / `Hiranniramol.json` / `Labuhn.json`。若新增数据集需编写相应的.json, 说明见 `data/config/README.md`。

### 4.1 DeepCRISPR（主数据集）

| 项 | 内容 |
|---|---|
| Source | Chuai et al. 2018, *Genome Biology*, "DeepCRISPR: optimized CRISPR guide RNA design by deep learning"，doi:[10.1186/s13059-018-1459-4](https://doi.org/10.1186/s13059-018-1459-4) |
| License | 遵循原始出版物与其数据仓库的条款；**使用前请到原文/原仓库确认** |
| Raw location | `data/raw/DeepCRISPR/{hct116,hek293t,hela,hl60}.csv` |
| Processed location | `data/processed/DeepCRISPR/` |
| Sequence field | `sgRNA`（23 nt） |
| Label field | `Normalized efficacy`（已是 [0,1]，不再缩放） |
| Sequence length | 23 nt（20 nt protospacer + 3 nt PAM） |
| PAM | NGG，位于第 21–23 位（已包含在张量内） |
| Cell lines | hct116 / hek293t / hela / hl60 |
| Environment features | CTCF, Dnase, H3K4me3, RRBS（逐位点二值可及性） |
| Preprocessing | `feature_engineering.py --config data/config/DeepCRISPR.json` → 23×8 = 184 维 |
| Split | `single`（细胞内 70/15/15）、`all`（真实 LOCO，训练池先剔除留出系同源序列）、`mixed`（跨细胞系 70/15/15，种子 42–45） |

### 4.2 Hiranniramol（外部复现数据集）

| 项 | 内容 |
|---|---|
| Source | Hiranniramol et al. 2020, *Bioinformatics* 36(9):2684–2689，doi:[10.1093/bioinformatics/btaa041](https://doi.org/10.1093/bioinformatics/btaa041) |
| License | 遵循原始出版物条款；使用前请到原文确认 |
| Raw location | `data/raw/Hiranniramol/Hiranniramol.CSV`（单文件） |
| Processed location | `data/processed/Hiranniramol/` |
| Sequence field | `gRNA`（20 nt） |
| Label field | `Edit Efficiency`（**原始为 0–100 百分制**，适配器自动 /100） |
| Sequence length | 构造为 23 nt（gRNA + 后 3 nt PAM） |
| PAM | 由 `Extended Target` 定位 gRNA 后取后 3 nt，要求 GG |
| Cell lines | 见原论文 |
| Environment features | **无**（纯序列，4 通道 / 92 维） |
| Preprocessing | `--config data/config/Hiranniramol.json --format hiranniramol` |
| Split | 仅 `single`（70/15/15） |

### 4.3 Labuhn（外部复现数据集）

| 项 | 内容 |
|---|---|
| Source | Labuhn et al. 2018, *Nucleic Acids Research* 46(3):1375–1385，doi:[10.1093/nar/gkx1268](https://doi.org/10.1093/nar/gkx1268) |
| License | 遵循原始出版物条款；使用前请到原文确认 |
| Raw location | `data/raw/Labuhn/Labuhn.CSV`（单文件） |
| Processed location | `data/processed/Labuhn/` |
| Sequence field | `sgRNA_sequence`（20 nt） |
| Label field | `KO_reporter_assay`（**已是 [0,1]**，不再缩放） |
| Sequence length | 构造为 23 nt |
| PAM | 由 `extended_spacer` 定位后取后 3 nt，要求 GG |
| Cell lines | 见原论文 |
| Environment features | **无**（纯序列，4 通道 / 92 维） |
| Preprocessing | `--config data/config/Labuhn.json --format labuhn` |
| Split | 仅 `single`；同一 sgRNA 的重复测量按均值合并（5 条） |

 **不同数据集的 label semantics 不同，不可当作同一个物理量比较,只能读作"同一流程在不同数据集上的可预测性"**，

---
# 项目流程

## 5. Feature schema（编写映射）

### 编写json:

若新增了数据集,需编写**`data/config/<数据集>.json`** ，在 JSON 内完成**新增环境通道**、**碱基符号**的映射与**序列长度**（`sequence_length` / `protospacer_length`）的调整。


### 验证"声明 == 实际":

```bash
# 校验全部数据集
python core/features/engineering/validate_feature_schema.py --all

# 校验单个数据集
python core/features/engineering/validate_feature_schema.py --data-set DeepCRISPR

```
该脚本会断言：通道数/序列长度/展平维度与张量一致、`channel_names` 的环境段顺序与 `environment_features`
声明一致、**由 metadata 序列重建的 one-hot 与张量逐元素相等**、环境取值 ⊆ {0,1}、
`layout` 与 `model_compatibility` 声明自洽。退出码 0 = 通过。

---

## 6. 预处理（Preprocessing）

**入口**：`core/features/engineering/feature_engineering.py`

**作用**：把原始 CSV 变成模型可直接读取的张量与编码声明，并把每一步的收编/剔除量记录成可审计的汇总表。完整步骤如下：

1. **输入发现**：`--raw-data` 可给单个文件或目录；给目录时**递归**查找 `*.csv` / `*.CSV`，文件名去扩展名即细胞系名；`--cell-lines` 只处理其中几个，且顺序即输出顺序（指定了不存在的名字会报错并列出可用名）。
2. **适配器选择**：显式 `--format {deepcrispr,hiranniramol,labuhn}`，留空则按列名自动识别，识别不出**直接报错**而不是猜。适配器负责三件事：规范列齐备性检查（缺列即报错并打印实际列名）、把原始列映射成统一列（序列 / 标签 / 坐标 / 元数据）、构造 23 nt 全长序列：
   - `deepcrispr`：序列列已是 23 nt 全长（含 PAM），要求 `Chromosome` + `Normalized efficacy`，可选 4 个表观列；
   - `hiranniramol`：20 nt `gRNA` + 53 nt `Extended Target`，在延长序列里**定位** gRNA 后取后 3 nt 作 PAM，要求 GG；
   - `labuhn`：20 nt `sgRNA_sequence` + 30 nt `extended_spacer`，同样定位取 PAM，并**按序列合并重复测量**（取均值）后一行一序列。
3. **适配层硬校验与剔除**（每类剔除量逐条记入 `adapter_drop_reasons`）：PAM 不以 GG 结尾默认**拒绝**（`--allow-non-gg-pam` 可放行）；序列含非 `[ACGT]` 字符剔除；spacer 不在延长序列区间内剔除；标签缺失或越界剔除；整行完全重复的记录删除（`remove_duplicate_rows`，只删整行全同的行）。
4. **标签归一化**：按配置的 `label_normalization` 走 `none`（只校验 y ∈ [0,1]）、`divide`（y / `divisor`）或 `minmax`（`(y − lo) / (hi − lo)`），并与 `label_normalized` 字段**交叉校验**，口径矛盾直接报错。`minmax_range` 必须来自训练集统计量，程序不替你算（否则等于把测试集信息泄漏进输入）。
5. **序列 one-hot 编码**：4 条序列通道（顺序由 `sequence_channels` 声明，本项目为 A/C/G/T）× `sequence_length`（23）→ `(N, 23, 4)`。
6. **环境（表观）通道编码**：`environment_features` 声明的每条轨道按**逐位点映射**编码成 `(N, 23, 1)`，再沿通道轴拼接。轨道长度必须等于 `sequence_length`；出现配置里未声明的字符**直接报错**，不静默填 0；三种轨道类型分别为逐位点二值（`per_position_binary`）、逐位点数值（`per_position_numeric`）、全局标量广播（`global_numeric`）。
7. **张量组装与落盘**：拼成 `(N, 23, C)` 张量、展平 `(N, 23×C)` 张量、`<cell>_labels.npy`，并同步写同名 CSV 便于人工查看；`<cell>_metadata.csv` 保存序列 + 标签 + `metadata_columns` 声明的元数据列（数据里没有的列会被跳过，不报错）。
8. **编码声明生成**：`feature_schema.json` 记录通道顺序与含义、序列长度 / protospacer 区间、PAM 是否在张量内、展平公式、`layout` 与 `model_compatibility`。
9. **收编统计**：`feature_engineering_summary.csv` 逐细胞系记录 `rows_in_raw` / `rows_dropped_by_adapter` / `adapter_drop_reasons` / `target_range` / `has_epigenetics` / `original_samples` / `duplicate_rows` / `final_samples` / `channel_count` / `feature_count`，用来回答"这批数据是怎么变少的"。
10. **多次运行写同一目录**：同一数据集可换 `--raw-data` 反复运行、输出到同一 `--output-dir`（`<cell>_*` 按细胞系各自落盘），用于拼装 LODO / 多来源训练集；若不同来源产生了**同名细胞系**，程序会直接报错（"会产生同名输出互相覆盖"）而不是静默覆盖。
11. **下游对账**：`validate_feature_schema.py` 把"配置声明"与"实际张量"逐项对账（通道数 / 序列长度 / 展平维度 / 由 metadata 重建的 one-hot 是否逐元素相等 / 环境取值 ⊆ {0,1}）；训练端只按 `feature_schema.json` 决定输入形状。

**输入参数**：

| 参数 | 说明 |
|---|---|
| `--raw-data` | 原始数据**文件或目录**（目录会递归查找 `*.csv/*.CSV`）。**必填** |
| `--output-dir` | 处理后数据输出目录。**必填**，不会被任何默认值覆盖 |
| `--config` | 数据集配置 JSON。**必填**；8 通道用 `data/config/DeepCRISPR.json`，纯序列用 `data/config/Hiranniramol.json` 或 `data/config/Labuhn.json` |
| `--format` | `deepcrispr` / `hiranniramol` / `labuhn`；**留空则按列名自动识别** |
| `--cell-lines` | 只处理这些文件（按文件名去扩展名匹配），顺序即输出顺序 |
| `--allow-non-gg-pam` | 允许 PAM 不以 GG 结尾（默认**拒绝**并在适配层报错） |

三个输入路径都必须显式给出，没有隐含默认值——这样不会误处理、也不会覆盖别的数据集。

**示例**：

```bash
# DeepCRISPR（8 通道，含表观遗传）
python core/features/engineering/feature_engineering.py \
  --raw-data data/raw/DeepCRISPR \
  --output-dir data/processed/DeepCRISPR \
  --config data/config/DeepCRISPR.json

# 外部数据集（4 通道，仅序列；显式给 --format）
python core/features/engineering/feature_engineering.py \
  --raw-data data/raw/Labuhn \
  --output-dir data/processed/Labuhn \
  --config data/config/Labuhn.json \
  --format labuhn
```

**实测输出**（Labuhn，417 样本）：

```text
Files processed this run: 1  (labuhn)
Original samples (this run): 417
Duplicate rows removed (this run): 0
Final samples (this run): 417
Channel count: 4
Feature count: 92

Schema saved to:
    data/processed/Labuhn/feature_schema.json
Summary saved to:
    data/processed/Labuhn/feature_engineering_summary.csv
```

**产物**：`feature_schema.json`、`<cell>_features_23x<n>.npy`、`<cell>_features_<dim>.npy`、`<cell>_labels.npy`、
`<cell>_metadata.csv`、`<cell>_23x<n>.csv`、`<cell>_<dim>.csv`（展平）、`feature_engineering_summary.csv`
（逐数据集记录 `rows_in_raw` / `rows_dropped_by_adapter` / `adapter_drop_reasons` / `target_range` / `final_samples`）。

---

## 7. 单run训练

**入口**：`train.py`（仓库根）

**作用**：跑**一次**受控实验——读 `data/processed/<数据集>/` → 按 `--split-type` 划分 → 建模型训练 → 落盘指标/预测/归因/权重。`data_digging.py`（§8）会按网格反复调用它。

**输入参数**：

| 训练超参数 | 对应输入参数 | 可选值 / 说明 |
|---|---|---|
| **模型** | `--model` | `linear` / `xgboost` / `mlp` / `cnn` / `transformer` |
| **数据集** | `--data-set` | `DeepCRISPR` / `Hiranniramol` / `Labuhn`（大小写不敏感）；或 `--data-dir data/processed/<数据集>` |
| **划分** | `--split-type` | `single` / `all`(LOCO) / `mixed` |
| **细胞系** | `--cell-line` / `--cell-lines` | 单细胞系或列表 |
| **环境组合** | `--environment` | `sequence`、`sequence_ctcf`、`sequence_ctcf_dnase` …（用 `_` 连接；`sequence` 表示纯序列） |
| **划分比例** | `--train-ratio` `--valid-ratio` `--test-ratio` | 默认 0.7 / 0.15 / 0.15 |
| **随机种子** | `--seed` | 默认 42 |
| **CNN 卷积核** | `--sequence-kernel` / `--environment-kernel` | 3 / 5 / 7 |
| **网络宽度** | `--hidden-dim1` `--hidden-dim2` `--conv-channels1` `--conv-channels2` | 默认 128 / 64 / 32 / 64 |
| **正则化** | `--dropout` `--weight-decay` | |
| **优化** | `--learning-rate` `--batch-size` `--epochs` | |
| **优化器** | `--optimizer` | `adam`（默认，等价于改造前的硬编码行为）/ `adamw` / `sgd` / `rmsprop` / `adagrad`；仅 mlp/cnn/transformer 生效 |
| **学习率调度** | `--scheduler` | `none`（默认，不创建调度器）/ `cosine` / `step` / `exponential` / `plateau`（按验证损失） |
| **激活函数** | `--activation` | `none`（默认＝沿用各模型原有激活：CNN/MLP 为 ReLU、Transformer 为 GELU）/ `relu` / `gelu` / `tanh` / `sigmoid` / `leaky_relu` / `elu` / `silu` |
| **数据加载并行** | `--num-workers` | DataLoader worker 数，默认 `0` |
| **绑定物理 GPU** | `--gpu-id` | 如 `0` 或 `0,1`；设置 `CUDA_VISIBLE_DEVICES` 后再训练。与 `--device` 的区别：`--device` 在**可见集合内**选序号，`--gpu-id` 改的是**可见集合本身**（多进程并行要隔离显存时用这个） |
| **早停** | `--patience` `--min-delta` | 默认 20 / 1e-6 |
| **标准化** | `--use-scaler` | 默认关闭 |
| **设备** | `--device` | 如 `cuda` / `cpu`（默认自动解析） |
| **输出路径** | `--results-dir` `--model-dir` `--logs-dir` | 相对项目根或绝对路径 |
| **实验名 / 批次名** | `--run-name` `--batch-name` | `--batch-name` 是产物分子目录名 |
| **自定义模型模块** | `--model-module` | 高级用法，指向自定义实现 |

> 三个输出目录的**默认值**是 `results/batches` / `models/weights` / `results/logs`（会再套一层 `--batch-name`）。

`--optimizer` / `--scheduler` / `--activation` / `--num-workers` / `--gpu-id` 在`data_digging.py` 中同名可用，会透传给每个训练子进程。

**示例**：

```bash
python train.py \
  --model linear --split-type single --cell-line hct116 --environment sequence \
  --data-set DeepCRISPR \
  --results-dir results/train_results/DeepCRISPR \
  --model-dir  models/DeepCRISPR \
  --logs-dir   logs/DeepCRISPR \
  --run-name demo_linear --seed 42
```

**实测输出**：

```text
2026-09-19 22:14:58 | INFO | Train shape: (2968, 161), Test shape: (636, 161)
2026-09-19 22:14:58 | INFO | Test Evaluation -> R2: 0.1320, Pearson: 0.3635, MAE: 0.1339
[✓] Experiment demo_linear Finished Successfully.
```

---

## 8. 数据挖掘（Data Digging）

**入口**：`data_digging.py`（仓库根）—— 全项目参数最完整的用户级 CLI，也是**平台总入口**。

**作用**：给定数据集与实验空间（模型 × 划分 × 环境组合 × 种子 × 卷积核），自动展开成实验网格，逐格调用 `train.py` 并汇总每个 run 的状态；支持 `--dry-run` 先看计划不训练。

**输入参数**：

| 参数 | 说明 |
|---|---|
| `--data-set` / `--data-dir` | 数据集（二选一，**必填**）；`--data-set` 列出当前可用数据集 |
| `--batch-name` | 批次名；**为空时直接存放在 `--results-dir` 根目录**（不套子目录） |
| `--training-scope-epis` | 要深入挖掘的表观特征，**自动展开全部环境组合**（与 `--environments` 互斥） |
| `--environments` | 显式环境组合列表，如 `sequence` / `sequence_ctcf` |
| `--models` | `linear` / `xgboost` / `mlp` / `transformer` / `cnn` |
| `--cell-lines` | 要跑的细胞系；缺省＝该 `--data-dir` 下实际发现的全部（不再限于 DeepCRISPR 的 4 个） |
| `--split-types` | `single` / `all` / `mixed` |
| `--mixed-seeds` | mixed 划分使用的种子，如 `42 43 44 45` |
| `--cnn-kernels` | `3` / `5` / `7` |
| `--results-dir` `--model-dir` `--logs-dir` | 输出位置（透传给 `train.py`） |
| `--workers` `--gpus` `--threads-per-worker` `--in-process` | 并行与设备 |
| `--dry-run` | 只打印计划，**不训练、不写文件** |
| 其余训练超参 | `--train-ratio` `--epochs` `--batch-size` `--learning-rate` `--dropout` `--weight-decay` `--patience` `--min-delta` `--hidden-dim1/2` `--conv-channels1/2` `--device` `--optimizer` `--scheduler` `--activation` `--num-workers` —— 与 §7 同名同义，逐项透传 |

**示例**：

```bash
# --dry-run（不训练、不写文件，先看实验计划）
python data_digging.py --data-set DeepCRISPR \
  --cell-lines hct116 --models linear --split-types single \
  --environments sequence --batch-name demo --dry-run

# 进行训练
python data_digging.py --data-set DeepCRISPR \
  --training-scope-epis CTCF --models linear xgboost cnn \
  --split-types single mixed --mixed-seeds 42 43 \
  --cnn-kernels 3 5 7 --batch-name demo --workers 4
```

**实测输出**（`--dry-run`，1 个实验、该实验尚未跑过）：

```text
[Dataset] DeepCRISPR -> /home/zhang/bioprogram/Submit/data/processed/DeepCRISPR
[Environments] 共 1 种: sequence
[Datasets] /home/zhang/bioprogram/Submit/data/processed/DeepCRISPR 下发现 4 个: hct116, hek293t, hela, hl60

======================================================================
Experiment Plan -> Total: 1 | Completed: 0 | Pending: 1
======================================================================
[*] Dry run completed.
```

> `Completed` 统计的是**磁盘上已存在**的 run；同一命令重复执行，见到的 Completed 会随之上升（幂等跳过）。


+ 注：深度学习应以GPU进行训练，CPU训练时间较长。
---

## 9. 评估（Evaluation）

**入口**：**无独立命令**——评估内嵌在训练的最后一步，训练完成即产出测试集与验证集两套指标。

**作用**：对每个 run 在 **test** 与 **validation** 两个划分上算 5 个指标（R² / RMSE / MAE / Pearson / Spearman），并把逐样本预测落盘，供后续所有 Effect / Importance / 统计分析复用。

**输入参数**：无新增 CLI；影响评估口径的只有：
- `--use-scaler`（是否在模型内做标准化，默认关）
- `--train-ratio` `--valid-ratio` `--test-ratio`（决定 test 集规模，默认 0.7/0.15/0.15）
- 早停 `--patience` `--min-delta`（决定回滚到哪个 epoch 的权重）

**口径约定**：`*_metrics.json` = **测试集**；`*_validation_metrics.json` = 验证集，**仅用于选最优 epoch，不作为性能口径**。

**示例**：

```bash
# 测试集指标（★ 唯一的性能口径）
cat results/train_results/DeepCRISPR/demo_linear/linear_regression_metrics.json
# 验证集指标
cat results/train_results/DeepCRISPR/demo_linear/linear_regression_validation_metrics.json
# 逐样本预测
head results/train_results/DeepCRISPR/demo_linear/linear_regression_predictions.csv
```

**实测输出**（`all_cnn_all_heldout_hct116_kernel_3`）：

```json
// cnn_metrics.json（测试集）
{"MSE": 0.028138, "RMSE": 0.167745, "MAE": 0.134335,
 "R2": 0.045792, "Pearson": 0.218467, "Spearman": 0.221151}
// cnn_validation_metrics.json（验证集）
{"MSE": 0.025327, "RMSE": 0.159144, "MAE": 0.122120,
 "R2": 0.045661, "Pearson": 0.215437, "Spearman": 0.189205}
```
```text
# cnn_predictions.csv
y_true,y_pred,error
0.1585093,0.26455086,-0.106041566
```

`*_info.txt` 另记录完整运行配置与划分审计：`n_train/n_valid/n_test`、`audit_train_test_sequence_overlap`、
`audit_train_test_revcomp_overlap`、`split_digest`、`data_fingerprint`、`code_fingerprint`、`env_stack_id`。

> **指标口径**：R² 以"永远预测均值"为基线；`SS_tot ≈ 0`（标签近乎常数）时返回 `NaN` 而非 0。
> `|R²| ≥ 10` 视为数值发散，在 LOCO / 证据矩阵等汇总处剔除（实测 1344 个 run 中 20 个测试集发散）。

---

## 10. 结果汇总（Result Collection）

**入口**：`python -m analysis.collect_results`（汇总表）+ `analysis/importance_extraction.py`（关键调控特征库）

**作用**：把散在 `results/train_results/<数据集>/<run>/` 里的 `*_metrics.json` / `*_validation_metrics.json` / `*_info.txt` 收成**批次级统一表**，供后续分析与报告复用；`importance_extraction` 再把 5 类模型的归因表白名单化，产出关键调控特征库。

**输入参数**：

| 参数 | 说明 |
|---|---|
| `--batch-dir` | 直接指定批次目录（**优先级最高**）。可以是"含 runs 的目录"，也可以是 `results/summary/<数据集>` 本身——两种布局都会被自动识别 |
| `--results-dir` | 结果根目录（默认 `results/batches`）；必须与训练时用的值**一致** |
| `--batch-name` | 批次名，与 `--results-dir` 组合成 `<results-dir>/<batch-name>` |
| `--latest` | 取 `--results-dir` 下最近修改的批次 |
| `--split-types` | 只生成所选划分的 CSV，如 `--split-types single all` |
| `importance_extraction`：`--batch_dir` / `--batch` / `--results_dir` / `--latest` / `--all_batches` | 指定要处理的批次；`--latest` 为缺省行为 |

**示例**：

```bash
# 指定批次目录（推荐；两种布局都能识别）
python -m analysis.collect_results --batch-dir results/summary/DeepCRISPR

# 用 results-dir + batch-name（必须与训练时一致）
python -m analysis.collect_results --results-dir results/batches --batch-name demo

# 最近修改的批次 / 只要 single 划分
python -m analysis.collect_results --latest
python -m analysis.collect_results --batch-dir results/summary/DeepCRISPR --split-types single

# 关键调控特征库（白名单归因 → key_regulatory_biomarkers.csv）
python analysis/importance_extraction.py --batch_dir results/summary/DeepCRISPR
```

**实测输出**：

```text
[*] 正在汇总模型指标，runs 根: results/train_results/DeepCRISPR
  [+] all_experiments.csv saved
  [+] single_cell_line_result.csv saved
  [+] all_cell_line_result.csv saved
  [+] mixed_cell_line_result.csv saved
  [+] Result CSV tables saved to: results/summary/DeepCRISPR/train_data
  [+] Sequence-only baseline table saved to: results/summary/DeepCRISPR/train_data
[✓] 成功生成 train_data -> results/summary/DeepCRISPR/train_data
```

**产物**：`<summary_root>/train_data/{all_experiments,single_cell_line_result,all_cell_line_result,mixed_cell_line_result,baseline}.csv`
与 `<summary_root>/feature_importance/{linear_coefficiency,xgboost_importance,mlp_importance,transformer_importance,cnn33_importance,cnn53_importance,cnn73_importance}.md` + `key_regulatory_biomarkers.csv`。
退出码：`0` 成功 / `1` 批次目录不存在 / `2` 用法错误。

---
## 11.报告生成与表格收录

**总入口**：`python -m analysis.pipeline`（与 §12 同一个引擎；报告与表格是它最先落盘的两批产物）

**作用**：把训练产物收敛成两类可交付文本 —— `reports/`（3 份 Markdown）与 `tables/`（20 张 CSV）。
报告、表格、图**共用同一份 `experiment_table.csv`**，所以三者的数字同源、可以互相对账。

**输入参数**：

| 参数 | 说明 |
|---|---|
| `--batch-dir` | 批次目录（**必填**）。既可是"含 runs 的目录"，也可是 `results/summary/<数据集>`；后者会自动到同级 `results/train_results/<数据集>` 找 runs |
| `--output` | 输出目录。默认：布局 A 为 `<batch>/summary`；若 `--batch-dir` 本身就是 `summary/<数据集>` 则**就地写回** |
| `--analysis-plan` | 可选的 `AnalysisPlan` JSON（如 `summary/<数据集>/analysis_plan.json`），用于复现某次分析的任务勾选 |

固定的输出约定：报告写在 `<output>/reports/`、表写在 `<output>/tables/`、图写在 `<output>/figures/{environment,plots}/`。

**示例**：

```bash
# 就地重跑：报告 + 表格（+ 图）一起写回交付目录
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR

# 只写到别处，不动交付产物
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR --output /tmp/check
```

**产物 → 生成程序**

#### `reports/`（3 份）

| 报告 | 生成函数 | 内容 |
|---|---|---|
| `overview.md` | `analysis.reports.markdown_report.build_overview_md` | 覆盖度七行 + **13 个任务**的 `selected / available / status / reason` |
| `data_quality.md` | `analysis.reports.markdown_report.build_data_quality_md` | `cell_line × environment` 覆盖矩阵 + 指标一致性明细 |
| `anomaly_report.md` | `analysis.reports.markdown_report.build_anomaly_md` | 两类异常的定义与明细 |

三个 `build_*_md` 只负责**正文**、不碰 IO；写盘统一由 `analysis.pipeline` 做。

#### `tables/`（20 张）

| 表 | 生成程序 |
|---|---|
| `experiment_table.csv` | `analysis.data.loaders.load_experiment_table`（**其余各表的底座**） |
| `metric_inconsistency.csv` | `analysis.data.validation.validate_metric_consistency` |
| `anomaly_report.csv` | `analysis.pipeline`（在 `metric_inconsistency` 前插一列 `anomaly_type`） |
| `prediction_summary.csv` | `analysis.prediction.performance_by_model_split` |
| `loco_performance.csv` | `analysis.prediction.loco_performance` |
| `environment_conditional_delta_r2.csv` | `analysis.environment.incremental_effect.summarize_conditional` |
| `environment_main_effects.csv` | `analysis.environment.incremental_effect.compute_main_effects` |
| `environment_nodes.csv` · `environment_edges.csv` · `environment_interactions.csv` · `environment_dag_report.csv` | `analysis.environment.factorial_dag.write_factorial_dag_artifacts`（**一次调用出四张**） |
| `bootstrap_results.csv` | `analysis.stats.tasks.bootstrap_environment_edges` |
| `bootstrap_main_effects.csv` | `analysis.stats.tasks.bootstrap_main_effects` |
| `bootstrap_cellline_effects.csv` | `analysis.stats.tasks.bootstrap_cellline_effects` |
| `permutation_results.csv` | `analysis.stats.tasks.permutation_environment_edges` / `permutation_main_effects` / `permutation_interactions`（**三张合成一张长表**，用 `test_type` 区分） |
| `anova_results.csv` | `analysis.stats.tasks.run_anova_tasks`（底层 `analysis.stats.hypothesis_tests.factorial_anova`） |
| `attribution_summary.csv` | `analysis.attribution.extractors.extract_attribution_table` |
| `motif_candidates.csv` · `motif_instances.csv` · `motif_enrichment.csv` | `analysis.sequence.motif.pipeline.run_and_write` |

#### 不在 `tables/` 下、但同属「表格收录」的两个目录

| 目录 | 生成程序 | 内容 |
|---|---|---|
| `train_data/`（5 张） | `analysis.collect_results` | 逐 run 指标汇总 + 纯序列基线 |
| `feature_importance/`（8 个） | `analysis.importance_extraction` | 7 个白名单归因表（含 `sig` 星级）+ `key_regulatory_biomarkers.csv` |

#### 任务 → 代表产物

`analysis.pipeline` 内部有一张 `task_artifacts` 登记表，把各任务映射到它的**代表产物**；
`analysis_status.json` 与 `overview.md` 里显示的 `(tables/xxx.csv (…))` 就来自它：

| 任务 | 代表产物 |
|---|---|
| `qc` | `reports/data_quality.md` |
| `prediction` | `tables/prediction_summary.csv` |
| `environment_conditional_effect` | `tables/environment_conditional_delta_r2.csv` |
| `environment_main_effect` | `tables/environment_main_effects.csv` |
| `environment_factorial_dag` | `tables/environment_edges.csv` |
| `bootstrap` | `tables/bootstrap_results.csv` |
| `hypothesis_testing` / `fdr_correction` | `tables/permutation_results.csv` |
| `environment_anova` | `tables/anova_results.csv` |
| `sequence_attribution` | `tables/attribution_summary.csv` |
| `motif_discovery` | `tables/motif_candidates.csv` |
| `motif_enrichment` | `tables/motif_enrichment.csv` |

---

## 12. 可视化（Visualization）

**总入口**：`python -m analysis.pipeline`（分析引擎，统计 / 证据 / 报告 / 图表）

**作用**：把统一的批次表一路推到**可交付产物**——校验 → 环境效应 → 归因 → motif → 统计检验 → 三份报告 + 两张主题图；每个任务都保持 `selected / available / status` 三态，任何一步不可用都**如实标 unavailable，不伪造**。

**输入参数**：

| 参数 | 说明 |
|---|---|
| `--batch-dir` | 批次目录（**必填**）。既可是"含 runs 的目录"，也可是 `results/summary/<数据集>`；后者会自动到同级 `results/train_results/<数据集>` 找 runs |
| `--output` | 输出目录。默认：布局 A 为 `<batch>/summary`；若 `--batch-dir` 本身就是 `summary/<数据集>` 则**就地写回** |
| `--analysis-plan` | 可选的 `AnalysisPlan` JSON（如 `summary/<数据集>/analysis_plan.json`），用于复现某次分析的任务勾选 |

固定的输出约定：报告写在 `<output>/reports/`、表写在 `<output>/tables/`、图写在 `<output>/figures/{environment,plots}/`。

**示例**：

```bash
# 就地重跑（交付结构；recommended）
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR

# 复现交付时的任务勾选
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR \
  --analysis-plan results/summary/DeepCRISPR/analysis_plan.json

# 写到别处（不动交付产物）
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR --output /tmp/check
```

**实测输出**（13 个任务）：

```text
[✓] pipeline finished -> /tmp/check
    tasks: qc=completed; prediction=completed; environment_conditional_effect=completed;
           environment_main_effect=completed; environment_factorial_dag=completed;
           sequence_attribution=completed; cnn_ism=completed;
           motif_discovery=completed; motif_enrichment=completed;
           bootstrap=completed; hypothesis_testing=completed; fdr_correction=completed;
           environment_anova=completed
```

**产物**：

- `reports/`：**`overview.md`** · **`data_quality.md`** · **`anomaly_report.md`**（三份交付报告）
  - `data_quality.md` 的 `cell-line × environment 覆盖` 表把 mixed 划分显示为 **`mixed`**
    （训练端把该 split 的 `cell_line` 记作字面量 `"none"`；仅在报告显示层改名，
    `tables/experiment_table.csv` 里的原始取值不动）。
  - **检测缺口的可见性**：若「指标表」与「目录扫描」两条路都拿不到数据，
    异常/一致性检测无法进行。此时产线**照常生成全部报告**，并在 `data_quality.md`
    （及 `anomaly_report.md` 的明细段）写入 **`[!] 未实现检测跳过`**，
    同时把 `qc` 任务标成 `unavailable` 且原因写进 `overview.md`——
    **不会**只在 stdout 打一行"跳过"就继续（那会让报告里的 `_无_` 被误读成"检查过了没问题"）。
    若指标表**存在但读不动**，则原样抛错，不降级成跳过。
- `tables/`：20 张 CSV（环境效应 / 归因 / motif / bootstrap / permutation / ANOVA…）
- `figures/environment/`：`conditional_delta_r2_heatmap.png`、`environment_main_effects.png` +
  `{factorial_dag,conditional_delta_r2,interactions}/` 各 63 张（`split × cell_line × model`）
- `figures/plots/`：**位置归因热图**（position × channel，per method）共 **7 张**
  `position_attribution_{cnn_ig,cnn_ism,mlp_ig,transformer_attention,xgboost_gain,xgboost_treeshap,linear_coefficient}.png`
  其中 6 张为 **8 × 23**（A/C/G/T/CTCF/Dnase/H3K4me3/RRBS × 位置 1–23），
  `linear_coefficient` 为 **7 × 23**（线性模型剔除全部 `*_T` 参照通道，无 `T` 行，见 §17.1）

**产物 → 生成程序**：

| 图（`figures/` 下） | 生成函数 | 张数 |
|---|---|---|
| `environment/conditional_delta_r2_heatmap.png` | `analysis.visualization.environment_plots.render` | 1 |
| `environment/environment_main_effects.png` | 同上（**同一次调用**产出这两张） | 1 |
| `environment/factorial_dag/{split}_{cell}_{model}.png` | `analysis.visualization.factorial_dag.render_factorial_dag` | **63** |
| `environment/conditional_delta_r2/{split}_{cell}_{model}.png` | `analysis.visualization.factorial_dag.render_conditional_delta_r2` | **63** |
| `environment/interactions/{split}_{cell}_{model}.png` | `analysis.visualization.factorial_dag.render_environment_interactions` | **63** |
| `plots/position_attribution_{method}.png` | `analysis.visualization.attribution_plots.render` | **7** |

**调用链**：`analysis.pipeline` → `analysis.visualization.render_all(...)`（总调度）
→ 再调 `factorial_dag` 的三个 renderer。`render_all` 只认两个主题目录
（`environment/` 与 `plots/`），入参就是内存里那几张表：

| 入参 | 来源表 | 给谁用 |
|---|---|---|
| `conditional_summary` | `tables/environment_conditional_delta_r2.csv` | `environment_plots.render` |
| `main_effects` | `tables/environment_main_effects.csv` | 同上 |
| `attribution_table` | `tables/attribution_summary.csv` | `attribution_plots.render` |
| `significance_table` | `feature_importance/key_regulatory_biomarkers.csv` | 位置归因热图的**显示门**（`FDR ≥ 0.10`（linear）或 `SNR < 0.8`（非线性）的格子留白）；传 `None` 则该门自动关闭，不影响其它图 |

---

## 13. 输出结构(具体解释见R2)

```text
data/processed/<数据集>/                 预处理产物（§6）
  feature_schema.json                    ★ 编码语义唯一权威（§5）
  feature_engineering_summary.csv        逐数据集适配/去重统计
  <cell>_features_23x<C>.npy             3D 张量 (N, 23, C)
  <cell>_features_<dim>.npy              展平张量 (N, dim)
  <cell>_labels.npy                      标签 y
  <cell>_metadata.csv                    序列 + 标签 + 该数据集的元数据列
  <cell>_23x<C>.csv / <cell>_<dim>.csv   同名 CSV 便于人工查看

results/train_results/<数据集>/<run>/    逐 run 产物（§7）
  <run> 形如 single_hct116_linear_sequence / mixed_cnn_all_seed_42_kernel_3
  <model>_metrics.json                   ★ 测试集指标（唯一性能口径）
  <model>_validation_metrics.json          验证集指标（仅用于选最优 epoch）
  <model>_predictions.csv                  测试集逐样本预测 (y_true, y_pred, error)
  <model>_validation_predictions.csv       验证集逐样本预测
  <model>_feature_importance.csv           归因白名单列（各模型列见 §17.0）
  <model>_training_history.csv             逐 epoch 训练曲线
  <model>_info.txt                         运行配置 + 划分审计 + 四类指纹（§14）
models/<数据集>/<run>/                   模型权重（.pkl / .pt / .json）
logs/<数据集>/<run>/                     训练日志

results/summary/<数据集>/                批次级汇总（§10 + §11）
  analysis_plan.json                     本次分析的任务勾选与阈值快照
  analysis_status.json                   每个任务的 selected / available / status / reason
  execution_log.json                     执行结果 + 产出 PNG 相对路径清单
  cnn7_validation.md / .csv              CNN7×CRISPRon 一致性验证交付物（见 R2）
  train_data/                            §10 汇总表（5 张）
  feature_importance/                    §10 关键调控特征库（7 个 .md + key_regulatory_biomarkers.csv）
  reports/                               §11 三份交付报告
    overview.md   data_quality.md   anomaly_report.md
  tables/                                §11 分析表（20 张 CSV，见下表）
  figures/
    environment/
      conditional_delta_r2_heatmap.png   背景 S × 新增环境 e 的条件 ΔR² 热图
      environment_main_effects.png       4 个环境因子的主效应
      factorial_dag/                     63 张（split × cell_line × model）
      conditional_delta_r2/              63 张
      interactions/                      63 张
    plots/
      position_attribution_<method>.png  位置归因热图 7 张（position × channel，无符号幅度）
                                         method ∈ cnn_ig · cnn_ism · mlp_ig ·
                                         transformer_attention · xgboost_gain ·
                                         xgboost_treeshap · linear_coefficient
```

**`tables/` 的 20 张表**（按四维度归类；括号内为行数）：

| 维度 | 表 |
|---|---|
| **Effect** | `environment_conditional_delta_r2`(2016) · `environment_main_effects`(252) · `environment_edges`(2671) · `environment_interactions`(378) · `anova_results`(94) · `motif_enrichment`(656) |
| **Importance** | `attribution_summary`(331200) · `motif_candidates`(656) |
| **Statistical Evidence** | `bootstrap_results`(7954) · `bootstrap_main_effects`(4) · `bootstrap_cellline_effects`(36) · `permutation_results`(3444) |
| **Robustness** | `loco_performance`(28) · `prediction_summary`(21) |
| **底座** | `experiment_table`(1344) · `environment_nodes`(1008) · `metric_inconsistency` · `anomaly_report` · `environment_dag_report`(63) · `motif_instances`(186737) |

> 四维度的定义、每张表的原理与目标见 **§15–§19「项目底层」**。

**两次运行的落点差异**：代码默认写 `results/batches/<batch>/`、`models/weights/<batch>/`、`results/logs/<batch>/`；
**本次提交**把三者分别指到 `results/train_results/<数据集>`、`models/<数据集>`、`logs/<数据集>`，因此没有 batch 层。
下游（§10 / §11）需要分别解析「runs 根」与「summary 根」——两者在同一套命令里都能正确识别。

---

## 14.完整流程代码:

把四个阶段连起来（以 DeepCRISPR 为例；`<数据集>` 替换为 `Hiranniramol` / `Labuhn` 时需要改用纯序列 config 与对应 `--format`）：

```bash
# 阶段 0：环境（§3）
pip install -r deploy/requirements.txt   # 依赖清单

# 阶段 1：预处理（§6）—— 原始 CSV → 张量 + feature_schema.json
python core/features/engineering/feature_engineering.py \
  --raw-data   data/raw/DeepCRISPR \
  --output-dir data/processed/DeepCRISPR \
  --config     data/config/DeepCRISPR.json

# 校验「schema 声明 == 实际张量」
python core/features/engineering/validate_feature_schema.py --all

# 阶段 2：数据挖掘（§8）—— 展开实验网格并逐格训练
#  先看计划（不训练、不写文件）
python data_digging.py --data-set DeepCRISPR \
  --training-scope-epis CTCF Dnase H3K4me3 RRBS \
  --models linear xgboost mlp transformer cnn \
  --split-types single all mixed --mixed-seeds 42 43 44 45 \
  --cnn-kernels 3 5 7 \
  --batch-name full --results-dir results/train_results/DeepCRISPR \
  --model-dir models/DeepCRISPR --logs-dir logs/DeepCRISPR --dry-run

#  正式训练（去掉 --dry-run）
python data_digging.py --data-set DeepCRISPR \
  --training-scope-epis CTCF Dnase H3K4me3 RRBS \
  --models linear xgboost mlp transformer cnn \
  --split-types single all mixed --mixed-seeds 42 43 44 45 \
  --cnn-kernels 3 5 7 \
  --batch-name full --results-dir results/train_results/DeepCRISPR \
  --model-dir models/DeepCRISPR --logs-dir logs/DeepCRISPR --workers 4

# 阶段 3：结果汇总（§10）
python -m analysis.collect_results --batch-dir results/summary/DeepCRISPR
python analysis/importance_extraction.py --batch_dir results/summary/DeepCRISPR

# 阶段 4：可视化与分析（§11）
python -m analysis.pipeline --batch-dir results/summary/DeepCRISPR

# 阶段 5（可选）：CNN7 × CRISPRon 的 Pos18 C→A 一致性验证
python analysis/candidates/wt_position18_selection.py           # 挑 WT 位点候选
python -m analysis.crispron_validation                          # 全流程（windows/crispron/model/table）
```

---

## 15. 可复现性

每个 run 的 `*_info.txt` 记录四类指纹 + 划分摘要，使结果可溯源到「数据 + 配置 + 代码 + 模型 + 随机种子」：

| 字段 | 含义 |
|---|---|
| `data_fingerprint` | 数据内容哈希（含特征文件大小） |
| `code_fingerprint` | **关键源文件 md5**：`train.py`、`core/data/cell_line_division.py`、`core/features/engineering/feature_engineering.py`（路径字符串本身也参与哈希） |
| `env_fingerprint` / `env_stack_id` | 数值栈版本与设备可见性；`env_stack_id` 是**不含 `cvd`** 的稳定标识（多卡节点上各 worker 的 `cvd` 不同，属预期行为） |
| `split_digest` | 划分摘要（sha256 前 16 位）= train/valid/test 的**序列集合 + 样本数**，不含任何路径，可核验"训练所用划分 == 分析所用划分" |
| `audit_train_test_sequence_overlap` / `audit_train_test_revcomp_overlap` / `audit_train_test_locus_overlap` | 泄漏审计；group-aware 模式下**非零即抛错中止** |

> 项目由于在超算平台上允许,一些字段无法复现。

**身份类与泄漏防控**（`core/data/cell_line_division.py`）：

- 身份键 = `min(sequence, revcomp(sequence))`（同一 sgRNA 及其反向互补视为同一身份类）
- 三种划分全部 group-aware：**同一序列不跨 train/valid/test**
- LOCO(`all`) 训练池先剔除留出系全部同源序列，再按 85/15 划分
- 划分处自证：重叠非零即抛错；`split_digest` 随结果落盘

**固定种子**：single / all 固定 42；mixed 42/43/44/45；bootstrap 2024；置换检验 B=1000（seed 2024）。

三个数据集全部来自公开文献，原始 CSV 随仓库交付在 `data/raw/`，**来源与标签语义见 §4**：

| 数据集 | 来源 | 样本 | 形态 |
|---|---|---|---|
| DeepCRISPR | Chuai et al. 2018, *Genome Biology* | **16,749**（hct116 4,239 / hek293t 2,333 / hela 8,101 / hl60 2,076） | 23 nt，**8 通道**（序列 4 + 表观 4） |
| Hiranniramol | Hiranniramol et al. 2020, *Bioinformatics* | **1,309** | 23 nt，4 通道（纯序列） |
| Labuhn | Labuhn et al. 2018, *Nucleic Acids Research* | **417**（原始 430，适配层剔除 8 条 spacer 不在 extended 区间 + 合并 5 条重复测量） | 23 nt，4 通道（纯序列） |

**标签语义不可跨数据集比较**：DeepCRISPR 是 `[0,1]` 归一化效率、Hiranniramol 原始为 0–100 百分制（适配器 /100）、Labuhn 是 `[0,1]` KO reporter assay。三者只能读作"同一流程在不同数据集上的可预测性"。

### 交付批次的当前状态

| 批次 | 规模 | 状态 |
|---|---|---|
| **DeepCRISPR** | **1,344 run** = 7 模型配置 × 16 环境组合 × 12（划分 × 种子）<br>划分各 448；`groups_aware=True`；全批**单一** `code_fingerprint` / `data_fingerprint` / `env_stack_id` | ✅ 交付：`results/train_results/DeepCRISPR/`（9,408 文件）+ `results/summary/DeepCRISPR/`（239 文件）+ `models/DeepCRISPR/` 1,344 个权重目录；13/13 任务 completed |
| Hiranniramol | 7 run（5 类模型 + cnn 3/5/7） | ✅ 交付：`results/train_results/Hiranniramol/`（49 文件）+ `results/summary/Hiranniramol/`（53 文件）+ `models/Hiranniramol/` 7 个权重目录；11/13 任务 completed，`fdr_correction` / `environment_anova` 如实 unavailable（无环境因子） |
| Labuhn | 7 run | ✅ 交付：`results/train_results/Labuhn/`（49 文件）+ `results/summary/Labuhn/`（53 文件）+ `models/Labuhn/` 7 个权重目录；11/13 任务 completed，`fdr_correction` / `environment_anova` 如实 unavailable（无环境因子） |

> 交付批次的三份报告（`overview.md` / `data_quality.md` / `anomaly_report.md`）记录了该批的完整任务状态；

---

# 项目底层

**本项目以 5 模型 7 配置 3 划分为框架，给出不同机器学习视角的互补观察**——同一个科学问题（"表观微环境与序列 motif 究竟为编辑效率贡献了多少、贡献在哪个位点、结论稳不稳"）被同一批数据以**三种独立的方式**反复回答：**换建模范式、换感受野尺度、换训练/测试协议**。每类模型都按**同一套接口**产出证据，最后**只在多个视角同向收敛时才认定为结论**。

三个维度各回答一个不同的问题：

| 框架维度 | 变化 | 回答的问题 | 规模 |
| :--- | :--- | :--- | :--- |
| **5 模型** | 归纳偏好：线性 / 树 / 稠密网 / 卷积 / 注意力 | "换一种**建模范式**，还看得见同一个效应吗？" | 5 类 |
| **7 配置** | 其中 CNN 再按**序列感受野 3 / 5 / 7** 展开 | "换一个**motif 尺度假设**，还看得见同一个 motif 吗？" | 5 + 2 = 7 |
| **3 划分** | 训练 / 测试怎么切：`single` / `all`(LOCO) / `mixed` | "换一种**评估协议**，结论还成立吗？" | 3 |

三者相乘、再叠加环境组合，就是交付批次的全部分量：
**7 配置 × 3 划分 × 16 环境组合 = 336 个单元**，每个单元在 4 个细胞系（`single` / `all`）或 4 个随机种子（`mixed`）上重复
→ **336 × 4 = 1,344 个受控 run**。

### 5 模型 7 配置：

| # | 配置 | 归纳偏好（它看得见什么） | 在本项目里的角色 |
| :-: | :--- | :--- | :--- |
| 1 | `linear` | **线性可加**；系数有方向、有量纲。是唯一具备经典参数检验前提（`t` / `p` / BH-FDR）的模型 | 可解释基线；**唯一允许报告显著性**的模型（§17.1） |
| 2 | `xgboost` | **分段常数 + 树状特征交互**；对阈值型、非单调关系敏感，不需要标准化 | 非线性与交互的主力；用 TreeSHAP 做样本级精确归因（§17.2） |
| 3 | `mlp` | **全局平滑非线性**；把展平后的全部输入当整体做稠密映射 | 检验"关系到底是不是分段的"；用 IG 做输入级归因（§17.3） |
| 4 | `cnn(3\|3)` | **局部模式，序列分支感受野 3 nt** | 短 motif（PAM 邻近 3–4 mer） |
| 5 | `cnn(5\|3)` | **局部模式，序列分支感受野 5 nt** | 中等长度 motif |
| 6 | `cnn(7\|3)` | **局部模式，序列分支感受野 7 nt** | 较长 motif |
| 7 | `transformer` | **任意位点间的长程依赖**；注意力权重可读出"谁在关注谁"与弥散/聚焦 | 检验远端位点是否真的参与决策（§17.5） |

### 3 划分：同一批数据，三种评估协议

| 划分 | 训练 / 测试切分 | 回答什么 | 不能回答什么 |
| :--- | :--- | :--- | :--- |
| **`single`** | **同一细胞系内部** 70 / 15 / 15；4 个细胞系各跑一遍，固定种子 42 | "在**已知宿主**内部，模型能做到多准"——模型与特征的上限基准 | 不能说明跨细胞系迁移；细胞内仍可能有同源序列跨集，靠 group-aware 身份类强制隔离 |
| **`all`**（LOCO） | **留出整个细胞系(Leave-One-Cell-Out)**做测试；训练池**先剔除留出系的全部同源序列**，再按 85 / 15 划分；4 个细胞系轮流留出，固定种子 42 | "换一个**从没见过的宿主细胞系**，还能不能预测"——真正的**跨域外推** | 只有 DeepCRISPR 能做（唯一带 4 个细胞系的数据集）；细胞系样本量悬殊（8,101 vs 2,076），方差天然更大 |
| **`mixed`** | 4 个细胞系**混成一个池**再 70 / 15 / 15，用 **4 个种子**（42/43/44/45） | "把细胞系当作同一个总体时，模型与环境效应**能否稳定复现**"——种子级可复现性 | `cell_line` 被池化成 `none`，**不能**用于跨细胞系异质性分析；也**不是**跨域泛化证据 |

### 视角要"接口一致"，才谈得上互补

7 个配置（× 3 种划分，共 21 种"配置×协议"组合）要求在**四个维度上各自出证据**：

- 每个模型都产出同一套 **Effect** 量（配对 ΔR²、环境主效应、ANOVA `effect`、motif OR）；
- 每个模型都只导出**白名单约束下**的 Importance 列，且**只有线性模型可以带显著性字段**——非线性模型一旦出现 `t_stat`/`p_value`/`fdr`，`core/xai/xai_importance.py` 会直接抛 `ValueError`；
- 每个模型的效应都要过 **Bootstrap CI → 符号翻转置换 → BH-FDR**（按科学问题分族校正）；

> 换句话说：**任何一个模型、任何一个尺度、任何一种划分给出的结论都只是一条假设**，
> 只有**跨范式、跨尺度、跨划分、跨种子、跨细胞系**都能复现的那部分才被当作发现。

**本项目依赖四维度对挖掘结论做不同维度的证据支持**：Effect 说明"变了多少"，Importance 说明"模型依赖哪里"，Statistical Evidence 说明"像不像噪声"，Robustness 说明"换个视角还站不站得住"。四者的关系是**并列且不可互相替代**的——任一项单独成立都不构成结论。

## 四维度：一个结论凭什么成立

| 维度 | 回答的问题 | 本项目使用的量 | 主要产物 |
| :--- | :--- | :--- | :--- |
| **Effect（效应量）** | 因素改变后，预测/活性改变了多少？方向如何？ | 配对 ΔR²、ΔRMSE/ΔMAE、环境主效应、交互 ΔR²、ANOVA `effect`、motif OR | `environment_{conditional_delta_r2,main_effects,edges,interactions}.csv`、`anova_results.csv`、`motif_enrichment.csv` |
| **Importance（预测贡献）** | 模型前向推理时多大程度依赖这个位点/通道？（**一律无符号幅度**） | IG、ISM、TreeSHAP、Attention，以及各自的 SNR | `attribution_summary.csv`、`motif_candidates.csv` |
| **Statistical Evidence（统计证据）** | 观察到的效应是真实信号还是纯随机噪声？ | p 值、FDR(q)、Bootstrap CI、F 统计量 | `bootstrap_*.csv`、`permutation_results.csv`、`anova_results.csv` |
| **Robustness（稳健性）** | 换模型 / 换种子 / 换细胞系 / 换划分后，结论还站得住吗？ | 方向一致率、LOCO、跨细胞系 CI 重叠 | `loco_performance.csv`、`bootstrap_cellline_effects.csv` |

---

## 16.模型性能指标(Performance metrics)

R²、RMSE、MAE、Pearson、Spearman —— 五项在 5 个模型里由**逐行同源的** `calculate_metrics(y_true, y_pred)` 计算（`core/models/{linear,xgboost,mlp,cnn,transformer}` 中实现完全一致），口径统一、可跨模型直接比较。

### 16.1 各指标的原理与目标

| 指标 | 公式 | 目标（回答什么） | 边界（不能说明什么） |
| :--- | :--- | :--- | :--- |
| **R²** | $R^2 = 1 - \dfrac{\mathrm{SS_{res}}}{\mathrm{SS_{tot}}} = 1 - \dfrac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}$ | 相对"永远预测均值"这个平凡基线，模型把误差削减了多少。是本项目**所有 Effect 维度量的基准**（ΔR² 就是两个 R² 之差） | 只对同分布、同标签尺度的测试集有意义；$\mathrm{SS_{tot}}\approx 0$（标签近乎常数）时返回 `NaN` 而非 0，绝不伪造 |
| **RMSE** | $\mathrm{RMSE}=\sqrt{\frac1n\sum_i (y_i-\hat y_i)^2}$ | 与标签同量纲的误差幅度；对大误差敏感（平方惩罚），用于**消融/边级**的 ΔRMSE | 与 MAE 同看才能判断"是否存在少量大错" |
| **MAE** | $\mathrm{MAE}=\frac1n\sum_i \lvert y_i-\hat y_i\rvert$ | 与标签同量纲的中位型误差；对离群不敏感 | 不反映误差分布尾部 |
| **Pearson** | $r=\dfrac{\sum_i (y_i-\bar y)(\hat y_i-\bar{\hat y})}{\sqrt{\sum_i (y_i-\bar y)^2}\sqrt{\sum_i (\hat y_i-\bar{\hat y})^2}}$ | 线性相关强度：模型是否抓住了**单调线性趋势** | 对系统性偏置/缩放不敏感（$y$ 与 $2y+5$ 的 $r$ 相同）；$\sigma=0$ 时返回 `NaN` |
| **Spearman** | $\rho = r\big(\mathrm{rank}(y),\ \mathrm{rank}(\hat y)\big)$，`rank(method="average")` | 秩相关：只看**排序**是否一致，等价于 Pearson 作用在秩上 | 不反映幅度准确性；用于判定"能否正确排序候选 gRNA" |

### 16.2 使用时必须注意的三个约定

- **口径约定**：`<model>_metrics.json` = **测试集**；`<model>_validation_metrics.json` = 验证集。验证集只用于选最优 epoch（早停回滚），**不作为性能口径**。
- **发散过滤**：`|R²| ≥ 10` 视为数值发散（线性模型在共线/近奇异设计下会出现 $10^{18}$ 量级），在 `loco_performance`、证据矩阵等汇总处**剔除**（阈值 `AnalysisConfig.consensus.unstable_effect_threshold = 10.0`）。实测 1344 个 run 里有 20 个测试集发散。
- **标签语义不可跨数据集比较**。

---

## 17.可解释性方法(XAI methods)

### 17.0 白名单（规定了"允许存在什么"）

`core/xai/xai_importance.py` 是归因输出的**唯一依据**，每个模型只能导出下列列（`Feature` / `Position` / `Channel` 为标识列，不算指标）：

| 模型 | 允许导出的指标列 | 是否有经典统计检验 |
| :--- | :--- | :--- |
| Linear Regression | `Linear_Coefficient`, `SE`, `t_stat`, `p_value`, `FDR` | ✅ 有（唯一允许） |
| XGBoost | `XGB_Gain`, `XGB_Weight`, `XGB_Cover`, `TreeSHAP`, `SHAP_SNR` | ❌ 无 |
| MLP | `MLP_IG`, `IG_SNR` | ❌ 无 |
| CNN | `CNN_IG`, `CNN_ISM`, `ISM_SNR` | ❌ 无 |
| Transformer | `Transformer_Attention`, `Attention_Entropy`, `Attention_SNR` | ❌ 无 |

---

### 17.1 线性回归：回归系数 + 经典参数推断

**原理.** 以最小二乘拟合 $\hat y = Xw$。为避免共线设计导致的数值爆炸，系数用 Moore–Penrose 伪逆求解（`np.linalg.pinv(X_bias, rcond=1e-15)`）：

$$w = (X^{\top}X)^{+}X^{\top}y,\qquad \mathrm{rank} = \#\{\sigma_i > \texttt{rcond}\cdot\sigma_{\max}\}$$

系数标准误由残差方差经伪逆传播得到：

$$\mathrm{Var}(w_i) = \mathrm{MSE_{res}}\sum_j \big[(X^{\top}X)^{+}\big]_{ij},\qquad \mathrm{SE}_i=\sqrt{\mathrm{Var}(w_i)},\qquad \mathrm{MSE_{res}}=\frac{\sum_i(y_i-\hat y_i)^2}{\mathrm{dof}},\quad \mathrm{dof}=N-\mathrm{rank}$$

$$t_i = \frac{w_i}{\mathrm{SE}_i},\qquad p_i = 2\Big(1-F_t\big(|t_i|;\ \mathrm{dof}\big)\Big)$$

其中 $F_t$ 是自由度 `dof` 的 t 分布 CDF；$p_i$ 为**双尾**。最后对同一 family（本模型全部系数）做 BH-FDR（见 §19.2）得到 `FDR`，显著性符号：`***` $q<0.001$、`**` $q<0.01$、`*` $q<0.05$、`.` $q<0.1$。

**目标.** 回答"**单位特征变动对预测的边际影响有多大，且这个影响是否显著偏离 0**"。`Linear_Coefficient` 是 Effect 维度的量（可正可负、有量纲意义）；`t_stat`/`p_value`/`FDR` 是 Statistical Evidence 维度的量。**这是本项目唯一有经典参数检验的模型**，因为线性回归满足高斯-马尔可夫前提，树/深度模型的参数没有可解释的抽样分布。

---

### 17.2 XGBoost：三种分裂重要度 + TreeSHAP

**原理.** XGBoost 是梯度提升树集成，模型本身不产生样本级归因；项目导出 5 列：

| 列 | 原理 | 目标 |
| :--- | :--- | :--- |
| `XGB_Gain` | 该特征作为分裂点带来的**平均损失下降**（`booster.get_score(importance_type="gain")`） | 分裂视角的"这个特征有多有用"；也是默认排序键 |
| `XGB_Weight` | 该特征被选为分裂点的**次数** | 使用频次（高频但低增益的特征会在这里露出来） |
| `XGB_Cover` | 该特征分裂覆盖的**样本权重和** | 影响面大小 |
| `TreeSHAP` | 全体评估样本的 **$\mathbb{E}[\lvert\varphi_i\rvert]$** —— SHAP 值的绝对值均值 | **特征净贡献**（见下） |
| `SHAP_SNR` | $\dfrac{\mathbb{E}[\lvert\varphi_i\rvert]}{\mathrm{std}(\varphi_i)+10^{-12}}$ | 该净贡献在样本间是否稳定 |

**SHAP 的原理与目标.** SHAP（SHapley Additive exPlanations）把单个预测分解为各特征的**净贡献**：

$$f(x) = \varphi_0 + \sum_{i=1}^{p}\varphi_i,\qquad
\varphi_i = \sum_{S\subseteq F\setminus\{i\}} \frac{|S|!\,(p-|S|-1)!}{p!}\,\Big[f\big(S\cup\{i\}\big) - f\big(S\big)\Big]$$

第一式是**局部精确性**（各特征贡献 + 基线 = 该样本预测值，所以"净"体现在贡献之间不重复计数）；第二式是**Shapley 值**，即"在所有可能的特征子集 $S$ 中，加入特征 $i$ 带来的平均边际增益"——这正是合作博弈论里对"公平分配总收益"的唯一满足 4 条公理（效率、对称、虚拟、可加）的解。

- **目标**：说明**每个特征对预测的净贡献**，且天然带方向（$\varphi_i>0$ 推高预测、$<0$ 压低），比 gain/weight/cover 更接近"模型依赖程度"。跨样本取 $|\varphi_i|$ 的均值即得到该特征的整体重要性。
- **本项目不自己算 Shapley**：直接用 XGBoost 原生 `booster.predict(dmat, pred_contribs=True)`（TreeSHAP，Lundberg 2017 的树结构精确多项式算法），返回矩阵最后一列是偏置 $\varphi_0$，代码里用 `[:, :-1]` 去掉。
- **边界**：SHAP 解释的是**这个已训练模型**的预测，不是生物学因果；`SHAP_SNR` 是 robustness 指标，不是显著性。

>  **已知静默路径**：TreeSHAP 计算被包在 `try/except: pass` 里（`core/models/xgboost/xgboost.py:336-358`），失败时 `SHAP_mean`/`SHAP_SNR` 会**静默留 `NaN`** 而不报错。

---

### 17.3 MLP：积分梯度（IG）

**原理.** 沿"基线 → 输入"的直线路径对梯度做路径积分（Sundararajan et al. 2017）：

$$\mathrm{IG}_i(x) = (x_i - x'_i)\times \frac1m\sum_{k=1}^{m}\frac{\partial F\big(x' + \tfrac{k}{m}(x-x')\big)}{\partial x_i}$$

- 基线 $x' = \mathbf{0}$（全零张量）；步数 $m=25$；实现上取 $\alpha=\mathrm{linspace}(0,1,m+1)$ 得到 $m+1$ 个点，梯度对前 $m$ 个点取平均（`grads[:-1].mean(0)`，右端点梯形修正）。
- **完备性公理**：$\sum_i \mathrm{IG}_i = F(x) - F(x')$，即归因之和精确等于"从基线到输入"的预测变化——这是 IG 相比原始梯度（gradient×input）的核心优势，避免梯度饱和导致的低估。
- 导出量：`MLP_IG` $= \mathbb{E}_n[\,|\mathrm{IG}_i|\,]$（跨样本绝对值均值,即使贡献方向不同也不会相互抵消），`IG_SNR` $= \dfrac{\mathbb{E}_n|\mathrm{IG}_i|}{\mathrm{std}_n(\mathrm{IG}_i)+10^{-12}}$。

**目标**：回答"**输入特征对黑盒 MLP 预测的贡献有多大**"，并用 SNR 说明该贡献在样本间是否一致，**不含方向**。

---

### 17.4 CNN：双分支 IG + 无符号 ISM

CNN 是双分支结构（序列分支 + 环境分支各自卷积），归因同样按"位点 × 通道"给出。

**(a) `CNN_IG` —— 同 MLP 的 IG**，只是作用在 $(L,C)=(23,C)$ 张量上，基线全零、$m=25$、导出 $\mathbb{E}_n\lvert\mathrm{IG}\rvert$。

**(b) `CNN_ISM` —— In-Silico Mutagenesis（虚拟饱和突变）**

**原理.** 对每个 $(l,c)$ 做**单通道扰动**：把该通道的值翻转（`>0 → 0`，`=0 → 1`），重新前向：

$$\Delta_{l,c} = \big|\hat y_{\text{mut}(l,c)} - \hat y_{\text{base}}\big|,\qquad
\texttt{CNN\_ISM}_{l,c}=\mathbb{E}_n[\Delta_{l,c}],\qquad
\texttt{ISM\_SNR}_{l,c}=\frac{\mathbb{E}_n[\Delta_{l,c}]}{\mathrm{std}_n(\Delta_{l,c})+10^{-12}}$$

**目标.** 回答"**把某个位点的某个通道改掉，模型预测会动多少**"——一种不依赖梯度的、对任意模型都成立的重要性度量，作为 IG 的交叉验证。

>  **语义边界**：对 one-hot 序列通道，单通道翻转产生的是**分布外输入**——例如把 `A=1` 翻成 `A=0` 而不把 C/G/T 任一置回 1，得到的既不是 A 也不是任何真实碱基。因此 `CNN_ISM` 度量的是**敏感性/重要性**，**不是碱基替换效应**。

---

### 17.5 Transformer：注意力强度 + 注意力熵

设模型有 $h$ 个注意力头、序列长度 $L$，注意力张量 $A\in\mathbb{R}^{h\times L\times L}$，其中每行由 $\mathrm{softmax}\big(QK^{\top}/\sqrt{d_k}\big)$ 给出。

**原理.** 先跨头平均，再对每个位点求"被所有位置查询时的平均被关注度"：

$$\bar A = \frac1h\sum_{h'} A_{h'},\qquad
\mathrm{incoming}_j = \frac1L\sum_{i=1}^{L}\bar A_{i,j},\qquad
\texttt{Transformer\_Attention}_j = \mathbb{E}_n[\mathrm{incoming}_j]$$

$$\texttt{Attention\_SNR}_j = \frac{\mathbb{E}_n[\mathrm{incoming}_j]}{\mathrm{std}_n(\mathrm{incoming}_j)+10^{-12}}$$

$$\texttt{Attention\_Entropy} = -\sum_{j=1}^{L}\bar A_j\log_2 \bar A_j\quad(\text{每个样本一个标量，取样本均值后广播到所有位点})$$

**目标.**
- `Transformer_Attention`：哪些**序列位置**在模型内部被反复关注（位点级重要性）。
- `Attention_Entropy`：注意力是**弥散**还是**聚焦**。熵接近 $\log_2 L$ 表示注意力均匀铺开（模型没有明确聚焦点）、接近 0 表示高度聚焦。它度量的是归因的**可靠程度**，不是效应大小。
- `Attention_SNR`：位点关注度在样本间是否稳定。

> **边界**：注意力权重 ≠ 重要性归因，它只是前向计算的一个中间量；项目在白名单里把它定位为**辅助归因**（`MotifDiscoveryConfig.supporting_methods`），不单独作为 motif 提取依据。

---

### 17.6 归因稳健性：SNR 的 4 档分档

**原理.** 所有非线性归因都用**同一个稳健性算子**：

$$\mathrm{SNR}_i = \frac{\mathbb{E}_n\big[\lvert\phi_i\rvert\big]}{\mathrm{std}_n(\phi_i) + 10^{-12}}$$

（$|\cdot|$ 对 IG/SHAP 而言；Attention 已是非负量故不加绝对值。）

**分档**（`analysis/config.py::AttributionRuleConfig`，实现在 `analysis/importance_extraction.py`）：

| 符号 | 阈值 | 含义 |
| :--- | :--- | :--- |
| `***` | $\mathrm{SNR} \ge 2.5$ | 强：该特征的归因幅度在样本间高度一致 |
| `**` | $\ge 1.8$ | 较强 |
| `*` | $\ge 1.2$ | 中 |
| `.` | $\ge 0.8$ | 弱 |
| （空） | $< 0.8$ | 不稳定 |

另有 `min_effect_size = 0.005` 作为"足够效应量"的补充下限（可配置，非普适阈值）。

**目标.** 回答"**这条归因是清晰信号还是样本间乱跳的噪声**"。分母是跨样本标准差，所以 SNR 高意味着"每个样本上都在说同一件事"。

> **边界**：SNR **不是** p 值、不做多重比较校正、不控制假阳性率。它属于 Robustness/Importance 维度，**不参与**任何统计显著性判定（`StatisticalRuleConfig` 的 docstring 明确写明其适用范围仅为重要性资产的标签与线性侧 FDR 分档）。

---

## 18.统计分析方法(Statistical methods)

### 18.1 ANOVA:析因方差分析（Type-II 边际 / 额外平方和）

**原理.** 对 $2^4$ 环境析因设计做**含区组的因子 ANOVA**：

- **设计矩阵**：把每个因子按哑变量展开（丢弃首水平避免共线），交互项 `a*b` 由两个因子的哑变量列逐元素相乘得到；区组因子为 `model` / `cell_line` / `split_type`。
- **Type-II 边际（额外平方和）**：检验某一项时，**只从完整模型里去掉该项**，而不是去掉所有包含它的项——主效应去掉自身及包含它的交互，交互项只去掉自身（保留其主效应）。这样在**不平衡设计**下不会因"主效应先进入"而得到顺序依赖的平方和（Type-I 的缺陷）。
- 残差平方和由最小二乘得到 $\mathrm{RSS} = \lVert y - X\beta\rVert^2$（`np.linalg.lstsq`），$\mathrm{dof_{den}} = n - p_{\text{full}}$。

$$\mathrm{SS_{term}} = \max\big(\mathrm{RSS_{reduced}} - \mathrm{RSS_{full}},\ 0\big),\qquad
\mathrm{df_{num}} = p_{\text{full}} - p_{\text{reduced}}$$

$$F = \frac{\mathrm{SS_{term}}/\mathrm{df_{num}}}{\mathrm{RSS_{full}}/\mathrm{dof_{den}}},\qquad
p = \Pr\big(F_{\mathrm{df_{num}},\,\mathrm{dof_{den}}} > F\big)$$

- **效应量（`effect_size`）**：代码计算的是 **partial $\eta^2$**（与 `AnovaRuleConfig.effect_size_metric = "partial_eta_squared"` 一致）：

  $$\eta^2_{\text{partial}} = \frac{\mathrm{SS_{term}}}{\mathrm{SS_{term}} + \mathrm{RSS_{full}}}$$

  **不是**经典 $\eta^2 = \mathrm{SS_{term}}/\mathrm{SS_{tot}}$。这里用 partial 是必要的：本批设计了 12 个 block 哑变量（`model`/`cell_line`/`split_type`），**block 因子占 $\mathrm{SS_{tot}}$ 的 74.9%**；若用 $\mathrm{SS_{tot}}$ 作分母，效应量会被系统性低估约 4 倍。
- **效应量（`effect`，on−off 调整差）**：对二值主效应，取**全部非参照水平回归系数之和** $\sum_{\ell\neq\ell_0}\beta_{\ell}$，即"打开该通道相对关闭时，留出 R² 的平均改变量"。
- **`effect` 的 CI**：对设计矩阵**按行重采样** $B$ 次（`ci_iterations = 400`），每次重新 `lstsq` 并重算 $\sum\beta_\ell$，取 $\alpha/2$ 与 $1-\alpha/2$ 分位数。

**可用性守卫**：`min_observations = 32`（样本数下限）、`min_residual_df = 5`（残差自由度下限）；设计饱和时输出 `(none)` 行并给出 `reason`。

**四维度归类.**

| 输出列 | 四维度 | 说明 |
| :--- | :--- | :--- |
| `effect`（on−off 调整差） | **Effect** | 控制区组后该通道对留出 R² 的边际改变量 |
| `effect_size`（partial $\eta^2$） | **Effect** | 排除 block 与其它项之后，该项单独解释的方差份额 |
| `F_statistic`, `p_value` | **Statistical Evidence** | 剥离区组方差后，额外平方和是否显著脱离零点 |
| `ci_low`, `ci_high` | **Statistical Evidence** | 该效应量的行级不确定性 |
| `n_obs`, `status`, `reason` | 底座 | 可估性守卫 |

**目标**：回答"**在控制了模型/细胞系/划分这些区组差异之后，某个表观通道（或其交互）还能额外解释多少方差，这个解释力是否显著**"。它是对 Effect 维度的**方差解释**视角补充，与"配对 ΔR²"的**预测增益**视角互为独立证据。

**输出规模**：`anova_results.csv` 共 **94 行** = `blocked_factorial`（全局区组设计，10 项 = 4 主效应 + 6 交互）+ `per_group_additive_main`（9 个 `(split, cell_line)` 分组上的加性主效应：`mixed/none` 组 28 行 = 7 模型 × 4 因子，其余 8 组各 7 行，合计 84 行）。

---

### 18.2 Permutation test:符号翻转随机化检验（Sign-flip）

**原理.** 对**配对差值**序列 $x_1,\dots,x_n$（例如同一样本在"父组合"与"子组合"下的逐样本损失差），检验

$$H_0:\ \mathbb{E}[x] = \mu_0 \quad(\text{默认}\ \mu_0=0),\qquad \text{统计量}\ T = \bar x = \frac1n\sum_i x_i$$

$H_0$ 下差值关于 0 **对称**，因此**符号可交换**。每次置换独立抽取随机符号 $s_i\in\{-1,+1\}$，计算置换统计量：

$$T^{*(b)} = \frac1n\sum_{i=1}^{n} s_i^{(b)}\,(x_i-\mu_0),\qquad b=1,\dots,B$$

$$p = \frac{\#\{b:\ T^{*(b)} \ge T_{\text{obs}}\} + 1}{B + 1}\quad(\texttt{greater};\ \texttt{less}\ \text{与}\ \texttt{two\_sided}\ \text{对称处理})$$

- $B = 1000$，`seed = 2024`；分子分母同时 $+1$ 是标准的**无偏修正**——它把观测值本身也算作一个可能的置换结果，避免 $p=0$ 这种不可能的精确值。
- **为什么必须翻转符号而不是重排位置**：直接对中心化值做位置重排，均值 $\bar x$ 是置换不变量，会得到退化的 $p$；代码注释里专门写了这一点。
- 对配对差值，该检验**等价于标准的配对置换检验**，且不依赖正态假设——这正是它相对 t 检验的价值。

**多重比较**：按 `family_key` 分族做 BH-FDR（见 §19.2）。`test_type` 有三类 family：`environment_edge`（边级）、`environment_interaction`（交互）、`environment_main_effect`（主效应）。

**目标.** 回答"**如果根本没有这个真实关系，出现这么大（或更大）的效应有多容易？**"——给出**不依赖分布假设**的经验 p 值，作为 Bootstrap CI 之外的第二种 Statistical Evidence。

**输出规模**：`permutation_results.csv` 3444 行 = 3378 `ok` + 66 `unavailable`。

---

### 18.3 Fisher 精确检验（Motif 富集）

**原理.** 对 motif 的 $2\times2$ 列联表做单侧 Fisher 精确检验（超几何分布）：

|  | 携带该 motif | 未携带 | 合计 |
| :--- | ---: | ---: | ---: |
| **前景**（高活性序列） | $a$ | $b$ | $a+b$ |
| **背景**（其余合格序列） | $c$ | $d$ | $c+d$ |

$$p = \Pr(X \ge a) = \sum_{k=a}^{\min(a+b,\,a+c)} \frac{\dbinom{a+b}{k}\dbinom{c+d}{a+c-k}}{\dbinom{n}{a+c}},\qquad n=a+b+c+d$$

效应量用**优势比**：

$$\mathrm{OR} = \frac{a\,d}{b\,c}\quad(\text{当 } b>0,\ c>0,\ a>0\ \text{时给出，否则 } \texttt{None})$$

另有 $\mathrm{enrichment} = \dfrac{a/(a+b)}{c/(c+d)}$（前景频率 / 背景频率）与 $\mathrm{effect} = \dfrac{a}{a+b} - \dfrac{c}{c+d}$（频率差）。

**目标**：回答"**某个序列 motif 是否在前景群体中过度出现**"，即富集显著性。OR 是 **Effect** 维度（富集倍数幅度），p/FDR 是 **Statistical Evidence** 维度。

**两组的定义（必须互斥，这是 Fisher 检验的前提）**：

- **前景** = 效率 **上三分位**（`enrichment_foreground_quantile = 0.67`）。阈值**逐细胞系**计算
  （各系效率尺度不同：hct116 0.302 / hek293t 0.263 / hela 0.302 / hl60 0.267），再把 4 个系的计数相加
  → `foreground_total = 5528`。
- **背景** = 前景的**补集**（`efficacy < 该细胞系阈值`）→ `background_total = 11221`。
  两者互斥且穷尽全部非空样本（`5528 + 11221 = 16749`）。

**边界**：这是**统计富集**，不等于生物学功能。另外，motif 候选的 `preferred_position` 绝大多数集中在**20 附近（PAM 邻近区）**，而 PAM 在数据里几乎是恒定的，因此 `odds_ratio` 会被这一恒定成分抬高；解读富集结果时应结合 `preferred_position` / `position_distribution`。

---

### 18.4 Bootstrap（见 §19.3 的 6 种 CI 类型）

Bootstrap 在本项目里**只作稳定性/不确定性证据**，不作显著性或因果证据（`analysis/stats/bootstrap.py` 的模块 docstring 开门见山写了这一条）。原理与分类见 §19.3。

---

## 19.统计量

### 19.1 p-value

**原理**：$p$ 值 = "在零假设 $H_0$ 成立的前提下，出现**当前或更极端**观测统计量的概率"：

$$p = \Pr\big(T(X) \succeq T(x_{\text{obs}})\ \big|\ H_0\big)$$

本项目有两个来源，口径不同、**不可互相替代**：

| 来源 | $H_0$ | 计算方式 | 产物列 |
| :--- | :--- | :--- | :--- |
| **符号翻转置换** | 配对差值的均值为 0（差值关于 0 对称） | 经验零分布，$B=1000$，$p=\frac{\#\{T^*\ge T\}+1}{B+1}$ | `permutation_results.p_value` / `.FDR` |
| **Factorial ANOVA F** | 该因子的额外平方和为 0 | 解析 $F$ 分布尾概率 | `anova_results.p_value` |
| **Fisher 精确检验** | motif 与前景/背景独立（OR=1） | 超几何精确尾概率 | `motif_enrichment.p_value` / `.FDR` |
| **线性回归 t 检验** | 该系数为 0 | 解析 t 分布双尾尾概率 | 线性 `p_value` / `FDR` |

**目标**：把"效应有多大"（Effect）与"这个效应有多不像噪声"（Statistical Evidence）分开。**p 小 ≠ 效应大**。`rule_discovery.py` 的微环境筛选即按此设计：先过统计门（细胞系级 CI 不跨 0）再过 effect 门（`|ΔR²| ≥ 0.01`）。

### 19.2 FDR（Benjamini–Hochberg q 值）

**原理**： 把 $m$ 个 p 值升序排列 $p_{(1)}\le\dots\le p_{(m)}$，BH 的**逐步上升（step-up）**过程给出

$$q_{(i)} = \min_{j \ge i}\ \Big(\frac{m}{j}\,p_{(j)}\Big),\qquad q_{(i)} \leftarrow \min\big(q_{(i)},\,1\big)$$

（代码实现为从大到小倒序遍历并维护 `min_q`，天然得到单调非降的 q 序列；数值与 `statsmodels.multipletests` 兼容。）

**目标**：控制**错误发现率**（被判定为发现的项里假阳性的期望比例）$\le \alpha$。相比 Bonferroni 控制"犯任何一个假阳性的概率"（FWER），BH 在检验数量大、真信号稀疏的基因组学场景下功效高得多。

**本项目的第一原则：family 隔离。** 只允许对**同一科学问题的假设族**做校正：

| family_key 前缀 | 包含什么 |
| :--- | :--- |
| `environment_edge\|<split>\|<cell>\|<model>` | 该上下文下所有环境边的置换检验 |
| `environment_main` / `environment_interaction` | 主效应 / 交互的置换检验 |
| `environment_anova` / `environment_anova_group` | 全局 / 分组 ANOVA |
| `motif_enrichment` | 全部 motif 的 Fisher 精确检验 |
| （线性模型）系数 family | 该模型的全部回归系数 |

`analysis/config.py::FdrFamilyConfig.min_family_size = 2`（族过小不校正，标注原因）。置换检验 189 个 family（analysis/pipeline.py 只统计这一类），加 ANOVA 的 64 个，分析层 fdr_family 共 253 个。

### 19.3 CI（置信区间）—— 本项目共 6 种类型，逐项说明

所有 CI 都是 **percentile bootstrap**：对同一批数据重采样 $B$ 次，取经验分布的 $\alpha/2$ 与 $1-\alpha/2$ 分位数

$$\mathrm{CI}_{1-\alpha} = \Big[\,Q_{\alpha/2}\big(\hat\theta^*_{1..B}\big),\ Q_{1-\alpha/2}\big(\hat\theta^*_{1..B}\big)\,\Big]$$

默认 $B=2000$、$\alpha=0.05$（即 95%）、`seed=2024`；`excludes_zero = not(ci_low ≤ 0 ≤ ci_high)`。**迭代数不足或样本 < 3 时返回 `available=False` / `status="unavailable"`，绝不用 0 或点估计冒充 CI。**

| # | 类型 | 原理（重采样单位） | 目标（回答什么） | 产物 |
| ---: | :--- | :--- | :--- | :--- |
| 1 | **通用估计量 CI** | 对一维数据重采样，任意估计量 $\hat\theta$ 的分位数区间 | 该估计量的整体不确定性 | `bootstrap_ci()`（`analysis/stats/bootstrap.py`） |
| 2 | **配对差 CI**（Paired difference） | **同一次** index 重采样同时作用于 baseline 与 expanded，重算 $\mathrm{metric}(b[idx])-\mathrm{metric}(a[idx])$ | 环境增量 Δ 的稳定性；配对消除了样本组成差异，是项目的 **Paired baseline 原则** | `bootstrap_difference_ci()`；`bootstrap_results.csv`（边级 × R²/MAE/RMSE 三指标 × 4 种子） |
| 3 | **配对逐样本多指标 CI**（向量化） | 用 multinomial **计数矩阵** $C\in\mathbb{N}^{B\times n}$（$C_{b,i}$ = 第 $b$ 次重采样中样本 $i$ 被抽中的次数）把"逐次重采样再算指标"改写为 **BLAS matvec**：$s_y = C\,y$、$s_{y^2}=C\,(y\odot y)$、$s_{e^2}=C\,e^2$、$s_{\lvert e\rvert}=C\,\lvert e\rvert$，于是<br>$R^2_b = 1-\dfrac{s_{e^2,b}}{s_{y^2,b}-s_{y,b}^2/n}$，$\ \mathrm{MAE}_b=\dfrac{s_{\lvert e\rvert,b}}{n}$，$\ \mathrm{RMSE}_b=\sqrt{\dfrac{s_{e^2,b}}{n}}$；<br>与逐次重采样**逐位等价**（实测 CI 在 $10^{-12}$ 内一致）；`_COUNT_CACHE` 按 $(n,B,\text{seed})$ 复用同一矩阵 | 同一个 CI 框架下批量给出 ΔR²/ΔMAE/ΔRMSE | `bootstrap_paired_metric_ci_fast()` / `bootstrap_paired_metrics_ci_fast()` |
| 4 | **Factor 级主效应的跨模型 CI** | **模型层** bootstrap：$\hat\theta$ = 跨模型主效应均值，重采样单位是模型而非样本 | "某个环境因子的**总体**效应有多确定" | `bootstrap_main_effects.csv`（4 行，`n_models` 列记录重采样单元数） |
| 5 | **细胞系级主效应 CI** | 同上，但按 cell line 分组计算 $\hat\theta$ | "某个环境因子的效应**在各细胞系内**是否稳定" | `bootstrap_cellline_effects.csv`（36 行 = all 16 + single 16 + mixed 4） |
| 6 | **ANOVA 效应量行级 CI** | 对 ANOVA 的**设计矩阵按行**重采样 $B=400$ 次，每次重新解最小二乘并重算 $\sum\beta_\ell$ | 单个 ANOVA 项上"on−off 调整差"的不确定性 | `anova_results.ci_low` / `.ci_high`（`ci_iterations = 400`） |

### 19.4 η²（方差解释比）

**原理**：方差分析里"该项解释的平方和占总平方和的比例"：

$$\eta^2 = \frac{\mathrm{SS_{term}}}{\mathrm{SS_{tot}}},\qquad \mathrm{SS_{tot}} = \sum_i (y_i-\bar y)^2$$

偏（partial）版本则把分母换成"该项 + 残差"：

$$\eta^2_{\text{partial}} = \frac{\mathrm{SS_{term}}}{\mathrm{SS_{term}} + \mathrm{RSS_{full}}}$$

> **本项目的 `anova_results.effect_size` 用的是 partial $\eta^2$**（第二个式子）。原因：本批的 ANOVA 里 `model`/`cell_line`/`split_type` 作为 **block 因子**进了模型，它们占了 $\mathrm{SS_{tot}}$ 的
> **74.9%**；用经典 $\eta^2$ 会把环境因子的效应量低估约 4 倍（实测比值 0.2512）。

两者的差别：**经典 $\eta^2$ 的分母含所有其它项**，因此在多因子/不平衡设计下，各 $\eta^2$ 之和 $\le 1$、互相"挤压"；**partial $\eta^2$ 的分母只含该项与误差**，各 partial $\eta^2$ 之和可以超过 1，衡量的是"排除其它项之后，这项单独还能解释多少"。

两者与 F 的关系（可用来互相反查）：

$$F = \frac{\eta^2_{\text{partial}}/\,\mathrm{df_{num}}}{(1-\eta^2_{\text{partial}})/\,\mathrm{dof_{den}}}$$

**目标**：回答"**这个因素解释实验结果变异的份额是多少**"——即 §18.1 ANOVA 里的 `effect_size` 列。它是 Effect 维度的**无量纲**补充：`effect` 告诉你"改变量有多少"（带标签量纲），$\eta^2$ 告诉你"在总变异里占多大比例"（无量纲，可跨因子比较）。

**边界**：$\eta^2$ 是**样本内**的方差解释比，会随设计（因子数、是否平衡）变化，不能当作总体效应量的无偏估计；小 $\eta^2$ + 大样本同样可以得到极小的 p 值。

---

## 附：底座的其它支撑机制

除上面四节，还有几项不属于"方法"但同样决定结论成立与否的底座机制：

| 机制 | 原理 | 目标 |
| :--- | :--- | :--- |
| **配对 cohort 校验** | `compute_paired_increment()`：baseline 与 expanded 的样本数必须一致，否则返回 `(NaN, paired_ok=False)` | ΔR² 只在**同一批样本**上做差才有意义；不一致时差异可能来自样本组成而非环境贡献，故标记为不可用于科学归因 |
| **发散过滤** | $\lvert\Delta\rvert$ 或 $\lvert R^2\rvert \ge 10$ 判为数值不稳定 | 防止 $10^{18}$ 量级的发散 run 主导均值/中位数；被排除的行数记入 `numerical_instability_excluded_from_evidence` |
| **多重种子** | single/all 固定 42；mixed 42/43/44/45 | 让"结论是否依赖某个种子"可检验 |
| **FDR family 隔离** | 见 §19.2 | 防止把不该混的假设放进一个校正族，从而虚增或虚减功效 |
| **四指纹** | `data_fingerprint` / `code_fingerprint` / `env_fingerprint` + `env_stack_id` / `split_digest`，随每个 run 落盘 | 事后证明"分析所用划分 == 训练所用划分"，并使结果可溯源到「数据 + 配置 + 代码 + 模型 + 随机种子」 |
| **越界断言** | `xai_importance.export_feature_table()` 对非线性模型强制剔除 t/p/FDR，残留即抛错 | 用代码防止"给深度模型编造显著性"这类学术红线问题 |

---

# R1.赛道二规律发现交付物（`rule_discovery.py`）

前面 `# 项目流程` 与 `# 项目底层` 描述的是**分析与证据链路**；本节描述**最终交付物的生成**。

## 这个程序做什么

`rule_discovery.py` **不训练模型、不重算统计量**。它只把已有的分析产物整合起来：

| 输入（只读，位于 `results/summary/<batch>/`） | 提供什么 |
| :--- | :--- |
| `tables/bootstrap_cellline_effects.csv` | 细胞系级主效应的 Bootstrap CI |
| `tables/environment_main_effects.csv` | 环境主效应 ΔR²（逐模型） |
| `feature_importance/key_regulatory_biomarkers.csv` | 归因 SNR 与 FDR（只用于展示支持情况） |
| `tables/motif_candidates.csv` | motif 级 BH-FDR / ISM 效应 / 支持度 |
| `tables/motif_instances.csv` | 样本级 ISM / IG 归因（逐实例复核） |
| `tables/motif_enrichment.csv` | Fisher 富集（备选 FDR 来源） |

输出**三类 CSV（每细胞系 2 个 + 跨细胞系 1 个）+ 1 个多 Sheet Excel**，写入 `results/赛道二_results/<batch>/`。
程序**不修改任何已有分析结果文件**。

## 运行

```bash
python rule_discovery.py                                   # 自动探测全部批次
python rule_discovery.py --data-set DeepCRISPR              # 按数据集名（大小写不敏感）
python rule_discovery.py --batch-dir results/summary/DeepCRISPR
python rule_discovery.py --print-config                     # 打印生效的评选标准
python rule_discovery.py --config-json my_thresholds.json   # 只覆盖想改的阈值
python rule_discovery.py --include-rejected                 # 审计模式：保留全部行 + selected 标志
```

必需输入缺失时程序**直接报错退出（exit 2）并列出缺哪个文件**，不会静默跳过。

## 输出结构

```text
results/赛道二_results/<batch>/
├── csv/
│   ├── <cell_line>_microenv.csv        # 每细胞系 1 个（DeepCRISPR 为 4 个）：单细胞系微环境候选
│   ├── <cell_line>_motif.csv           # 每细胞系 1 个（DeepCRISPR 为 4 个）：单细胞系 motif 候选
│   └── microenv_and_motif.csv          # 1 个：跨细胞系泛化候选
└── excel/
    └── 赛道二_results.xlsx             # 上述全部 CSV 各占一个 Sheet
```

Excel Sheet 名（以DeepCRISPR为例）：`HCT116_microenv` / `HCT116_motif` / `HEK293T_microenv` / `HEK293T_motif` / `HeLa_microenv` / `HeLa_motif` / `HL60_microenv` / `HL60_motif` / `microenv_and_motif`。
CSV 是**原始交付文件**，Excel 只是同一批数据的汇总视图，两者内容一致。

> Excel 由 `xlsxwriter` 写出（缺失时回退 `openpyxl`）。注意 `xlsxwriter` 是**只写**引擎：
> 若要用 `pandas.read_excel()` 把它读回来，需要额外装 `openpyxl`。用 Excel / LibreOffice 直接打开不受影响。

## 筛选顺序（三个文件各自的判据）

三个文件都遵循 **Statistical Evidence → Effect**，组合文件再叠加一层 **Robustness**。

### `<cell_line>_microenv.csv` — 单细胞系微环境通道

| 层 | 判据 | 数据来源 |
| :--- | :--- | :--- |
| ① Statistical Evidence | **细胞系级主效应 Bootstrap CI 不跨 0**（且 `status=ok`、`n_bootstrap ≥ 200`） | `bootstrap_cellline_effects.csv` |
| ② Effect | **`\|ΔR²\| ≥ 0.01`**（先按模型跨 split×cell 求均值，再对模型等权平均 —— 与原 `evidence_matrix.overall_effect` 同一算法，但不再产出那张表） | `environment_main_effects.csv` |
| 附 Importance（**只列举，不参与筛选**） | SNR ≥ 2.5 的非线性模型数；FDR < 0.001 的特征数 | `key_regulatory_biomarkers.csv` |

字段（24 列）：

| 字段 | 说明 |
| :--- | :--- |
| `cell_line` | 该行所属细胞系（单细胞系文件内恒定） |
| `split_type` | 该行 CI / 主效应取自哪条 split 线（`all` / `single` / `mixed`） |
| `environment_factor` | 被检验的环境因子名（`ctcf` / `dnase` / `h3k4me3` / `rrbs`） |
| `splits_available` | 该细胞系实际拥有的全部 split 线（筛选**之前**写入，供 `all_splits` 判据使用） |
| `bootstrap_CI_lower` / `bootstrap_CI_upper` | 细胞系级主效应的 Bootstrap 百分位区间上下界 |
| `CI_pass` | 第一层判据是否通过：CI 不跨 0（且 `status=ok`、`n_bootstrap ≥` 阈值） |
| `ci_estimate` | 该区间的点估计（跨模型主效应均值） |
| `bootstrap_status` | CI 那一步的结果：`ok` / `unavailable` |
| `bootstrap_n_iterations` | 该 CI 的重采样次数 |
| `bootstrap_n_models` | 该 CI 的跨模型重采样单元数 |
| `main_r2_delta` | 第二层判据用的环境主效应 ΔR²（跨模型等权平均） |
| `n_models` | 参与该均值的模型配置数 |
| `model_effects` | 逐模型主效应明细（`模型:值`，`;` 分隔） |
| `effect_pass` | 第二层判据是否通过：`\|ΔR²\| ≥` 阈值，且**只在第一层已通过的行上**评估 |
| `selected` | 最终是否入选（与 `effect_pass` 等价，两层串联的结果） |
| `nonlinear_model_support` | Importance（只列举）：命中 SNR 阈值的非线性模型**个数**，计数口径由 `microenv_importance_model_level` 决定 |
| `snr_support_threshold` | 上述计数使用的 SNR 阈值 |
| `snr_support` | 被计入 `nonlinear_model_support` 的模型名（`;` 分隔） |
| `snr_support_configs` | 命中 SNR 阈值的**全部**模型配置名（不论计数口径，供追溯用） |
| `fdr_lt_threshold_count` | 该通道下 `fdr <` 阈值的特征行数（本批只有 `linear` 报告 FDR） |
| `fdr_support_threshold` | 上述计数使用的 FDR 阈值 |
| `source_file` | 该行的输入表文件名（`bootstrap_cellline_effects.csv` / `environment_main_effects.csv`） |
| `generated_time` | 生成时间（UTC） |

> `effect_pass` 与 `selected` 是**递进**的：`effect_pass` 表示"在第一层已通过的行上又过了第二层"，
> 因此 `selected == effect_pass`，而"满足第二层条件但第一层没过"的行不会被计入。

### `<cell_line>_motif.csv` — 单细胞系 motif

| 层 | 判据 | 数据来源 |
| :--- | :--- | :--- |
| ① Statistical Evidence | **BH-FDR < 0.05** | `motif_candidates.csv`（可用 `motif_fdr_source` 切到 `motif_enrichment.csv`） |
| ② Effect | **`\|ISM effect\| ≥ 0.01`** | `motif_candidates.mean_effect`（可用 `motif_ism_effect_source` 切到逐实例均值） |
| 附 Attribution support（只列举） | `IG_score`（逐实例 IG 均值）、`SHAP_score`、`model_support`、`model_consistency` | `motif_instances.csv` / `motif_candidates.csv` |

字段（31 列）：

| 字段 | 说明 |
| :--- | :--- |
| `cell_line` | 该 motif 所属细胞系 |
| `split_type` | 该 motif 来自哪条 split 线（`all` / `single`） |
| `environment` | 该 motif 来自哪个环境组合（本项目只在纯序列上下文提 motif） |
| `splits_available` | 该细胞系实际拥有的全部 split 线（筛选**之前**写入） |
| `motif_id` | motif 主键，编码 `<架构>_<细胞系>_<环境>_<方法>_<序号>` |
| `motif` | **共识序列**（取 `motif_candidates.consensus`，已是确定碱基）——跨细胞系对齐用的键 |
| `position` | 主峰位置（取 `preferred_position`） |
| `position_start` / `position_end` | 实例在 23 nt 序列上的起止位置（1-based 闭区间） |
| `position_mean` | 实例起始位置的均值 |
| `BH_FDR` | 第一层判据用的 q 值，取自 `fdr_source` 指定的那张表 |
| `FDR_pass` | 第一层判据是否通过：`BH_FDR <` 阈值（阈值见 `motif_fdr_threshold`） |
| `fdr_source` | FDR 口径：`candidate`（取 `motif_candidates.FDR`）或 `enrichment`（取 `motif_enrichment.FDR`） |
| `enrichment_FDR` | 另一来源（`motif_enrichment`）的 q 值，供切换口径时对照，不参与筛选 |
| `ISM_effect` | 第二层判据用的 ISM 效应，取自 `ism_effect_source` 指定的口径 |
| `ISM_effect_abs` | `\|ISM_effect\|` |
| `ISM_effect_pass` | 第二层判据是否通过：`\|ISM_effect\| ≥` 阈值，且**只在第一层已通过的行上**评估 |
| `ism_effect_source` | 效应口径：`candidate`（motif 级 `mean_effect`）或 `instances`（逐实例 ISM 均值） |
| `ISM_effect_instances` | 逐实例 ISM 均值（`instances` 口径下与 `ISM_effect` 同值） |
| `n_instances` | 该 motif 的支撑实例数 |
| `IG_score` | Attribution support（只列举）：逐实例 IG 均值 |
| `SHAP_score` | Attribution support（只列举）：SHAP 值；本批无此归因，**整列为空** |
| `model_support` | Attribution support：`模型来源/核宽配置/归因方法` 去重后 `;` 连接 |
| `model_consistency` | Attribution support：该 motif 的跨模型归因一致性得分（来自 `motif_candidates`） |
| `support_count` | 命中该 motif 的 seqlet 条数 |
| `enrichment` | 富集倍数（源自 `motif_enrichment` 的副本，与那边逐字相同） |
| `odds_ratio` | Fisher 2×2 表的优势比（同上，副本） |
| `p_value` | Fisher 精确检验 p 值（未校正，副本） |
| `selected` | 最终是否入选（与 `ISM_effect_pass` 等价） |
| `source_file` | 该行的输入表文件名（`motif_candidates` / `motif_instances` / `motif_enrichment`） |
| `generated_time` | 生成时间（UTC） |

> 本项目的 motif 发现**只由 CNN 系列驱动**。

### `microenv_and_motif.csv` — 跨细胞系泛化

在单细胞系已通过筛选的候选上，只增加一条 Robustness 判据：

$$cellline\_ratio = \frac{supporting\ celllines}{total\ celllines}, \qquad cellline\_ratio > 0.75$$

| 字段 | 说明 |
| :--- | :--- |
| `feature_type` | `microenvironment` 或 `motif` |
| `feature_name` | 环境因子名（如 `ctcf`）或 motif consensus 序列（如 `GAGG`） |
| `supporting_celllines` / `supporting_splits` / `n_supporting` / `n_total_celllines` | 支持来源 |
| `cellline_ratio` / `cellline_ratio_pass` | 跨细胞系比例与结论 |
| `mean_effect` / `effect_variance` / `effect_by_cellline` | 效应量与跨细胞系离散度 |
| `statistical_summary` | 各细胞系的 CI（微环境）或去重后的 FDR（motif） |
| `source_files` / `generated_time` | 溯源 |

> `splits_available`（前两个文件里都有）记录**该细胞系实际拥有哪几条 split 线**，
> 它在筛选**之前**写入，因此即使 CSV 只保留通过筛选的行，也仍能判断
> `cellline_support_mode="all_splits"` 所需的"是否每条 split 都通过"。

> **键的选择**：微环境以 **environment factor** 为键；motif 以 **consensus 序列**为键
> ——因为 `motif_id` 里含细胞系名（如 `motif_cnn33_hela_sequence_cnn_ig_001`），
> 逐细胞系唯一，跨细胞系对齐必须用序列。
>
> **默认阈值下的算术后果**：本仓库有 4 个细胞系，`ratio > 0.75` 等价于"必须 4/4 全支持"；
> 想允许 3/4，把 `cellline_ratio_inclusive` 设为 `true`（即 `>= 0.75`）。

## 可配置的评选标准

**所有阈值集中在 `rule_discovery.py` 的 `RuleDiscoveryConfig` 类**，改配置即可调整评选标准，
不需要改核心逻辑；也可以用 `--config-json` 只覆盖想改的字段。

> 用户可以通过 `rule_discovery.py` 中 `RuleDiscoveryConfig` 类调整筛选阈值和评选标准。

| 字段 | 默认值 | 含义 |
| :--- | :--- | :--- |
| `microenv_ci_required` | `True` | 是否启用第一层 CI 判据（`False` = 跳过，不推荐） |
| `microenv_min_bootstrap_iterations` | `200` | CI 参与判定所需的最少 bootstrap 次数 |
| `microenv_ci_must_exclude_zero` | `True` | CI 是否必须不跨 0 |
| `microenv_effect_aggregation` | `"model_mean"` | `model_mean` = 跨模型等权平均；`per_model` = 每模型一行 |
| `microenv_min_abs_main_r2_delta` | `0.01` | Effect 闸门 |
| `microenv_unstable_effect_threshold` | `10.0` | \|ΔR²\| 超此值的模型行先剔除（同 `consensus.unstable_effect_threshold`） |
| `microenv_effect_must_match_ci_sign` | `True` | 要求 Effect 与 CI 点估计同号（两者本是同一统计量，此项用于捕捉来源漂移） |
| `microenv_require_ci_status_ok` | `True` | 要求 bootstrap 行的 `status == "ok"` |
| `microenv_snr_threshold` | `2.5` | Importance：SNR 展示阈值 |
| `microenv_fdr_threshold` | `0.001` | Importance：FDR 展示阈值 |
| `microenv_importance_model_level` | `"family"` | 计数口径：`family`（≤4）或 `config`（≤6） |
| `motif_fdr_threshold` | `0.05` | motif 第一层 BH-FDR 阈值 |
| `motif_fdr_source` | `"candidate"` | FDR 取自 `motif_candidates` 还是 `motif_enrichment` |
| `motif_abs_ism_effect_threshold` | `0.01` | motif 第二层 \|ISM effect\| 阈值 |
| `motif_ism_effect_source` | `"candidate"` | 效应取自 motif 级 `mean_effect` 还是逐实例均值 |
| `motif_min_support_count` | `0` | 可选：最少实例支撑数 |
| `motif_require_status_ok` | `False` | `True` = 只保留 `status == "ok"` 的 motif 行 |
| `microenv_nonlinear_models` | 4 个家族 | 哪些模型算"非线性"（仅影响 Importance 的计数口径） |
| `cellline_ratio_threshold` | `0.75` | 跨细胞系比例阈值 |
| `cellline_ratio_inclusive` | `False` | `False` = `>`；`True` = `>=` |
| `cellline_support_mode` | `"any_split"` | 单个细胞系算"支持"的条件：`any_split` = 任一 split 通过；`all_splits` = 该细胞系**拥有的每条 split 都通过** |
| `include_rejected_rows` | `False` | `True` = 保留全部输入行并附 `selected`（审计） |
| `target_cell_lines` | 4 个细胞系 | 参与交付的细胞系；若配置的这些细胞系在本批次一个都不存在，改用该批次实际的细胞系 |
| `output_root` | `results/赛道二_results` | 交付物根目录 |

其余字段与筛选无关，只影响输入定位与输出外观：
`summary_root`（批次汇总根）、`batch_dirs` / `dataset_names`（指定批次）、
`tables_subdir`（表所在子目录）、6 个 `*_file`（输入文件名）、
`target_cell_lines` / `cell_line_display`（细胞系与显示名）、
`microenv_factor_to_channel`（因子名到归因 channel 的映射）、
`csv_encoding` / `excel_filename` / `excel_max_col_width`。

## 科学声明

> 本结果文件用于赛事展示和候选规律整理，其筛选标准用于提高候选发现效率，并非严格意义上的生物学因果证明。所有候选规律仍需结合独立实验验证或进一步机制研究确认。


# R2.results/Summary/<batch>文件说明

## 目录一览

```text
results/summary/<数据集>/                批次级汇总（§10 + §11）
  analysis_plan.json                     本次分析的任务勾选与阈值快照
  analysis_status.json                   每个任务的 selected / available / status / reason
  execution_log.json                     执行结果 + 产出 PNG 相对路径清单
  cnn7_validation.md / .csv              CNN7×CRISPRon 一致性验证交付物

  train_data/                            §10 汇总表（5 张 CSV）
  feature_importance/                    §10 关键调控特征库（7 个 .md + key_regulatory_biomarkers.csv）
  reports/                               §11 三份交付报告
    overview.md   data_quality.md   anomaly_report.md
  tables/                                §11 分析表（**20 张** CSV，见下）
  figures/
    environment/
      conditional_delta_r2_heatmap.png   背景 S × 新增环境 e 的条件 ΔR² 热图
      environment_main_effects.png       4 个环境因子的主效应
      factorial_dag/                     63 张（split × cell_line × model）
      conditional_delta_r2/              63 张
      interactions/                      63 张
    plots/
      position_attribution_<method>.png  位置归因热图 7 张（position × channel，无符号幅度）
                                         method ∈ cnn_ig · cnn_ism · mlp_ig ·
                                         transformer_attention · xgboost_gain ·
                                         xgboost_treeshap · linear_coefficient
```

## JSON

### analysis_plan.json

+ 来源程序: `analysis.pipeline`（写出）／命令行 `--analysis-plan`（读入）

+ 内容: 本次分析的**任务勾选 + 阈值快照**。顶层键：`run_qc` /
  `run_prediction_analysis` / `environment` / `sequence` / `statistics`。
  它是"这次跑了什么"的唯一记录，重跑时传它即可**复现同一批任务**。


### analysis_status.json

+ 来源程序: `analysis.pipeline` → `ExecutionPlan.to_status_dict()`

+ 内容: 每个任务的 `task_id` / `selected`/ `available`/ `status`/ `reason` / `artifact`。

### execution_log.json

+ 来源程序: `analysis.pipeline`

+ 内容: `executed`+ `completed_at` + `status_counts` +`figures`。

---

## figures

### environment/conditional_delta_r2_heatmap.png

+ 来源程序: `analysis.visualization.environment_plots`（`render`）

+ 说明: 一张热图：行 = 背景组合 `S`，列 = 新增环境因子 `e`，格子 = 条件 ΔR²（只画 ΔR²，1 张）。
  数值直接取自 `tables/environment_conditional_delta_r2.csv`。

### environment/environment_main_effects.png

+ 来源程序: `analysis.visualization.environment_plots`

+ 说明: 4 个环境因子的主效应柱状图（把背景平均掉之后），数值来自
  `tables/environment_main_effects.csv`。

### environment/factorial_dag/（63 张）

+ 来源程序: `analysis.visualization.factorial_dag.render_factorial_dag`

+ 说明: 每个 `(split_type, cell_line, model)` 一张因子组合格图（16 节点 / 32 边），
  节点按 `status` 着色，缺数据的组合显式标灰。63 = 3 split × 5 cell_line/model 组合
  （`hct116` `hek293t` `hela` `hl60` `none`）× 7 model。输入是
  `environment_nodes.csv` + `environment_edges.csv`。

### environment/conditional_delta_r2/（63 张）

+ 来源程序: `analysis.visualization.factorial_dag.render_conditional_delta_r2`

+ 说明: 与 `factorial_dag/` **相同的 63 个上下文**，每个一张「背景 S × 新增 e」的条件 ΔR² 图。输入
  `environment_edges.csv`。

### environment/interactions/（63 张）

+ 来源程序: `analysis.visualization.factorial_dag.render_environment_interactions`

+ 说明: 与 `factorial_dag/` **相同的 63 个上下文**，每个一张两因子交互图。输入
  `environment_interactions.csv`。

### plots/position_attribution_<method>.png（7 张）

+ 来源程序: `analysis.visualization.attribution_plots`（`AttributionPlotStyle` + `render`）

+ 说明: 位置归因热图（position × channel），7 种归因方法各一张，数值取
  `tables/attribution_summary.csv` 的 `importance` 按 `(channel, position)` 取均值。
  渲染时带**显示门**：`FDR ≥ 0.10`（linear）或 `SNR < 0.8`（非线性）的格子不画；
  linear 的异常系数（|coef| 超阈值）画成**黑色**且不参与色域归一化。

---

## report

### overview.md

+ 来源程序: `analysis.reports.markdown_report.build_overview_md`

+ 说明: 覆盖度七行（experiment count / valid / models / cell lines / environments /
  splits / metric inconsistency rows）+ **13 个任务的状态清单**（符号：`✓` completed ·
  `⊘` skipped · `⚠` unavailable · `✗` failed）+ 两句固定执行说明。
  数据源：`tables/experiment_table.csv` 经 `coverage_summary()` + `analysis_status.json`。

### data_quality.md

+ 来源程序: `analysis.reports.markdown_report.build_data_quality_md`

+ 说明: 批次级数据质量：`cell_line × environment` **覆盖矩阵**（单元格 = run 数；
  `mixed` 是训练端 `none` 的显示层改名）+ 指标一致性明细。本批 `metric_inconsistency` 0 行。

### anomaly_report.md

+ 来源程序: `analysis.reports.markdown_report.build_anomaly_md`

+ 说明: 两类异常的定义与明细：
  `metric_inconsistency`（同 cohort 下 ΔR²/ΔRMSE 同号）与
  `numerical_instability_excluded_from_evidence`（\|ΔR²\| 超阈值、**不参与跨模型聚合**的行数，本批 12 行）。

---

## train_data/ 与 feature_importance/

### train_data/all_experiments.csv

+ 来源程序: `analysis.collect_results`

+ 说明: 全部 1344 个 run 的逐 run 指标（含验证集与测试集两套）。是
  `tables/experiment_table.csv` 的上游，也是 `tables/experiment_table.source_table`
  指向的那个文件。

+ 列说明: `model` / `split_type` / `cell_line` / `environment` / `random_seed` 含义见
  [tables 的通用约定](#tables)；`n_train` / `n_valid` / `n_test` = 三个划分的样本数；
  `R2` / `MAE` / `RMSE` / `MSE` / `Pearson` / `Spearman` = **测试集**指标；
  `validation_R2` / `validation_MAE/RMSE` / `validation_Pearson/Spearman` = 验证集指标；
  `run_name` = 唯一键。

### train_data/single_cell_line_result.csv · all_cell_line_result.csv · mixed_cell_line_result.csv

+ 来源程序: `analysis.collect_results`

+ 说明: 按三种 `split_type` 分别做的模型级聚合。列是测试集/验证集两组指标的合并写法：
  `MAE/RMSE` / `Pearson/Spearman`（**同一列里装了两个指标**）、`R2`、`delta_R2`，
  以及对应的 `validation_MAE/RMSE` / `validation_Pearson/Spearman` / `validation_R2` 列。

### train_data/baseline.csv

+ 来源程序: `analysis.collect_results`

+ 说明: 纯序列基线（不含任何表观通道），用于回答"加了环境到底有没有用"。
  列与 `all_experiments.csv` 的指标列同名同义。

### feature_importance/*.md（7 个）

+ 来源程序: `analysis.importance_extraction`

+ 说明: 白名单归因表：`linear_coefficiency.md`、`xgboost_importance.md`、
  `mlp_importance.md`、`cnn{33,53,73}_importance.md`、`transformer_importance.md`。
  各含 `sig` 星级列。

### feature_importance/key_regulatory_biomarkers.csv

+ 来源程序: `analysis.importance_extraction`（入口 `generate_key_regulatory_biomarkers`）

+ 说明: 关键调控特征库（长表，**91747 行 × 9 列**）。一行 =
  一个 `(split_type, cell_line, environment, model, feature)` 组合的一条记录，
  且**已被过滤到「有星级」的行**（见下方须知 ①）。


| 列 | 含义 |
| :--- | :--- |
| `split_type` | 训练方式：`all` / `mixed` / `single`，与 `tables/` 同义 |
| `cell_line` | 细胞系；`mixed` 划分不绑定细胞系，记作 `none` |
| `environment` | 环境通道组合串（16 个取值，`sequence` … `all`） |
| `model` | **7 个模型配置**，此处写作 `cnn33` / `cnn53` / `cnn73` / `linear` / `mlp` / `transformer` / `xgboost`；与 `tables/` 的 `cnn(3\|3)` 写法不同 |
| `feature` | **已规范化为 1-based 的 `pos<N>_<通道>`**（如 `pos18_C`），全表 **184** 个取值 = 8 通道 × 23 位。与 `attribution_summary.feature` 的两套编号**不同** —— 这里已统一，可直接解析。`Bias` 行与位点越界的行已被丢弃 |
| `significance` | 星级。**两类模型的判据不同**：`linear` 看 **BH-FDR**（`***` q<0.001、`**` <0.01、`*` <0.05、`.` <0.10）；其余 6 个模型看 **SNR**（`***` ≥2.5、`**` ≥1.8、`*` ≥1.2、`.` ≥0.8）。**非线性模型的星级是「归因稳健性」，不是统计显著性** |
| `contribution` | **各模型主力归因指标的原值**，逐模型不同：`cnn`=`CNN_ISM` · `mlp`=`MLP_IG` · `transformer`=`Transformer_Attention` · `xgboost`=`XGB_Gain` · `linear`=`Linear_Coefficient`。 **量纲不可跨模型比较**：linear 是带符号系数（实测可到 ±1e11），其余是归因幅度 |
| `snr` | 逐模型不同：`linear` 放 **`t_stat`**（可正可负，实测 −9.53 ~ 16.1）；其余 5 个放各自的 SNR —— `cnn`=`ISM_SNR` · `mlp`=`IG_SNR` · `transformer`=`Attention_SNR` · `xgboost`=`SHAP_SNR` |
| `fdr` | **只有 `linear` 行有值**（实测 3337 / 91747，非空行全部 `model=linear`），其余 6 个模型整列为空。族 = **该 run 自己的全部回归系数**，BH 在**训练端** `core/models/linear/linear_regression.py` 内完成 |

**三条读表须知**

1. **本表已被过滤**：`generate_key_regulatory_biomarkers` 先取 `sig_score > 0`
   （即 `linear` 的 FDR<0.10、非线性的 SNR≥0.8），所以它**不是** 184 特征 × 全部 run 的完整笛卡尔积。
   **不能拿行数反推实验总数**，也不能用它统计"某特征在所有 run 里的分布"。
2. **`mixed` 行是跨 seed 的均值**：`mixed` 有 4 个种子，同一
   `(split, cell, environment, model, feature)` 会先对 `sig_score` / `contribution` / `snr` / `fdr`
   取均值，再由**均值**重新判定星级（阈值 2.5 / 1.8 / 1.0 / 0.5）。
   因此 `mixed` 行的星级是"平均后的等级"，与 `all` / `single` 行的"单次判定"不完全同质。
3. **排序规则**：按 `split_type → cell_line → environment → model → sig_score(降序) → |contribution|(降序)`。

### cnn7_validation.md / .csv

+ 来源程序: `python -m analysis.crispron_validation`（`analysis.crispron_validation.table`）

+ 回答什么问题: 本项目 CNN7（`sequence_kernel=7`）在 Pos18 C→A 这个已知关键位点上，其
  序列–效率方向性是否与第三方模型 CRISPRon 一致。中间产物在 `results/crispron_validation/`。

+ 列说明: CSV 7 列，md 是同一批数据的汇总视图（总序列数 + 两个主指标的同正号占比 +
  逐细胞系一致性）。命名约定 `A-B` = `A − B`；`ism` = 对 ISM 突变后序列的预测值。

| 列 | 含义 |
| :--- | :--- |
| `cell_line` | 细胞系（`hct116` / `hek293t` / `hela` / `hl60`） |
| `sgRNA` | 23 nt 序列（20 nt protospacer + 3 nt PAM） |
| `y_true-ism` | 真实编辑效率 − CNN7 对 **C18A 突变序列**的预测效率。**只作噪声参照** |
| `y_true-crispron_c18a` | 真实编辑效率 − CRISPRon 对 **C18A 突变序列**的预测。**只作噪声参照** |
| `cnn_pred-ism` | CNN7 对 **WT** 的预测 − 对 **C18A 突变序列**的预测。两个主指标之一 |
| `crispron_wt-crispron_c18a` | CRISPRon 对 **WT** 的预测 − 对 **C18A 突变体**的预测。两个主指标之一 |
| `consistency` | `cnn_pred-ism` 与 `crispron_wt-crispron_c18a` **同号为 `+`、异号为 `-`、含 0 记 `0`、缺失记空串** |

---

## tables

> 标注 `†` 的列在**多张表里同名不同义**，见末尾速查表。

### experiment_table.csv

+ 来源程序: `analysis.collect_results`（产出 `train_data/all_experiments.csv`）→
  `analysis.environment.factorial_dag` 统一为长表

+ 回答什么问题: 全部 run 的统一实验表（**所有其它表的底座**）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `model` | `linear` `mlp` `transformer` `xgboost` `cnn(3\|3)` `cnn(5\|3)` `cnn(7\|3)` | 7 个模型配置；`cnn(k\|e)` = 序列核宽 k、环境核宽 e |
| `split_type` | `all` · `single` · `mixed` | 数据划分方式。`all` = 留一细胞系（LOCO）；`single` = 系内划分；`mixed` = 打散所有细胞系 |
| `cell_line` | `hct116` `hek293t` `hela` `hl60` `none` | `none` 只出现在 `mixed`（该划分不绑定细胞系） |
| `environment` | `sequence` … `all`（16 个） | 参与输入的环境通道组合。`sequence` = 纯序列，`all` = 四者全开 |
| `random_seed` | `42` `43` `44` `45` | 划分与训练的随机种子。`all` / `single` 只用 42；`mixed` 用 42~45（四个独立划分，**不是重复测量**） |
| `n_train` | 1453 ~ 12313 | 训练集样本数 |
| `n_valid` | 311 ~ 2554 | 验证集样本数 |
| `n_test` | 312 ~ 8101 | 测试集样本数 |
| `R2` / `MAE` / `RMSE` / `MSE` / `Pearson` / `Spearman` | 实数 | **测试集**的 6 个指标 |
| `source_table` | `results/summary/DeepCRISPR/train_data/all_experiments.csv` | 批次路径常量（记录本表从哪里汇总而来） |
| `run_name` | `all_cnn_all_heldout_hct116_kernel_3` …（1344 个） | 连接 `results/train_results/<数据集>/<run_name>/` 目录的**唯一键** |

### attribution_summary.csv

+ 来源程序: `analysis.attribution.extractors.extract_attribution_table`

+ 回答什么问题: 每个模型对每个 (位置 × 通道) 的归因幅度与稳健性（长表）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `feature` | `<通道>_pos_<0-22>` 或 `pos<1-23>_<通道>` |  **各模型的原生标签，两套位置编号并存**：`cnn_ig`/`cnn_ism`/`transformer_attention` 来自 `<ch>_pos_<l>`（**0-based**，0~22）；`linear_coefficient`/`mlp_ig`/`xgboost_gain`/`xgboost_treeshap` 来自 `pos<i>_<ch>`（**1-based**，1~23）。同一核苷酸因此有两个不同字符串。**跨方法比较请用数值列 `position` 或 `(channel, position)`，不要 join `feature`** |
| `channel` | `A` `C` `G` `T` `CTCF` `Dnase` `H3K4me3` `RRBS` | 序列四通道 + 4 个表观通道。这里是**混合大小写**（`Dnase` 而非 `DNASE`），比较前需统一大小写 |
| `position` | `1` ~ `23` | **已规范化的 1-based 位置**，7 种方法完全一致。跨方法比较**用这一列** |
| `model` | `cnn` `linear` `mlp` `transformer` `xgboost` | 5 个模型族（三个 CNN 归在一起，用 `architecture` 区分） |
| `architecture` | `cnn33` `cnn53` `cnn73` | CNN 的三个核宽配置；非 CNN 行为空 |
| `split_type` / `cell_line` / `environment` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 16 个环境串 | 实验网格坐标 |
| `method` | `cnn_ig` `cnn_ism` `mlp_ig` `transformer_attention` `xgboost_gain` `xgboost_treeshap` `linear_coefficient` | 7 种归因方法 |
| `importance` | 实数（331200 行全部有值） | 归因幅度（**无符号** magnitude） |
| `snr` | 实数，−9.53 ~ 1.06e6 | 归因信噪比。 **只有 5 种方法有值**（`cnn_ism` `linear_coefficient` `mlp_ig` `transformer_attention` `xgboost_treeshap`）；`cnn_ig` 与 `xgboost_gain` **整列为空** |
| `effect` | 实数 | **有符号**效应。 **只有 `linear_coefficient` 有值**（OLS 系数），其余 6 种方法整列为空 |
| `attention_entropy` | 实数 | **只有 `transformer_attention` 有值**。注意力分布的熵 —— 熵高 = 注意力分散 = 该位置不可信。这就是 README 里 `attention_entropy` 要分两处讲的原因 |
| `source_file` | `cnn_feature_importance.csv` `linear_regression_weights.csv` `mlp_feature_importance.csv` `transformer_feature_importance.csv` `xgboost_feature_importance.csv` | 逐 run 的原生归因文件名。**它决定了 `feature` 用哪套命名** |

### metric_inconsistency.csv

+ 来源程序: `analysis.data.validation.validate_metric_consistency`

+ 回答什么问题: 哪些实验的 ΔR² 与 ΔRMSE 同号（理论矛盾）。本批 **0 行**

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `row` | 整数 | 该实验在 `experiment_table.csv` 里的行号（0-based） |
| `experiment` | 字符串 | 等价于 `run_name`，直接可读的实验标识 |
| `flags` † | `metric_inconsistency_same_increase` · `metric_inconsistency_same_decrease` | `;` 分隔的标记串。可能值只有这两个（外加"无标记"，但无标记的行不会被写进本表）。**只标记，不删除实验** |
| `delta_r2` | 实数 | 相对同 `(split_type, cell_line, model, seed)` 的 `sequence` 基线的 ΔR² |
| `delta_rmse` | 实数 | 同一基线的 ΔRMSE |

> 为什么"同号"就是矛盾：同一 cohort 下 R²=1−SSE/SST、RMSE=√(SSE/n)，两者**同源**。
> 加一个环境通道让预测变好时 R² 必升、RMSE 必降，所以 ΔR² 与 ΔRMSE **必须反号**。
> 配对严格限定在**同一 seed** 内（`mixed` 有 4 个 seed，不跨 seed 比较）。

### anomaly_report.csv

+ 来源程序: `analysis.pipeline`（在 `metric_inconsistency` 结果上插入前缀列）

+ 回答什么问题: 上表的机器可读版（带 `anomaly_type` 前缀）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `anomaly_type` | `metric_inconsistency` | 异常类别前缀。**本表 = `metric_inconsistency` 表 + 这一列**，其余列逐字相同 |
| `row` | 整数 | 同 `metric_inconsistency.row` |
| `experiment` | 字符串 | 同 `metric_inconsistency.experiment` |
| `flags` † | 同 `metric_inconsistency.flags` | 与 `metric_inconsistency.flags` **同义**（同一份数据），只是外面多套了一个类别前缀 |
| `delta_r2` / `delta_rmse` | 实数 | 同 `metric_inconsistency` |

### environment_conditional_delta_r2.csv

+ 来源程序: `analysis.environment.incremental_effect`（`summarize_conditional`）

+ 回答什么问题: 在某背景 S 下再加环境 e，预测能力增加多少

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 7 个模型配置 | 实验网格坐标 |
| `environment_added` | `ctcf` `dnase` `h3k4me3` `rrbs` | 被**新加进来**的那个因子 e |
| `background` | `sequence` … （15 个） | 背景组合串（含 e 之前的所有因子） |
| `background_set` | `ctcf` · `ctcf,dnase` · …（14 个） | 背景的**因子集合**（逗号分隔、无序）。与 `background` 的区别：这里不编码顺序 |
| `delta_r2_mean` / `delta_mae_mean` / `delta_rmse_mean` | 实数 | 条件增量在多个 seed / 重复上的均值 |
| `n_paired` † | `1` · `4` | 参与该均值的**配对观测数**。这里 = 背景 × seed 的可用组合数。 本表 **1 是合法值**（只有一个配对也能算均值），但无法给出不确定性 |

### environment_main_effects.csv

+ 来源程序: `analysis.environment.incremental_effect`（`compute_main_effects`）

+ 回答什么问题: 把所有背景平均后，e 平均有多大作用

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 7 个模型配置 | 实验网格坐标 |
| `environment` | `ctcf` `dnase` `h3k4me3` `rrbs` | 被平均的那个因子 e |
| `main_r2_delta` / `main_mae_delta` / `main_rmse_delta` | 实数 | 主效应（三个指标） |
| `n_seeds` | 整数 | 参与平均的 seed 数 |
| `n_backgrounds_avg` | 整数 | 被平均掉的背景组合数 |


### environment_edges.csv

+ 来源程序: `analysis.environment.factorial_dag.build_environment_edges` →
  `merge_edge_ci`（合入真实 bootstrap CI）

+ 回答什么问题: 加一个环境因子后各指标如何变化（DAG 的边）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 7 个模型配置 | 实验网格坐标 |
| `parent_combination` → `child_combination` | `sequence` → `sequence_ctcf` 等 | 一条边的父节点与子节点（子 = 父再开一个因子） |
| `added_environment` | `ctcf` `dnase` `h3k4me3` `rrbs` | 这条边新增的因子 |
| `parent_r2` / `child_r2` / … / `child_mae` | 实数 | 父/子节点在**配对 seed** 上的指标均值（先配对再平均，不是各算各的） |
| `delta_r2` / `delta_rmse` / `delta_mae` / `delta_pearson` / `delta_spearman` | 实数 | 子 − 父。5 个增量 |
| `n_paired` † | `0` `1` `3` `4` | **父与子都有合格实验的 seed 数**。 与 `environment_conditional_delta_r2.n_paired` 不是一回事：那边是背景×seed 的组合数，这边是 seed 数；且本表 **0 是合法值**（表示这条边算不出来，此时 `status=unavailable`） |
| `status` † | `ok` · `unavailable` | **这条边的 Δ 指标算不算得出来**。`unavailable` 表示父或子至少一侧没有合格实验 |
| `warning` † | `CI_AVAILABLE` · `INVALID_EXPERIMENTS_EXCLUDED;CI_AVAILABLE` · `INVALID_EXPERIMENTS_EXCLUDED;child_all_invalid;CI_UNAVAILABLE` · `INVALID_EXPERIMENTS_EXCLUDED;parent_all_invalid;CI_UNAVAILABLE` | **`;` 分隔的标记串**（本表专用格式）。可能出现：`INVALID_EXPERIMENTS_EXCLUDED`（有发散行被剔除）、`parent_all_invalid` / `child_all_invalid`（某一侧全是无效实验）、`no_paired_seed`（没有共同 seed）、`METRIC_INCONSISTENCY`（该边出现 ΔR²/ΔRMSE 同号）、`CI_AVAILABLE` / `CI_UNAVAILABLE`（CI 有没有算出来）。**与 `environment_nodes.warning` 格式和语义都不同**，见该表 |
| `ci_low` / `ci_high` | 实数或空 | 配对 per-seed bootstrap 的百分位区间 |
| `ci_excludes_zero` | `True` / `False` / 空 | 区间是否不跨 0 —— **判断"这个效应稳不稳"的主判据** |
| `n_bootstrap` | 整数（0 表示没跑） | 重采样次数 |
| `bootstrap_status` † | `ok` · `unavailable`（代码里还可能出 `not_run`） | **CI 那一步**的结果：`ok` = CI 算出来了；`unavailable` = 算不出来（此时 `ci_*` 为空）；`not_run` = 边本身算出来了但 CI 尚未接通（本批数据里没有 `not_run` 残留）。 与同表的 `status` **不是一回事**：`status` 说的是"Δ 指标算不算得出"，`bootstrap_status` 说的是"CI 算不算得出" |

### environment_nodes.csv

+ 来源程序: `analysis.environment.factorial_dag.build_environment_nodes`

+ 回答什么问题: 16 个环境组合**各自**的表现与数据完整性

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 7 个模型配置 | 实验网格坐标 |
| `combination` | `sequence` … `all`（16 个） | 该节点代表的环境组合（2⁴ 个） |
| `environment_count` | `0` ~ `4` | 该组合含几个环境因子 |
| `CTCF` / `DNase` / `H3K4me3` / `RRBS` | `True` / `False` | 该组合是否含此因子（与 `combination` 冗余，便于筛选） |
| `r2` / `rmse` / `mae` / `pearson` / `spearman` | 实数或空 | 该组合在合格实验上的指标均值 |
| `sample_count` | 整数 | 落在这个组合上的**全部**实验行数（未过滤） |
| `eligible_count` | 整数 | 其中**合格**（没有发散/缺失指标）的行数。指标均值只用这些行算 |
| `n_invalid` | 整数 | `sample_count − eligible_count`，被剔除的行数 |
| `status` † | `ok` · `unavailable` | **这个环境组合有没有合格实验**。`ok` = 至少 1 行合格；`unavailable` = 一行都没有或全不合格（此时 5 个指标列全空） |
| `warning` † | 空 · `invalid_experiments_excluded:R2_diverged(\|R2\|>10)` · `all experiments invalid:R2_diverged(\|R2\|>10)` · `no experiment for this combination` | **`前缀:明细` 格式**（本表专用）。`invalid_experiments_excluded:` = `status` 仍是 `ok`，但有行被剔除；`all experiments invalid:` = `status=unavailable`；`no experiment for this combination` = 压根没数据。本批 988/1008 行为空。**与 `environment_edges.warning` 的 `;` 分隔标记串格式不同，不可互相套用** |

### environment_interactions.csv

+ 来源程序: `analysis.environment.incremental_effect`（`compute_pair_interactions`）→
  `analysis.environment.factorial_dag` 按 (split, cell, model, A, B) 聚合

+ 回答什么问题: A、B 同时出现时是否有超出加和的额外作用

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`：`all`·`single`·`mixed` / 4 系+`none` / 7 个模型配置 | 实验网格坐标 |
| `factor_a` / `factor_b` | `ctcf` `dnase` `h3k4me3` / `dnase` `h3k4me3` `rrbs` | 参与交互的两个因子（无序对） |
| `interaction_r2` | 实数 | 交互项效应：I(A,B) = ΔR²(A,B) − ΔR²(A) − ΔR²(B)。> 0 = 协同，< 0 = 拮抗 |
| `n_seeds` | 整数 | 参与聚合的 seed 数 |
| `n_with_b` | 4（本批） | 计算交互时**含因子 b** 的配对观测数 |
| `n_without_b` | 4（本批） | **不含因子 b** 的配对观测数。两个数都记下来，是为了让"配对是否平衡"可被审计 —— 两者差太多时交互估计不可靠 |

### environment_dag_report.csv

+ 来源程序: `analysis.environment.factorial_dag.build_dag_report`

+ 回答什么问题: 每个上下文的因子组合完整性（理论/实测节点与边、缺失组合）

+ 列说明: 每个 `(split_type, cell_line, model)` 一行，**全部是计数/审计列**：

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `theoretical_nodes` | `16` | 理论上应有的节点数 = 2⁴（4 个环境因子） |
| `observed_nodes` | 0 ~ 16 | 其中 `status == "ok"` 的节点数 |
| `unavailable_nodes` | `theoretical_nodes − observed_nodes` | 缺数据的节点数（本批 0 ~ 8） |
| `theoretical_edges` | `32` | 理论上应有的边数 = 4 × 2³（每个节点各开一个因子） |
| `valid_edges` | 0 ~ 32 | `status == "ok"` 的边数。**已按 `(parent_combination, child_combination)` 去重** —— `mixed` 有 4 个 seed，边表是 4 行/边，不去重会让本列超过 32、`unavailable_edges` 变成负数（修前 `mixed` 是 −96） |
| `unavailable_edges` | `theoretical_edges − valid_edges` | 缺数据的边数。去重修复后**恒为非负**（本批 0 ~ 20） |
| `metric_inconsistency_edges` | 0 ~ 32 | `warning` 里含 `METRIC_INCONSISTENCY` 的边数（同样按 (parent, child) 去重） |
| `invalid_experiments` | 整数 | 该上下文 16 个节点的 `n_invalid` 之和 = 被剔除的发散/无效实验行总数 |
| `unavailable_combinations` | `sequence_dnase, sequence_ctcf_dnase, …` | 该上下文下 `status != "ok"` 的节点名，**逗号分隔**；全齐时为空串 |

### bootstrap_results.csv

+ 来源程序: `analysis.stats.tasks.bootstrap_environment_edges`

+ 回答什么问题: 该效应估计值稳不稳（**配对 per-sample** bootstrap）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` / `random_seed` | 同 `experiment_table`（`random_seed` 同样只作划分/训练种子用） | 实验网格坐标 |
| `parent_combination` / `child_combination` / `added_environment` | 同 `environment_edges`：父→子组合串 + 新增因子名 | 被检验的那条 DAG 边 |
| `metric` † | `R2` `RMSE` `MAE` | 被 bootstrap 的**指标名**。**本表专有** —— 一张长表里装了 3 个指标，读的时候必须按 `metric` 过滤；同族的 `bootstrap_main_effects` / `bootstrap_cellline_effects` 只做 R²，没有这一列 |
| `estimate` † | 实数或空 | 配对 per-sample bootstrap 的**点估计 = 子 − 父** 的指标差（如 `metric=R2` 时即 ΔR²）。重采样单元是**测试样本（sgRNA）**，与同族另外两张表的 `estimate` 口径不同 |
| `ci_low` / `ci_high` | 实数或空 | 百分位区间 |
| `excludes_zero` | `True` / `False` / 空 | 区间是否不跨 0 |
| `n_bootstrap` | 整数 | 重采样次数 |
| `seed` | 整数 | bootstrap 自身的随机种子（与 `random_seed` 是两回事：后者是数据划分种子） |
| `alpha` | `0.05` | 区间显著性水平 |
| `n_samples` | 整数 | 参与配对的**测试样本数** |
| `status` † | `ok` · `unavailable` | **这次 bootstrap 有没有产出 CI**。`unavailable` 的三类原因写在 `reason` 里 |
| `reason` † | 空 · `invalid_experiment_excluded (diverged/missing metric)` · `prediction_artifact_missing` · `cohort_mismatch (n_parent=…, n_child=…)` · `insufficient paired samples (<3)` | 只在 `status=unavailable` 时有值。本批 7899/7954 行为空 —— 空 = 成功，不是缺数据 |

### bootstrap_main_effects.csv

+ 来源程序: `analysis.stats.tasks.bootstrap_main_effects`

+ 回答什么问题: 某环境因子的**总体**效应不确定性（model-level bootstrap）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `feature` | `ctcf` `dnase` `h3k4me3` `rrbs` | 被检验的环境因子。本表用 `feature` 这个列名装**环境因子**，与 `attribution_summary.feature`（位置×通道特征名）**完全无关** |
| `estimate` † | 实数或空 | 该因子的**跨模型平均主效应**（先在每个模型内把 `main_r2_delta` 跨 `split × cell_line` 平均，再对模型取均值）。重采样单元是**模型配置**（n = 6~7），**不是** sgRNA 样本 |
| `ci_low` / `ci_high` / `excludes_zero` | 实数 / 布尔 | 模型级 bootstrap 区间 |
| `n_bootstrap` | 整数 | 重采样次数 |
| `seed` | 整数 | bootstrap 种子 |
| `n_models` | 6 ~ 7 | 参与 bootstrap 的**模型个数**，也就是重采样单元数。读区间宽度时必须带上这个 n |
| `status` † | `ok` · `unavailable` | **这个因子的总体 CI 有没有算出来**。本批 4 行全 `ok` |
| `reason` † | 空 · `需要 >=3 个模型的 main effect 才能 bootstrap` | 只在 `unavailable` 时有值。本批全空 |

### bootstrap_cellline_effects.csv

+ 来源程序: `analysis.stats.tasks.bootstrap_cellline_effects`

+ 回答什么问题: 某环境因子的效应在**不同细胞系**内是否稳定

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` | 同 `experiment_table`，但本表**没有 `model` 列** | 细胞系维度（本表**没有** `model` 列 —— 模型被平均掉了，见 `n_models`） |
| `factor` | `ctcf` `dnase` `h3k4me3` `rrbs` | 被检验的环境因子。本表用 `factor`，而 `bootstrap_main_effects` 用 `feature` 指同一件事 —— **两张表的列名不统一** |
| `estimate` † | 实数或空 | 该 `(split_type, cell_line, factor)` 内 `main_r2_delta` 的跨模型均值。重采样单元同样是**模型**（`n_models`），**不是** sgRNA / 样本；**不能**读作总体抽样不确定性 |
| `ci_low` / `ci_high` / `excludes_zero` | 实数 / 布尔 | 跨模型稳定性区间 |
| `n_bootstrap` | 整数 | 重采样次数 |
| `n_models` | 6 ~ 7 | 重采样单元数 |
| `status` † | `ok`（本批全部） | **这个 (split, cell_line, factor) 的 CI 有没有算出来**。可能值还有 `unavailable`（模型数 < 3） |

> **本表没有 `n_obs` / `family_key` / `fdr_family` / `fdr_status` 列**。
> 这四个列名实际出现在 `anova_results.csv`（`n_obs` / `family_key` / `fdr_family` /
> `fdr_status` 全有）与 `permutation_results.csv`（后三个），见各自条目。

### permutation_results.csv

+ 来源程序: `analysis.stats.tasks.permutation_environment_edges` /
  `permutation_interactions` / `permutation_main_effects`

+ 回答什么问题: "若根本没有真实关系，这么大的效应容易出现吗"（符号翻转置换 + 族内 BH-FDR）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `split_type` / `cell_line` / `model` / `random_seed` | 同 `experiment_table`（`random_seed` 同样只作划分/训练种子用） | 实验网格坐标 |
| `test_type` | `environment_main_effect` · `environment_edge` · `environment_interaction` | 三类置换检验，对应三类效应。**同一张表混装三类检验，读之前必须先按本列过滤** |
| `parent_combination` / `child_combination` / `added_environment` / `factor` | 同 `environment_edges`，另有 `factor`（**含交互项**，如 `ctcf*dnase`） | 被检验的效应标识（不同 `test_type` 填的列不同） |
| `null_hypothesis` † | `mean conditional ΔR² over backgrounds = 0 (sign-flip randomization over background values)` · `mean per-sample interaction I(ctcf,dnase) = 0 (sign-flip randomization over pooled background contrasts)` …（8 个取值） | **零假设的完整文字**，随 `test_type` 与因子变化。把 H₀ 写进数据里，是为了避免"读了 p 值却不知道检验的是什么" |
| `alternative` † | `greater (adding the environment reduces squared error)` · `two-sided` | **备择方向的完整文字**。边检验是单侧 `greater`，交互检验是双侧 —— 单双侧不同，p 值不可直接互比 |
| `observed_effect` | 实数 | 观测到的效应量（置换分布要与之比较的那个值） |
| `p_value` | 实数，最小 `0.000999…` | 置换 p 值。分辨率受置换次数限制（`p` 最小约为 `1/(n_permutations+1)`） |
| `n_permutations` | 整数 | 置换次数 |
| `seed` | 整数 | 置换自身的随机种子 |
| `n_values` | 整数 | 参与置换的观测值个数 |
| `n_values_excluded` | 整数 | 其中被剔除的（发散/无效）个数 |
| `family_key` † | `environment_edge\|all\|hct116\|cnn(3\|3)` …（189 个） | 多重比较的"族"标识 = `<检验类型>\|<split>\|<cell_line>\|<model>`。**前缀与 `test_type` 对应但不同名**：`environment_edge` / `environment_interaction` 同名，主效应为 `environment_main`（`test_type` 记 `environment_main_effect`） |
| `status` † | `ok` · `unavailable` | **这次置换检验跑没跑成**。`unavailable` 的原因写在 `reason` |
| `reason` † | 空 · `invalid_experiment_excluded` · `缺少可配对的 4 个组合预测 (S0, S0+a, S0+b, S0+a+b)` · `需要 >=3 个背景/seed 条件增量 (已剔除 6 个发散/无效值)` | 只在 `unavailable` 时有值。本批 3378/3444 行为空 |
| `fdr_family` † | 与 `family_key` **取值完全相同** | `family_key` 的**改名副本**：物理上再加一列，只为让下游按统一列名做 BH 校正。**值一字不差** |
| `FDR` | 实数或空 | 族内 BH 校正后的 q 值。族由 `fdr_family` 定义，族大小 < 2 时不校正（留空） |
| `fdr_status` † | `ok`（本批全部） | **这一族的 FDR 有没有校正成功**。另一可能值是 `not_applicable (family size N < 2)`（族太小）。本表本批全是 `ok`，而 `anova_results` 里有一大半是 `not_applicable` —— 两张表同名不同分布 |

### anova_results.csv

+ 来源程序: `analysis.stats.hypothesis_tests`（Type-II 边际 F）→
  `analysis.stats.tasks.run_anova_tasks`（接 BH-FDR）

+ 回答什么问题: 各因子解释了多少变异

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `model_scope` | `blocked_factorial` · `per_group_additive_main` | **两类完全不同的 ANOVA 混在一张表里**：`blocked_factorial` = 16 组合的阻塞析因（检验因子是否影响效率，10 个项）；`per_group_additive_main` = 逐个 (split×cell_line×model) 族内的加性主效应（84 行）。**两类的 F 统计量与自由度不可直接比较**，读表必须先按本列过滤 |
| `split_type` / `cell_line` / `model` | 同 `experiment_table`，**另有 `ALL`** | `ALL` = 该行是跨细胞系 / 跨模型池化后的汇总 |
| `factor` | `(none)` · `ctcf` · `ctcf*dnase` · …（11 个） | 被检验的项。`*` = 交互项；`(none)` = 零模型/截距行（设计饱和导致无可估项时的兜底行） |
| `effect` | 实数 | 该项的效应估计（系数尺度）。 与 `attribution_summary.effect`（归因有符号效应）和 `motif_enrichment.effect`（频率差）**是同名的三种不同东西** |
| `F_statistic` | 实数或空 | Type-II 边际 F |
| `p_value` | 实数或空 | 该 F 的 p 值（**尚未校正**；校正结果在 `FDR`） |
| `df_num` / `df_den` | 实数 | 分子 / 分母自由度 |
| `effect_size` | 实数，最大 0.0536 | **偏 eta 方**：`SS_term / (SS_term + RSS_full)`。不是 `SS_term / SS_total` —— 后者会把全部平方和都算进分母，把阻塞因子的占比压成 25%，见 §19.4 |
| `ci_low` / `ci_high` | 实数或空 | 效应的 bootstrap 区间 |
| `n_obs` † | `8` `10` `16` `58` `64` `1324` | **进入该方差分析模型矩阵的观测数**（行数）。只在 `anova_results` 里出现；它和 `environment_nodes.sample_count`（落在这个组合上的实验行数）以及 `bootstrap_*` 的各种 n **都不是一回事** |
| `family_key` † | `environment_anova_group\|all\|hct116\|cnn(3\|3)` …（64 个） | 族标识 = `<问题类型>\|<split>\|<cell_line>\|<model>`。**前缀是 `environment_anova_group`**，而 `permutation_results` 是 `environment_edge` / `environment_interaction` / `environment_main_effect` —— 两张表的族**不通用** |
| `status` † | `ok` · `unavailable` | **这个 ANOVA 项的 F/p 算不算得出来**。`unavailable` 两类：设计不足（`F statistic not estimable`）与样本不足（`insufficient_data: n=X < 32`）。本批 38 `ok` / 56 `unavailable` |
| `reason` † | 空 · `insufficient_data: n=8 < 32` · `insufficient_data: n=10 < 32` · `insufficient_data: n=16 < 32` | 只在 `unavailable` 时有值。门限是 `AnalysisConfig` 里"每组至少 32 行" |
| `FDR` | 实数或空 | 族内 BH 校正后的 q 值（本批 `blocked_factorial` 的 10 个项全部 0.999602） |
| `fdr_family` † | 与 `family_key` 取值完全相同 | `family_key` 的**改名副本**，值一字不差 |
| `fdr_status` † | `ok` · `not_applicable (family size 0 < 2)` | **这一族的 FDR 有没有校正成功**。族大小 < 2 时标 `not_applicable`（本批 56 行）—— 因为 `unavailable` 的项不参与校正，其所在族只剩 0~1 个可校正项。与 `permutation_results.fdr_status` 同名，但那边本批全是 `ok` |

### prediction_summary.csv

+ 来源程序: `analysis.prediction.performance_by_model_split`

+ 回答什么问题: 不同模型在不同 split 下是否表现稳定

+ 列说明: 每个 `(model, split_type)` 一行，列是 6 个指标的 **mean / std 成对**写法：
  `R2_mean` / `R2_std`、`MAE_mean` / `MAE_std`、`RMSE_mean` / `RMSE_std`、
  `Pearson_mean` / `Pearson_std`、`Spearman_mean` / `Spearman_std`，外加
  `n_experiments`（参与聚合的实验数）。`std` 跨的是不同细胞系/环境，**不是**重复测量误差。
  `loco_performance.csv` 里的 `*_mean` 列**名字相同但聚合范围不同**，见下。

### loco_performance.csv

+ 来源程序: `analysis.prediction.loco_performance`

+ 回答什么问题: 模型能否泛化到没见过的细胞系（LOCO）

+ 列说明: 每个 `(model, cell_line)` 一行 —— **没有 `split_type` 列**（本表天然只对应
  `all` 留一划分）。列是 `R2_mean` / `R2_median`、`MAE_mean` / `MAE_median`、
  `RMSE_mean` / `RMSE_median`、`Pearson_mean` / `Pearson_median`、
  `Spearman_mean` / `Spearman_median`，外加 `n_experiments` 与 `n_valid`。
  **`*_mean` 与 `prediction_summary.csv` 的同名列同名不同义**：这里是同一细胞系内
  跨环境的均值，那里是跨细胞系/环境的均值。

### motif_candidates.csv

+ 来源程序: `analysis.sequence.motif.pipeline.run_and_write`（`run_motif_discovery`）

+ 回答什么问题: 提取出的 motif 候选：共识序列、出现位点、支撑实例数、携带者 vs 背景效率差

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `motif_id` † | `motif_cnn33_hct116_all_cnn_ig_073` …（656 个） | 主键，编码 `<架构>_<细胞系>_<环境>_<方法>_<序号>`。 另外两张 motif 表用它做外键，含义相同 |
| `model_source` | `cnn` | 恒定：motif 只从 CNN 归因里挖 |
| `model_variant` | `cnn33` `cnn53` `cnn73` | CNN 核宽配置 |
| `attribution_method` | `cnn_ig` `cnn_ism` | 只用 CNN 的两种归因 |
| `split_type` / `environment` | `all` `single` / `all` `sequence` | 实验网格坐标（motif 只在纯序列实验中挖） |
| `sequence` | `AAAG` …（133 个） | 实测到的具体序列实例（共识的一个代表） |
| `consensus` | `AAAG` …（133 个） | 由实例位置频率矩阵取 argmax 得到的**共识序列**（已是确定碱基） |
| `iupac` | `AAAG` …（133 个） | 共识的 **IUPAC 表示**（允许简并码），是真正用于匹配的串 |
| `human_pattern` | `(A/G)G(A/C/G/T)(A/T)(C/G)CCATGG` … | 人类可读的 IUPAC 描述（把简并码展开成 `(A/G)` 形式） |
| `regex` | `AAAG` …（133 个） | `iupac` 的**正则等价形式**，便于外部工具直接匹配 |
| `length` | 整数 | motif 长度（nt） |
| `position_start` / `position_end` | 1 ~ 23 | 实例在 23nt 序列上的起止位置（1-based 闭区间） |
| `position_mean` / `position_std` | 实数 | 实例起始位置的均值 / 标准差 |
| `position_distribution` | `14:106` · `14:4;20:288` | **`位置:实例数`**，多峰用 `;` 分隔。比单个 `position_mean` 更能揭示双峰 |
| `preferred_position` | `20` · `14;20` · …（26 个） | 主峰位置（实例数最多的那个位置）；多峰时用 `;` 连接。**20 = protospacer 最后一位 = PAM 邻近区** |
| `support_count` | 整数 | **实例总数**（seqlet 条数，含同一序列的多个归因实例） |
| `sample_support` | 整数 | **不同 `(cell_line, sample_id)` 的去重个数**，即"有多少条不同的 sgRNA 支持它"。 与 `support_count` 不同：前者会被同一条序列的多个实例重复计数，后者不会。门槛用 `MotifDiscoveryConfig.min_sample_support` |
| `mean_effect` | 实数 | **携带者 vs 背景的实测效率差**：`mean(efficacy of 携带该 motif 的序列) − mean(efficacy of 同细胞系内不携带的序列)`。与 `motif_enrichment.effect`（**频率差**）同名不同义，两者都在各自表里 |
| `effect_direction` | `+` / `-` | `mean_effect` 的符号。**来源是实测效率对比，不是 attribution 符号** —— CNN 的 attribution 是无符号幅度，无法提供方向 |
| `enrichment` † | 实数 | 富集倍数 = 前景命中率 / 背景命中率。**本列是从 `motif_enrichment.csv` 连接过来的副本**，与那边逐字相同 |
| `odds_ratio` † | 实数 | Fisher 2×2 表的优势比 `(a·d)/(b·c)`。同样是**从 `motif_enrichment.csv` 连接来的副本** |
| `p_value` † | 实数 | Fisher 精确检验 p 值（未校正）。同样是连接来的副本 |
| `FDR` † | 实数 | `motif_enrichment` 族内的 BH q 值。同样是连接来的副本 |
| `status` † | `ok` · `exploratory` |  **本表的取值集与其它表不同，且语义完全不同**：这里说的是**共识序列的质量** —— `ok` = 简并位点占比 ≤ `MotifDiscoveryConfig.max_ambiguous_fraction`（0.34）；`exploratory` = 超过该上限，共识太模糊、只能当探索性结果（每上下文最多保留 `max_exploratory_per_context = 3` 条）。本批 652 `ok` / 4 `exploratory`。**与"算不算得出来"无关** |

### motif_instances.csv

+ 来源程序: `analysis.sequence.motif.pipeline.run_and_write`

+ 回答什么问题: 每个 motif 的**逐实例**明细（一条 seqlet 一行，186737 行）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `motif_id` † | 同 `motif_candidates.motif_id` | 外键 |
| `sample_id` | 整数 | 该实例来自**数据集 metadata 的行号**。配合 `cell_line` 可反查是哪条 sgRNA |
| `cell_line` / `split_type` / `environment` | 同 `experiment_table` / `motif_candidates` | 实例来源的实验坐标 |
| `model` / `model_variant` / `attribution_method` | `cnn` / `cnn33`… / `cnn_ig` `cnn_ism` | 归因来源 |
| `position_start` / `position_end` | 1 ~ 23 | 该实例在序列上的位置 |
| `sequence` | `AAAG` …（302 个） | 该实例的实测序列 |
| `attribution_score` † | 实数 | **该实例的归因幅度**（取自 `attribution_summary.importance`，无符号 magnitude）。聚类时按它降序处理，让高分 seqlet 先占位。⚠️ 与 `motif_candidates` 里的汇总量不同：这里是一条实例一个值，没有跨实例平均 |
| `ism_effect` | 实数 | 若该实例来自 `cnn_ism`，这里放 ISM 归因值；否则为空 |
| `ig_effect` | 实数 | 若该实例来自 `cnn_ig`，这里放 IG 归因值；否则为空。**两列互斥**，保证换方法时数值不被混用 |
| `direction` | `unsigned` | 恒定：当前批次的 attribution 是无符号幅度。与 `motif_candidates.effect_direction`（来自实测效率）**不是一回事** |
| `efficacy` | 实数 | 该序列在数据集里的实测标签（`Normalized efficacy`），直接连回 metadata |

### motif_enrichment.csv

+ 来源程序: `analysis.sequence.motif.core.enrichment_for_motif`

+ 回答什么问题: 某 motif 是否在"高效序列"里过度出现（Fisher 精确检验）

+ 列说明:

| 列 | 取值 | 含义 |
| :--- | :--- | :--- |
| `motif_id` † | 同 `motif_candidates.motif_id` | 外键 |
| `foreground_count` | 整数 | **前景**（高效）序列中携带该 motif 的条数（4 个细胞系求和） |
| `foreground_total` | `5528` | 前景序列总数。阈值是**逐细胞系**的效率分位数 `MotifDiscoveryConfig.enrichment_foreground_quantile`（0.67），因为各系效率尺度不同；四个系的上分位序列数相加 = 5528 |
| `background_count` | 整数 | **背景**序列中携带该 motif 的条数 |
| `background_total` | `11221` | 背景序列总数。背景 = **前景的补集**（`efficacy < 阈值`），两者互斥且穷尽 → `5528 + 11221 = 16749` = 全部非空样本。 这是 Fisher 精确检验的前提：若背景用"全部合格序列"（含前景），2×2 表两行重叠，p 值会系统性偏向 1。`MotifDiscoveryConfig.enrichment_background` 现在**只接受** `"complement_of_foreground"`，取别的值直接 `raise ValueError` |
| `effect` | 实数 | **频率差** = `foreground_count/foreground_total − background_count/background_total`。 与 `motif_candidates.mean_effect`（效率差）同名不同义 |
| `enrichment` † | 实数 | **倍数** = 前景命中率 / 背景命中率。背景命中率为 0 时留空（不伪造无穷大） |
| `odds_ratio` † | 实数 | Fisher 2×2 表的优势比 `(a·d)/(b·c)`，其中 a=`foreground_count`、b=`foreground_total−foreground_count`、c=`background_count`、d=`background_total−background_count`。任一分母为 0 时留空 |
| `p_value` † | 实数 | Fisher 精确检验 p（单侧富集）。**本表是全流程唯一用 Fisher 产生 p 值的地方** |
| `FDR` † | 实数 | `motif_enrichment` 族内 BH 校正后的 q 值（656 个 motif 同属一族） |
| `status` † | `ok`（本批全部） | **这次富集检验跑没跑成**。可能值还有 `unavailable`（前景/背景序列不足，或携带者少于 `enrichment_min_carriers`） |
| `reason` † | 空（本批全部为空） | 只在 `status=unavailable` 时有值：`insufficient carriers/eligible sequences` |


---

## 术语表

正文里出现的缩写：

* `ΔR²` / `ΔRMSE` / `ΔMAE` = 加入环境因子前后的指标差（子 − 父）
* `CI` = 百分位 bootstrap 置信区间；`CI` 不跨 0 是"效应稳定"的主判据
* `BH-FDR` = Benjamini-Hochberg 假发现率；只在**同一族内**校正
* `LOCO` = Leave-One-Cell-line-Out（留一细胞系）
* `partial η²` = 偏 eta 方，`SS_term / (SS_term + RSS_full)`
* `SS_total` = 总平方和；`RSS_full` = 全模型残差平方和
* `PAM` = 原间隔区邻近基序（本仓库为 `NGG`，占 sgRNA 的第 21–23 位）
* `protospacer` / `spacer` = PAM 之前的 20 nt 靶向序列
* `ISM` = in-silico mutagenesis（逐位点虚拟突变）
* `IG` = integrated gradients（积分梯度）
* `DAG` = 有向无环图（此处指 2⁴ 环境因子组合格）
* `SNR` = 归因信噪比
* `seqlet` = 归因热图上切出的一个局部连续高归因窗口（motif 的实例前身）
* `consensus` = 由实例位置频率矩阵逐位取 argmax 得到的共识序列
* `run` = 一次完整的"模型 × 划分 × 细胞系 × 环境 × seed"训练
* `cohort` = 参与配对的一组实验（同 `split_type` / `cell_line` / `model` / `random_seed`）

---

## 科学声明

> 本结果文件用于赛事展示和候选规律整理，其筛选标准用于提高候选发现效率，并非严格意义上的生物学因果证明。所有候选规律仍需结合独立实验验证或进一步机制研究确认。

# R3. 平台外部验证说明

> 由于完成平台建设所用时间消耗过长，无法对挖掘结果进行湿实验验证，故引入 2 个外部数据集和 1 个第三方预测平台做**干实验闭环**。

## R3.0 为什么要做外部验证

本平台的全部结论都来自 DeepCRISPR 单一批次的计算证据链（Effect / Importance / Statistical
Evidence / Robustness）。计算证据能回答的是"**在同一批数据里这个结论稳不稳**"，回答不了
"**换一批数据、换一个模型，这个结论还在不在**"。湿实验不可得，因此用两条**独立于 DeepCRISPR
建模过程**的路径做交叉检验：

| 路径 | 外部对象 | 检验什么 |
| :--- | :--- | :--- |
| **数据集层面** | Hiranniramol（1,309 条）、Labuhn（417 条） | 同一套 `core/` + `analysis/` 代码换数据后，**序列层面的位置/化学结论**是否重现 |
| **模型层面** | 第三方预测平台 CRISPRon | 在**同一个位点**上，独立模型的**方向判断**是否与本项目 CNN7 一致 |

两条路径的性质不同：数据集层面检验"结论是不是数据特异的"，模型层面检验"结论是不是模型特异的"。
两者都通过，才说明该结论至少**不是单一数据 + 单一模型的产物**。

## R3.1 数据集层面的外部复现（Hiranniramol / Labuhn）

**复现口径**：不改动 `core/` 与 `analysis/` 任何代码，只新增两份 `data/config/*.json`
（§5 的承重键审计全通过），然后走完整流程：
`feature_engineering` → `collect_results` → `importance_extraction` → `analysis.pipeline`
→ `rule_discovery`。两个数据集都是**纯序列**（4 通道 / 92 维），没有表观遗传通道。

**结果如实**：11/13 个分析任务 completed；`fdr_correction` 与 `environment_anova` 因**没有环境因子**
而如实标 `unavailable`（未伪造显著性）。这本身就是"结论不可用时如实降级"的验证。

### 被照应上的 DeepCRISPR 挖掘结果

1. **PAM 邻近区（18–21 位）是效率的主要决定区**
   - DeepCRISPR 的序列通道归因（`key_regulatory_biomarkers`）按位置统计，行数前三名正是
     **20 位（2,709 行）、18 位（2,677 行）、19 位（2,670 行）**；
   - Hiranniramol 的 top-6 特征全部落在 **19–21 位**（`pos20_G` / `pos19_T` / `pos20_T` /
     `pos19_G` / `pos21_G` / `pos18_T`）；
   - Labuhn 的 top-2 是 `pos20_G` / `pos20_C`，top-6 里还有 `pos18_C`。
   - 三个数据集在**样本来源、标签量纲、物种/实验体系都不同**的前提下，把重要性集中到了同一段区域。

2. **Pos18 是一个正向关键位点**
   - `pos18_C` 在三个数据集中**都**出现在关键调控特征库里：DeepCRISPR 91,747 行中 796 行、
     Hiranniramol 421 行中 5 行、Labuhn 390 行中 5 行；
   - 其中 Hiranniramol 的 `pos18_C` 拿到了**线性模型 BH-FDR 三星**（`***`），Labuhn 的
     `pos18_C` 也进入其 top-6（XGBoost 归因最强）；
   - 与 §R3.2 的 CRISPRon 方向性对照共同构成"该位点正向"的跨模型、跨数据证据。

3. **motif 主峰位置一致地落在 PAM 邻位**
   - `preferred_position` 含 **20 位**的候选占比：DeepCRISPR **424/656 = 64.6%**、
     Hiranniramol **29/32 = 90.6%**、Labuhn **14/14 = 100%**；
   - 这与 README §18.3 里"motif 候选的 `preferred_position` 绝大多数集中在 20 附近
     （PAM 邻近区）"的说明同向，且在两个独立数据集上更极端。

4. **PAM 邻位的碱基化学同向（G/C 富集有利）**
   - Hiranniramol 的高效侧 motif 共识为 `GGGG` / `GAGG` / `AGGG` / `GCGG`（G-rich，
     `mean_effect` 为正、Fisher FDR 极小），低效侧为 `TTGG` / `CTGG` / `TGGG` / `TCGG`
     （`effect_direction = -`）；
   - 方向与 DeepCRISPR 侧"PAM 邻近区 G/C 相关特征被选中"的结论一致。

5. **统计骨架可迁移且不会伪造显著性**
   - Bootstrap 配对 CI、符号翻转置换检验、BH-FDR 族隔离在纯序列数据集上**照常运行**，
     只是因没有可校正的族而如实产出 `not_applicable`/`unavailable`；
   - 说明这套统计机制不依赖 DeepCRISPR 的表观通道结构。

### **没有**被外部照应的部分（同样重要）

| 维度 | 为什么无法外部验证 |
| :--- | :--- |
| **表观微环境结论**（CTCF / Dnase / H3K4me3 / RRBS 的贡献与交互） | 两个外部数据集**没有表观遗传通道**，这一维度仍然只有 DeepCRISPR 内部证据 |
| **跨细胞系泛化比例**（`cellline_ratio > 0.75`） | Hiranniramol / Labuhn 各只有一个"细胞系"，4/4 判据在该数据上退化；其 `microenv_and_motif` 行只验证流程可用性，不构成泛化证据 |
| **motif 富集的统计显著性** | Labuhn 只有 417 条，motif 富集未过 FDR 门槛，只能读作"**未出现反例**"；Hiranniramol 样本量也远小于 DeepCRISPR |
| **标签量纲可比性** | Hiranniramol 原始为 0–100 百分制、Labuhn 为 KO reporter assay，只能读作"同一流程在不同数据集上的可预测性"，不能横向比较数值 |

## R3.2 模型层面的外部对照（第三方预测平台 CRISPRon）

**对象**：`analysis.crispron_validation`（CNN7 × CRISPRon 的 Pos18 C→A 一致性验证），
交付物为 `results/summary/DeepCRISPR/cnn7_validation.md` 与 `.csv`。

**设计**（先把"比什么"钉死，再比）：

- 取 DeepCRISPR **四个细胞系全部**第 18 位原始碱基为 `C` 的序列，共 **5,080 条**；
- 对每条序列做**同一处碱基替换** `C → A`：这是**合法的 one-hot 碱基替换**
  （把 C 通道置 0、A 通道置 1），落在模型的输入分布内；
- 对每条序列计算两个增量并比较**方向**：
  - `cnn_pred-ism` = 本项目 CNN7（`sequence_kernel=7`，纯序列，四系全部数据既训练又评估）对
    WT 的预测 − 对 C18A 突变序列的预测；
  - `crispron_wt-crispron_c18a` = CRISPRon 对 WT 的预测 − 对 C18A 突变体的预测；
  - `consistency` = 两者**同号为 `+`、异号为 `-`**。

**三个主指标（实测）**：

| 指标 | 数值 | 含义 |
| :--- | ---: | :--- |
| `cnn_pred-ism` 正号占比 | **0.7911**（4019/5080） | 本项目 CNN7 判"该突变有害"的比例 |
| `crispron_wt-crispron_c18a` 正号占比 | **0.9039**（4592/5080） | CRISPRon 判"该突变有害"的比例 |
| `consistency` 正号率 | **0.7624**（3873/5080） | 两者方向一致的序列比例 |

两个增量的均值分别为 **0.0913** 与 **0.1007**，**尺度接近**，因此可以直接比较方向而不必先做归一化。

**逐细胞系一致性**：

| cell_line | 序列数 | `cnn_pred-ism` 正号占比 | `crispron_wt-crispron_c18a` 正号占比 | `consistency` |
| :--- | ---: | ---: | ---: | ---: |
| hct116 | 1377 | 0.8235 | 0.9092 | 0.7938 |
| hek293t | 618 | 0.6489 | 0.9078 | **0.6149** |
| hela | 2483 | 0.8284 | 0.9041 | 0.7986 |
| hl60 | 602 | 0.7093 | 0.8870 | 0.6927 |

**解读（三条，按确定性排序）**：

1. **方向性显著高于随机**：两次独立建模（本项目 CNN7 与第三方 CRISPRon）都给出
   "把 Pos18 由 C 换成 A 会降低预测效率"的多数判断（0.79 / 0.90），一致率 0.76 —— 说明这个
   结论**不是本项目模型的特异假象**，这是本次外部验证最硬的一条。
2. **一致性并非满分，且 CRISPRon 方向性更强**（0.90 vs 0.79）：两个模型对"**哪些**序列更敏感"
   的排序并不完全重合，提示该位点的效应存在**序列上下文依赖**，单一模型的位点级结论不宜过度外推。
3. **低样本细胞系一致性最低**：hek293t 只有 2,333 条训练数据、一致性 0.6149，明显低于 hela
   （0.7986）/ hct116（0.7938）。这与"样本量不足时方向判断更不可靠"的预期一致，也解释了为什么
   平台坚持在低样本上下文里如实标注不确定性。

**口径纪律（不可混用的两条算子）**：

- 本验证的 `cnn_pred-ism` 用的是**合法 one-hot 碱基替换**，因此度量的是**碱基替换效应**；
- 平台内的 `CNN_ISM` 归因（§17.4）是**单通道翻转**（`>0 → 0`，`=0 → 1`），对 one-hot 序列通道会产生
  分布外输入（`[0,0,0,0]`），度量的是**敏感性/重要性**，**不是**碱基替换效应；
- 两者名字相近但语义不同，**不可互相替代**；本验证全程不读取 `CNN_ISM` 归因产物。

**两个 `y_true-*` 列只作噪声参照**：`y_true-ism` 正号占比 0.5850、`y_true-crispron_c18a` 0.2319。
它们把真实标签卷了进来，符号会被标签的序列间波动翻转（`y_true-ism` 的标准差 0.1342 大于
`cnn_pred-ism` 的均值 0.0913），因此**不能**当作模型或 CRISPRon 的性能指标，详见 `cnn7_validation.md`。

## R3.3 结论汇总：哪些 DeepCRISPR 挖掘结果被外部照应

| DeepCRISPR 的挖掘结论 | 外部照应来源 | 强度 | 依据 |
| :--- | :--- | :--- | :--- |
| PAM 邻近区（18–21 位）是效率的主要决定区 | Hiranniramol + Labuhn | **强** | 三数据集的重要性都集中在同一段位置；motif 主峰含 20 位占比 64.6% / 90.6% / 100% |
| Pos18 是正向关键位点 | CRISPRon + Hiranniramol + Labuhn | **中–强** | 方向一致率 0.7624；`pos18_C` 在三数据集均入关键特征库，Hiranniramol 达 BH-FDR `***` |
| PAM 邻位 G/C 富集有利 | Hiranniramol | **中** | G-rich motif 在高效侧、T/C-rich 在低效侧，FDR 极小 |
| 统计骨架（配对 Bootstrap CI / 置换检验 / 族隔离 BH-FDR）可迁移 | 两数据集流程级 | **中** | 机制照常运行，无环境因子时如实降级而不伪造显著性 |
| 表观微环境（CTCF / Dnase / H3K4me3 / RRBS）的贡献与交互 | — | **未验证** | 外部数据集无表观通道 |
| 跨细胞系泛化比例 | — | **未验证** | 外部数据集单细胞系，判据退化 |
| 具体 motif 的富集显著性 | — | **未验证** | 外部数据集样本量不足，未能过 FDR |

## R3.4 局限与边界

- 这是**干实验闭环**，不是湿实验证据；"一致"只说明两个独立计算路径同向，**不构成生物学因果**。
- CRISPRon 是第三方模型，其训练语料与归纳偏置对我们是黑箱；方向一致**不能完全排除两者共享**
  某些训练数据或先验偏置，因此一致率应读作"**同向性强**"而非"独立确证"。
- 平台内部对 Pos18 的结论来自"CNN7 + 全部数据训练"这一配置；换成其它配置或加入表观通道后
  是否仍成立，本验证未覆盖。
- 三个数据集的标签语义不同（DeepCRISPR 为 `[0,1]` 归一化效率、Hiranniramol 原为 0–100 百分制、
  Labuhn 为 `[0,1]` KO reporter assay），只能读作"同一流程在不同数据集上的可预测性"。
- Labuhn 仅 417 条、Hiranniramol 仅 1,309 条，两者都无法支撑"表观微环境"这类需要 2⁴ 析因设计的
  统计推断；因此本文把"未验证"与"已验证"并列写出，而不是用弱证据凑成强结论。


# R4.平台建设说明

> 平台计划建设Web App界面供可视化操作，但由于时间紧凑，未完成Debug，仍存在很多问题，故不在正式项目中展示。
