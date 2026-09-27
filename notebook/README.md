# `notebook/` — 项目核心流程演示

本目录放**只读型**演示 notebook：不训练模型、不改任何产物，只按交付目录里已落盘的产物走一遍核心链路，
用来给评审/二次开发者一个可交互的入口。

| 文件 | 说明 |
| :--- | :--- |
| `core_flow_hct116_single_all_cnn73.ipynb` | 以 `HCT116 · single · all（全环境）· CNN(7\|3)`（run 名 `single_hct116_cnn_all_kernel_7`）为例，展示：① 训练数据（张量 / 标签 / 划分 / 编码声明）→ ② 生成了哪些文件（run 级 7 个 + 批次级五类目录）→ ③ 可视化（只看热图）→ ④ 本次跳过的统计处理与原因 → ⑤ 复现命令 |

## 打开方式

```bash
pip install jupyter            # 或直接用 VS Code 打开 .ipynb
jupyter notebook notebook/core_flow_hct116_single_all_cnn73.ipynb
```

notebook 里已经内嵌了**真实执行输出**，不运行也能看；要重新运行请先确保仓库根下已有
`results/train_results/`、`results/summary/`、`data/processed/` 三棵子树（即已跑过一遍主流程）。

## 路径约定

notebook 用一行代码自适应工作目录：

```python
ROOT = Path.cwd().parent if Path.cwd().name == "notebook" else Path.cwd()
```

因此无论从仓库根还是从 `notebook/` 启动（或在 VS Code 中打开）都能定位到仓库根；图片用相对路径
`../results/summary/...` 嵌入，随 notebook 位置解析。

## 范围说明

- 可视化只展示与本配置归因方法对应的 **CNN IG 位置归因热图**
  （`figures/plots/position_attribution_cnn_ig.png`）；条件 ΔR² 热图因该表当前含数值发散行
  （`|ΔR²|` ≥ 10 的行占 55/2016，最大 3.97e20）而不在 notebook 内展示，原因写在第 ④ 节。
- 演示配置是**单细胞系 + 单划分**，因此析因 ANOVA、两因子交互、跨细胞系泛化比例等需要
  跨上下文/跨细胞系的设计**不在本 notebook 展开**（原因见第 ④ 节）。
  这些产物在交付批次 `results/summary/DeepCRISPR/` 里都有，可直接打开核对。
- 热图是**批次级**产物（`results/summary/DeepCRISPR/figures/`），不是单 run 产物；notebook 里已注明。
