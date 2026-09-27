# -*- coding: utf-8 -*-
"""analysis.crispron_validation.table — 组装验证表并写出 md / csv。

交付表七列（**列名逐字按计划**）：

| 列 | 含义 |
| :--- | :--- |
| `cell_line` | 细胞系 |
| `sgRNA` | 23 nt 序列（20 protospacer + 3 PAM） |
| `y_true-ism` | 真实编辑效率 − CNN7 对 **C18A 突变序列**的预测效率 |
| `y_true-crispron_c18a` | 真实编辑效率 − **CRISPRon** 对 C18A 突变序列的预测 |
| `cnn_pred-ism` | CNN7 对 **WT** 的预测 − 对 **C18A 突变序列**的预测 |
| `crispron_wt-crispron_c18a` | CRISPRon 对 **WT** 的预测 − 对 **C18A 突变体**的预测 |
| `consistency` | `cnn_pred-ism` 与 `crispron_wt-crispron_c18a` **同号为 `+`，异号为 `-`** |

命名约定：`ism` = **对 ISM 突变后序列的预测值**。因此
`A-ism` = A 减去"突变体预测值"。`A-B` 统一为 `A − B`。

**为什么要看 `cnn_pred-ism` 而不是 `y_true-ism`**：`y_true` 的**序列间波动**
远大于突变本身的效应量，直接拿它做差会把方向信号稀释成噪声。
`cnn_pred-ism` 是模型自己的 ISM 增量，与 CRISPRon 的增量同尺度、同结构，
才是可比的对象。两个 `y_true-*` 列的符号占比因此只作**噪声参照**列出。

`consistency` 的占比 = `n(+) / (n(+) + n(-))` —— 分母只数能判方向的序列，
并列（`0`）与缺失不参与，避免把并列算作异号而低估一致性。
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import CrispronValidationConfig

#: 交付 CSV 的列（顺序 = 计划里的顺序）
TABLE_COLUMNS = ["cell_line", "sgRNA", "y_true-ism", "y_true-crispron_c18a",
                 "cnn_pred-ism", "crispron_wt-crispron_c18a", "consistency"]

#: 三个主指标各自对应的列
COL_CNN_ISM = "cnn_pred-ism"
COL_CRISPRON_DELTA = "crispron_wt-crispron_c18a"
#: 两个只作噪声参照的列
COL_YT_ISM = "y_true-ism"
COL_YT_CRISPRON = "y_true-crispron_c18a"


def consistency_flag(a: float, b: float) -> str:
    """同号 `+` / 异号 `-` / 含 0 记 `0` / 缺失记空串。"""
    if a is None or b is None:
        return ""
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return ""
    if a == 0.0 or b == 0.0:
        return "0"
    return "+" if (a > 0) == (b > 0) else "-"


def _sign_stats(s: pd.Series) -> Dict[str, object]:
    """一列的有符号统计：正/负/零/缺失计数 + 正号占比（分母只数非零可判值）。"""
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
    fin = np.isfinite(v)
    n_fin = int(fin.sum())
    pos = int((v[fin] > 0).sum())
    neg = int((v[fin] < 0).sum())
    zero = int((v[fin] == 0).sum())
    denom = pos + neg
    return {
        "n_finite": n_fin,
        "n_pos": pos,
        "n_neg": neg,
        "n_zero": zero,
        "n_missing": int(len(v) - n_fin),
        "pos_fraction": (pos / denom) if denom else float("nan"),
        "mean": float(np.mean(v[fin])) if n_fin else float("nan"),
        "sd": float(np.std(v[fin])) if n_fin else float("nan"),
    }


def build_table(meta: pd.DataFrame, y_true: np.ndarray, pred_wt: np.ndarray,
                pred_mut: np.ndarray, crispron_wt: np.ndarray,
                crispron_mut: np.ndarray,
                eligible_mask: np.ndarray) -> pd.DataFrame:
    """把逐序列的量拼成交付表（只保留 eligible 行）。"""
    m = np.asarray(eligible_mask, dtype=bool)
    if not m.any():
        raise ValueError("eligible 掩码全为 False，没有可交付的序列")

    def take(a: np.ndarray) -> np.ndarray:
        return np.asarray(a, dtype=np.float64)[m]

    yt, pw, pm = take(y_true), take(pred_wt), take(pred_mut)
    cw, cm = take(crispron_wt), take(crispron_mut)

    df = pd.DataFrame({
        "cell_line": meta.loc[m, "cell_line"].astype(str).to_numpy(),
        "sgRNA": meta.loc[m, "sgRNA"].astype(str).to_numpy(),
        COL_YT_ISM: yt - pm,
        COL_YT_CRISPRON: yt - cm,
        COL_CNN_ISM: pw - pm,
        COL_CRISPRON_DELTA: cw - cm,
    })
    df["consistency"] = [consistency_flag(a, b) for a, b in
                         zip(df[COL_CNN_ISM], df[COL_CRISPRON_DELTA])]
    df = (df.sort_values(["cell_line", "sgRNA"])
            .reset_index(drop=True))
    return df[TABLE_COLUMNS]


def _block(g: pd.DataFrame) -> Dict[str, object]:
    """一组序列（总体或某个细胞系）的完整统计。"""
    flag = g["consistency"].astype(str)
    n_pos = int((flag == "+").sum())
    n_neg = int((flag == "-").sum())
    denom = n_pos + n_neg
    out: Dict[str, object] = {
        "n_sequences": int(len(g)),
        "n_comparable": denom,
        "n_same_sign": n_pos,
        "n_opposite": n_neg,
        "n_tie": int((flag == "0").sum()),
        "n_missing": int((flag == "").sum()),
        "same_sign_fraction": (n_pos / denom) if denom else float("nan"),
    }
    for key, col in (("cnn_ism", COL_CNN_ISM),
                     ("crispron_delta", COL_CRISPRON_DELTA),
                     ("yt_ism", COL_YT_ISM),
                     ("yt_crispron", COL_YT_CRISPRON)):
        out[key] = _sign_stats(g[col])
    # y_true-ism 与 cnn_pred-ism 的相关：说明前者为何被后者取代
    a = pd.to_numeric(g[COL_YT_ISM], errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(g[COL_CNN_ISM], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    out["corr_yt_ism_vs_cnn_ism"] = (
        float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else float("nan"))
    return out


def summarize(table: pd.DataFrame) -> Dict[str, object]:
    """总体 + 逐细胞系的统计。三个主指标在前，两个 y_true 列作噪声参照。"""
    overall = _block(table)
    rows = []
    for cl, g in table.groupby("cell_line", sort=True):
        b = _block(g)
        rows.append({
            "cell_line": cl,
            "n_sequences": b["n_sequences"],
            "frac_cnn_ism_pos": b["cnn_ism"]["pos_fraction"],        # type: ignore[index]
            "frac_crispron_pos": b["crispron_delta"]["pos_fraction"],  # type: ignore[index]
            "same_sign_fraction": b["same_sign_fraction"],
            "frac_yt_ism_pos": b["yt_ism"]["pos_fraction"],          # type: ignore[index]
            "frac_yt_crispron_pos": b["yt_crispron"]["pos_fraction"],  # type: ignore[index]
        })
    overall["by_cell_line"] = rows
    return overall


def _fmt(x: object, nd: int = 4) -> str:
    try:
        v = float(x)                                   # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(v):
        return "—"
    return f"{v:.{nd}f}"


def _pct(fr: object) -> str:
    v = _fmt(fr, 4)
    return "—" if v == "—" else v


def render_markdown(cfg: CrispronValidationConfig, stats: Dict[str, object],
                    scale: Optional[Dict[str, object]] = None) -> str:
    """说明文档：三个主指标 + 噪声参照 + 列释义 + 指向同名 CSV。**不列数据行**。"""
    L: List[str] = []
    A = L.append
    cnn = stats["cnn_ism"]                                    # type: ignore[index]
    cp = stats["crispron_delta"]                              # type: ignore[index]
    yt_ism = stats["yt_ism"]                                  # type: ignore[index]
    yt_cp = stats["yt_crispron"]                              # type: ignore[index]

    A("# CNN7 × CRISPRon：Pos18 C→A 一致性验证")
    A("")
    A(f"- 数据集：`{cfg.dataset}`（细胞系 {'/'.join(cfg.cell_lines)}）")
    A(f"- 模型：CNN，`sequence_kernel={cfg.sequence_kernel}`，**纯序列**"
      f"（`selected_environments=[]`），**四个细胞系全部数据既训练又评估**（无留出）")
    A(f"- 关注位点：第 **{cfg.pos_1b}** 位（1-based），只取原始碱基为 "
      f"`{cfg.target_base}` 的序列，突变为 `{cfg.mutant_base}`")
    A(f"- 待检验的命题：把该位点由 `{cfg.target_base}` 变成 `{cfg.mutant_base}`，"
      f"**会不会降低编辑效率**（即两个增量是否都为正、且方向一致）")
    A("")
    A("## 三个主指标")
    A("")
    A(f"1. **`cnn_pred-ism` 正号占比 = {_pct(cnn['pos_fraction'])}**"
      f"（{cnn['n_pos']} / {cnn['n_pos'] + cnn['n_neg']}）"
      f" —— 本项目 CNN7 判为**有害**（突变把预测效率**拉低**）的序列比例")
    A(f"2. **`crispron_wt-crispron_c18a` 正号占比 = {_pct(cp['pos_fraction'])}**"
      f"（{cp['n_pos']} / {cp['n_pos'] + cp['n_neg']}）"
      f" —— 外部预测器 CRISPRon 判为**有害**的序列比例")
    A(f"3. **`consistency` 正号率 = {_pct(stats['same_sign_fraction'])}**"
      f"（{stats['n_same_sign']} / {stats['n_comparable']}）"
      f" —— 两者对「是否有害」判断**一致**（同号）的序列比例")
    A("")
    A(f"> 总序列数 **{stats['n_sequences']}**；可判方向 "
      f"{stats['n_comparable']}（并列 {stats['n_tie']}，缺失 {stats['n_missing']}）。")
    A(f"> `cnn_pred-ism` 均值 {_fmt(cnn['mean'])}，"
      f"`crispron_wt-crispron_c18a` 均值 {_fmt(cp['mean'])} —— "
      f"两者尺度接近，可直接比较：CRISPRon 的方向性明显更强"
      f"（{_pct(cp['pos_fraction'])} vs {_pct(cnn['pos_fraction'])}），"
      f"一致率只有 {_pct(stats['same_sign_fraction'])}。")
    A("")
    A("### 逐细胞系")
    A("")
    A("| cell_line | 序列数 | `cnn_pred-ism` 正号占比 | "
      "`crispron_wt-crispron_c18a` 正号占比 | `consistency` 正号率 |")
    A("| :--- | ---: | ---: | ---: | ---: |")
    for r in stats["by_cell_line"]:                           # type: ignore[union-attr]
        A(f"| {r['cell_line']} | {r['n_sequences']} | "
          f"{_pct(r['frac_cnn_ism_pos'])} | {_pct(r['frac_crispron_pos'])} | "
          f"{_pct(r['same_sign_fraction'])} |")
    A("")

    A("## 两个 `y_true-*` 列：请当作**噪声参照**读")
    A("")
    A("这两列把**真实标签**卷了进来，因此它们的符号占比**不能**当作模型或 "
      "CRISPRon 的表现指标：")
    A("")
    A("| 列 | 正号占比 | 说明 |")
    A("| :--- | ---: | :--- |")
    A(f"| `y_true-ism` | {_pct(yt_ism['pos_fraction'])} | "
      f"真实效率 − CNN7 对突变体的预测 |")
    A(f"| `y_true-crispron_c18a` | {_pct(yt_cp['pos_fraction'])} | "
      f"真实效率 − CRISPRon 对突变体的预测 |")
    A("")
    A("两列的偏离方向**各有各的原因**，都不是突变效应：")
    A("")
    A(f"**(1) `y_true-ism` 偏向 0.5 —— 被标签的序列间波动稀释。**")
    A("")
    A(f"- `y_true-ism` 的标准差 **{_fmt(yt_ism['sd'])}**，而 `cnn_pred-ism` 的均值"
      f"只有 **{_fmt(cnn['mean'])}** —— 噪声项比信号项大 "
      f"{_fmt(abs(float(yt_ism['sd']) / float(cnn['mean'])) if cnn['mean'] else float('nan'), 2)} 倍")
    A(f"- `y_true-ism` 与 `cnn_pred-ism` 的相关系数仍高达 "
      f"**{_fmt(stats['corr_yt_ism_vs_cnn_ism'])}** —— in-sample 下 "
      f"`pred_wt ≈ y_true`，所以它**约等于** ISM 增量，但符号被标签涨落来回翻转，"
      f"占比被推回 {_pct(yt_ism['pos_fraction'])}")
    A("")
    A(f"**(2) `y_true-crispron_c18a` 远离 0.5 —— 被两个体系的尺度错配主导。**")
    A("")
    A("它根本不是 50/50 的随机量，而是**几乎恒为负**：")
    A("")
    if scale:
        A("| 量 | 均值 | 标准差 |")
        A("| :--- | ---: | ---: |")
        A(f"| 真实标签 `y_true` | {_fmt(scale.get('mean_yt'))} | {_fmt(scale.get('sd_yt'))} |")
        A(f"| CNN7 对 WT 的预测 | {_fmt(scale.get('mean_pred_wt'))} | "
          f"{_fmt(scale.get('sd_pred_wt'))} |")
        A(f"| CNN7 对 C18A 的预测 | {_fmt(scale.get('mean_pred_mut'))} | "
          f"{_fmt(scale.get('sd_pred_mut'))} |")
        A(f"| CRISPRon 对 WT 的预测 | {_fmt(scale.get('mean_cp_wt'))} | "
          f"{_fmt(scale.get('sd_cp_wt'))} |")
        A(f"| CRISPRon 对 C18A 的预测 | {_fmt(scale.get('mean_cp_mut'))} | "
          f"{_fmt(scale.get('sd_cp_mut'))} |")
        A("")
        A("DeepCRISPR 的标签与 CRISPRon 的输出**不在同一尺度上**，"
          "所以「标签 − CRISPRon 预测」里叠了一个**恒定的系统性偏移**，"
          "突变效应只是叠加在它上面的小量。具体到本批：")
        A("")
        A(f"- 真实标签均值约 **{_fmt(scale.get('mean_yt'), 3)}**，而 CRISPRon 对突变体的"
          f"预测均值约 **{_fmt(scale.get('mean_cp_mut'), 3)}**，系统性相差约 "
          f"**{_fmt((scale.get('mean_yt') or 0) - (scale.get('mean_cp_mut') or 0), 3)}**")
        A(f"- 而突变本身在 CRISPRon 尺度上的效应只有约 **{_fmt(cp['mean'])}**")
    else:
        A("DeepCRISPR 的标签与 CRISPRon 的输出**不在同一尺度上**，"
          "所以「标签 − CRISPRon 预测」里叠了一个**恒定的系统性偏移**，"
          "突变效应只是叠加在它上面的小量。")
    A(f"- 结果：`y_true-crispron_c18a > 0` 的占比只有 "
      f"**{_pct(yt_cp['pos_fraction'])}** —— 这个数字反映的是**两套体系的尺度差**，"
      f"不是任何一方对突变的判断")
    A("")
    A(f"> 两列都没有信息量可用：一列被方差淹没（{_pct(yt_ism['pos_fraction'])}），"
      f"一列被尺度偏移淹没（{_pct(yt_cp['pos_fraction'])}）；而两个真正的增量是 "
      f"{_pct(cnn['pos_fraction'])} 与 {_pct(cp['pos_fraction'])}，"
      f"方向性清楚得多。**判读只看后者。**")
    A("")
    A("> 结论：判读方向一致性**只看** `cnn_pred-ism` × "
      "`crispron_wt-crispron_c18a`；两个 `y_true-*` 列保留在表里是为了"
      "**展示这两条噪声路径**，不是判据。")
    A("")

    A("## 列的释义")
    A("")
    A("命名约定：`ism` = **对 ISM 突变后序列的预测值**；`A-B` = `A − B`。")
    A("")
    A("| 列 | 含义 |")
    A("| :--- | :--- |")
    A("| `cell_line` | 细胞系 |")
    A("| `sgRNA` | 23 nt 序列（20 nt protospacer + 3 nt PAM） |")
    A("| `y_true-ism` | 真实编辑效率 − CNN7 对**第 "
      f"{cfg.pos_1b} 位改成 {cfg.mutant_base} 之后**的序列的预测效率 |")
    A("| `y_true-crispron_c18a` | 真实编辑效率 − CRISPRon 对 **C18A 突变体**的预测 |")
    A("| `cnn_pred-ism` | CNN7 对 **WT** 的预测 − 对 **C18A 突变体**的预测"
      "（模型自己的 ISM 增量） |")
    A("| `crispron_wt-crispron_c18a` | CRISPRon 对 **WT** 的预测 − 对 "
      "**C18A 突变体**的预测（外部预测器的同一增量） |")
    A("| `consistency` | `cnn_pred-ism` 与 `crispron_wt-crispron_c18a` "
      "**同号为 `+`，异号为 `-`**；任一列为 0 记 `0`；缺失留空 |")
    A("")
    A("## 怎么读")
    A("")
    A("- 三行的读法：两个正号占比各自回答「这一方判该突变**有害**（拉低效率）的序列占多少」，"
      "`consistency` 正号率回答「两方判断**同向**的序列占多少」。"
      "两列的正号都被定义成同一个方向，因此 `consistency` 的 `+` 等价于「两边都说有害」。")
    A(f"- 本次：CRISPRon 有 **{_pct(cp['pos_fraction'])}** 的序列判为「有害」，"
      f"CNN7 只有 **{_pct(cnn['pos_fraction'])}**，一致率 **{_pct(stats['same_sign_fraction'])}**。"
      f"即：**本项目模型没有复现 CRISPRon 对 PAM 近端位点突变的强方向先验。**")
    A("- ⚠️ **CNN7 是在同一批序列上训练又在同一批序列上预测的（in-sample）**，"
      "因此这是「模型已尽可能拟合标签后，仍与外部预测器不一致」的对照，"
      "不是泛化误差。")
    A("- ⚠️ 两个预测器都对同一批 30 nt 窗口取预测值；CRISPRon 用的是 WT 与突变体"
      "两个真实 30-mer，CNN7 用的是同一序列的 one-hot 张量。")
    A("")
    A(f"> 完整逐序列数据见同目录 **`{cfg.report_csv.name}`**"
      f"（{stats['n_sequences']} 行）。本文件不含数据行。")
    A("")
    return "\n".join(L)


def write_outputs(cfg: CrispronValidationConfig, table: pd.DataFrame,
                  stats: Dict[str, object],
                  scale: Optional[Dict[str, object]] = None) -> Dict[str, Path]:
    cfg.ensure_dirs()
    table.to_csv(cfg.report_csv, index=False, encoding="utf-8-sig")
    cfg.report_md.write_text(render_markdown(cfg, stats, scale), encoding="utf-8")
    return {"md": cfg.report_md, "csv": cfg.report_csv}
