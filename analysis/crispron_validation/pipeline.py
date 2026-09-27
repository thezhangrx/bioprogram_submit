# -*- coding: utf-8 -*-
"""analysis.crispron_validation.pipeline — 编排四个步骤并落盘中间产物。

步骤之间**只通过磁盘上的表交接**，因此任何一步都可以单独重跑：

    windows    -> work_dir/windows_<dataset>.csv
    crispron   -> work_dir/crispron_wt.csv, crispron_mut.csv
    model      -> work_dir/predictions_<dataset>.csv
    table      -> results/summary/<dataset>/cnn7_validation.{md,csv}

所有按序列的交接都以 `(cell_line, sgRNA)` 为键做 merge，
不依赖"两个文件行序恰好一致"这种脆弱假设。
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import CrispronValidationConfig
from .crispron import CrispronRunner, ensure_wt_scores
from .model import Cnn7AllDataModel, load_all_data, mutate_pos18
from .table import build_table, summarize, write_outputs
from .windows import WindowBuilder

STEPS = ("windows", "crispron", "model", "table")


class CrispronValidationPipeline:
    def __init__(self, cfg: CrispronValidationConfig) -> None:
        self.cfg = cfg
        cfg.ensure_dirs()
        self.windows: Optional[pd.DataFrame] = None

    # ---------------- 1. 窗口 ----------------
    def step_windows(self, force: bool = False) -> pd.DataFrame:
        print("=" * 76)
        print("步骤 1/4：构建 30nt 野生型与 C18A 突变型窗口")
        print("=" * 76)
        self.windows = WindowBuilder(self.cfg).build(force=force)
        el = self.windows[self.windows["eligible"]]
        print(f"  可做 C→A 的序列: {len(el)}，唯一 WT 窗口 {el['window30'].nunique()}，"
              f"唯一突变窗口 {el['window30_mut'].nunique()}")
        return self.windows

    def _load_windows(self) -> pd.DataFrame:
        if self.windows is None:
            if not self.cfg.windows_path.exists():
                raise FileNotFoundError(
                    f"缺 {self.cfg.windows_path}；请先跑 --steps windows")
            self.windows = pd.read_csv(self.cfg.windows_path, dtype={"Chromosome": str})
        return self.windows

    # ---------------- 2. CRISPRon ----------------
    def step_crispron(self, force: bool = False) -> Dict[str, Path]:
        print()
        print("=" * 76)
        print("步骤 2/4：调用 CRISPRon（WT 与 C18A 突变体各一份）")
        print("=" * 76)
        w = self._load_windows()
        el = w[w["eligible"]]
        if el.empty:
            raise ValueError("没有任何可做 C→A 的序列，无法继续。")
        wt_windows = sorted(set(el["window30"].astype(str)))
        mut_windows = sorted(set(el["window30_mut"].astype(str)))
        runner = CrispronRunner(self.cfg)

        wt = ensure_wt_scores(self.cfg, wt_windows, runner, force=force)
        pd.DataFrame({"window30": list(wt), "crispron": [wt[k] for k in wt]}
                     ).to_csv(self.cfg.crispron_wt_path, index=False,
                              encoding="utf-8-sig")
        cover_wt = el["window30"].astype(str).isin(wt).mean()
        print(f"  [WT] 覆盖 {cover_wt:.1%}（{len(wt)}/{len(wt_windows)} 个窗口）")

        mut = runner.run(mut_windows, tag="c18a", force=force)
        pd.DataFrame({"window30": list(mut), "crispron": [mut[k] for k in mut]}
                     ).to_csv(self.cfg.crispron_mut_path, index=False,
                              encoding="utf-8-sig")
        cover_mut = el["window30_mut"].astype(str).isin(mut).mean()
        print(f"  [C18A] 覆盖 {cover_mut:.1%}（{len(mut)}/{len(mut_windows)} 个窗口）")
        return {"wt": self.cfg.crispron_wt_path, "mut": self.cfg.crispron_mut_path}

    # ---------------- 3. CNN7 ----------------
    def step_model(self, force: bool = False) -> Path:
        print()
        print("=" * 76)
        print("步骤 3/4：四个细胞系全部数据训练 CNN7（train = test）")
        print("=" * 76)
        out = self.cfg.work_dir / f"predictions_{self.cfg.dataset}.csv"
        if out.exists() and not force:
            print(f"  预测表已存在: {out.name}（加 --force 可重训）")
            return out

        w = self._load_windows()
        X, y, meta = load_all_data(self.cfg)

        # 行对齐自检：windows 表与 metadata 必须逐行同源
        if len(w) != len(meta):
            raise ValueError(f"windows 行数 {len(w)} != metadata 行数 {len(meta)}")
        key_w = (w["cell_line"].astype(str) + "|" + w["sgRNA"].astype(str)).to_numpy()
        key_m = (meta["cell_line"].astype(str) + "|" + meta["sgRNA"].astype(str)).to_numpy()
        if not np.array_equal(key_w, key_m):
            raise ValueError("windows 表与 metadata 的 (cell_line, sgRNA) 顺序不一致")
        eligible = w["eligible"].to_numpy(dtype=bool)

        m = Cnn7AllDataModel(self.cfg)
        res = m.fit(X, y)
        pred_wt = np.asarray(res["predictions"], dtype=np.float64)

        # 只把 eligible 行的第 18 位改成 A，其余行原样
        X_mut = mutate_pos18(X, eligible, self.cfg.pos_index0, self.cfg.mutant_base)
        pred_mut = m.predict(X_mut)

        pd.DataFrame({
            "cell_line": meta["cell_line"].astype(str).to_numpy(),
            "sgRNA": meta["sgRNA"].astype(str).to_numpy(),
            "y_true": np.asarray(y, dtype=np.float64),
            "pred_wt": pred_wt,
            "pred_c18a": np.asarray(pred_mut, dtype=np.float64),
            "eligible": eligible,
        }).to_csv(out, index=False, encoding="utf-8-sig")
        print(f"  -> {out}")
        return out

    # ---------------- 4. 交付表 ----------------
    def step_table(self) -> Dict[str, Path]:
        print()
        print("=" * 76)
        print("步骤 4/4：组装验证表并写 cnn7_validation.{md,csv}")
        print("=" * 76)
        w = self._load_windows()
        pred_path = self.cfg.work_dir / f"predictions_{self.cfg.dataset}.csv"
        for p in (pred_path, self.cfg.crispron_wt_path, self.cfg.crispron_mut_path):
            if not p.exists():
                raise FileNotFoundError(f"缺 {p}；请先跑对应步骤")

        pred = pd.read_csv(pred_path)
        wt = pd.read_csv(self.cfg.crispron_wt_path)
        mut = pd.read_csv(self.cfg.crispron_mut_path)

        # `eligible` 在窗口表与预测表里各有一份：先交叉核对两份必须一致（同一判据
        # 由两条独立路径算出），核对通过后丢掉预测表那份，避免 merge 出 _x/_y。
        if "eligible" in pred.columns:
            w_sorted = w.sort_values(["cell_line", "sgRNA"]).reset_index(drop=True)
            p_sorted = pred.sort_values(["cell_line", "sgRNA"]).reset_index(drop=True)
            a = w_sorted["eligible"].to_numpy(dtype=bool)
            b = p_sorted["eligible"].to_numpy(dtype=bool)
            if len(a) != len(b) or not np.array_equal(a, b):
                raise ValueError("窗口表与预测表的 eligible 标记不一致，拒绝继续")
            print("  eligible 交叉核对: 窗口表 == 预测表 ✓")
            pred = pred.drop(columns=["eligible"])

        # 全部以 (cell_line, sgRNA) / window30 为键 merge，不靠行序
        d = w.merge(pred, on=["cell_line", "sgRNA"], how="left", validate="one_to_one")
        d = d.merge(wt.rename(columns={"crispron": "cp_wt"}), on="window30", how="left")
        # 突变体分数同样以 `window30` 为键保存（CRISPRon 的输入就是那个 30-mer），
        # 这里改名到窗口表的 `window30_mut` 再 merge。
        d = d.merge(mut.rename(columns={"window30": "window30_mut",
                                        "crispron": "cp_mut"}),
                    on="window30_mut", how="left")

        el = d["eligible"].fillna(False).to_numpy(dtype=bool)
        table = build_table(d, d["y_true"].to_numpy(), d["pred_wt"].to_numpy(),
                            d["pred_c18a"].to_numpy(),
                            d["cp_wt"].to_numpy(), d["cp_mut"].to_numpy(), el)
        if table.empty:
            raise ValueError("组装出的表为空（可能 CRISPRon 覆盖率为 0）")
        stats = summarize(table)

        # 绝对值尺度：`y_true-crispron_c18a` 的符号几乎完全由"标签 vs CRISPRon"
        # 的系统性尺度差决定，所以把这些均值一并写进文档，供读者自行核对。
        el_d = d[el]
        scale = {}
        for name, col in (("yt", "y_true"), ("pred_wt", "pred_wt"),
                          ("pred_mut", "pred_c18a"), ("cp_wt", "cp_wt"),
                          ("cp_mut", "cp_mut")):
            v = pd.to_numeric(el_d[col], errors="coerce").to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            scale[f"mean_{name}"] = float(np.mean(v)) if v.size else float("nan")
            scale[f"sd_{name}"] = float(np.std(v)) if v.size else float("nan")
        paths = write_outputs(self.cfg, table, stats, scale)

        cnn = stats["cnn_ism"]                                # type: ignore[index]
        cp = stats["crispron_delta"]                           # type: ignore[index]
        yt_ism = stats["yt_ism"]                               # type: ignore[index]
        yt_cp = stats["yt_crispron"]                           # type: ignore[index]
        print(f"  总序列数 {stats['n_sequences']}；可判方向 {stats['n_comparable']}"
              f"（并列 {stats['n_tie']}，缺失 {stats['n_missing']}）")
        print(f"  [主指标] cnn_pred-ism 正号占比   {cnn['pos_fraction']:.4f} "
              f"({cnn['n_pos']}/{cnn['n_pos'] + cnn['n_neg']})")
        print(f"  [主指标] crispron 增量 正号占比  {cp['pos_fraction']:.4f} "
              f"({cp['n_pos']}/{cp['n_pos'] + cp['n_neg']})")
        print(f"  [主指标] consistency 正号率      {stats['same_sign_fraction']:.4f} "
              f"({stats['n_same_sign']}/{stats['n_comparable']})")
        print(f"  [噪声参照] y_true-ism 正号占比   {yt_ism['pos_fraction']:.4f}")
        print(f"  [噪声参照] y_true-crispron 正号占比 {yt_cp['pos_fraction']:.4f}")
        print(f"  -> {paths['md']}")
        print(f"  -> {paths['csv']}")
        return paths

    # ---------------- 驱动 ----------------
    def run(self, steps: Optional[list] = None, force: bool = False) -> Dict[str, object]:
        t0 = time.time()
        todo = list(steps) if steps else list(STEPS)
        unknown = [s for s in todo if s not in STEPS]
        if unknown:
            raise ValueError(f"未知步骤 {unknown}；可用 {list(STEPS)}")
        done: Dict[str, object] = {}
        if "windows" in todo:
            done["windows"] = self.step_windows(force=force)
        if "crispron" in todo:
            done["crispron"] = self.step_crispron(force=force)
        if "model" in todo:
            done["model"] = self.step_model(force=force)
        if "table" in todo:
            done["table"] = self.step_table()
        print(f"\n完成，用时 {(time.time()-t0)/60:.2f} min。")
        return done
