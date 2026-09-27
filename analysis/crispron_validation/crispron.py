# -*- coding: utf-8 -*-
"""analysis.crispron_validation.crispron — 调用已安装的 CRISPRon，得到逐序列预测值。

CRISPRon 的最终输出是 `crispron.csv`（列 `ID, 30mer, CRISPRon`），
`CRISPRon` 是 **indel 频率百分数（0-100）**，本模块统一 /100 对齐到 [0,1]。

按 **30mer 列** 对齐而不是按 ID —— `get_30mers_from_fa.py` 会给 ID 加
`_p_<pos>` / `_m_<pos>` 后缀，且同一输入若含多个合法 PAM 会产出多行。

WT 预测有一条约定的**快速路径**：旧外部验证归档里已经算过 DeepCRISPR 全部
野生型窗口（12,465 个）。该文件存在就直接复用；不存在则照样实跑，因此
本模块对归档**没有运行时依赖**。
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

from .config import CrispronValidationConfig


class CrispronRunner:
    """把一组 30 nt 窗口喂给 `deploy/crispron/`，拿回 ``{window: 预测值}``。"""

    def __init__(self, cfg: CrispronValidationConfig) -> None:
        self.cfg = cfg

    # ---------------- 安装自检 ----------------
    def check_install(self) -> None:
        missing = [p for p in (self.cfg.crispron_script, self.cfg.crispron_venv_bin,
                               self.cfg.crispron_wrappers, self.cfg.crispron_main)
                   if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "CRISPRon 安装不完整，缺少:\n  "
                + "\n  ".join(str(m) for m in missing)
                + f"\n请确认 {self.cfg.crispron_root} 是按官方步骤装好的。")

    # ---------------- 单批次运行 ----------------
    def run(self, windows: Sequence[str], tag: str, force: bool = False
            ) -> Dict[str, float]:
        """对 ``windows`` 里的唯一 30-mer 跑 CRISPRon，结果缓存到 work_dir。"""
        self.check_install()
        uniq = sorted({str(w).upper().strip() for w in windows if str(w).strip()})
        out_dir = self.cfg.work_dir / "crispron_raw" / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        merged = out_dir / "crispron.csv"

        if not force and merged.exists():
            got = self._read_cache(merged, uniq)
            if len(got) == len(uniq):
                print(f"  [{tag}] 结果已缓存且完整（{len(uniq)} 条），跳过运行")
                return got

        print(f"  [{tag}] 运行 CRISPRon：{len(uniq)} 条唯一 30-mer，"
              f"分块 {self.cfg.crispron_chunk_size}")
        scores: Dict[str, float] = {}
        n_chunks = (len(uniq) + self.cfg.crispron_chunk_size - 1) // self.cfg.crispron_chunk_size
        for ci in range(n_chunks):
            chunk = uniq[ci * self.cfg.crispron_chunk_size:(ci + 1) * self.cfg.crispron_chunk_size]
            part_dir = out_dir / f"chunk_{ci:03d}"
            part_csv = part_dir / "crispron.csv"
            if force or not part_csv.exists():
                part_dir.mkdir(parents=True, exist_ok=True)
                fasta = part_dir / "input.fa"
                with fasta.open("w", encoding="utf-8") as fh:
                    for i, w in enumerate(chunk):
                        fh.write(f">sq{ci:03d}_{i:06d}\n{w}\n")
                self._invoke(fasta, part_dir)
            got = self._parse(part_csv, chunk)
            missing = [w for w in chunk if w not in got]
            if missing:
                print(f"    [!] 块 {ci}: {len(missing)}/{len(chunk)} 条无预测值"
                      f"（窗口内没有合法 PAM 或 CRISPRon 未输出）")
            scores.update(got)
            print(f"    块 {ci+1}/{n_chunks}: 累计 {len(scores)}/{len(uniq)}", flush=True)

        pd.DataFrame({"window30": list(scores),
                      "crispron": [scores[k] for k in scores]}
                     ).to_csv(merged, index=False, encoding="utf-8-sig")
        return scores

    def _invoke(self, fasta: Path, out_dir: Path) -> None:
        env = dict(os.environ)
        env["PATH"] = f"{self.cfg.crispron_venv_bin}:{self.cfg.crispron_wrappers}:" \
                      + env.get("PATH", "")
        # CRISPRon.sh 用 `dirname $0` 找 data/，必须用绝对路径调用
        cmd = ["bash", str(self.cfg.crispron_script.resolve()),
               str(fasta.resolve()), str(out_dir.resolve())]
        proc = subprocess.run(cmd, cwd=str(self.cfg.crispron_main), env=env,
                              capture_output=True, text=True,
                              timeout=int(self.cfg.crispron_timeout))
        if proc.returncode != 0:
            tail = (proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-3000:]
            raise RuntimeError(f"CRISPRon 退出码 {proc.returncode}\n{tail}")
        if not (out_dir / "crispron.csv").exists():
            raise RuntimeError("CRISPRon 未产出 crispron.csv\n"
                               f"stdout 尾部: {(proc.stdout or '')[-1500:]}")

    # ---------------- 解析 ----------------
    @staticmethod
    def _parse(csv_path: Path, wanted: Sequence[str]) -> Dict[str, float]:
        if not csv_path.exists():
            return {}
        d = pd.read_csv(csv_path)
        if "30mer" not in d.columns or "CRISPRon" not in d.columns:
            return {}
        want = {str(w).upper() for w in wanted}
        scores: Dict[str, float] = {}
        # `30mer` / `CRISPRon` 不是合法 Python 标识符，itertuples 会改名，
        # 因此按列名取值。
        for raw_window, raw_score in zip(d["30mer"], d["CRISPRon"]):
            w = str(raw_window).upper().strip()
            if w not in want:
                continue
            try:
                val = float(raw_score) / 100.0
            except (TypeError, ValueError):
                continue
            if not np.isfinite(val):
                continue
            if w in scores and abs(scores[w] - val) > 1e-6:
                raise RuntimeError(f"同一窗口 {w} 得到两个不同预测值: "
                                   f"{scores[w]} vs {val}")
            scores[w] = val
        return scores

    @staticmethod
    def _read_cache(path: Path, wanted: Sequence[str]) -> Dict[str, float]:
        d = pd.read_csv(path)
        if not {"window30", "crispron"}.issubset(d.columns):
            return {}
        want = {str(w).upper() for w in wanted}
        return {str(w).upper(): float(s) for w, s in zip(d["window30"], d["crispron"])
                if str(w).upper() in want}


def seed_wt_from_archive(cfg: CrispronValidationConfig) -> Optional[pd.DataFrame]:
    """旧外部验证归档里的 DeepCRISPR 野生型 CRISPRon 结果（可选快速路径）。"""
    p = cfg.archive_work_dir / "crispron_raw" / cfg.dataset / "crispron.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
    except Exception:                                           # noqa: BLE001
        return None
    if not {"window30", "crispron"}.issubset(d.columns):
        return None
    d = d.copy()
    d["window30"] = d["window30"].astype(str).str.upper()
    return d.drop_duplicates(subset=["window30"], keep="first")


def ensure_wt_scores(cfg: CrispronValidationConfig, windows: Sequence[str],
                     runner: CrispronRunner, force: bool = False) -> Dict[str, float]:
    """野生型预测：先用归档快速路径补齐，缺的再实跑。"""
    uniq = sorted({str(w).upper() for w in windows if str(w)})
    scores: Dict[str, float] = {}
    if not force:
        arch = seed_wt_from_archive(cfg)
        if arch is not None:
            scores = {w: float(s) for w, s in zip(arch["window30"], arch["crispron"])
                      if w in set(uniq)}
            print(f"  [WT] 归档复用 {len(scores)}/{len(uniq)} 条")
    missing = [w for w in uniq if w not in scores]
    if missing:
        print(f"  [WT] 归档未覆盖 {len(missing)} 条，实跑 CRISPRon")
        scores.update(runner.run(missing, tag="wt", force=force))
    return scores
