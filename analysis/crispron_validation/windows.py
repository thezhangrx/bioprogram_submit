# -*- coding: utf-8 -*-
"""analysis.crispron_validation.windows — 30 nt 窗口构建与 Pos18 C→A 突变窗口。

CRISPRon 要的是

    prefix(4) + protospacer(20) + PAM(3) + suffix(3) = 30 nt

而 DeepCRISPR 的 metadata 只给 23 nt（20 + PAM）。所以必须补侧翼：

* 优先复用 `work_dir/windows_<dataset>.csv`（本流程上次的产物）；
* 其次从 `archive_work_dir` 的旧窗口表播种（**可选加速**，文件不在就跳过）；
* 都没有时，用位点坐标从 **hg19** 取 30 nt（UCSC API，磁盘缓存，可断点续跑）。

三条路径产出的窗口都要过同一道**回验**：30 nt 的中间 23 nt 必须逐字等于
数据集的 `sgRNA`。不过的行标 `matched=False` 并带上原因，绝不静默使用。

突变窗口不重新取序列 —— 直接把 WT 窗口第 18 位（1-based）改成目标碱基。
对 23 nt 而言第 18 位在窗口里的 0-based 下标是 `4 + 17 = 21`；
PAM 在 24~26 位，不在突变范围内，因此 PAM 仍是 NGG。
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from analysis.leakage import revcomp

from .config import (CRISPRON_LEN, DOWNSTREAM_NT, PAM_LEN, SEQ_LEN, SPACER_LEN,
                     UPSTREAM_NT, CrispronValidationConfig)

UCSC_SEQ_URL = ("https://api.genome.ucsc.edu/getData/sequence"
                "?genome=hg19;chrom={chrom};start={start0};end={end}")

#: 输出列
WINDOW_COLUMNS = ["cell_line", "sgRNA", "Chromosome", "Start", "End", "Strand",
                  "pos_base", "window30", "window30_mut", "eligible", "note"]


def window_bounds(start_1b: int, end_1b: int, strand: str) -> Tuple[int, int]:
    """hg19 正链上的取窗区间 ``[lo, hi)``（0-based 半开）。

    负链时侧翼方向相反：取出的正链序列还要反向互补，才是"以靶标为正向"的 30-mer。
    口径与 `analysis/candidates/wt_position18_selection._crispron_window_bounds` 一致。
    """
    start0 = int(start_1b) - 1
    end = int(end_1b)
    if str(strand).strip() == "-":
        return start0 - DOWNSTREAM_NT, end + UPSTREAM_NT
    return start0 - UPSTREAM_NT, end + DOWNSTREAM_NT


def _http_get_dna(chrom: str, start0: int, end: int, timeout: int = 30) -> str:
    url = UCSC_SEQ_URL.format(chrom=chrom, start0=start0, end=end)
    with urllib.request.urlopen(url, timeout=timeout) as fh:
        return str(json.load(fh).get("dna", "")).upper()


def mutate_at(window: str, index0: int, base: str) -> str:
    """把窗口 ``index0`` 处的碱基替换为 ``base``。越界或非 ACGT 时抛错。"""
    w = str(window).upper()
    b = str(base).upper()
    if not (0 <= int(index0) < len(w)):
        raise IndexError(f"突变位点 {index0} 超出窗口长度 {len(w)}")
    if b not in set("ACGT"):
        raise ValueError(f"目标碱基非法: {base!r}")
    return w[:int(index0)] + b + w[int(index0) + 1:]


class WindowBuilder:
    """把 (cell_line, sgRNA) 映射到 WT / C18A 两个 30 nt 窗口。"""

    def __init__(self, cfg: CrispronValidationConfig,
                 http_workers: int = 8, http_retries: int = 3) -> None:
        self.cfg = cfg
        self.http_workers = int(http_workers)
        self.http_retries = int(http_retries)

    # ---------------- 数据集的位点表 ----------------
    def _load_metadata(self) -> pd.DataFrame:
        frames = []
        for cl in self.cfg.cell_lines:
            f = self.cfg.data_dir / f"{cl}_metadata.csv"
            if not f.exists():
                raise FileNotFoundError(f"缺少 {f}")
            d = pd.read_csv(f)
            need = {"sgRNA"}
            missing = need - set(d.columns)
            if missing:
                raise KeyError(f"{f} 缺少列 {sorted(missing)}")
            d = d.copy()
            d["cell_line"] = cl
            frames.append(d)
        out = pd.concat(frames, ignore_index=True)
        out["sgRNA"] = out["sgRNA"].astype(str).str.upper().str.strip()
        out["pos_base"] = out["sgRNA"].str[self.cfg.pos_index0]
        return out

    # ---------------- hg19 抓取（带磁盘缓存）----------------
    def _load_hg19_cache(self) -> Dict[Tuple[str, int, int, str], str]:
        cache: Dict[Tuple[str, int, int, str], str] = {}
        p = self.cfg.hg19_cache_path
        if p.exists():
            c = pd.read_csv(p, dtype={"chrom": str})
            for r in c.itertuples():
                cache[(str(r.chrom), int(r.start), int(r.end), str(r.strand))] = str(r.window)
        return cache

    def _save_hg19_cache(self, cache: Dict[Tuple[str, int, int, str], str]) -> None:
        rows = [{"chrom": k[0], "start": k[1], "end": k[2], "strand": k[3], "window": v}
                for k, v in sorted(cache.items())]
        pd.DataFrame(rows).to_csv(self.cfg.hg19_cache_path, index=False,
                                  encoding="utf-8-sig")

    def _seed_hg19_cache_from_archive(self) -> int:
        """从归档的旧缓存播种（可选加速）。返回播种条数。"""
        src = self.cfg.archive_work_dir / "hg19_window_cache.csv"
        if not src.exists() or self.cfg.hg19_cache_path.exists():
            return 0
        try:
            d = pd.read_csv(src, dtype={"chrom": str})
        except Exception:                                       # noqa: BLE001
            return 0
        d.to_csv(self.cfg.hg19_cache_path, index=False, encoding="utf-8-sig")
        return int(len(d))

    def _fetch_hg19(self, meta: pd.DataFrame) -> Dict[Tuple[str, int, int, str], str]:
        cache = self._load_hg19_cache()
        seeded = self._seed_hg19_cache_from_archive()
        if seeded:
            cache = self._load_hg19_cache()
            print(f"    从归档播种 hg19 缓存 {seeded} 条")

        keys = [(str(r.Chromosome), int(r.Start), int(r.End), str(r.Strand).strip())
                for r in meta.itertuples()]
        uniq = list(dict.fromkeys(keys))
        todo = [k for k in uniq if k not in cache]
        print(f"    hg19 窗口: 需要 {len(uniq)} 个唯一位点, "
              f"缓存命中 {len(uniq) - len(todo)}, 需联网 {len(todo)}")
        if not todo:
            return cache

        t0 = time.time()
        done = 0

        def worker(key):
            chrom, start, end, strand = key
            lo, hi = window_bounds(start, end, strand)
            last = ""
            for attempt in range(self.http_retries):
                try:
                    return key, _http_get_dna(chrom, lo, hi)
                except Exception as exc:                        # noqa: BLE001
                    last = str(exc)
                    time.sleep(0.6 * (attempt + 1))
            return key, f"__ERROR__{last}"

        with ThreadPoolExecutor(max_workers=self.http_workers) as pool:
            for key, win in pool.map(worker, todo):
                cache[key] = win
                done += 1
                if done % 500 == 0:
                    el = time.time() - t0
                    rate = done / max(el, 1e-9)
                    print(f"      {done}/{len(todo)}  {rate:.1f}/s  "
                          f"已用 {el/60:.1f} min  预计剩余 "
                          f"{(len(todo)-done)/max(rate,1e-9)/60:.1f} min", flush=True)
                    self._save_hg19_cache(cache)
        self._save_hg19_cache(cache)
        print(f"    hg19 抓取完成: {done} 个新窗口, 用时 {(time.time()-t0)/60:.1f} min")
        return cache

    # ---------------- 主入口 ----------------
    def build(self, force: bool = False) -> pd.DataFrame:
        cfg = self.cfg
        out_path = cfg.windows_path
        if out_path.exists() and not force:
            d = pd.read_csv(out_path, dtype={"Chromosome": str})
            print(f"  窗口表已存在: {out_path.name} ({len(d)} 行)")
            return d

        print(f"  构建 30nt 窗口 -> {out_path.name}")
        meta = self._load_metadata()
        need_locus = {"Chromosome", "Start", "End", "Strand"}.issubset(meta.columns)

        # 优先用归档里的窗口表（已含 window30），否则走 hg19
        windows: Optional[pd.Series] = None
        arch = cfg.archive_work_dir / f"sequences_{cfg.dataset}.csv"
        if arch.exists():
            try:
                a = pd.read_csv(arch, dtype={"Chromosome": str})
                if {"cell_line", "sgRNA", "window30"}.issubset(a.columns):
                    key = (a["cell_line"].astype(str).str.lower() + "|"
                           + a["sgRNA"].astype(str).str.upper().str.strip())
                    amap = dict(zip(key, a["window30"].astype(str)))
                    mk = (meta["cell_line"].astype(str).str.lower() + "|" + meta["sgRNA"])
                    windows = mk.map(amap)
                    hit = windows.notna().sum()
                    print(f"    从归档播种窗口表: 命中 {hit}/{len(meta)}")
            except Exception as exc:                            # noqa: BLE001
                print(f"    [!] 归档窗口表读取失败，改用 hg19: {exc}")

        if windows is None or windows.isna().any():
            if not need_locus:
                raise KeyError(
                    f"{cfg.data_dir} 的 metadata 缺少坐标列，且归档窗口表不可用；"
                    "无法构建 CRISPRon 窗口")
            cache = self._fetch_hg19(meta)
            raw = []
            for r in meta.itertuples():
                k = (str(r.Chromosome), int(r.Start), int(r.End), str(r.Strand).strip())
                w = cache.get(k, "")
                if w.startswith("__ERROR__") or len(w) != CRISPRON_LEN:
                    raw.append("")
                else:
                    raw.append(revcomp(w) if k[3] == "-" else w)
            hg = pd.Series(raw, index=meta.index)
            windows = hg if windows is None else windows.fillna(hg)

        df = meta.copy()
        df["window30"] = windows.astype(str).str.upper()
        df["note"] = ""
        # 回验：窗口中间 23 nt 必须逐字等于 sgRNA
        bad_len = df["window30"].str.len() != CRISPRON_LEN
        df.loc[bad_len, "note"] = "window_missing_or_wrong_length"
        mid = df["window30"].str.slice(UPSTREAM_NT, UPSTREAM_NT + SEQ_LEN)
        mismatch = (~bad_len) & (mid != df["sgRNA"])
        df.loc[mismatch, "note"] = "middle_23nt_mismatch"

        matched = df["note"].eq("")
        df["window30_mut"] = ""
        df.loc[matched, "window30_mut"] = [
            mutate_at(w, cfg.window_pos_index0, cfg.mutant_base)
            for w in df.loc[matched, "window30"]]
        df["eligible"] = matched & df["pos_base"].eq(cfg.target_base)

        # 突变必须只改关注位点、且不动 PAM
        if df["eligible"].any():
            e = df[df["eligible"]]
            assert (e["window30"].str.slice(UPSTREAM_NT, UPSTREAM_NT + SEQ_LEN)
                    .str[cfg.pos_index0] == cfg.target_base).all(), "eligible 行原始碱基不是目标碱基"
            assert (e["window30_mut"].str[cfg.window_pos_index0] == cfg.mutant_base).all(), "突变未生效"
            same = (e["window30"].str.slice(0, cfg.window_pos_index0)
                    == e["window30_mut"].str.slice(0, cfg.window_pos_index0)).all()
            same &= (e["window30"].str.slice(cfg.window_pos_index0 + 1)
                     == e["window30_mut"].str.slice(cfg.window_pos_index0 + 1)).all()
            assert same, "突变影响了关注位点之外的位置"
            pam_ok = e["window30_mut"].str.slice(UPSTREAM_NT + SPACER_LEN,
                                                 UPSTREAM_NT + SPACER_LEN + PAM_LEN).str.endswith("GG").all()
            assert pam_ok, "突变破坏了 PAM（NGG）"

        cols = [c for c in WINDOW_COLUMNS if c in df.columns]
        out = df[cols]
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        n_ok = int(matched.sum())
        n_el = int(df["eligible"].sum())
        print(f"    回验通过 {n_ok}/{len(df)} ({n_ok/max(len(df),1):.1%}); "
              f"其中 pos{cfg.pos_1b}={cfg.target_base} 可做 C→A: {n_el}")
        notes = df.loc[~matched, "note"].value_counts().to_dict()
        if notes:
            print(f"    未通过原因: {notes}")
        return out

    # ---------------- 取子集 ----------------
    def eligible(self, df: pd.DataFrame) -> pd.DataFrame:
        """返回可做 C18A 的序列（回验通过 且 原始碱基 == target_base），按细胞系排序。"""
        e = df[df["eligible"]].copy()
        e = e.sort_values(["cell_line", "sgRNA"]).reset_index(drop=True)
        return e
