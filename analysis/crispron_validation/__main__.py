# -*- coding: utf-8 -*-
"""analysis.crispron_validation.__main__ — 命令行入口。

用法::

    python -m analysis.crispron_validation                    # 全流程
    python -m analysis.crispron_validation --steps windows
    python -m analysis.crispron_validation --steps crispron
    python -m analysis.crispron_validation --steps model
    python -m analysis.crispron_validation --steps table
    python -m analysis.crispron_validation --force            # 忽略已有缓存，全部重跑
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

from .config import CrispronValidationConfig
from .pipeline import STEPS, CrispronValidationPipeline


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.crispron_validation",
        description="CNN7（四细胞系全数据训练）× CRISPRon 的 Pos18 C→A 一致性验证",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--steps", nargs="+", default=None, choices=list(STEPS),
                   help=f"只跑指定步骤（默认全部：{' '.join(STEPS)}）")
    p.add_argument("--force", action="store_true",
                   help="忽略已有缓存与已训模型，全部重跑")
    p.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数")
    p.add_argument("--seed", type=int, default=None, help="覆盖随机种子")
    p.add_argument("--sequence-kernel", type=int, default=None, choices=[3, 5, 7],
                   help="覆盖序列核长度（默认 7）")
    p.add_argument("--http-workers", type=int, default=8,
                   help="hg19 抓取并发（仅在没有窗口缓存时需要）")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    cfg = CrispronValidationConfig()
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    if args.seed is not None:
        cfg.random_seed = int(args.seed)
    if args.sequence_kernel is not None:
        cfg.sequence_kernel = int(args.sequence_kernel)
        cfg.run_name = f"cnn{cfg.sequence_kernel}_alltrain_sequence"

    print(cfg.describe())
    print(f"  工作目录: {cfg.work_dir}")
    print(f"  交付目录: {cfg.summary_dir}")
    print()
    CrispronValidationPipeline(cfg).run(steps=args.steps, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
