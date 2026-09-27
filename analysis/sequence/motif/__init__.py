"""analysis.sequence.motif — Sequence Motif Discovery (从已有 attribution 提取候选 motif)。

流程 (严格按任务书):
    attribution (CNN ISM/IG; Transformer IG 若存在)
        -> seqlet extraction (高归因 + 局部连续窗口, 非 top-N 单点)
        -> positive/negative separation (有符号 attribution 时; 否则方向由 carrier/background
           measured efficacy 对比给出, direction_source 显式记录)
        -> clustering (sequence similarity + attribution similarity)
        -> motif candidate (support / consensus / IUPAC / human pattern / regex)
        -> position & region
        -> enrichment (可选, Fisher exact + BH-FDR, 明确 background)
        -> cross-model & cross-kernel evidence

只读已有 artifact; 不重新训练, 不重算 XAI。
"""
from analysis.sequence.motif.iupac import (consensus_from_frequencies, human_pattern,  # noqa: F401
                                          information_content, iupac_code, matches_iupac,
                                          pfm_from_instances, regex_pattern)
