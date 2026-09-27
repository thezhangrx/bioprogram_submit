# 00 Overview

- experiment count: **1344**
- valid experiments: **1344**
- models: cnn(3|3), cnn(5|3), cnn(7|3), linear, mlp, transformer, xgboost
- cell lines: hct116, hek293t, hela, hl60, none
- environments: 16
- splits: all, mixed, single
- metric inconsistency (ΔR²/ΔRMSE 同号) rows: **0**

## Analysis Plan (实际选择)

- ✓ qc [selected=True, available=True, status=completed]
- ✓ prediction [selected=True, available=True, status=completed]
- ✓ environment_conditional_effect [selected=True, available=True, status=completed]  (tables/environment_conditional_delta_r2.csv (2016 条件增量行 × 3 指标))
- ✓ environment_main_effect [selected=True, available=True, status=completed]
- ✓ environment_factorial_dag [selected=True, available=True, status=completed]  (63 DAG, 63 conditional-ΔR², 63 interaction figures (ablation view 已移出交付范围 2026-09-25))
- ✓ sequence_attribution [selected=True, available=True, status=completed]
- ✓ cnn_ism [selected=True, available=True, status=completed]
- ✓ motif_discovery [selected=True, available=True, status=completed]  (tables/motif_candidates.csv (656 motifs; 559773 seqlets))
- ✓ motif_enrichment [selected=True, available=True, status=completed]  (tables/motif_enrichment.csv (656 tests, 136 pass FDR<0.05))
- ✓ bootstrap [selected=True, available=True, status=completed]  (tables/bootstrap_results.csv (2633 ΔR² CIs); paired per-sample bootstrap)
- ✓ hypothesis_testing [selected=True, available=True, status=completed]  (tables/permutation_results.csv (3378 tests, BH-FDR by family))
- ✓ fdr_correction [selected=True, available=True, status=completed]  (BH-FDR applied to 189 p-value families (不与其他问题混用 family))
- ✓ environment_anova [selected=True, available=True, status=completed]  (tables/anova_results.csv (38 terms))

## 执行说明

- 本报告由统一 experiment table 生成; 所有图/表引用同一数据源。
- PFI/LOFO 等若训练端未生成, 一律标记 Unavailable, 不回头修改训练系统。
