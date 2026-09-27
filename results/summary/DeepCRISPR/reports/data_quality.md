# 01 Data Quality (批次级)

- experiments: 1344 | valid (R² finite): 1344

## cell-line × environment 覆盖

| cell_line | all | sequence | sequence_ctcf | sequence_ctcf_dnase | sequence_ctcf_dnase_h3k4me3 | sequence_ctcf_dnase_rrbs | sequence_ctcf_h3k4me3 | sequence_ctcf_h3k4me3_rrbs | sequence_ctcf_rrbs | sequence_dnase | sequence_dnase_h3k4me3 | sequence_dnase_h3k4me3_rrbs | sequence_dnase_rrbs | sequence_h3k4me3 | sequence_h3k4me3_rrbs | sequence_rrbs |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hct116 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 |
| hek293t | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 |
| hela | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 |
| hl60 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 | 14 |
| mixed | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 | 28 |

## Metric inconsistency (同 cohort ΔR²/ΔRMSE 同号)

_无_

> Eligible/Limited/Ineligible 的逐环境判定由训练前的数据质控流程输出（本仓库未随附该程序）;
此处为批次级状态 (训练后只读)。
