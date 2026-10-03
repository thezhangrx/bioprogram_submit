#!/usr/bin/env bash
# =============================================================================
# run.sh — 端到端复跑脚本
#
#   读取映射 → 特征工程 → 数据 QC（接入层 + 批次层）→ 训练 → 结果汇总
#   → 分析（统计检验 / motif / 环境图 / 报告）→ 文档 + CSV + 可视化 + 交付物
#
# 用法
#   bash run.sh                 # 全流程 = prep + train + summary + analysis + deliverable
#   bash run.sh prep            # ① 读取映射 + 特征工程 + 接入层 QC（schema 对账）
#   bash run.sh train           # ② 受控实验训练（幂等：已完成的 run 自动跳过）
#   bash run.sh summary         # ③ 结果汇总 + 关键调控特征库
#   bash run.sh analysis        # ④ 分析层：Data QC、统计检验、motif、环境图、报告
#   bash run.sh deliverable     # ⑤ 赛道二交付物（CSV + Excel）
#   bash run.sh help            # 显示本帮助
#
# 可用环境变量
#   DATASETS="DeepCRISPR Hiranniramol Labuhn"   要复跑的数据集
#   LOG_DIR=logs/run_<时间戳>                    本脚本日志目录
#
# 说明
#   各流程程序按自身默认参数执行，仅显式给出三类必需参数：
#     a) feature_engineering 的 --raw-data / --output-dir / --config（无默认值，必填）；
#     b) data_digging 的 --results-dir / --model-dir / --logs-dir
#        （默认值已与交付结构一致，此处显式给出以固定落点）；
#     c) 无表观通道的数据集限定 --split-types single，以与仓库交付的 7 个 run 一致。
#   analysis 阶段几乎不打印进度，本脚本每 30 秒汇报一次"已用时间 + 产物计数"，
#   因此控制台长时间只有心跳属正常现象。
#
# 产物落点（细节见 README）
#   data/processed/<数据集>/                     张量 + feature_schema.json + 接入层 QC 汇总
#   results/train_results/<数据集>/<run>/        每次 run 的 7 个产物
#   models/<数据集>/<run>/                       模型权重与配置
#   logs/<数据集>/<run>/                          训练日志
#   results/summary/<数据集>/train_data/         批次级统一表（CSV）
#   results/summary/<数据集>/feature_importance/ 关键调控特征库
#   results/summary/<数据集>/tables/             分析表（CSV）
#   results/summary/<数据集>/reports/            报告（Markdown，含 Data QC）
#   results/summary/<数据集>/figures/            可视化（PNG）
#   results/赛道二_results/                      赛道二交付物（CSV + Excel）
#   logs/run_<时间戳>/                           本脚本各阶段日志
# =============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# -----------------------------------------------------------------------------
# 配置
# -----------------------------------------------------------------------------
DATASETS_STR="${DATASETS:-DeepCRISPR Hiranniramol Labuhn}"
read -r -a DATASETS <<< "$DATASETS_STR"

if [[ -x "./.venv/bin/python" ]]; then
  PY="./.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
elif command -v python >/dev/null 2>&1; then
  PY="python"
else
  echo "[run.sh] 未找到 Python 解释器" >&2
  exit 1
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/run_${RUN_ID}}"
HEARTBEAT_SECONDS=30          # 长任务心跳间隔（秒）
export PYTHONUNBUFFERED=1     # 日志按行落盘，便于 tail 观察
WATCH_DIR=""                  # 各阶段执行前设定，用于心跳汇报产物计数

# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------
log()  { printf '\n\033[1m[%s]\033[0m %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '\n[%s] 提示：%s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '\n[run.sh] 失败：%s\n' "$*" >&2; exit 1; }

watch_summary() {  # 采样产物计数，作为"静默长任务"的进度信号
  local d="$1"
  if [[ ! -d "$d" ]]; then printf '目录尚未创建'; return 0; fi
  printf 'csv=%s md=%s png=%s' \
    "$(find "$d" -name '*.csv' 2>/dev/null | wc -l)" \
    "$(find "$d" -name '*.md' 2>/dev/null | wc -l)" \
    "$(find "$d" -name '*.png' 2>/dev/null | wc -l)"
}

run() {  # run <日志文件> <命令...>：后台执行 + 定时心跳，结束回显退出码与日志尾部
  local logfile="$1"; shift
  local start=$SECONDS next_report=$HEARTBEAT_SECONDS elapsed
  "$@" >>"$logfile" 2>&1 &
  local pid=$!
  printf '    启动：PID %s｜日志 %s\n' "$pid" "$logfile"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1                      # 1 秒轮询：命令一结束立即继续，不被心跳周期拖住
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
    elapsed=$((SECONDS-start))
    if (( elapsed >= next_report )); then
      if [[ -n "$WATCH_DIR" ]]; then
        printf '    … 运行中 %ss｜%s 产物：%s\n' "$elapsed" "$WATCH_DIR" "$(watch_summary "$WATCH_DIR")"
      else
        printf '    … 运行中 %ss\n' "$elapsed"
      fi
      next_report=$((next_report + HEARTBEAT_SECONDS))
    fi
  done
  local rc=0
  wait "$pid" || rc=$?
  printf '    结束：用时 %ss｜退出码 %s\n' "$((SECONDS-start))" "$rc"
  if [[ -s "$logfile" ]]; then tail -n 4 "$logfile" | sed 's/^/      | /'; fi
  return "$rc"
}

ensure_summary_layout() {  # 建立交付布局的目录骨架：runs 与 summary 是同级目录，
  local ds="$1"             # 解析器据此从 summary/<数据集> 定位到 train_results/<数据集>
  mkdir -p "results/summary/${ds}/train_data" \
           "results/summary/${ds}/tables" \
           "results/summary/${ds}/reports" \
           "results/summary/${ds}/figures" \
           "results/summary/${ds}/feature_importance"
}

require_runs() {
  local ds="$1"
  [[ -d "results/train_results/${ds}" ]] || \
    die "缺少训练产物 results/train_results/${ds}（先跑 bash run.sh train）"
  local n
  n="$(find "results/train_results/${ds}" -maxdepth 1 -mindepth 1 -type d | wc -l)"
  [[ "$n" -gt 0 ]] || die "results/train_results/${ds} 下没有 run 目录"
  echo "$n"
}

has_environment() {  # 该数据集是否带表观微环境通道（读配置声明）
  "$PY" -c 'import json,sys; print(str(json.load(open(sys.argv[1])).get("has_environment", False)).lower())' \
    "data/config/$1.json"
}

summarize_ingest_qc() {  # 打印接入层 QC 汇总（只读）
  local f="$1"
  if [[ ! -f "$f" ]]; then warn "未找到 $f"; return 0; fi
  "$PY" - "$f" <<'PYEOF'
import csv, sys
path = sys.argv[1]
with open(path, encoding="utf-8") as fh:
    rows = list(csv.reader(fh))
if not rows:
    print("    (空表)"); raise SystemExit(0)
head = rows[0]
keys = ("cell_line", "source_format", "rows_in_raw", "rows_dropped_by_adapter",
        "adapter_drop_reasons", "final_samples", "channel_count", "has_epigenetics")
for row in rows[1:]:
    print("    " + " | ".join(f"{k}={v}" for k, v in zip(head, row) if k in keys))
PYEOF
}

usage() { sed -n '3,25p' "$0" | sed 's/^# \{0,1\}//'; }

preflight() {  # 解释器与依赖自检：尽早失败，而不是训练到一半报 ImportError
  log "环境自检"
  printf '  解释器：%s（Python %s）\n' "$PY" "$("$PY" -c 'import sys; print(sys.version.split()[0])')"
  "$PY" - <<'PYEOF'
import importlib.util, sys
required = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "torch",
            "matplotlib", "seaborn"]
excel_engines = ["xlsxwriter", "openpyxl"]
missing = [m for m in required if importlib.util.find_spec(m) is None]
available_excel = [m for m in excel_engines if importlib.util.find_spec(m) is not None]
if missing:
    print("  缺少必需依赖：" + ", ".join(missing))
    print("  处理方式：安装依赖后重跑（见 README §14 阶段 0：pip install -r deploy/requirements.txt）")
    sys.exit(1)
print("  依赖自检通过；Excel 写出引擎：" + (", ".join(available_excel) if available_excel
      else "无（缺 xlsxwriter / openpyxl，⑤ 交付物阶段的 xlsx 将失败）"))
PYEOF
}

# -----------------------------------------------------------------------------
# ① 读取映射 + 特征工程 + 接入层 Data QC
# -----------------------------------------------------------------------------
stage_prep() {
  log "① 读取映射 + 特征工程 + 接入层 Data QC"
  local ds
  for ds in "${DATASETS[@]}"; do
    [[ -d "data/raw/${ds}" ]] || die "缺少原始数据目录 data/raw/${ds}"
    [[ -f "data/config/${ds}.json" ]] || die "缺少配置 data/config/${ds}.json"
    log "  特征工程：${ds}（适配器由列名自动识别）"
    WATCH_DIR="data/processed/${ds}"
    run "$LOG_DIR/prep_${ds}.log" \
      "$PY" core/features/engineering/feature_engineering.py \
        --raw-data   "data/raw/${ds}" \
        --output-dir "data/processed/${ds}" \
        --config     "data/config/${ds}.json"
  done

  log "  接入层 Data QC：校验「schema 声明 == 实际张量」"
  WATCH_DIR=""
  run "$LOG_DIR/prep_validate.log" \
    "$PY" core/features/engineering/validate_feature_schema.py --all

  log "  适配统计：每细胞系原始行数 / 剔除行数 / 剔除原因 / 最终样本数 / 通道数"
  for ds in "${DATASETS[@]}"; do
    summarize_ingest_qc "data/processed/${ds}/feature_engineering_summary.csv" \
      | tee -a "$LOG_DIR/prep_summary.log"
  done
  log "① 完成：张量、feature_schema.json 与接入层 QC 汇总见 data/processed/<数据集>/"
}

# -----------------------------------------------------------------------------
# ② 受控实验训练（幂等：已完成的 run 跳过）
# -----------------------------------------------------------------------------
stage_train() {
  log "② 受控实验训练"
  local ds args
  for ds in "${DATASETS[@]}"; do
    [[ -d "data/processed/${ds}" ]] || die "缺少处理后数据 data/processed/${ds}（先跑 bash run.sh prep）"
    args=(--data-set "$ds"
          --results-dir "results/train_results/${ds}"
          --model-dir   "models/${ds}"
          --logs-dir    "logs/${ds}")
    if [[ "$(has_environment "$ds")" != "true" ]]; then
      # 无表观通道的数据集：仓库交付的批次为单细胞系划分（7 个 run）；
      # 程序默认会跑 3 种划分共 42 个 run，这里限定为 single 以与交付一致。
      args+=(--split-types single)
    fi
    log "  训练：${ds}"
    WATCH_DIR="results/train_results/${ds}"
    run "$LOG_DIR/train_${ds}.log" "$PY" data_digging.py "${args[@]}"
  done
  log "② 完成：run 产物 results/train_results/<数据集>/<run>/，模型 models/<数据集>/，日志 logs/<数据集>/"
}

# -----------------------------------------------------------------------------
# ③ 结果汇总 + 关键调控特征库
# -----------------------------------------------------------------------------
stage_summary() {
  log "③ 结果汇总 + 关键调控特征库"
  local ds n
  for ds in "${DATASETS[@]}"; do
    n="$(require_runs "$ds")"
    ensure_summary_layout "$ds"
    WATCH_DIR="results/summary/${ds}"
    log "  汇总模型指标：${ds}（${n} 个 run）"
    run "$LOG_DIR/summary_${ds}.log" \
      "$PY" -m analysis.collect_results --batch-dir "results/summary/${ds}"
    log "  关键调控特征库：${ds}"
    run "$LOG_DIR/summary_${ds}.log" \
      "$PY" analysis/importance_extraction.py --batch_dir "results/summary/${ds}"
  done
  log "③ 完成：results/summary/<数据集>/train_data/ 与 feature_importance/"
}

# -----------------------------------------------------------------------------
# ④ 分析层：批次层 Data QC + 统计检验 + motif + 环境图 + 报告
# -----------------------------------------------------------------------------
stage_analysis() {
  log "④ 分析层（含批次层 Data QC、统计检验、motif 发现、可视化与报告）"
  local ds
  for ds in "${DATASETS[@]}"; do
    require_runs "$ds" >/dev/null
    ensure_summary_layout "$ds"
    WATCH_DIR="results/summary/${ds}"
    log "  分析引擎：${ds}（本阶段几乎不打印进度，下面靠心跳与产物计数观察）"
    run "$LOG_DIR/analysis_${ds}.log" \
      "$PY" -m analysis.pipeline --batch-dir "results/summary/${ds}"
  done
  log "④ 完成：tables/（CSV）、reports/（含 data_quality.md）、figures/（PNG）"
}

# -----------------------------------------------------------------------------
# ⑤ 赛道二交付物（CSV + Excel）
# -----------------------------------------------------------------------------
stage_deliverable() {
  log "⑤ 赛道二交付物（规则发现：统计门 → 效应门 → 跨细胞系泛化门）"
  WATCH_DIR="results/赛道二_results"
  run "$LOG_DIR/deliverable_config.log" \
    "$PY" rule_discovery.py --print-config
  run "$LOG_DIR/deliverable.log" \
    "$PY" rule_discovery.py
  log "⑤ 完成：results/赛道二_results/<数据集>/{csv,excel}"
}

# -----------------------------------------------------------------------------
# 产物清单
# -----------------------------------------------------------------------------
count_dirs() { find "$1" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | wc -l; }
count_files() { find "$1" -name "$2" 2>/dev/null | wc -l; }

inventory() {
  log "产物清单"
  local ds
  for ds in "${DATASETS[@]}"; do
    printf '  %-12s runs=%-5s models=%-4s tables=%-4s reports=%-3s figures=%-4s\n' \
      "$ds" \
      "$(count_dirs "results/train_results/${ds}")" \
      "$(count_dirs "models/${ds}")" \
      "$(count_files "results/summary/${ds}/tables" '*.csv')" \
      "$(count_files "results/summary/${ds}/reports" '*.md')" \
      "$(count_files "results/summary/${ds}/figures" '*.png')"
  done
  printf '  %-12s %s 个 CSV，%s 个 Excel\n' "交付物" \
    "$(count_files "results/赛道二_results" '*.csv')" \
    "$(count_files "results/赛道二_results" '*.xlsx')"
  printf '  %-12s %s\n' "脚本日志" "$LOG_DIR"
}

# -----------------------------------------------------------------------------
# 入口
# -----------------------------------------------------------------------------
main() {
  local stage="${1:-all}"
  case "$stage" in
    help|-h|--help) usage; exit 0 ;;
  esac
  mkdir -p "$LOG_DIR"
  log "run.sh 启动 | 解释器=${PY} | 数据集=${DATASETS[*]} | 日志=${LOG_DIR}"
  preflight
  case "$stage" in
    prep)        stage_prep ;;
    train)       stage_train ;;
    summary)     stage_summary ;;
    analysis)    stage_analysis ;;
    deliverable) stage_deliverable ;;
    all)
      stage_prep
      stage_train
      stage_summary
      stage_analysis
      stage_deliverable
      ;;
    *) usage; die "未知阶段：$stage" ;;
  esac
  inventory
  log "全部完成"
}

main "$@"
