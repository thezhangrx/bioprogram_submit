"""analysis.pipeline — 证据分析引擎编排入口 (只编排; 科学计算在各模块)。

用法:
    python -m analysis.pipeline --batch-dir <batch> --output <out>
    python -m analysis.pipeline --batch-dir <batch> --analysis-plan plan.json
"""
from __future__ import annotations


# --- 项目根引导: 保证从任意工作目录运行/被导入都能解析 core、analysis、workflows ---
import sys as _sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import json

import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from analysis import __version__ as ENGINE_VERSION
from analysis.config import AnalysisConfig
from analysis.data.loaders import coverage_summary, load_experiment_table
from analysis.data.validation import validate_metric_consistency
from analysis.plans import (AnalysisPlan, Capabilities, ExecutionPlan, ExecutionTask,
                           default_plan, validate_analysis_plan)
from analysis.reports.markdown_report import (build_anomaly_md, build_data_quality_md,
                                             build_overview_md)


@dataclass
class PipelineArtifacts:
    output_dir: Path
    tables_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def create(cls, output: str | Path) -> "PipelineArtifacts":
        """产物根目录下的三个子目录。

        2026-09-13 起产物根为 <batch>/summary（旧为 <batch>/analyse_out），
        为避免出现 summary/summary 套娃，报告子目录命名为 reports/：
            <batch>/summary/tables    全部分析表 CSV
            <batch>/summary/reports   overview.md / data_quality.md / anomaly_report.md
            <batch>/summary/figures   分析引擎生成的图
        """
        out = Path(output)
        for sub in ("tables", "reports", "figures"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        return cls(out, out / "tables", out / "reports", out / "figures")


def sort_for_output(df: pd.DataFrame, keys) -> pd.DataFrame:
    """按给定键稳定排序后落盘，保证同一批次重复运行得到逐字节相同的 CSV。

    配对枚举走的是 ``set``（字符串哈希随机化），不排序会让
    ``bootstrap_results.csv`` / ``permutation_results.csv`` 的行序在两次运行间漂移。
    """
    if df is None or df.empty:
        return df
    present = [k for k in keys if k in df.columns]
    if not present:
        return df
    return df.sort_values(present, kind="mergesort").reset_index(drop=True)


def resolve_motif_data_root(runs_root: Path) -> Optional[str]:
    """按 runs 根的数据集名定位 ``data/processed/<数据集>``（motif 需要 `<cell>_metadata.csv`）。

    交付结构下 runs 根即 ``results/train_results/<数据集>``，与 ``data/processed/<数据集>`` 同名；
    非数据集名（旧布局 A 的批次名）时返回 ``None``，退回 ``MotifDiscoveryConfig.data_root``。
    """
    from core.common.paths import available_datasets, resolve_data_dir

    name = Path(runs_root).name
    if name in available_datasets():
        return str(resolve_data_dir(data_set=name))
    return None


def capabilities_from_table(table: pd.DataFrame, has_environment_cols: bool = True,
                            batch_dir: Optional[str | Path] = None) -> Capabilities:
    def contains(keyword: str) -> bool:
        if "model" not in table.columns:
            return False
        return bool(table["model"].astype(str).str.lower().str.contains(keyword).any())

    return Capabilities(
        n_cell_lines=int(table["cell_line"].nunique()) if "cell_line" in table.columns else 0,
        n_environment_factors=4 if has_environment_cols else 0,
        n_models=int(table["model"].nunique()) if "model" in table.columns else 0,
        has_environment_features=has_environment_cols,
        has_cnn=contains("cnn"),
        has_transformer=contains("trans"),
        has_xgboost=contains("xgb"),
        has_mlp=contains("mlp"),
        has_linear=contains("linear"),
        environment_factorial_observations=16 if has_environment_cols else 0,
        replication_per_cellline=4,
        has_bootstrap_results=bool(_has_paired_predictions(batch_dir, table)),
    )


def _has_paired_predictions(batch_dir: Optional[str | Path], table: pd.DataFrame,
                            sample_limit: int = 6) -> bool:
    """探测是否存在 per-sample 预测 artifact (bootstrap/permutation 的真实前置条件)。"""
    if batch_dir is None or table is None or table.empty or "run_name" not in table.columns:
        return False
    from analysis.stats.tasks import predictions_path
    for run in list(table["run_name"].dropna().astype(str).unique())[:sample_limit]:
        if predictions_path(batch_dir, run) is not None:
            return True
    return False


#: 位置归因热图显示门的输入（显著性/稳健性特征库）。
SIGNIFICANCE_TABLE_REL = "feature_importance/key_regulatory_biomarkers.csv"


def _load_significance_table(batch_dir: Optional[str | Path]) -> Optional[pd.DataFrame]:
    """读取显著性特征库，供 `figures/plots/` 做显示门（FDR>=0.10 / SNR<0.8 留白）。

    读不到就返回 None（显示门自动关闭），**不阻断**出图——图画少一层过滤总比整组图失败好。
    """
    if batch_dir is None:
        return None
    path = Path(batch_dir) / SIGNIFICANCE_TABLE_REL
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[!] 读取显著性表失败（{(path)}）：{exc}；位置归因热图将不做显示门")
        return None


def run_analysis(
    batch_dir: str | Path,
    output: str | Path,
    plan: Optional[AnalysisPlan] = None,
    config: Optional[AnalysisConfig] = None,
) -> Dict[str, Any]:
    """Phase-2 垂直切片: 统一表 -> 校验 -> overview/status/plan 产物。

    后续阶段将在此编排器中挂载 environment/sequence/cell-line/evidence 任务,
    每个任务保持 selected/available/status 三态, GUI 只读 analysis_status.json。
    """
    plan = plan or default_plan()
    config = config or AnalysisConfig()
    artifacts = PipelineArtifacts.create(output)

    # 0) 解析两棵子树：当前交付结构把 runs 与 summary 拆成同级目录
    #    （results/train_results/<数据集> 与 results/summary/<数据集>）。
    #    runs_root 供所有"扫 <run>/ 目录"的地方使用（归因/配对预测）；
    #    load_experiment_table / write_factorial_dag_artifacts 内部自带两布局识别。
    from analysis.data.loaders import resolve_runs_root

    runs_root = resolve_runs_root(batch_dir)

    # 1) 统一实验表 (只读)
    #    先探测两个来源（候选指标表 / 目录扫描）。两者都拿不到数据时，本函数**不再
    #    print 一行"跳过"就继续**（那样报告里的"_无_"会被误读成"检查过了没问题"），
    #    而是照常生成全部产物，并把 [!] 未实现检测跳过 写进 reports/data_quality.md。
    from analysis.data.loaders import (empty_experiment_table,
                                       probe_experiment_sources)
    from analysis.reports.markdown_report import ANOMALY_DETECTION_SKIPPED

    detection_note: str | None = None
    sources = probe_experiment_sources(batch_dir)
    if sources["available"]:
        table = load_experiment_table(batch_dir)
    elif sources["errors"]:
        # 表存在但读不动：这是真错误，不要降级成"跳过"
        raise ValueError(sources["errors"][0])
    else:
        table = empty_experiment_table()
        detection_note = ANOMALY_DETECTION_SKIPPED
    table.to_csv(artifacts.tables_dir / "experiment_table.csv", index=False)

    # 2) 批次级校验
    anomalies = validate_metric_consistency(table)
    anomalies.to_csv(artifacts.tables_dir / "metric_inconsistency.csv", index=False)

    # 3) availability -> execution plan
    caps = capabilities_from_table(table, has_environment_cols=True, batch_dir=runs_root)
    task_states = validate_analysis_plan(plan, caps)
    execution_tasks: List[ExecutionTask] = [
        ExecutionTask(
            task_id=st["task_id"], selected=bool(st["selected"]),
            available=bool(st["available"]), status=str(st["status"]),
            reason=str(st.get("reason", "")),
        )
        for st in task_states.values()
    ]

    # 4) 执行 (本构建已实现: qc/prediction/environment conditional+main)
    from analysis.environment.incremental_effect import (compute_conditional_increments,
                                                        compute_main_effects,
                                                        summarize_conditional)
    from analysis.prediction import loco_performance, performance_by_model_split

    executed: Dict[str, str] = {}
    completed_at = datetime.now(timezone.utc).isoformat()

    task_artifacts = {
        "qc": "reports/data_quality.md",
        "prediction": "tables/prediction_summary.csv",
        "environment_conditional_effect": "tables/environment_conditional_delta_r2.csv",
        "environment_main_effect": "tables/environment_main_effects.csv",
        "environment_factorial_dag": "tables/environment_edges.csv",
        "bootstrap": "tables/bootstrap_results.csv",
        "hypothesis_testing": "tables/permutation_results.csv",
        "fdr_correction": "tables/permutation_results.csv",
        "environment_anova": "tables/anova_results.csv",
        "sequence_attribution": "tables/attribution_summary.csv",
        "motif_discovery": "tables/motif_candidates.csv",
        "motif_enrichment": "tables/motif_enrichment.csv",
    }

    def _finish(task_id: str, status: str = "completed", reason: str = ""):
        for task in execution_tasks:
            if task.task_id == task_id:
                task.status = status
                # reason 语义：**只在没做成时**才有意义（skipped/unavailable/failed 的原因）。
                # 成功时不传 reason 就**清空**，否则会沿用 `validate_analysis_plan` 里
                # 那次 availability 探测的说明，产出
                # "status=completed (environment features unavailable)" 这种自相矛盾的报告行。
                task.reason = reason or ("" if status == "completed" else task.reason)
                task.completed_at = completed_at
                if task_id in task_artifacts:
                    task.artifact = task_artifacts[task_id]
                executed[task_id] = status
                return

    # 仅当 selected & available 才执行; 否则保持 skipped/unavailable
    state = {t.task_id: t for t in execution_tasks}

    def _should_run(task_id: str) -> bool:
        t = state.get(task_id)
        return bool(t and t.status == "pending" and t.selected and t.available)

    if _should_run("qc"):
        # 产物照常生成；但检测确实没跑成时，任务状态标 unavailable 并把原因写进
        # overview.md（而不是只在 stdout 打一行"跳过"）。
        if detection_note:
            _finish("qc", "unavailable",
                    f"异常/一致性检测无可用指标数据（{detection_note}）；"
                    f"缺口已记入 reports/data_quality.md")
        else:
            _finish("qc")

    prediction_df, loco_df = None, None
    prediction_ok = _should_run("prediction")
    if prediction_ok:
        try:
            prediction_df = performance_by_model_split(table)
            loco_df = loco_performance(table)
            prediction_df.to_csv(artifacts.tables_dir / "prediction_summary.csv", index=False)
            loco_df.to_csv(artifacts.tables_dir / "loco_performance.csv", index=False)
            _finish("prediction")
        except Exception as exc:  # noqa: BLE001
            _finish("prediction", "failed", f"prediction analysis failed: {exc}")

    cond_summary, main_df = None, None
    if _should_run("environment_conditional_effect"):
        try:
            cond_raw = compute_conditional_increments(table)
            cond_summary = summarize_conditional(cond_raw)
            cond_summary.to_csv(artifacts.tables_dir / "environment_conditional_delta_r2.csv",
                                index=False)
            _finish("environment_conditional_effect", "completed",
                    f"tables/environment_conditional_delta_r2.csv "
                    f"({len(cond_summary)} 条件增量行 × 3 指标)")
        except Exception as exc:  # noqa: BLE001
            _finish("environment_conditional_effect", "failed",
                    f"conditional effect failed: {exc}")

    if _should_run("environment_main_effect"):
        if cond_raw is None:
            _finish("environment_main_effect", "unavailable",
                    "environment_main_effect requires conditional effect data")
            main_df = None
        else:
            try:
                main_df = compute_main_effects(cond_raw)
                main_df.to_csv(artifacts.tables_dir / "environment_main_effects.csv", index=False)
                _finish("environment_main_effect")
            except Exception as exc:  # noqa: BLE001
                _finish("environment_main_effect", "failed", f"main effect failed: {exc}")

    # ------------------------------------------------------------------
    # 统计工具接线 (bootstrap / permutation / BH-FDR / ANOVA) —— 只读已有产物
    # 顺序: 必须先于 factorial DAG (edges 要带 CI) 与 evidence matrix (CI 参与 tier)
    # ------------------------------------------------------------------
    stats_bundle = {"bootstrap": None, "permutation": None, "anova": None,
                    "main_ci": None, "cellline_ci": None, "fdr": None}
    if _should_run("bootstrap"):
        try:
            from analysis.stats.tasks import (bootstrap_cellline_effects,
                                              bootstrap_environment_edges,
                                              bootstrap_main_effects)
            bs = bootstrap_environment_edges(runs_root, table, config=config)
            bs = sort_for_output(bs, ("split_type", "cell_line", "model", "random_seed",
                                      "parent_combination", "child_combination", "metric"))
            bs.to_csv(artifacts.tables_dir / "bootstrap_results.csv", index=False)
            main_ci = bootstrap_main_effects(main_df, config=config) if main_df is not None \
                else None
            if main_ci is not None:
                main_ci.to_csv(artifacts.tables_dir / "bootstrap_main_effects.csv", index=False)
            cell_ci = bootstrap_cellline_effects(main_df, config=config) if main_df is not None \
                else None
            if cell_ci is not None:
                cell_ci.to_csv(artifacts.tables_dir / "bootstrap_cellline_effects.csv",
                               index=False)
            stats_bundle.update({"bootstrap": bs, "main_ci": main_ci, "cellline_ci": cell_ci})
            n_ci = int(((bs["metric"] == "R2") & (bs["status"] == "ok")).sum())
            _finish("bootstrap", "completed",
                    f"tables/bootstrap_results.csv ({n_ci} ΔR² CIs); "
                    f"paired per-sample bootstrap")
        except Exception as exc:  # noqa: BLE001
            _finish("bootstrap", "failed", f"bootstrap wiring failed: {exc}")

    if _should_run("hypothesis_testing"):
        try:
            from analysis.stats.tasks import (apply_fdr, permutation_environment_edges,
                                             permutation_interactions,
                                             permutation_main_effects)
            parts = [permutation_environment_edges(runs_root, table, config=config)]
            if cond_summary is not None:
                parts.append(permutation_main_effects(cond_summary, config=config))
            parts.append(permutation_interactions(runs_root, table, config=config))
            perm = pd.concat([p for p in parts if p is not None and not p.empty],
                             ignore_index=True) if any(
                p is not None and not p.empty for p in parts) else pd.DataFrame()
            perm = apply_fdr(perm, config=config)
            perm = sort_for_output(perm, ("test_type", "split_type", "cell_line", "model",
                                          "random_seed", "parent_combination",
                                          "child_combination", "factor"))
            perm.to_csv(artifacts.tables_dir / "permutation_results.csv", index=False)
            stats_bundle["permutation"] = perm
            stats_bundle["fdr"] = perm
            n_ok = int((perm["status"] == "ok").sum()) if not perm.empty else 0
            _finish("hypothesis_testing", "completed",
                    f"tables/permutation_results.csv ({n_ok} tests, BH-FDR by family)")
        except Exception as exc:  # noqa: BLE001
            _finish("hypothesis_testing", "failed", f"permutation wiring failed: {exc}")

    anova_enabled = bool(getattr(plan.statistics, "anova", True))
    if state.get("environment_anova") is not None:
        if not anova_enabled or not plan.environment.anova:
            _finish("environment_anova", "skipped", "user_disabled")
        elif _should_run("environment_anova"):
            try:
                from analysis.stats.tasks import run_anova_tasks
                anova = run_anova_tasks(table, config=config)
                anova.to_csv(artifacts.tables_dir / "anova_results.csv", index=False)
                stats_bundle["anova"] = anova
                ok = int((anova["status"] == "ok").sum()) if not anova.empty else 0
                if ok == 0:
                    reason = (anova["reason"].iloc[0] if not anova.empty else "no estimable term")
                    _finish("environment_anova", "unavailable",
                            f"insufficient_data: {reason}")
                else:
                    _finish("environment_anova", "completed",
                            f"tables/anova_results.csv ({ok} terms)")
            except Exception as exc:  # noqa: BLE001
                _finish("environment_anova", "failed", f"ANOVA failed: {exc}")

    # FDR 不是独立分析: 只有存在可校正 family 时才执行 (否则如实记录)
    if state.get("fdr_correction") is not None and state["fdr_correction"].selected:
        perm_df = stats_bundle.get("permutation")
        n_families = 0
        if perm_df is not None and not perm_df.empty and "fdr_status" in perm_df.columns:
            n_families = int(perm_df.loc[perm_df["fdr_status"] == "ok", "fdr_family"].nunique())
        if n_families > 0:
            _finish("fdr_correction", "completed",
                    f"BH-FDR applied to {n_families} p-value families "
                    f"(不与其他问题混用 family)")
        else:
            _finish("fdr_correction", "unavailable",
                    "no corrigible p-value family available in this batch")

    # Environment Factorial DAG (2^4 lattice): normalized table -> nodes/edges -> figures.
    # 必须在下方的 "pending -> unavailable" 汇总之前执行。
    if _should_run("environment_factorial_dag"):
        try:
            from analysis.environment.factorial_dag import write_factorial_dag_artifacts
            from analysis.visualization.factorial_dag import (
                render_conditional_delta_r2, render_environment_interactions,
                render_factorial_dag)
            dag_paths = write_factorial_dag_artifacts(
                batch_dir, artifacts.tables_dir,
                threshold=config.consensus.unstable_effect_threshold)
            dag_nodes = pd.read_csv(dag_paths["nodes"])
            dag_edges = pd.read_csv(dag_paths["edges"])
            env_fig_dir = artifacts.figures_dir / "environment"   # 交付结构定稿 2026-09-25
            dag_figs = render_factorial_dag(dag_nodes, dag_edges, env_fig_dir)
            dag_delta_figs = render_conditional_delta_r2(dag_edges, env_fig_dir)
            dag_int_figs = render_environment_interactions(
                pd.read_csv(dag_paths["interactions"]), env_fig_dir)
            _finish("environment_factorial_dag", "completed",
                    f"{len(dag_figs)} DAG, {len(dag_delta_figs)} conditional-ΔR², "
                    f"{len(dag_int_figs)} interaction figures "
                    f"(ablation view 已移出交付范围 2026-09-25)")
        except Exception as exc:  # noqa: BLE001
            _finish("environment_factorial_dag", "failed", f"factorial DAG failed: {exc}")

    attribution_table = None
    if _should_run("sequence_attribution"):
        try:
            from analysis.attribution.extractors import extract_attribution_table
            attribution_table = extract_attribution_table(runs_root)
            # tables/attribution_summary.csv 是 **motif discovery 的输入**，必须保留。
            # 注：该归因的**报告**（旧 04_sequence_motifs.md 与新 04_sequence_and_motifs.md）
            # 均已移出交付范围（2026-09-25），只保留本 CSV 与其下游 motif 产物。
            attribution_table.to_csv(artifacts.tables_dir / "attribution_summary.csv", index=False)
            _finish("sequence_attribution")
            has_cnn_rows = bool(
                attribution_table["model"].astype(str).str.contains("cnn").any()
                if not attribution_table.empty else False)
            if _should_run("cnn_ism") and has_cnn_rows:
                _finish("cnn_ism")
            elif state.get("cnn_ism") and state["cnn_ism"].selected and not has_cnn_rows:
                _finish("cnn_ism", "unavailable", "no CNN importance outputs in batch")
        except Exception as exc:  # noqa: BLE001
            _finish("sequence_attribution", "failed", f"attribution extraction failed: {exc}")

    # Sequence Motif Discovery (attribution -> seqlet -> clustering -> consensus -> enrichment)
    motif_result: Dict[str, object] = {}
    if state.get("motif_discovery") is not None:
        seq_state = state["motif_discovery"]
        if not plan.sequence.motif_discovery:
            _finish("motif_discovery", "skipped", "user_disabled")
        elif seq_state.status == "pending":
            try:
                from analysis.sequence.motif.pipeline import run_and_write
                # 只产数据表（motif_candidates / motif_instances / motif_enrichment）；
                # figures/motif/ 与 motif_evidence.csv 已随人为标签体系移出。
                motif_result = run_and_write(
                    attribution_table if attribution_table is not None else pd.DataFrame(),
                    runs_root, artifacts.tables_dir,
                    data_root=resolve_motif_data_root(Path(runs_root)),
                    config=config, discovery=True,
                    enrichment=bool(plan.sequence.motif_enrichment))
                n_motifs = int(motif_result.get("summary", {}).get("n_motifs", 0))
                if n_motifs == 0:
                    _finish("motif_discovery", "unavailable",
                            "; ".join(motif_result.get("unavailable_reasons", []))
                            or "no motif passed support thresholds")
                else:
                    _finish("motif_discovery", "completed",
                            f"tables/motif_candidates.csv ({n_motifs} motifs; "
                            f"{motif_result.get('summary', {}).get('n_seqlets')} seqlets)")
                enr_state = state.get("motif_enrichment")
                if enr_state is not None:
                    if not plan.sequence.motif_enrichment:
                        _finish("motif_enrichment", "skipped", "user_disabled")
                    elif n_motifs == 0:
                        _finish("motif_enrichment", "unavailable",
                                "motif_enrichment requires motif candidates")
                    else:
                        n_ok = sum(1 for e in motif_result.get("enrichment", [])
                                   if e.get("status") == "ok")
                        n_sig = sum(1 for e in motif_result.get("enrichment", [])
                                    if e.get("FDR") is not None
                                    and float(e["FDR"]) < config.motif.enrichment_fdr)
                        _finish("motif_enrichment", "completed",
                                f"tables/motif_enrichment.csv ({n_ok} tests, "
                                f"{n_sig} pass FDR<{config.motif.enrichment_fdr})")
            except Exception as exc:  # noqa: BLE001
                _finish("motif_discovery", "failed", f"motif discovery failed: {exc}")

    # 数值不稳定上下文行数（数据事实，供 anomaly_report 记录；
    # 原先还兼作"证据矩阵排除数"，证据矩阵本身已随 Tier 移出）
    matrix_excluded = int(
        (pd.to_numeric(main_df["main_r2_delta"], errors="coerce").abs()
         >= config.consensus.unstable_effect_threshold).sum()
    ) if (main_df is not None and "main_r2_delta" in main_df.columns) else 0

    # 尚未实现但被选中的任务: 明确 unavailable (绝不伪造)
    for task in execution_tasks:
        if task.status == "pending" and task.selected:
            task.status = "unavailable"
            task.reason = "analysis module not yet implemented in this build (phase roadmap)"
            task.completed_at = completed_at
            executed[task.task_id] = "unavailable"

    ep = ExecutionPlan(plan=plan, tasks=execution_tasks)
    ep.completed_at = datetime.now(timezone.utc).isoformat()

    # 5) 产物
    coverage = coverage_summary(table)
    status_dict = ep.to_status_dict()
    status_dict.update({
        "engine_version": ENGINE_VERSION,
        "batch": str(batch_dir),
        "runs_root": str(runs_root),
        "config": config.as_dict(),
    })
    plan.save(artifacts.output_dir / "analysis_plan.json")
    ep.save_status(artifacts.output_dir / "analysis_status.json")

    overview_md = build_overview_md(coverage, plan, status_dict, anomalies)
    (artifacts.reports_dir / "overview.md").write_text(overview_md, encoding="utf-8")

    # 批次级 qc 报告 (01/08 + anomaly_report.csv; 机器可读见 tables/)
    anomaly_report = anomalies.copy() if anomalies is not None else pd.DataFrame()
    if not anomaly_report.empty:
        if "flags" in anomaly_report.columns:
            anomaly_report.insert(0, "anomaly_type", anomaly_report["flags"].map(
                lambda f: (f.split(";")[0].rsplit("_", 1)[0]
                           if isinstance(f, str) and f else "metric_inconsistency")))
        anomaly_report.to_csv(artifacts.tables_dir / "anomaly_report.csv", index=False)
    else:
        pd.DataFrame(columns=["anomaly_type", "row", "experiment", "flags",
                              "delta_r2", "delta_rmse"]) \
            .to_csv(artifacts.tables_dir / "anomaly_report.csv", index=False)
    (artifacts.reports_dir / "data_quality.md").write_text(
        build_data_quality_md(coverage, table, anomalies, detection_note),
        encoding="utf-8")
    (artifacts.reports_dir / "anomaly_report.md").write_text(
        build_anomaly_md(anomalies, matrix_excluded, detection_note), encoding="utf-8")

    (artifacts.output_dir / "execution_log.json").write_text(
        json.dumps({"executed": executed,
                    "completed_at": ep.completed_at,
                    "status_counts": {
                        s: sum(1 for t in execution_tasks if t.status == s)
                        for s in {t.status for t in execution_tasks}
                    }}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 6) 图形渲染 + Importance–ΔR² 二维证据图 (只消费上面已落盘的表格数据; 失败不阻断已完成的科学计算)
    if executed.get("visualization") is None:
        execution_log_path = artifacts.output_dir / "execution_log.json"

        def _patch_log(key: str, value: str, refresh_figures: bool = True) -> None:
            log = json.loads(execution_log_path.read_text(encoding="utf-8"))
            log.setdefault("executed", {})[key] = value
            if refresh_figures:
                log["figures"] = sorted(
                    str(p.relative_to(artifacts.figures_dir))
                    for p in artifacts.figures_dir.rglob("*.png"))
            execution_log_path.write_text(
                json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")

        # 6a) Importance / Attribution – ΔR² 已**停用**（2026-09-20）。
        #     该分析产出的 4 个文件（tables/importance_vs_delta_r2.csv、
        #     summary/07_importance_vs_delta_r2.md、summary/importance_metric_selection.md、
        #     summary/importance_delta_r2_summary.json）与 figures/07_evidence/*.png
        #     已从交付物中移除，对应程序 analysis/visualization/importance_delta.py 与
        #     analysis/importance_metrics.py 已移入 Delete/_moved_paper/。
        #     保留此注释是为了让后来者知道这里曾有一个分析步骤、为何不再运行。

        # 6c) 常规图组
        try:
            from analysis.visualization import render_all
            render_all(
                artifacts.figures_dir,
                conditional_summary=cond_summary, main_effects=main_df,
                attribution_table=attribution_table,
                significance_table=_load_significance_table(batch_dir))
            executed["visualization"] = "completed"
            _patch_log("visualization", "completed")
        except Exception as exc:  # noqa: BLE001
            executed["visualization"] = f"failed: {exc}"
            _patch_log("visualization", executed["visualization"], refresh_figures=False)
    return status_dict


def main() -> None:
    ap = argparse.ArgumentParser(description="CRISPR evidence analysis engine (analysis.pipeline)")
    ap.add_argument("--batch-dir", type=str, required=True, help="训练结果批次目录")
    ap.add_argument("--output", type=str, default="",
                    help="输出目录 (默认：布局 A 为 <batch>/summary；"
                         "若 --batch-dir 本身就是 summary/<数据集> 则就地写回)")
    ap.add_argument("--analysis-plan", type=str, default="", help="可选 AnalysisPlan json")
    args = ap.parse_args()

    from analysis.data.loaders import resolve_summary_root

    batch = Path(args.batch_dir)
    output = Path(args.output) if args.output else resolve_summary_root(batch)
    plan = AnalysisPlan.load(args.analysis_plan) if args.analysis_plan else default_plan()
    status = run_analysis(batch_dir=batch, output=output, plan=plan)
    print(f"[✓] pipeline finished -> {output}")
    print(f"    tasks: " + "; ".join(
        f"{t['task_id']}={t['status']}" for t in status.get("tasks", [])
    ))


if __name__ == "__main__":
    main()
