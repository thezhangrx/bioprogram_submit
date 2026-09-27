"""analysis.plans — AnalysisPlan / Validator / ExecutionPlan / 状态。

三层状态模型:
    selected  用户是否选择
    available 数据/依赖是否支持
    status    实际执行状态 (pending/running/completed/skipped/unavailable/failed)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"


@dataclass
class EnvironmentPlan:
    enabled: bool = True
    conditional_effect: bool = True
    main_effect: bool = True
    interaction: bool = False
    anova: bool = False


@dataclass
class SequencePlan:
    enabled: bool = True
    position_attribution: bool = True
    ism: bool = True
    motif_discovery: bool = True
    motif_enrichment: bool = False


@dataclass
class StatisticsPlan:
    bootstrap: bool = True              # paired ΔR² bootstrap CI
    hypothesis_testing: bool = True      # permutation test (真实零假设)
    permutation_test: bool = True        # 与 hypothesis_testing 同义, 显式别名
    fdr_correction: bool = True          # 只有在存在可校正 p-value family 时才执行
    anova: bool = True                   # factorial ANOVA (与 environment.anova 同时为真才执行)


@dataclass
class AnalysisPlan:
    """用户/向导生成的方案 (只描述"做什么")。"""

    run_qc: bool = True
    run_prediction_analysis: bool = True
    environment: EnvironmentPlan = field(default_factory=EnvironmentPlan)
    sequence: SequencePlan = field(default_factory=SequencePlan)
    statistics: StatisticsPlan = field(default_factory=StatisticsPlan)

    def to_json_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def _sub_plan(cls, klass, data: Any):
        """用已知字段构造子计划；**未知键直接忽略**。

        这样旧版 `analysis_plan.json`（还带 `cell_line` / `evidence` 段，
        或将来新增过又删掉的开关）仍能加载，而不是抛 TypeError。
        """
        if not isinstance(data, dict):
            return klass()
        known = {f.name for f in fields(klass)}
        return klass(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisPlan":
        return cls(
            run_qc=bool(data.get("run_qc", True)),
            run_prediction_analysis=bool(data.get("run_prediction_analysis", True)),
            environment=cls._sub_plan(EnvironmentPlan, data.get("environment")),
            sequence=cls._sub_plan(SequencePlan, data.get("sequence")),
            statistics=cls._sub_plan(StatisticsPlan, data.get("statistics")),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisPlan":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def default_plan() -> AnalysisPlan:
    return AnalysisPlan()


# ---------------------------------------------------------------------------
# Availability / Capability
# ---------------------------------------------------------------------------
@dataclass
class Capabilities:
    """由实际数据推导的可用性 (不随用户选择改变)。"""

    n_cell_lines: int = 0
    n_environment_factors: int = 0
    n_models: int = 0
    has_environment_features: bool = False
    has_cnn: bool = False
    has_transformer: bool = False
    has_xgboost: bool = False
    has_mlp: bool = False
    has_linear: bool = False
    has_pfi: bool = False
    has_lofo: bool = False
    has_bootstrap_results: bool = False   # 存在可配对的 per-sample 预测 (bootstrap artifact)
    environment_factorial_observations: int = 0
    replication_per_cellline: int = 0
    reasons: Dict[str, str] = field(default_factory=dict)


def validate_analysis_plan(plan: AnalysisPlan, caps: Capabilities) -> Dict[str, Any]:
    """
    依赖 + 数据可得性校验。返回每个可选任务的
    {selected, available, status, reason}。绝不抛错代替判断。
    """
    tasks: Dict[str, Dict[str, Any]] = {}

    def task(task_id: str, selected: bool, available: bool, reason: str = "", dependency_ok: bool = True):
        tasks[task_id] = {
            "task_id": task_id,
            "selected": bool(selected),
            "available": bool(available) and bool(dependency_ok),
            "reason": reason,
            "status": (
                "skipped"
                if (not selected or not (available and dependency_ok))
                else "pending"
            ),
        }

    task("qc", plan.run_qc, True)
    task("prediction", plan.run_prediction_analysis, True)
    task("environment_conditional_effect",
         plan.environment.enabled and plan.environment.conditional_effect,
         caps.has_environment_features and caps.environment_factorial_observations >= 2,
         reason=caps.reasons.get("environment", "environment features unavailable"))
    task("environment_main_effect",
         plan.environment.enabled and plan.environment.main_effect,
         caps.has_environment_features and caps.n_environment_factors >= 2,
         reason=caps.reasons.get("environment", ""))
    task("environment_factorial_dag",
         plan.environment.enabled and plan.environment.conditional_effect,
         caps.has_environment_features and caps.environment_factorial_observations >= 2,
         reason=caps.reasons.get("environment", "environment features unavailable"))
    task("sequence_attribution", plan.sequence.enabled and plan.sequence.position_attribution, True)
    task("cnn_ism", plan.sequence.enabled and plan.sequence.ism, caps.has_cnn,
         reason="no CNN outputs in batch" if not caps.has_cnn else "")
    task("motif_discovery", plan.sequence.enabled and plan.sequence.motif_discovery,
         caps.has_cnn or caps.has_transformer,
         reason=("user_disabled" if not (plan.sequence.enabled and plan.sequence.motif_discovery)
                 else "insufficient_data: no compatible sequence attribution (CNN ISM/IG)"))
    task("motif_enrichment",
         plan.sequence.enabled and plan.sequence.motif_enrichment,
         caps.has_cnn or caps.has_transformer,
         dependency_ok=plan.sequence.motif_discovery,
         reason=("user_disabled" if not (plan.sequence.enabled and plan.sequence.motif_enrichment)
                 else "motif_enrichment requires motif_discovery"))
    task("bootstrap", plan.statistics.bootstrap, caps.has_bootstrap_results,
         reason=("user_disabled" if not plan.statistics.bootstrap
                 else "insufficient_data: no paired per-sample prediction artifact"))
    task("hypothesis_testing",
         plan.statistics.hypothesis_testing and plan.statistics.permutation_test,
         caps.has_bootstrap_results,
         reason=("user_disabled" if not (plan.statistics.hypothesis_testing
                                         and plan.statistics.permutation_test)
                 else "insufficient_data: no paired per-sample prediction artifact"))
    task("fdr_correction", plan.statistics.fdr_correction, True,
         reason=("user_disabled" if not plan.statistics.fdr_correction
                 else "executed only when corrigible p-value families exist"))
    task("environment_anova",
         plan.environment.enabled and plan.environment.anova and plan.statistics.anova,
         caps.n_environment_factors >= 2 and caps.environment_factorial_observations >= 2,
         reason=("user_disabled" if not (plan.environment.anova and plan.statistics.anova)
                 else "insufficient_data: factorial environment observations < 2"))
    return tasks


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------
@dataclass
class ExecutionTask:
    task_id: str
    selected: bool
    available: bool
    status: str = ExecutionStatus.PENDING.value
    dependencies: List[str] = field(default_factory=list)
    reason: str = ""
    artifact: str = ""          # 该任务产出的结果文件 (前端 Artifact Viewer 直接读取)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class ExecutionPlan:
    plan: AnalysisPlan
    tasks: List[ExecutionTask]
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: Optional[str] = None

    def to_status_dict(self) -> Dict[str, Any]:
        return {
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "tasks": [
                {
                    "task_id": t.task_id,
                    "selected": t.selected,
                    "available": t.available,
                    "status": t.status,
                    "reason": t.reason,
                    "artifact": getattr(t, "artifact", ""),
                }
                for t in self.tasks
            ],
        }

    def save_status(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(
            json.dumps(self.to_status_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return p
