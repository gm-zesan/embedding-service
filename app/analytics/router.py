import time
import logging
from typing import Optional, List, Dict, Any, Literal
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, status
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

# Phase 3.x Generic Semantic Engine
from .models_v2 import SemanticQueryPlan
from .planner_v2 import SemanticPlannerV2
from .validator_v2 import SemanticValidator
from .compiler_v2 import AnalyticsCompilerV2

# Phase 3 Control Baseline (Fallback Engine)
from .planner import AnalyticsPlanner
from .compiler import AnalyticsCompiler

# Shared Safe Execution and Formatting
from .executor import AnalyticsExecutor, SecurityViolationError
from .formatter import AnalyticsFormatter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Engine instances
planner_v2 = SemanticPlannerV2()
validator_v2 = SemanticValidator()
compiler_v2 = AnalyticsCompilerV2()

planner_v1 = AnalyticsPlanner()
compiler_v1 = AnalyticsCompiler()

executor = AnalyticsExecutor()
formatter = AnalyticsFormatter()


class AnalyticsQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language business intelligence question")
    workspace_id: int = Field(..., description="Authenticated multi-tenant workspace context")
    engine: Literal["semantic"] = Field("semantic", description="Engine to use")


class AnalyticsQueryResponse(BaseModel):
    success: bool
    intent: str
    report: str
    sql: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    is_security_rejection: bool = False
    is_ambiguous: bool = False
    latency_ms: float = 0.0
    engine: str = "v2_semantic"
    provider_used: Optional[str] = None
    fallback_triggered: bool = False


def _execute_v1_fallback(
    query: str, workspace_id: int, t_start: float, reason: str = ""
) -> AnalyticsQueryResponse:
    """Executes query using Phase 3 Baseline engine as fail-safe fallback."""
    try:
        plan_v1, llm_ms = planner_v1.plan(query)
        if plan_v1.is_security_rejection:
            report = formatter.format(query, plan_v1, [], latency_ms=llm_ms)
            return AnalyticsQueryResponse(
                success=True,
                intent=plan_v1.intent,
                report=report,
                sql=None,
                rows=[],
                is_security_rejection=True,
                is_ambiguous=False,
                latency_ms=round(llm_ms, 2),
                engine="v1_fallback",
                fallback_triggered=True,
            )
        if plan_v1.needs_clarification:
            report = formatter.format(query, plan_v1, [], latency_ms=llm_ms)
            return AnalyticsQueryResponse(
                success=True,
                intent=plan_v1.intent,
                report=report,
                sql=None,
                rows=[],
                is_security_rejection=False,
                is_ambiguous=True,
                latency_ms=round(llm_ms, 2),
                engine="v1_fallback",
                fallback_triggered=True,
            )
        sql, params = compiler_v1.compile(plan_v1, workspace_id=workspace_id)
        rows, exec_ms = executor.execute(sql, params)
        total_latency = round(llm_ms + exec_ms, 2)
        report = formatter.format(query, plan_v1, rows, latency_ms=total_latency)
        return AnalyticsQueryResponse(
            success=True,
            intent=plan_v1.intent,
            report=report,
            sql=sql,
            rows=rows,
            is_security_rejection=False,
            is_ambiguous=False,
            latency_ms=total_latency,
            engine="v1_fallback",
            fallback_triggered=True,
        )
    except Exception as e_v1:
        total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
        logger.error(f"[Analytics API] Fallback v1 failed: {e_v1}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics query execution failed (v2 reason: {reason}; fallback error: {str(e_v1)})",
        )


@router.post("/query", response_model=AnalyticsQueryResponse)
def handle_analytics_query(req: AnalyticsQueryRequest):
    t_start = time.perf_counter()
    query = req.query.strip()
    workspace_id = req.workspace_id

    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query string cannot be empty.",
        )

    # -------------------------------------------------------------
    # Primary Pipeline: Phase 3.x Generic Semantic Analytics Engine
    # -------------------------------------------------------------
    try:
        plan_v2, meta = planner_v2.plan(query)
        llm_latency = meta.get("latency_ms", 0.0)

        # 1. Early Security Rejection
        if plan_v2.is_security_rejection:
            report = formatter.format(query, plan_v2, [], latency_ms=llm_latency)
            return AnalyticsQueryResponse(
                success=True,
                intent="security_rejection",
                report=report,
                sql=None,
                rows=[],
                is_security_rejection=True,
                is_ambiguous=False,
                latency_ms=round(llm_latency, 2),
                engine="v2_semantic",
                provider_used=meta.get("provider_used"),
                fallback_triggered=meta.get("fallback_triggered", False),
            )

        # 2. Early Clarification Handling
        if plan_v2.needs_clarification:
            report = formatter.format(query, plan_v2, [], latency_ms=llm_latency)
            return AnalyticsQueryResponse(
                success=True,
                intent="ambiguous_query",
                report=report,
                sql=None,
                rows=[],
                is_security_rejection=False,
                is_ambiguous=True,
                latency_ms=round(llm_latency, 2),
                engine="v2_semantic",
                provider_used=meta.get("provider_used"),
                fallback_triggered=meta.get("fallback_triggered", False),
            )

        # 3. Semantic Validation
        val_res = validator_v2.validate_plan(plan_v2)
        if not val_res.is_valid:
            logger.warning(
                f"[Phase 3.x] Semantic validation warnings: {val_res.errors}. Triggering fallback to Phase 3 baseline..."
            )
            return _execute_v1_fallback(query, workspace_id, t_start, reason="semantic_validation_failed")

        # 4. Deterministic SQL Compilation
        sql, params = compiler_v2.compile(plan_v2, workspace_id=workspace_id)

        # 5. Safe Read-Only Execution
        rows, exec_latency = executor.execute(sql, params)
        total_latency = round(llm_latency + exec_latency, 2)

        # 6. Report Formatting
        report = formatter.format(query, plan_v2, rows, latency_ms=total_latency)

        # Derive clean high-level intent label for backward compatibility
        intent_label = f"{plan_v2.domain.value}_query"
        if plan_v2.measures:
            intent_label = f"{plan_v2.domain.value}_{plan_v2.measures[0].name}"
        elif plan_v2.derived_metrics:
            intent_label = f"{plan_v2.domain.value}_{plan_v2.derived_metrics[0].type.value}"

        return AnalyticsQueryResponse(
            success=True,
            intent=intent_label,
            report=report,
            sql=sql,
            rows=rows,
            is_security_rejection=False,
            is_ambiguous=False,
            latency_ms=total_latency,
            engine="v2_semantic",
            provider_used=meta.get("provider_used"),
            fallback_triggered=meta.get("fallback_triggered", False),
        )

    except SecurityViolationError as sve:
        logger.warning(f"[Analytics API] Security violation blocked: {sve}")
        total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
        return AnalyticsQueryResponse(
            success=True,
            intent="security_blocked",
            report=f"🛡️ **Security Boundary Enforced**\n\n> {str(sve)}",
            sql=None,
            rows=[],
            is_security_rejection=True,
            is_ambiguous=False,
            latency_ms=total_latency,
            engine="v2_semantic",
            fallback_triggered=False,
        )
    except Exception as e_v2:
        logger.warning(
            f"[Phase 3.x] v2 pipeline failed ({e_v2}). Executing graceful fallback to Phase 3 baseline...",
            exc_info=True,
        )
        return _execute_v1_fallback(query, workspace_id, t_start, reason=str(e_v2))
