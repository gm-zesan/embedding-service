import time
import logging
from typing import Optional, List, Dict, Any, Literal
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, status
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

# Phase 3.x Generic Semantic Engine
from .models import SemanticQueryPlan
from .planner import SemanticPlannerV2
from .validator import SemanticValidator
from .compiler import AnalyticsCompilerV2


# Shared Safe Execution and Formatting
from .executor import AnalyticsExecutor, SecurityViolationError
from .formatter import AnalyticsFormatter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analytics", tags=["analytics"])

# Engine instances
planner = SemanticPlannerV2()
validator = SemanticValidator()
compiler = AnalyticsCompilerV2()


executor = AnalyticsExecutor()
formatter = AnalyticsFormatter()


class AnalyticsQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language business intelligence question")
    workspace_id: int = Field(..., description="Authenticated multi-tenant workspace context")
    engine: Literal["semantic"] = Field("semantic", description="Engine to use")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation turns")


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
        plan, meta = planner.plan(query, history=req.history)
        llm_latency = meta.get("latency_ms", 0.0)

        # 1. Early Security Rejection
        if plan.is_security_rejection:
            report = formatter.format(query, plan, [], latency_ms=llm_latency)
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
        if plan.needs_clarification:
            report = formatter.format(query, plan, [], latency_ms=llm_latency)
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
        val_res = validator.validate_plan(plan)
        if not val_res.is_valid:
            logger.warning(f"[Phase 3.x] Semantic validation failed: {val_res.errors}")
            error_details = "\n".join(f"- {err}" for err in val_res.errors)
            report = (
                f"ℹ️ **Query Interpretation Notice**\n\n"
                f"Could not compute analytics for: *\"{query}\"*\n\n"
                f"**Reason:**\n{error_details}\n\n"
                f"*Please ask a question relating to supported metrics like sales amount, orders count, cash collections, or outstanding dues.*"
            )
            total_latency = round((time.perf_counter() - t_start) * 1000.0, 2)
            return AnalyticsQueryResponse(
                success=True,
                intent="validation_notice",
                report=report,
                sql=None,
                rows=[],
                is_security_rejection=False,
                is_ambiguous=False,
                latency_ms=total_latency,
                engine="v2_semantic",
                provider_used=meta.get("provider_used"),
                fallback_triggered=meta.get("fallback_triggered", False),
            )

        # 4. Deterministic SQL Compilation
        sql, params = compiler.compile(plan, workspace_id=workspace_id)

        # 5. Safe Read-Only Execution
        rows, exec_latency = executor.execute(sql, params)
        total_latency = round(llm_latency + exec_latency, 2)

        # 6. Report Formatting
        report = formatter.format(query, plan, rows, latency_ms=total_latency)

        # Derive clean high-level intent label for backward compatibility
        intent_label = f"{plan.domain.value}_query"
        if plan.measures:
            intent_label = f"{plan.domain.value}_{plan.measures[0].name}"
        elif plan.derived_metrics:
            intent_label = f"{plan.domain.value}_{plan.derived_metrics[0].type.value}"

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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Phase 3.x] Pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics query execution failed: {str(e)}",
        )
