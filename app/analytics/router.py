import os
import shutil
import tempfile
import time
import logging
from typing import Optional, List, Dict, Any, Literal
# pyrefly: ignore [missing-import]
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

# Phase 3.x Generic Semantic Engine
from .models import SemanticQueryPlan
from .planner import SemanticPlannerV2
from .validator import SemanticValidator
from .compiler import AnalyticsCompilerV2
from .fuzzy_resolver import fuzzy_resolver
from .schema_discovery import auto_catalog
from .excel_engine import excel_engine

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
    engine: Literal["semantic", "excel"] = Field("semantic", description="Engine to use")
    file_id: Optional[str] = Field(default=None, description="Optional uploaded Excel virtual database file ID")
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


class ExcelQueryRequest(BaseModel):
    workspace_id: int = Field(..., description="Authenticated workspace context")
    question: str = Field(..., description="Natural language question about the uploaded Excel file")
    file_id: Optional[str] = Field(default=None, description="Uploaded file ID (uses latest if empty)")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation turns")


@router.get("/schema")
def get_discovered_schema():
    """Returns dynamic schema discovery and auto-cataloged metadata."""
    return auto_catalog.discover_and_catalog()


@router.post("/excel/upload")
async def upload_excel_file(
    file: UploadFile = File(...),
    workspace_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
):
    """
    Ingests an uploaded Excel (.xlsx, .xls) or CSV file.
    Creates an isolated SQLite virtual database where each tab is an analytical table.
    """
    filename = file.filename or "uploaded_spreadsheet.xlsx"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {ext}. Only .xlsx, .xls, and .csv are supported.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        metadata = excel_engine.ingest_excel_file(
            file_path=tmp_path,
            original_filename=filename,
            workspace_id=workspace_id,
            conversation_id=conversation_id,
        )
        return {
            "success": True,
            "message": f"Successfully ingested '{filename}' with {metadata['sheets_count']} sheets and {metadata['total_rows']} rows.",
            "file_id": metadata["file_id"],
            "filename": metadata["filename"],
            "sheets": list(metadata["tables"].keys()),
            "total_rows": metadata["total_rows"],
            "schema_summary": metadata["schema_catalog"],
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.post("/excel/query")
def query_excel_database(req: ExcelQueryRequest):
    """
    Queries an uploaded Excel virtual database using natural language.
    """
    res = excel_engine.query_excel(
        workspace_id=req.workspace_id,
        file_id=req.file_id or "",
        question=req.question,
        history=req.history,
    )
    return res


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
    # Check if query is targeting an uploaded Excel virtual database
    # -------------------------------------------------------------
    q_lower = query.lower()
    is_excel_keyword = (
        req.engine == "excel" or 
        bool(req.file_id) or 
        any(kw in q_lower for kw in (
            "sheet", "tab", "excel", "file e", "spreadsheet", 
            "salary", "expense", "expenses", "employee", "employees", "payroll", "vendor"
        ))
    )
    if is_excel_keyword:
        excel_res = excel_engine.query_excel(
            workspace_id=workspace_id,
            file_id=req.file_id or "",
            question=query,
            history=req.history,
        )
        if excel_res.get("success") and excel_res.get("rows"):
            return AnalyticsQueryResponse(
                success=True,
                intent="excel_file_query",
                report=excel_res["report"],
                sql=excel_res.get("sql"),
                rows=excel_res.get("rows", []),
                is_security_rejection=False,
                is_ambiguous=False,
                latency_ms=excel_res.get("latency_ms", 0.0),
                engine="excel_virtual_db",
                provider_used="excel_sql_planner",
                fallback_triggered=False,
            )

    # -------------------------------------------------------------
    # Primary Pipeline: Phase 3.x Generic Semantic Analytics Engine
    # -------------------------------------------------------------
    try:
        # 1. Semantic Planning with Live Auto-Catalog
        plan, meta = planner.plan(query, history=req.history)
        llm_latency = meta.get("latency_ms", 0.0)

        # 2. Early Security Rejection (with Excel fallback if not malicious mutation)
        if plan.is_security_rejection:
            if plan.rejection_reason == "out_of_domain":
                # Check if Excel virtual database can answer it
                excel_fallback = excel_engine.query_excel(
                    workspace_id=workspace_id,
                    file_id=req.file_id or "",
                    question=query,
                    history=req.history,
                )
                if excel_fallback.get("success") and excel_fallback.get("rows"):
                    return AnalyticsQueryResponse(
                        success=True,
                        intent="excel_file_fallback",
                        report=excel_fallback["report"],
                        sql=excel_fallback.get("sql"),
                        rows=excel_fallback.get("rows", []),
                        is_security_rejection=False,
                        is_ambiguous=False,
                        latency_ms=excel_fallback.get("latency_ms", 0.0),
                        engine="excel_virtual_db",
                        provider_used="excel_sql_planner",
                        fallback_triggered=True,
                    )

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

        # 3. Early Clarification Handling
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

        # 4. Phase 1: Fuzzy Entity Normalization (Resolves misspellings like 'hasn', 'laptp', 'bksh')
        plan = fuzzy_resolver.normalize_plan(plan, workspace_id=workspace_id)

        # 5. Semantic Validation
        val_res = validator.validate_plan(plan)
        if not val_res.is_valid:
            # Check Excel fallback before returning validation notice
            excel_fallback = excel_engine.query_excel(
                workspace_id=workspace_id,
                file_id=req.file_id or "",
                question=query,
                history=req.history,
            )
            if excel_fallback.get("success") and excel_fallback.get("rows"):
                return AnalyticsQueryResponse(
                    success=True,
                    intent="excel_file_fallback",
                    report=excel_fallback["report"],
                    sql=excel_fallback.get("sql"),
                    rows=excel_fallback.get("rows", []),
                    is_security_rejection=False,
                    is_ambiguous=False,
                    latency_ms=excel_fallback.get("latency_ms", 0.0),
                    engine="excel_virtual_db",
                    provider_used="excel_sql_planner",
                    fallback_triggered=True,
                )

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

        # 6. Deterministic SQL Compilation
        sql, params = compiler.compile(plan, workspace_id=workspace_id)

        # 7. Safe Read-Only Execution
        rows, exec_latency = executor.execute(sql, params)
        total_latency = round(llm_latency + exec_latency, 2)

        # 8. Report Formatting
        report = formatter.format(query, plan, rows, latency_ms=total_latency)

        # Derive clean high-level intent label
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
