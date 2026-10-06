import os
import shutil
import tempfile
import time
import logging
from typing import Optional, List, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from pydantic import BaseModel, Field

# Phase 3.x Generic Semantic Engine
from .models import SemanticQueryPlan
from .planner import SemanticPlannerV2
from .validator import SemanticValidator
from .compiler import AnalyticsCompilerV2
from .fuzzy_resolver import fuzzy_resolver
from .schema_discovery import auto_catalog

# Generic Data Source Registry & Engine
from .data_source import DataSource, Dataset, DataSourceFormat, ResolutionStatus
from .source_registry import source_registry
from .source_resolver import source_resolver
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
    engine: Literal["semantic", "excel", "data_source"] = Field("semantic", description="Engine to use")
    file_id: Optional[str] = Field(default=None, description="Optional uploaded data source or file ID")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation turns")


class AnalyticsQueryResponse(BaseModel):
    success: bool
    intent: str
    report: str
    sql: Optional[str] = None
    rows: Optional[List[Dict[str, Any]]] = None
    is_security_rejection: bool = False
    is_ambiguous: bool = False
    is_cross_source_unsupported: bool = False
    clarification_options: Optional[List[str]] = None
    candidates: Optional[List[Dict[str, Any]]] = None
    filename: Optional[str] = None
    source_id: Optional[str] = None
    latency_ms: float = 0.0
    engine: str = "v2_semantic"
    provider_used: Optional[str] = None
    fallback_triggered: bool = False


class ExcelQueryRequest(BaseModel):
    workspace_id: int = Field(..., description="Authenticated workspace context")
    question: str = Field(..., description="Natural language question about the uploaded source")
    file_id: Optional[str] = Field(default=None, description="Optional source/file ID")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation turns")


@router.get("/schema")
def get_discovered_schema():
    """Returns dynamic schema discovery and auto-cataloged metadata."""
    return auto_catalog.discover_and_catalog()


# ── Generic Data Source Endpoints ─────────────────────────────────

@router.get("/sources/list")
def list_workspace_sources(workspace_id: int):
    """Lists all registered analytical data sources for a workspace."""
    sources = source_registry.list_sources(workspace_id)
    return {
        "success": True,
        "workspace_id": workspace_id,
        "count": len(sources),
        "sources": [s.model_dump() for s in sources],
    }


@router.delete("/sources/{source_id}")
def delete_workspace_source(source_id: str, workspace_id: int):
    """Deletes an analytical data source and its storage files with workspace authorization."""
    deleted = source_registry.delete_source(workspace_id, source_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source {source_id} not found in workspace {workspace_id}",
        )
    return {"success": True, "message": f"Source {source_id} deleted successfully."}


@router.post("/sources/upload")
@router.post("/excel/upload")
async def upload_analytical_source(
    file: UploadFile = File(...),
    workspace_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
):
    """
    Ingests an uploaded analytical document (XLSX, XLS, CSV).
    Creates an isolated SQLite virtual database where each sheet/table is registered.
    """
    filename = file.filename or "uploaded_data.xlsx"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in (".xlsx", ".xls", ".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format: {ext}. Supported formats: .xlsx, .xls, .csv",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        finally:
            file.file.close()

    try:
        result = excel_engine.ingest_excel_file(
            file_path=tmp_path,
            original_filename=filename,
            workspace_id=int(workspace_id),
            conversation_id=conversation_id,
        )
        return result
    except Exception as e:
        logger.error(f"[Source Ingestion] Failed to ingest {filename}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to ingest spreadsheet file: {str(e)}",
        )
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@router.post("/sources/query")
@router.post("/excel/query")
async def query_analytical_source(req: ExcelQueryRequest):
    """
    Query uploaded analytical sources using natural language with format-aware resolution,
    conversational context, and ambiguity detection.
    """
    try:
        result = excel_engine.query_excel(
            workspace_id=req.workspace_id,
            file_id=req.file_id,
            question=req.question,
            history=req.history,
        )
        return result
    except Exception as e:
        logger.error(f"[Source Query] Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute document query: {str(e)}",
        )


# ── Unified Analytics Dispatch Router ─────────────────────────────

@router.post("/query", response_model=AnalyticsQueryResponse)
async def query_analytics(req: AnalyticsQueryRequest):
    """
    Unified analytics endpoint:
    1. If engine == 'excel' or 'data_source', or if explicit file_id is provided, dispatches to Source Analytics Engine.
    2. Otherwise, executes semantic natural language SQL pipeline on production DB,
       with safe fallback to document analytics if query refers to uploaded datasets.
    """
    t_start = time.perf_counter()
    query = req.query.strip()
    workspace_id = req.workspace_id

    # 1. Direct Data Source Analytics Request
    if req.engine in ("excel", "data_source") or req.file_id:
        doc_res = excel_engine.query_excel(
            workspace_id=workspace_id,
            file_id=req.file_id,
            question=query,
            history=req.history,
        )
        return AnalyticsQueryResponse(
            success=doc_res.get("success", False),
            intent="document_source_query",
            report=doc_res.get("report", ""),
            sql=doc_res.get("sql"),
            rows=doc_res.get("rows", []),
            is_security_rejection=False,
            is_ambiguous=doc_res.get("is_ambiguous", False),
            is_cross_source_unsupported=doc_res.get("is_cross_source_unsupported", False),
            clarification_options=doc_res.get("clarification_options"),
            candidates=doc_res.get("candidates"),
            filename=doc_res.get("filename"),
            source_id=doc_res.get("source_id"),
            latency_ms=doc_res.get("latency_ms", 0.0),
            engine="document_virtual_db",
            provider_used="document_sql_planner",
            fallback_triggered=False,
        )

    # 2. Check if query explicitly mentions an uploaded source/file/sheet
    sources_in_ws = source_registry.list_sources(workspace_id)
    if sources_in_ws:
        res = source_resolver.resolve(
            query=query,
            workspace_id=workspace_id,
            explicit_source_id=req.file_id,
            context=None,
        )
        if res.status == ResolutionStatus.RESOLVED and res.source:
            doc_res = excel_engine.query_excel(
                workspace_id=workspace_id,
                file_id=res.source.source_id,
                question=query,
                history=req.history,
            )
            if doc_res.get("success") and doc_res.get("rows"):
                return AnalyticsQueryResponse(
                    success=True,
                    intent="document_source_query",
                    report=doc_res["report"],
                    sql=doc_res.get("sql"),
                    rows=doc_res.get("rows", []),
                    is_security_rejection=False,
                    is_ambiguous=False,
                    latency_ms=doc_res.get("latency_ms", 0.0),
                    engine="document_virtual_db",
                    provider_used="document_sql_planner",
                    fallback_triggered=False,
                )

    # -------------------------------------------------------------
    # Primary Pipeline: Phase 3.x Generic Semantic Analytics Engine
    # -------------------------------------------------------------
    try:
        plan, meta = planner.plan(query, history=req.history)
        llm_latency = meta.get("latency_ms", 0.0)

        if plan.is_security_rejection:
            if plan.rejection_reason == "out_of_domain":
                doc_fallback = excel_engine.query_excel(
                    workspace_id=workspace_id,
                    file_id=req.file_id,
                    question=query,
                    history=req.history,
                )
                if doc_fallback.get("success") and doc_fallback.get("rows"):
                    return AnalyticsQueryResponse(
                        success=True,
                        intent="document_source_fallback",
                        report=doc_fallback["report"],
                        sql=doc_fallback.get("sql"),
                        rows=doc_fallback.get("rows", []),
                        is_security_rejection=False,
                        is_ambiguous=False,
                        latency_ms=doc_fallback.get("latency_ms", 0.0),
                        engine="document_virtual_db",
                        provider_used="document_sql_planner",
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

        # Fuzzy Entity Normalization
        plan = fuzzy_resolver.normalize_plan(plan, workspace_id=workspace_id)

        # Semantic Validation
        val_res = validator.validate_plan(plan)
        if not val_res.is_valid:
            doc_fallback = excel_engine.query_excel(
                workspace_id=workspace_id,
                file_id=req.file_id,
                question=query,
                history=req.history,
            )
            if doc_fallback.get("success") and doc_fallback.get("rows"):
                return AnalyticsQueryResponse(
                    success=True,
                    intent="document_source_fallback",
                    report=doc_fallback["report"],
                    sql=doc_fallback.get("sql"),
                    rows=doc_fallback.get("rows", []),
                    is_security_rejection=False,
                    is_ambiguous=False,
                    latency_ms=doc_fallback.get("latency_ms", 0.0),
                    engine="document_virtual_db",
                    provider_used="document_sql_planner",
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

        sql, params = compiler.compile(plan, workspace_id=workspace_id)
        rows, exec_latency = executor.execute(sql, params)
        total_latency = round(llm_latency + exec_latency, 2)
        report = formatter.format(query, plan, rows, latency_ms=total_latency)

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
