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
from app import config

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
    currency_hint: Optional[str] = Field(default=None, description="Optional explicit currency hint (e.g. USD, EUR, BDT)")


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
    currency_hint: Optional[str] = Field(default=None, description="Optional explicit currency hint (e.g. USD, EUR, BDT)")


class SourceResolveRequest(BaseModel):
    query: str = Field(..., description="User question or query text")
    workspace_id: int = Field(..., description="Tenant workspace ID")
    explicit_source_id: Optional[str] = Field(default=None, description="Optional explicit source ID")


class SourceResolveResponse(BaseModel):
    success: bool
    status: str  # "resolved", "ambiguous", "not_found", "unsupported"
    source_id: Optional[str] = None
    filename: Optional[str] = None
    confidence: float = 0.0
    reason: str = ""
    clarification_message: Optional[str] = None
    clarification_options: Optional[List[str]] = None
    candidates: Optional[List[Dict[str, Any]]] = None


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


@router.post("/sources/resolve", response_model=SourceResolveResponse)
def resolve_analytical_source(req: SourceResolveRequest):
    """
    Resolves the most relevant generic analytics data source for a user query.
    1. If explicit_source_id is given, validates it against workspace ownership.
    2. If single source in workspace, auto-resolves with 100% confidence.
    3. Searches Typesense analytics_catalog collection with strict workspace filter.
    4. Evaluates confidence & ambiguity policy (single clear winner vs multiple close candidates).
    5. Falls back to schema-fingerprint and column matching in source_resolver if Typesense has no hits.
    """
    workspace_id = req.workspace_id
    query = req.query.strip()
    explicit_id = req.explicit_source_id

    # 1. Manual source selection override
    if explicit_id:
        src = source_registry.get_source(workspace_id, explicit_id)
        if src:
            return SourceResolveResponse(
                success=True,
                status="resolved",
                source_id=src.source_id,
                filename=src.original_filename,
                confidence=1.0,
                reason=f"Explicit source override: {src.original_filename}",
            )
        return SourceResolveResponse(
            success=False,
            status="not_found",
            reason=f"Explicit source {explicit_id} not found in workspace {workspace_id}",
            clarification_message="নির্দিষ্ট করা ফাইলটি পাওয়া যায়নি বা এই ওয়ার্কস্পেসের আওতাভুক্ত নয়।",
        )

    # 2. Check all sources in workspace
    all_sources = source_registry.list_sources(workspace_id)
    if not all_sources:
        return SourceResolveResponse(
            success=False,
            status="not_found",
            reason="No generic analytics documents found in workspace.",
            clarification_message="এই ওয়ার্কস্পেসে কোনো স্প্রেডশিট ডেটাসেট পাওয়া যায়নি।",
        )

    if len(all_sources) == 1:
        single_src = all_sources[0]
        return SourceResolveResponse(
            success=True,
            status="resolved",
            source_id=single_src.source_id,
            filename=single_src.original_filename,
            confidence=1.0,
            reason=f"Single active source in workspace: {single_src.original_filename}",
        )

    # 3. Search Typesense analytics_catalog
    candidates = []
    try:
        from app.typesense_engine import get_typesense_client, search_analytics_metadata
        ts_client = get_typesense_client()
        hits = search_analytics_metadata(ts_client, query, workspace_id, top_k=5)
        for h in hits:
            doc = h.get("document", {})
            score = h.get("text_match", 0)
            src_id = doc.get("source_id")
            fn = doc.get("original_filename")
            if src_id and source_registry.get_source(workspace_id, src_id):
                candidates.append({
                    "source_id": src_id,
                    "filename": fn,
                    "score": score,
                    "summary": doc.get("dataset_summary"),
                })
    except Exception as e:
        logger.warning(f"Typesense analytics search failed, falling back to local resolver: {e}")

    # Minimum BM25 text match score required to avoid noise/stopword false positives (not a probability)
    MIN_TEXT_MATCH_SCORE = 15
    # Threshold margin: top score must be >= 1.20x of the second score to declare a clear winner
    AMBIGUITY_SCORE_MARGIN = 1.20

    # Filter out weak/negligible matches (e.g. score < 15 or 0)
    valid_candidates = [c for c in candidates if c.get("score", 0) >= MIN_TEXT_MATCH_SCORE]

    # Evaluate Typesense hits if available
    if valid_candidates:
        top_cand = valid_candidates[0]
        top_score = top_cand.get("score", 0)

        # Ambiguity Policy:
        # If there's only 1 valid candidate above minimum threshold
        if len(valid_candidates) == 1:
            return SourceResolveResponse(
                success=True,
                status="resolved",
                source_id=top_cand["source_id"],
                filename=top_cand["filename"],
                confidence=0.9,
                reason=f"Resolved via Typesense analytics catalog: {top_cand['filename']} (BM25 score: {top_score})",
                candidates=valid_candidates,
            )
        else:
            second_cand = valid_candidates[1]
            second_score = second_cand.get("score", 0)

            # Safe comparison / zero-score handling:
            # Check if top score beats runner-up by at least AMBIGUITY_SCORE_MARGIN (1.20x)
            is_clear_winner = False
            if second_score == 0 and top_score >= MIN_TEXT_MATCH_SCORE:
                is_clear_winner = True
            elif second_score > 0 and (top_score >= AMBIGUITY_SCORE_MARGIN * second_score):
                is_clear_winner = True

            if is_clear_winner:
                return SourceResolveResponse(
                    success=True,
                    status="resolved",
                    source_id=top_cand["source_id"],
                    filename=top_cand["filename"],
                    confidence=0.9,
                    reason=f"Resolved via Typesense analytics catalog: {top_cand['filename']} (BM25 score: {top_score} vs {second_score})",
                    candidates=valid_candidates,
                )
            else:
                # Ambiguous: multiple close candidates within the 1.20x margin or tied
                options = [f"{c['filename']} থেকে উত্তর দাও" for c in valid_candidates[:3]]
                items_str = "\n".join([f"{idx+1}. `{c['filename']}`" for idx, c in enumerate(valid_candidates[:3])])
                return SourceResolveResponse(
                    success=False,
                    status="ambiguous",
                    confidence=0.5,
                    reason=f"Multiple datasets match the query with close BM25 scores ({top_score} vs {second_score}). User clarification required.",
                    clarification_message=f"🤔 **একাধিক ফাইল পাওয়া গেছে**:\n{items_str}\n\nঅনুগ্রহ করে কোন ফাইলটি দেখতে চান সিলেক্ট করুন।",
                    clarification_options=options,
                    candidates=valid_candidates,
                )

    # 4. Fallback: Local schema and column matching via source_resolver
    resolver_res = source_resolver.resolve(query=query, workspace_id=workspace_id)
    if resolver_res.status == ResolutionStatus.RESOLVED and resolver_res.source_id:
        src = source_registry.get_source(workspace_id, resolver_res.source_id)
        return SourceResolveResponse(
            success=True,
            status="resolved",
            source_id=resolver_res.source_id,
            filename=src.original_filename if src else resolver_res.source_id,
            confidence=0.85,
            reason=resolver_res.reason or "Resolved via schema and column match",
        )
    elif resolver_res.status == ResolutionStatus.AMBIGUOUS:
        return SourceResolveResponse(
            success=False,
            status="ambiguous",
            reason=resolver_res.reason,
            clarification_message=resolver_res.clarification_message,
            clarification_options=resolver_res.clarification_options,
            candidates=resolver_res.candidates,
        )

    return SourceResolveResponse(
        success=False,
        status="not_found",
        reason="No matching generic analytics dataset found for query.",
        clarification_message="আপনার প্রশ্নের সাথে সম্পর্কিত কোনো স্প্রেডশিট বা টেবিল পাওয়া যায়নি।",
    )


@router.delete("/sources/{source_id}")
def delete_workspace_source(source_id: str, workspace_id: int):
    """Deletes an analytical data source and its storage files with workspace authorization."""
    deleted = source_registry.delete_source(workspace_id, source_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source {source_id} not found in workspace {workspace_id}",
        )
    # Also remove from Typesense analytics catalog
    try:
        from app.typesense_engine import get_typesense_client, delete_analytics_metadata_by_source
        ts_client = get_typesense_client()
        delete_analytics_metadata_by_source(ts_client, workspace_id=workspace_id, source_id=source_id)
    except Exception as e:
        logger.warning(f"Error removing source {source_id} from Typesense analytics catalog: {e}")

    return {"success": True, "message": f"Source {source_id} deleted successfully."}


@router.post("/sources/upload")
@router.post("/excel/upload")
async def upload_analytical_source(
    file: UploadFile = File(...),
    workspace_id: int = Form(...),
    conversation_id: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
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

    raw_max = getattr(config, "ANALYTICS_MAX_UPLOAD_SIZE_BYTES", 20 * 1024 * 1024)
    try:
        max_bytes = int(raw_max)
        if max_bytes <= 0:
            max_bytes = 20 * 1024 * 1024
    except (ValueError, TypeError):
        max_bytes = 20 * 1024 * 1024

    chunk_size = 64 * 1024  # 64 KiB
    total_bytes = 0

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp_path = tmp.name

    try:
        try:
            while True:
                # Bounded read: reads up to chunk_size but never more than 1 byte past max_bytes
                remaining = (max_bytes - total_bytes) + 1
                to_read = min(chunk_size, remaining)
                chunk = file.file.read(to_read)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > max_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"Uploaded file exceeds maximum allowed size of {max_bytes} bytes "
                            f"({max_bytes // (1024 * 1024)} MiB)."
                        ),
                    )
                tmp.write(chunk)
        finally:
            tmp.close()
            file.file.close()

        result = excel_engine.ingest_excel_file(
            file_path=tmp_path,
            original_filename=filename,
            workspace_id=int(workspace_id),
            conversation_id=conversation_id,
            source_id=source_id,
        )
        return result
    except HTTPException:
        # Re-raise standard HTTP exceptions (e.g. 413) without converting to 422
        raise
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
            currency_hint=req.currency_hint,
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
            currency_hint=req.currency_hint,
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
        report = formatter.format(query, plan, rows, latency_ms=total_latency, currency_hint=req.currency_hint)

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
