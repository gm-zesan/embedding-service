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
    RETIRED ENDPOINT: Direct legacy Excel SQL execution is permanently disabled.
    All spreadsheet analytics must use the Contract 3.0 Virtual Execution Gateway (/analytics/v3/execute).
    """
    logger.warning(
        f"[Legacy Route Rejected] Direct Excel query attempted on retired endpoint for workspace {req.workspace_id}"
    )
    return {
        "success": False,
        "status": "retired",
        "report": "Direct legacy Excel query execution is permanently retired. All spreadsheet queries must be authorized through Contract 3.0 (/analytics/v3/execute).",
        "rows": [],
        "error": "ENDPOINT_RETIRED",
    }


# ── Unified Analytics Dispatch Router ─────────────────────────────

@router.post("/query", response_model=AnalyticsQueryResponse)
async def query_analytics(req: AnalyticsQueryRequest):
    """
    Unified analytics endpoint:
    1. If engine == 'excel' or 'data_source', or if explicit file_id is provided, rejects direct legacy spreadsheet execution.
    2. Otherwise, executes semantic natural language SQL pipeline on production DB.
    """
    t_start = time.perf_counter()
    query = req.query.strip()
    workspace_id = req.workspace_id

    # 1. Direct Data Source Analytics Request (Permanently Retired)
    if req.engine in ("excel", "data_source") or req.file_id:
        logger.warning(
            f"[Legacy Dispatch Rejected] Direct engine={req.engine} attempted on retired endpoint for workspace {workspace_id}"
        )
        return AnalyticsQueryResponse(
            success=False,
            intent="document_source_query",
            report="Direct legacy spreadsheet query execution is permanently retired. Use Contract 3.0 (/analytics/v3/execute).",
            sql=None,
            rows=[],
            is_security_rejection=True,
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
            logger.warning(
                f"[Legacy Dispatch Blocked] Document source query attempted via ungoverned /analytics/query for workspace {workspace_id}"
            )
            return AnalyticsQueryResponse(
                success=False,
                intent="document_source_query",
                report="Direct legacy spreadsheet query execution is permanently retired. All spreadsheet queries must be authorized through Contract 3.0 (/analytics/v3/execute).",
                sql=None,
                rows=[],
                is_security_rejection=True,
            )

    # -------------------------------------------------------------
    # Primary Pipeline: Phase 3.x Generic Semantic Analytics Engine
    # -------------------------------------------------------------
    try:
        plan, meta = planner.plan(query, history=req.history)
        llm_latency = meta.get("latency_ms", 0.0)

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


# ===========================================================================
# Contract 3.0.0 Endpoints (Phase 3 Milestone 3.0 Foundation)
# ===========================================================================

from app.analytics.contract_v3 import (
    CONTRACT_VERSION_3_0,
    AnalyticsPrepareRequest,
    PreparedQueryArtifact,
    ExecutionAuthorizationGrant,
    AnalyticsExecuteRequest,
    AnalyticsExecutionResponse,
    DynamicSchemaCatalog,
)
from app.analytics.virtual_compiler import virtual_compiler, VirtualCompilerError
from app.analytics.virtual_catalog import virtual_catalog_builder
from app.analytics.virtual_planner import virtual_semantic_planner
from app.analytics.execution_gateway import virtual_execution_gateway, ExecutionGatewaySecurityError


class PlanCandidateRequest(BaseModel):
    workspace_id: int = Field(..., gt=0)
    dataset_id: str
    question: str
    history: Optional[List[Dict[str, str]]] = None


@router.post("/v3/plan-candidate", response_model=SemanticQueryPlan)
def plan_query_candidate(req: PlanCandidateRequest):
    """
    Candidate generation endpoint: Generates a SemanticQueryPlan AST from question
    without any SQL generation.
    """
    try:
        source = source_registry.get_source(req.workspace_id, req.dataset_id)
        db_path = source.db_path if source else None
        if not db_path:
            storage_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "storage", "excel_databases")
            clean_dataset = "".join(c for c in req.dataset_id if c.isalnum() or c in ("-", "_"))[:64]
            candidate_path = os.path.join(storage_dir, f"ws_{req.workspace_id}_{clean_dataset}.sqlite")
            if os.path.exists(candidate_path):
                db_path = candidate_path

        if not db_path or not os.path.exists(db_path):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset {req.dataset_id} database not found.")

        generation_id = getattr(source, "generation_id", None) or req.dataset_id
        catalog = virtual_catalog_builder.build_from_sqlite(
            db_path=db_path,
            workspace_id=req.workspace_id,
            dataset_id=req.dataset_id,
            generation_id=str(generation_id),
        )

        plan = virtual_semantic_planner.plan_query(
            question=req.question,
            catalog=catalog,
            history=req.history,
        )
        return plan
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Contract 3.0 Plan Candidate] Error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v3/prepare", response_model=PreparedQueryArtifact)
def prepare_virtual_query(req: AnalyticsPrepareRequest):
    """
    Stage A: Compiles SemanticQueryPlan into PreparedQueryArtifact.
    DOES NOT EXECUTE SQL.
    """
    try:
        catalog = req.catalog
        if not catalog:
            source = source_registry.get_source(req.workspace_id, req.dataset_id)
            db_path = source.db_path if source else None
            if not db_path:
                storage_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "storage", "excel_databases")
                clean_dataset = "".join(c for c in req.dataset_id if c.isalnum() or c in ("-", "_"))[:64]
                candidate_path = os.path.join(storage_dir, f"ws_{req.workspace_id}_{clean_dataset}.sqlite")
                if os.path.exists(candidate_path):
                    db_path = candidate_path

            if not db_path or not os.path.exists(db_path):
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Database for dataset {req.dataset_id} not found.")

            catalog = virtual_catalog_builder.build_from_sqlite(
                db_path=db_path,
                workspace_id=req.workspace_id,
                dataset_id=req.dataset_id,
                generation_id=req.generation_id,
                relationship_graph=req.relationship_graph,
                source_descriptors=req.source_descriptors,
            )

        artifact = virtual_compiler.compile(
            plan=req.query_plan,
            catalog=catalog,
            relationship_graph=req.relationship_graph,
        )
        artifact.request_id = req.request_id
        return artifact
    except VirtualCompilerError as vce:
        logger.warning(f"[Contract 3.0 Prepare] Virtual compiler error: {vce}")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(vce))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Contract 3.0 Prepare] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/v3/execute", response_model=AnalyticsExecutionResponse)
def execute_virtual_query(req: AnalyticsExecuteRequest):
    """
    Stage C: Executes authorized prepared query against isolated virtual SQLite.
    Validates HMAC signature, expiry, replay cache, and query fingerprint.
    """
    try:
        response = virtual_execution_gateway.execute_authorized_query(req)
        return response
    except ExecutionGatewaySecurityError as egse:
        logger.warning(f"[Contract 3.0 Execute] Security error: {egse} (code={egse.code})")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=f"{str(egse)} (code={egse.code})")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Contract 3.0 Execute] Unexpected error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
