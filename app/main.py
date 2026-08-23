import logging
import time
import uuid

from contextlib import asynccontextmanager
from contextvars import ContextVar

# pyrefly: ignore [missing-import]
from fastapi import FastAPI, HTTPException, Request, Depends
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]
from fastapi.responses import JSONResponse

from app.embedding import (
    embed,
    embed_batch,
    load_model,
    warmup_model,
    unload_model,
    get_model,
    get_embedding_dimension,
)
from app.models import (
    EmbedRequest,
    BatchEmbedRequest,
    EmbedResponse,
    BatchEmbedResponse,
    ErrorResponse,
    SearchRequest,
    SearchResponse,
    SyncFAQRequest,
    SyncFAQResponse,
    DeleteFAQResponse,
)
from app.config import (
    API_KEY,
    APP_ENV,
    MAX_REQUEST_SIZE,
    MODEL_NAME,
    ALLOWED_ORIGINS,
)
from app.typesense_engine import (
    get_typesense_client,
    ensure_faq_collection,
    upsert_faq_document,
    delete_faq_document,
)
from app.retrieval_engine import search_knowledge_base

# ---------------------------------------------------------------------------
# Request-ID logging context
# ---------------------------------------------------------------------------
_request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIDFilter(logging.Filter):
    """Injects *request_id* from context into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_ctx.get() or "-"
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(request_id)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
# Attach the filter to the root logger so every module benefits.
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIDFilter())

logger = logging.getLogger(__name__)

_start_time = time.time()

# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


async def verify_api_key(request: Request):
    """Dependency — validate ``X-API-Key`` header.

    In ``local`` environment (``APP_ENV=local``) the check is skipped so
    developers can work without configuring a key.
    """
    if APP_ENV == "local" or not API_KEY:
        return
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
        )


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Embedding & Retrieval Service | APP_ENV=%s", APP_ENV)

    try:
        load_model()
        warmup_model()
        # Initialize Typesense FAQ collection schema
        ts_client = get_typesense_client()
        ensure_faq_collection(ts_client)
    except Exception:
        logger.exception("Failed during startup initialization")
    yield
    logger.info("Shutting down Embedding Service")
    unload_model()


app = FastAPI(
    title="Embedding & Retrieval Service",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    _request_id_ctx.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def add_process_time(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response


@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > MAX_REQUEST_SIZE:
        return JSONResponse(
            status_code=413,
            content=ErrorResponse(detail="Request too large").model_dump(),
        )
    return await call_next(request)


# ---------------------------------------------------------------------------
# Exception Handlers
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(detail=exc.detail).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(detail=str(exc)).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(detail="Internal server error").model_dump(),
    )


# ---------------------------------------------------------------------------
# Monitoring Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", tags=["Monitoring"])
def health():
    """Health check with model metadata, Typesense status, and uptime."""
    model = get_model()
    ts_ok = False
    try:
        ts_res = get_typesense_client().operations.is_healthy()
        ts_ok = bool(ts_res)
    except Exception:
        pass

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME if model is not None else None,
        "embedding_dimension": get_embedding_dimension(),
        "typesense_healthy": ts_ok,
        "device": str(model.device) if model is not None else None,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


# ---------------------------------------------------------------------------
# Knowledge Retrieval Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/api/v1/search",
    response_model=SearchResponse,
    tags=["Knowledge Retrieval"],
    dependencies=[Depends(verify_api_key)],
)
async def search_faqs(request: SearchRequest):
    """
    Search FAQs using adaptive hybrid search (Dense 768-d vector + BM25 keyword + optional LLM query expansion).
    """
    return await search_knowledge_base(
        query=request.query,
        workspace_id=request.workspace_id,
        top_k=request.top_k,
    )


@app.post(
    "/api/v1/faqs/sync",
    response_model=SyncFAQResponse,
    tags=["Knowledge Retrieval"],
    dependencies=[Depends(verify_api_key)],
)
def sync_faq(request: SyncFAQRequest):
    """
    Generate embedding for FAQ question+answer and upsert document into Typesense.
    """
    combined_text = f"{request.question.strip()} {request.answer.strip()}"
    vector = embed(combined_text)

    doc = {
        "id": str(request.id),
        "workspace_id": int(request.workspace_id),
        "question": request.question.strip(),
        "answer": request.answer.strip(),
        "priority": int(request.priority),
        "is_active": bool(request.is_active),
        "embedding": vector,
    }

    client = get_typesense_client()
    upsert_faq_document(client, doc)

    return SyncFAQResponse(
        status="synced",
        id=str(request.id),
        workspace_id=int(request.workspace_id),
    )


@app.delete(
    "/api/v1/faqs/{faq_id}",
    response_model=DeleteFAQResponse,
    tags=["Knowledge Retrieval"],
    dependencies=[Depends(verify_api_key)],
)
def delete_faq(faq_id: str):
    """
    Delete FAQ document and its embedding from Typesense.
    """
    client = get_typesense_client()
    delete_faq_document(client, faq_id)

    return DeleteFAQResponse(status="deleted", id=faq_id)


# ---------------------------------------------------------------------------
# Embedding Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/embed",
    response_model=EmbedResponse,
    tags=["Embedding"],
    dependencies=[Depends(verify_api_key)],
)
def embedding(request: EmbedRequest, req: Request):
    """Generate an embedding vector for a single text input."""
    start = time.time()
    vector = embed(request.text)
    elapsed = time.time() - start
    logger.info(
        "embed | endpoint=%s | client=%s | duration=%.4fs | dim=%d",
        req.url.path,
        req.client.host if req.client else "unknown",
        elapsed,
        len(vector),
    )
    return EmbedResponse(embedding=vector, dimensions=len(vector))


@app.post(
    "/embed-batch",
    response_model=BatchEmbedResponse,
    tags=["Embedding"],
    dependencies=[Depends(verify_api_key)],
)
def embedding_batch(request: BatchEmbedRequest, req: Request):
    """Generate embedding vectors for a batch of text inputs."""
    batch_size = len(request.texts)
    start = time.time()
    vectors = embed_batch(request.texts)
    elapsed = time.time() - start
    logger.info(
        "embed-batch | endpoint=%s | client=%s | batch=%d | duration=%.4fs",
        req.url.path,
        req.client.host if req.client else "unknown",
        batch_size,
        elapsed,
    )
    return BatchEmbedResponse(embeddings=vectors, dimensions=len(vectors[0]))
