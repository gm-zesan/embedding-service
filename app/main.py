import logging
import time
import uuid

from contextlib import asynccontextmanager
from contextvars import ContextVar

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
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
)
from app.config import (
    API_KEY,
    APP_ENV,
    MAX_REQUEST_SIZE,
    MODEL_NAME,
    ALLOWED_ORIGINS,
)

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
    if APP_ENV == "local":
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
    logger.info("Starting Embedding Service | APP_ENV=%s", APP_ENV)

    # Fail fast when API_KEY is missing in non-local environments.
    if APP_ENV != "local" and not API_KEY:
        raise RuntimeError(
            "API_KEY is required when APP_ENV is not 'local'. "
            "Set API_KEY in .env or use APP_ENV=local for development."
        )

    try:
        load_model()
        warmup_model()
    except Exception:
        logger.exception("Failed to load model on startup")
    yield
    logger.info("Shutting down Embedding Service")
    unload_model()


app = FastAPI(
    title="Embedding Service",
    version="1.0.0",
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
    """Health check with model metadata and process uptime."""
    model = get_model()
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_name": MODEL_NAME if model is not None else None,
        "embedding_dimension": get_embedding_dimension(),
        "device": str(model.device) if model is not None else None,
        "uptime_seconds": round(time.time() - _start_time, 2),
    }


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
    """Generate an embedding vector for a single text input.

    The returned vector is L2-normalised (unit length) for use with
    cosine-similarity search.
    """
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
    """Generate embedding vectors for a batch of text inputs.

    Each returned vector is L2-normalised (unit length).
    """
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