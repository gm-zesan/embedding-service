import logging
import time
import gc
import threading

import torch
from sentence_transformers import SentenceTransformer

from app.config import MODEL_NAME, EMBED_BATCH_SIZE

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock: threading.Lock = threading.Lock()
_embedding_dimension: int = 0


def get_model() -> SentenceTransformer | None:
    """Return the loaded model instance, or *None* if not yet loaded."""
    return _model


def get_embedding_dimension() -> int:
    """Return the cached embedding dimension (avoids querying the model at runtime)."""
    return _embedding_dimension


def load_model() -> SentenceTransformer:
    """Load the embedding model into memory.

    Idempotent — safe to call multiple times.  Returns the existing instance
    if already loaded.
    """
    global _model, _embedding_dimension
    if _model is not None:
        return _model

    logger.info("Loading model: %s", MODEL_NAME)
    start = time.time()
    _model = SentenceTransformer(MODEL_NAME)
    elapsed = time.time() - start
    _embedding_dimension = _model.get_sentence_embedding_dimension()
    logger.info(
        "Model loaded | name=%s | device=%s | dim=%d | duration=%.2fs",
        MODEL_NAME,
        _model.device,
        _embedding_dimension,
        elapsed,
    )
    return _model


def warmup_model() -> None:
    """Run a tiny embedding to warm up the model and avoid cold-start latency."""
    model = get_model()
    if model is None:
        return
    logger.info("Running model warmup ...")
    model.encode(
        "warmup",
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    logger.info("Model warmup complete")


def unload_model() -> None:
    """Release the model and free associated GPU memory."""
    global _model, _embedding_dimension
    if _model is not None:
        logger.info("Releasing model resources")
        _model = None
        _embedding_dimension = 0
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def embed(text: str) -> list[float]:
    """Generate a single embedding vector for *text*.

    Raises
    ------
    ValueError
        If *text* is empty or whitespace-only.
    RuntimeError
        If the model has not been loaded (call :func:`load_model` first).
    """
    if not text or not text.strip():
        raise ValueError("text must be a non-empty string")

    model = get_model()
    if model is None:
        raise RuntimeError("Model is not loaded")

    with _model_lock:
        vector = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    return vector.tolist()


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a batch of texts.

    Raises
    ------
    ValueError
        If *texts* is empty or contains empty entries.
    RuntimeError
        If the model has not been loaded.
    """
    if not texts:
        raise ValueError("texts must be a non-empty list")

    for i, t in enumerate(texts):
        if not t or not t.strip():
            raise ValueError(f"texts[{i}] is empty or whitespace-only")

    model = get_model()
    if model is None:
        raise RuntimeError("Model is not loaded")

    batch_count = len(texts)
    logger.debug("Embedding batch (count=%d)", batch_count)

    start = time.time()
    with _model_lock:
        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=EMBED_BATCH_SIZE,
        )
    elapsed = time.time() - start
    throughput = batch_count / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Batch complete | count=%d | duration=%.3fs | throughput=%.1f texts/s",
        batch_count,
        elapsed,
        throughput,
    )

    return [v.tolist() for v in vectors]