from typing import List, Optional
# pyrefly: ignore [missing-import]
from pydantic import BaseModel, Field

from app.config import MAX_TEXT_LENGTH, MAX_BATCH_SIZE


class EmbedRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Input text to embed. Must be a non-empty string.",
    )


class BatchEmbedRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description=(
            "List of texts to embed. "
            f"Must contain between 1 and {MAX_BATCH_SIZE} non-empty items."
        ),
    )


class EmbedResponse(BaseModel):
    embedding: list[float]
    dimensions: int


class BatchEmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dimensions: int


class ErrorResponse(BaseModel):
    detail: str


# -- Knowledge Retrieval Models --

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH)
    workspace_id: Optional[int] = None
    top_k: int = Field(5, ge=1, le=50)


class SearchResultItem(BaseModel):
    id: str
    question: str
    answer: str
    priority: int = 0
    score: float
    match_type: str = "hybrid"
    keyword_score: float = 0.0
    semantic_score: float = 0.0


class RetrievalTelemetry(BaseModel):
    request_id: Optional[str] = None
    workspace_id: Optional[int] = None
    first_pass_score: float = 0.0
    first_pass_top_id: Optional[str] = None
    expansion_triggered: bool = False
    expanded_query: Optional[str] = None
    second_pass_score: Optional[float] = None
    second_pass_top_id: Optional[str] = None
    final_score: float = 0.0
    first_pass_latency_ms: float = 0.0
    expansion_latency_ms: float = 0.0
    second_pass_latency_ms: float = 0.0
    total_retrieval_latency_ms: float = 0.0
    returned_faq_ids: List[str] = []
    reranker_applied: bool = False
    reranker_reason: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchResultItem]
    query: str
    expanded_query: Optional[str] = None
    expansion_applied: bool = False
    total_found: int
    telemetry: Optional[RetrievalTelemetry] = None



class SyncFAQRequest(BaseModel):
    id: int | str
    workspace_id: int
    question: str
    answer: str
    priority: int = 0
    is_active: bool = True
    lexicon_terms: List[str] = []


class SyncFAQResponse(BaseModel):
    status: str = "synced"
    id: str
    workspace_id: int


class DeleteFAQResponse(BaseModel):
    status: str = "deleted"
    id: str
