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