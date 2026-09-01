# pyrefly: ignore [missing-import]
from dotenv import load_dotenv
import os

load_dotenv()

# -- Environment --
APP_ENV = os.getenv("APP_ENV", "local")

# -- Model --
MODEL_NAME = os.getenv("MODEL_NAME", "paraphrase-multilingual-mpnet-base-v2")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

# -- Server --
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8001))
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "").split(",")
    if o.strip()
]

# -- Security --
API_KEY = os.getenv("API_KEY", "")

# -- Limits --
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "8192"))
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "100"))
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10 MB

# -- Typesense Settings --
TYPESENSE_HOST = os.getenv("TYPESENSE_HOST", "127.0.0.1")
TYPESENSE_PORT = int(os.getenv("TYPESENSE_PORT", "8108"))
TYPESENSE_PROTOCOL = os.getenv("TYPESENSE_PROTOCOL", "http")
TYPESENSE_API_KEY = os.getenv("TYPESENSE_API_KEY", "")
TYPESENSE_COLLECTION = os.getenv("TYPESENSE_COLLECTION", "faqs")

# -- Retrieval & Adaptive Expansion --
RETRIEVAL_EXPANSION_THRESHOLD = float(os.getenv("RETRIEVAL_EXPANSION_THRESHOLD", "0.65"))
RETRIEVAL_LLM_PROVIDER = os.getenv("RETRIEVAL_LLM_PROVIDER", "deepseek")
RETRIEVAL_LLM_FALLBACK_PROVIDER = os.getenv("RETRIEVAL_LLM_FALLBACK_PROVIDER", "openrouter")
RETRIEVAL_LLM_API_KEY = os.getenv("RETRIEVAL_LLM_API_KEY", os.getenv("LLM_EXPANSION_API_KEY", ""))
RETRIEVAL_LLM_BASE_URL = os.getenv("RETRIEVAL_LLM_BASE_URL", os.getenv("LLM_EXPANSION_BASE_URL", "https://api.deepseek.com/v1"))
RETRIEVAL_LLM_MODEL = os.getenv("RETRIEVAL_LLM_MODEL", os.getenv("LLM_EXPANSION_MODEL", "deepseek-chat"))

# Backward compatibility aliases
LLM_EXPANSION_API_KEY = RETRIEVAL_LLM_API_KEY
LLM_EXPANSION_BASE_URL = RETRIEVAL_LLM_BASE_URL
LLM_EXPANSION_MODEL = RETRIEVAL_LLM_MODEL

# -- Post-Retrieval Candidate Reranker --
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1")
RERANKER_CLOSE_DELTA = float(os.getenv("RERANKER_CLOSE_DELTA", "0.06"))
RERANKER_MULTI_ENTITY_ENABLED = os.getenv("RERANKER_MULTI_ENTITY_ENABLED", "true").lower() in ("true", "1")
