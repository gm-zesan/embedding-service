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