# Python AI & Semantic Retrieval Microservice

[![FastAPI](https://img.shields.io/badge/FastAPI-0.116+-009688.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-EE4C2C.svg)](https://pytorch.org)
[![Sentence--Transformers](https://img.shields.io/badge/Sentence--Transformers-5.1+-FFD21E.svg)](https://sbert.net)
[![Typesense](https://img.shields.io/badge/Typesense-29.0-D82C20.svg)](https://typesense.org)
[![Port](https://img.shields.io/badge/Port-8001-blue.svg)](#)

A high-performance standalone Python FastAPI microservice providing **Multilingual Vector Embeddings** (via `paraphrase-multilingual-mpnet-base-v2`), **Hybrid Vector/BM25 Semantic Retrieval** (via Typesense), and **Adaptive Query Expansion** for the Multi-Source Chatbot platform.

---

## 🏛️ Service Overview & Port

- **Service Port:** `8001`
- **Swagger Documentation:** `http://127.0.0.1:8001/docs`
- **Redoc UI:** `http://127.0.0.1:8001/redoc`
- **Health Check:** `http://127.0.0.1:8001/health`
- **Dependencies:** Typesense (`http://127.0.0.1:8108`), MySQL (`3306`)

---

## 📋 System Prerequisites

Ensure you have Python 3.10+ (preferably Python 3.12) installed:

- **macOS / Linux:** `brew install python@3.12`
- **Windows:** Download Python 3.12 installer from [python.org](https://www.python.org/).  
  *(⚠️ Make sure to check **"Add python.exe to PATH"** during installation).*

---

## 🚀 Step-by-Step Manual Setup & Execution

### 1. For macOS / Linux (Terminal)

```bash
# 1. Navigate to service directory
cd /path/to/Entrepreneurs_Automation/python-ai-service

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
cp -n .env.example .env

# 5. Start the FastAPI server with Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

---

### 2. For Windows (PowerShell)

```powershell
# 1. Navigate to service directory
cd C:\path\to\Entrepreneurs_Automation\python-ai-service

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# (If script execution is disabled, run: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Configure environment
if (!(Test-Path .env)) { Copy-Item .env.example .env }

# 5. Start the FastAPI server with Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

---

### 3. For Windows (Command Prompt / CMD)

```cmd
:: 1. Navigate to service directory
cd C:\path\to\Entrepreneurs_Automation\python-ai-service

:: 2. Create and activate virtual environment
python -m venv venv
call venv\Scripts\activate.bat

:: 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

:: 4. Configure environment
if not exist .env copy .env.example .env

:: 5. Start the FastAPI server with Uvicorn
uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
```

---

## ⚙️ Environment Variables Configuration (`.env`)

Verify your `.env` contains the correct connectivity parameters matching Typesense and Laravel:

```ini
APP_ENV=local
HOST=127.0.0.1
PORT=8001
ALLOWED_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Embedding Model
MODEL_NAME=paraphrase-multilingual-mpnet-base-v2
EMBED_BATCH_SIZE=32

# Typesense Connection (Docker port 8108)
TYPESENSE_HOST=127.0.0.1
TYPESENSE_PORT=8108
TYPESENSE_PROTOCOL=http
TYPESENSE_API_KEY=m_JSgPavO2LV4XR0iDxE5BzHxfq4Xym52BPQrNnsd2Y
TYPESENSE_COLLECTION=faqs

# Adaptive Query Expansion Threshold
RETRIEVAL_EXPANSION_THRESHOLD=0.65

# Optional LLM Provider (DeepSeek / OpenAI)
LLM_PROVIDER=deepseek
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat

# Database Settings (MySQL connection matching Laravel)
DB_CONNECTION=mysql
DB_HOST=127.0.0.1
DB_PORT=3306
DB_DATABASE=chatbot_core
DB_USERNAME=root
DB_PASSWORD=root1234
```

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Service and model health probe |
| `POST` | `/embed` | Generates 768-dimensional dense vector embeddings for input text |
| `POST` | `/retrieval/search` | Performs hybrid BM25 + dense vector semantic retrieval with tenant isolation |
| `POST` | `/retrieval/sync` | Ingests and synchronizes FAQs and knowledge chunks into Typesense |
| `POST` | `/retrieval/delete` | Purges documents or FAQs from the vector index |

---

## 🧪 Testing

To run the automated tests:
```bash
pytest
```
