# Embedding Service

Python FastAPI embedding service for the Multi-Source Chatbot.

---

## Requirements

- Python 3.12+
- pip
- Virtual Environment

---

## Create Virtual Environment

```bash
py -3.12 -m venv venv
```

---

## Activate Virtual Environment

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configure Environment

Copy the example environment file.

```bash
copy .env.example .env
```

Update values if necessary.

---

## Run the Application

```bash
uvicorn app.main:app --reload --port 8001
```

The service will be available at:

```
http://127.0.0.1:8001
```

---

## API Documentation

Swagger UI

```
http://127.0.0.1:8001/docs
```

---

## Health Check

```
GET http://127.0.0.1:8001/health
```

---

## Deactivate Virtual Environment

```bash
deactivate
```
