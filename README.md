# Backend (Flask) - legal_ai

This directory contains the Python Flask backend used for document processing, OCR, and LLM interactions.

## Quick start (local)

1. Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Set environment variables (example):

```powershell
$env:FLASK_APP = "app.py"
$env:FLASK_ENV = "development"
# If using Google GenAI:
$env:GOOGLE_API_KEY = "your_google_api_key"
# If using Supabase:
$env:SUPABASE_URL = "https://..."
$env:SUPABASE_KEY = "your_supabase_key"
```

4. Run the app:

```powershell
flask run --host=0.0.0.0 --port=5000
```

## Important files
- `app.py` — Flask routes and API endpoints.
- `llm_service.py` — LLM interaction helpers and prompt logic.
- `ocr_service.py` — OCR and PDF parsing helpers.
- `text_cleaner.py` — Text normalization utilities.
- `uploads/` — Uploaded documents (local only).

## Security
- This repository intentionally keeps secrets out of version control. Do not commit `.env` or service account credentials. Use an external secrets store or the parent repo's `..\repo-secrets` directory used in earlier workflows.

## Notes
- The backend directory may be intentionally untracked in the repository root to keep local files private. If you want to push backend files back to the repo, remove `backend/` from `.gitignore` and commit.
