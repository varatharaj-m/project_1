# CyberShield AI

CyberShield AI is a Flask web app for multimodal digital safety screening across text, news claims, images, video, and audio.

The included lightweight text and news detectors are transparent baselines. Image, video, audio transcription, and synthetic-audio authenticity modules expose adapters and return explicit `unavailable` statuses until dedicated model weights are configured. The app never treats an unavailable model as proof that content is authentic or safe.

## Windows PowerShell setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
Copy-Item .env.example .env
python scripts/init_db.py
python run.py
```

Open http://127.0.0.1:5000.

## API examples

```powershell
$body = @{ username="analyst"; email="analyst@example.com"; password="password123" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/auth/register -Method Post -Body $body -ContentType "application/json" -SessionVariable s

$text = @{ text="This claim is shocking and they don't want you to know" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/analyze/news -Method Post -Body $text -ContentType "application/json" -WebSession $s
```

## Models and datasets

Use `scripts/download_models.py` as the explicit entry point for model download planning. Store licensed datasets in `datasets/` and trained or downloaded weights in `model_weights/`. Do not report training metrics unless scripts are run against real datasets.

## Migrations

For local development, `python scripts/init_db.py` creates the SQLite schema. For production, configure MySQL in `DATABASE_URL`, run `flask db init` once, then `flask db migrate` and `flask db upgrade`.

## Tests

```powershell
pytest
```

Tests use mocked/lightweight services and SQLite in memory.
