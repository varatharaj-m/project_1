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

## Local training data and image inference

Raw datasets and generated model weights are kept out of Git. Obtain datasets and checkpoints from their original sources under the applicable terms, then follow [datasets/README.md](datasets/README.md) to prepare local inputs and [ml/README.md](ml/README.md) to train the text/news models. Install the optional TensorFlow dependency in `requirements-image.txt` for image/audio inference and bounded video-frame screening. External deepfake outputs stay review-only until validated on representative labeled data. Speech transcription uses the bundled `imageio-ffmpeg` decoder; the configured Whisper model downloads on first use if it is not cached.

## API examples

```powershell
$body = @{ username="analyst"; email="analyst@example.com"; password="password123" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/auth/register -Method Post -Body $body -ContentType "application/json" -SessionVariable s

$text = @{ text="This claim is shocking and they don't want you to know" } | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5000/api/v1/analyze/news -Method Post -Body $text -ContentType "application/json" -WebSession $s
```

## Models and datasets

| Capability | Current model/service | Important limitation |
|---|---|---|
| Cyberbullying | Trained TF-IDF classifier or local keyword baseline | Screening labels, not judgments about intent |
| Fake-news screening | ISOT TF-IDF classifier or lexical baseline | No external sources are retrieved; not fact verification |
| Image | Configured image checkpoint or explicit unavailable/review status | Supplied external checkpoint is unvalidated and can misclassify real photos |
| Video | Bounded frame sampling through the image model | Not a video-trained detector; results require review |
| Audio speech | Whisper transcription | Transcription is not currently forwarded to cyberbullying screening |
| Synthetic audio | Adapter placeholder | Dedicated inference is not implemented in the current detector |

The model configuration, artifact expectations, licensing, training, and evaluation steps are documented in [model_weights/README.md](model_weights/README.md), [ml/README.md](ml/README.md), and [datasets/README.md](datasets/README.md). Model weights and raw data stay out of Git; use the included scripts to recreate locally. Use `scripts/download_models.py` for model download planning.

## Migrations

For local development, `python scripts/init_db.py` creates the SQLite schema. For production, configure MySQL in `DATABASE_URL`, run `flask db init` once, then `flask db migrate` and `flask db upgrade`.

## Tests

```powershell
pytest
```

Tests use mocked/lightweight services and SQLite in memory.
