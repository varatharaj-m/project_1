import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _project_path(value):
    path = Path(value)
    return path if path.is_absolute() else BASE_DIR / path


def _optional_project_path(value):
    return str(_project_path(value)) if value else ""


def _database_url(value):
    if not value:
        return f"sqlite:///{(BASE_DIR / 'instance' / 'cybershield.db').as_posix()}"
    prefix = "sqlite:///"
    if value.startswith(prefix):
        database_path = value[len(prefix):]
        path = Path(database_path)
        if database_path and not path.is_absolute():
            return f"sqlite:///{_project_path(database_path).as_posix()}"
    return value


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-me")
    SQLALCHEMY_DATABASE_URI = _database_url(os.getenv("DATABASE_URL"))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 25 * 1024 * 1024))
    UPLOAD_DIR = _project_path(os.getenv("UPLOAD_DIR", "instance/uploads"))
    WTF_CSRF_ENABLED = os.getenv("WTF_CSRF_ENABLED", "false").lower() == "true"
    JSON_SORT_KEYS = False
    TEXT_MAX_CHARS = int(os.getenv("TEXT_MAX_CHARS", "5000"))
    NEWS_MAX_CHARS = int(os.getenv("NEWS_MAX_CHARS", "12000"))
    MODEL_DEVICE = os.getenv("MODEL_DEVICE", "cpu")
    TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "lightweight-keyword-baseline")
    TEXT_MODEL_PATH = _optional_project_path(
        os.getenv("TEXT_MODEL_PATH", "model_weights/text_classifier.joblib")
    )
    FAKE_NEWS_MODEL_NAME = os.getenv("FAKE_NEWS_MODEL_NAME", "tfidf-logreg-baseline")
    IMAGE_MODEL_PATH = _optional_project_path(os.getenv("IMAGE_MODEL_PATH", ""))
    ALLOW_NONCOMMERCIAL_MODELS = os.getenv("ALLOW_NONCOMMERCIAL_MODELS", "false").lower() == "true"
    VIDEO_MODEL_PATH = _optional_project_path(os.getenv("VIDEO_MODEL_PATH", ""))
    AUDIO_MODEL_PATH = _optional_project_path(os.getenv("AUDIO_MODEL_PATH", ""))
    WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL_NAME", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SECRET_KEY = "testing"
    WTF_CSRF_ENABLED = False
    IMAGE_MODEL_PATH = ""
    VIDEO_MODEL_PATH = ""
    AUDIO_MODEL_PATH = ""
    WHISPER_MODEL_NAME = ""
    ALLOW_NONCOMMERCIAL_MODELS = False
