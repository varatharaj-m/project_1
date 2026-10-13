import os
import uuid
from pathlib import Path

from flask import current_app
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


class ValidationError(ValueError):
    pass


def validate_extension(filename, allowed):
    suffix = Path(filename or "").suffix.lower()
    if suffix not in allowed:
        raise ValidationError(f"Unsupported file extension: {suffix or 'none'}.")
    return suffix


def save_upload(file_storage, subdir, allowed_extensions):
    if not file_storage or not file_storage.filename:
        raise ValidationError("A file upload is required.")
    suffix = validate_extension(file_storage.filename, allowed_extensions)
    storage_dir = Path(current_app.config["UPLOAD_DIR"]) / subdir
    storage_dir.mkdir(parents=True, exist_ok=True)
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file_storage.filename)}"
    path = storage_dir / safe_name
    file_storage.save(path)
    if os.path.getsize(path) == 0:
        path.unlink(missing_ok=True)
        raise ValidationError("Uploaded file is empty.")
    return str(path), suffix


def validate_image_signature(path):
    try:
        with Image.open(path) as image:
            detected = (image.format or "").lower()
    except UnidentifiedImageError as exc:
        raise ValidationError("File signature is not a supported image type.") from exc
    if detected not in {"jpeg", "png", "gif", "webp", "bmp"}:
        raise ValidationError("File signature is not a supported image type.")
    return detected
