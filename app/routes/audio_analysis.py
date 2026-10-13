from flask import Blueprint, jsonify, request

from app.routes.analysis_helpers import persist_analysis, response_for
from app.routes.auth import login_required
from app.services.audio_deepfake_detector import AudioDeepfakeDetector
from app.services.audio_transcriber import AudioTranscriber
from app.services.file_validation import AUDIO_EXTENSIONS, ValidationError, save_upload

bp = Blueprint("audio_analysis_api", __name__, url_prefix="/api/v1/analyze")


@bp.post("/audio")
def analyze_audio():
    user, error = login_required()
    if error:
        return error
    try:
        file = request.files.get("file")
        path, _suffix = save_upload(file, "audio", AUDIO_EXTENSIONS)
        results = [AudioTranscriber().analyze(path), AudioDeepfakeDetector().analyze(path)]
        analysis = persist_analysis(user, "audio", results, filename=file.filename, storage_path=path)
        return jsonify(response_for(analysis))
    except ValidationError as exc:
        return jsonify({"success": False, "error": "invalid_upload", "message": str(exc)}), 400
