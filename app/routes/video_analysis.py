from flask import Blueprint, jsonify, request

from app.routes.analysis_helpers import persist_analysis, response_for
from app.routes.auth import login_required
from app.services.file_validation import VIDEO_EXTENSIONS, ValidationError, save_upload
from app.services.video_detector import VideoDetector

bp = Blueprint("video_analysis_api", __name__, url_prefix="/api/v1/analyze")


@bp.post("/video")
def analyze_video():
    user, error = login_required()
    if error:
        return error
    try:
        file = request.files.get("file")
        path, _suffix = save_upload(file, "videos", VIDEO_EXTENSIONS)
        result = VideoDetector().analyze(path)
        analysis = persist_analysis(user, "video", [result], filename=file.filename, storage_path=path)
        return jsonify(response_for(analysis)), 200 if result["availability_status"] != "invalid_input" else 400
    except ValidationError as exc:
        return jsonify({"success": False, "error": "invalid_upload", "message": str(exc)}), 400
