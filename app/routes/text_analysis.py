from flask import Blueprint, jsonify, request

from app.routes.analysis_helpers import persist_analysis, response_for
from app.routes.auth import login_required
from app.services.text_detector import TextDetector

bp = Blueprint("text_analysis_api", __name__, url_prefix="/api/v1/analyze")


@bp.post("/text")
def analyze_text():
    user, error = login_required()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    result = TextDetector().analyze(text)
    status_code = 400 if result["availability_status"] == "invalid_input" else 200
    analysis = persist_analysis(user, "text", [result], text_content=text)
    return jsonify(response_for(analysis)), status_code
