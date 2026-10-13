from flask import Blueprint, jsonify, request

from app.routes.analysis_helpers import persist_analysis, response_for
from app.routes.auth import login_required
from app.services.fake_news_detector import FakeNewsDetector

bp = Blueprint("news_analysis_api", __name__, url_prefix="/api/v1/analyze")


@bp.post("/news")
def analyze_news():
    user, error = login_required()
    if error:
        return error
    data = request.get_json(silent=True) or {}
    text = data.get("text") or data.get("claim") or data.get("article") or ""
    result = FakeNewsDetector().analyze(text)
    status_code = 400 if result["availability_status"] == "invalid_input" else 200
    analysis = persist_analysis(user, "news", [result], text_content=text)
    return jsonify(response_for(analysis)), status_code
