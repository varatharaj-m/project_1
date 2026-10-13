from flask import Blueprint, jsonify

from app.models import Analysis
from app.routes.auth import login_required
from app.services.report_service import ReportService

bp = Blueprint("reports_api", __name__, url_prefix="/api/v1")


@bp.get("/reports/<int:analysis_id>")
def report(analysis_id):
    user, error = login_required()
    if error:
        return error
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first_or_404()
    return jsonify({"success": True, "report": ReportService().build(analysis)})
