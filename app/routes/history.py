from flask import Blueprint, jsonify, request

from app.extensions import db
from app.models import Analysis
from app.routes.auth import login_required

bp = Blueprint("history_api", __name__, url_prefix="/api/v1")


@bp.get("/analyses")
def list_analyses():
    user, error = login_required()
    if error:
        return error
    page = max(1, request.args.get("page", default=1, type=int))
    per_page = min(50, max(1, request.args.get("per_page", default=10, type=int)))
    content_type = request.args.get("content_type")
    query = Analysis.query.filter_by(user_id=user.id)
    if content_type:
        query = query.filter_by(content_type=content_type)
    pagination = query.order_by(Analysis.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    return jsonify({"success": True, "items": [item.to_dict(include_results=True) for item in pagination.items], "total": pagination.total})


@bp.get("/analyses/<int:analysis_id>")
def get_analysis(analysis_id):
    user, error = login_required()
    if error:
        return error
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first_or_404()
    return jsonify({"success": True, "analysis": analysis.to_dict(include_results=True)})


@bp.delete("/analyses/<int:analysis_id>")
def delete_analysis(analysis_id):
    user, error = login_required()
    if error:
        return error
    analysis = Analysis.query.filter_by(id=analysis_id, user_id=user.id).first_or_404()
    db.session.delete(analysis)
    db.session.commit()
    return jsonify({"success": True})
