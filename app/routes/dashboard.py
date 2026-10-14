import os
from pathlib import Path

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session

from app.models import Analysis, AnalysisResult
from app.routes.auth import current_user, login_required
from app.services.model_registry import refresh_model_registry, registry

bp = Blueprint("dashboard", __name__)
api = Blueprint("dashboard_api", __name__, url_prefix="/api/v1/dashboard")


@bp.get("/")
def index():
    return render_template("dashboard.html", user=current_user())


@bp.get("/dashboard")
def dashboard_page():
    return render_template("dashboard.html", user=current_user())


@bp.get("/text")
def text_page():
    return render_template("text_analysis.html")


@bp.get("/news")
def news_page():
    return render_template("news_analysis.html")


@bp.get("/media")
def media_page():
    return render_template("media_analysis.html")


@bp.get("/history")
def history_page():
    return render_template("history.html")


@bp.get("/reports")
def reports_page():
    return render_template("reports.html")


def _mask_api_key(value):
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return f"{value[:4]}...{value[-4:]}"


def _resolve_openai_api_key():
    if session.get("openai_api_key"):
        return session["openai_api_key"]
    return os.getenv("OPENAI_API_KEY", current_app.config.get("OPENAI_API_KEY", ""))


def _persist_openai_api_key(api_key):
    env_path = Path(current_app.root_path).resolve().parent / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    updated = False
    for index, line in enumerate(lines):
        if line.startswith("OPENAI_API_KEY="):
            lines[index] = f"OPENAI_API_KEY={api_key}"
            updated = True
            break
    if not updated:
        lines.append(f"OPENAI_API_KEY={api_key}")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.environ["OPENAI_API_KEY"] = api_key
    current_app.config["OPENAI_API_KEY"] = api_key


@bp.get("/settings")
def settings_page():
    return redirect("/dashboard")


@api.get("/settings")
def get_settings():
    api_key = _resolve_openai_api_key()
    source = "session" if session.get("openai_api_key") else "environment" if os.getenv("OPENAI_API_KEY") else "none"
    return jsonify({
        "success": True,
        "openai_api_key_configured": bool(api_key),
        "source": source,
        "openai_api_key_preview": _mask_api_key(api_key),
        "persisted": bool(os.getenv("OPENAI_API_KEY")),
    })


@api.post("/settings")
def save_settings():
    payload = request.get_json(silent=True) or {}
    api_key = (payload.get("api_key") or "").strip()
    persist = bool(payload.get("persist"))

    if not api_key:
        return jsonify({"success": False, "error": "api_key_required"}), 400

    if persist:
        _persist_openai_api_key(api_key)
        session.pop("openai_api_key", None)
        persisted = True
    else:
        session["openai_api_key"] = api_key
        current_app.config["OPENAI_API_KEY"] = api_key
        persisted = False

    return jsonify({
        "success": True,
        "persisted": persisted,
        "source": "environment" if persisted else "session",
        "openai_api_key_preview": _mask_api_key(api_key),
    })


@api.get("/stats")
def stats():
    user, error = login_required()
    if error:
        return error
    refresh_model_registry()
    analyses = Analysis.query.filter_by(user_id=user.id).all()
    ids = [item.id for item in analyses]
    high_priority = AnalysisResult.query.filter(AnalysisResult.analysis_id.in_(ids), AnalysisResult.risk_category.in_(["high", "needs_review"])).count() if ids else 0
    by_type = {}
    risk = {}
    for analysis in analyses:
        by_type[analysis.content_type] = by_type.get(analysis.content_type, 0) + 1
    activity = {}
    high_risk_recent = []
    for analysis in analyses:
        day = analysis.created_at.date().isoformat() if analysis.created_at else "unknown"
        activity[day] = activity.get(day, 0) + 1
        for result in analysis.results:
            risk[result.risk_category] = risk.get(result.risk_category, 0) + 1
            if result.risk_category in {"high", "needs_review"} and len(high_risk_recent) < 6:
                high_risk_recent.append(
                    {
                        "analysis_id": analysis.id,
                        "content_type": analysis.content_type,
                        "risk_category": result.risk_category,
                        "module": result.module_name,
                        "model_status": result.availability_status,
                        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
                    }
                )
    available_models = sum(1 for item in registry.health().values() if item.get("status") == "available")
    return jsonify({
        "success": True,
        "total_analyses": len(analyses),
        "analyses_by_type": by_type,
        "high_priority": high_priority,
        "risk_distribution": risk,
        "activity_by_day": activity,
        "high_risk_recent": high_risk_recent,
        "available_models": available_models,
        "recent": [item.to_dict(include_results=True) for item in analyses[-5:]][::-1],
        "model_health": registry.health(),
    })
