from flask import Blueprint, jsonify

from app.services.model_registry import refresh_model_registry, registry

bp = Blueprint("health_api", __name__, url_prefix="/api/v1")


@bp.get("/health")
def health():
    refresh_model_registry()
    return jsonify({"success": True, "status": "ok", "models": registry.health()})
