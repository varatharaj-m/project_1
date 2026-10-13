import logging
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request

from .config import Config
from .extensions import cors, db, migrate


class RequestIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


def create_app(config_object=Config):
    app = Flask(__name__)
    app.config.from_object(config_object)

    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)

    db.init_app(app)
    migrate.init_app(app, db)
    cors.init_app(app, resources={r"/api/*": {"origins": app.config.get("CORS_ORIGINS", "*")}})

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s request_id=%(request_id)s %(message)s",
    )
    for handler in logging.getLogger().handlers:
        handler.addFilter(RequestIdFilter())

    from .routes import register_blueprints

    register_blueprints(app)

    @app.before_request
    def assign_request_id():
        g.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    @app.after_request
    def add_request_id(response):
        response.headers["X-Request-ID"] = getattr(g, "request_id", "")
        return response

    @app.errorhandler(404)
    def not_found(_error):
        return jsonify({"success": False, "error": "not_found", "request_id": g.get("request_id")}), 404

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"success": False, "error": "file_too_large", "request_id": g.get("request_id")}), 413

    @app.errorhandler(Exception)
    def internal_error(error):
        app.logger.exception("Unhandled error", extra={"request_id": g.get("request_id", "-")})
        if app.config.get("TESTING"):
            raise error
        return jsonify({"success": False, "error": "internal_server_error", "request_id": g.get("request_id")}), 500

    return app
