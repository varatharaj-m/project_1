from flask import Blueprint, jsonify, request, session

from app.extensions import db
from app.models import User

bp = Blueprint("auth_api", __name__, url_prefix="/api/v1/auth")
LOCAL_USERNAME = "local_workspace"
LOCAL_EMAIL = "local@cybershield.local"


def local_workspace_user():
    user = User.query.filter_by(username=LOCAL_USERNAME).first()
    if user:
        return user
    user = User(username=LOCAL_USERNAME, email=LOCAL_EMAIL, role="analyst")
    user.set_password("local-workspace-disabled-login")
    db.session.add(user)
    db.session.commit()
    return user


def current_user():
    user_id = session.get("user_id")
    return db.session.get(User, user_id) if user_id else local_workspace_user()


def login_required():
    return current_user(), None


@bp.post("/register")
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if len(username) < 3 or "@" not in email or len(password) < 8:
        return jsonify({"success": False, "error": "invalid_registration"}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({"success": False, "error": "user_exists"}), 409
    user = User(username=username, email=email)
    user.set_password(password)
    db.session.add(user)
    db.session.commit()
    session["user_id"] = user.id
    return jsonify({"success": True, "user": user.to_dict()}), 201


@bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    identifier = (data.get("email") or data.get("username") or "").strip().lower()
    password = data.get("password") or ""
    user = User.query.filter((User.email == identifier) | (User.username == identifier)).first()
    if not user or not user.check_password(password):
        return jsonify({"success": False, "error": "invalid_credentials"}), 401
    session["user_id"] = user.id
    return jsonify({"success": True, "user": user.to_dict()})


@bp.post("/logout")
def logout():
    session.clear()
    return jsonify({"success": True})


@bp.get("/me")
def me():
    user, error = login_required()
    if error:
        return error
    return jsonify({"success": True, "user": user.to_dict()})
