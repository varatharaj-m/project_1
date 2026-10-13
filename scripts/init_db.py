import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import create_app
from app.extensions import db
from app.models import Analysis, AnalysisResult, AuditLog, User  # noqa: F401

app = create_app()

with app.app_context():
    db.create_all()
    print("Database initialized.")
