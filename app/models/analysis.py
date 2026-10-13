from datetime import datetime, timezone

from app.extensions import db


class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)
    content_type = db.Column(db.String(32), nullable=False, index=True)
    original_filename = db.Column(db.String(255))
    safe_storage_path = db.Column(db.String(512))
    text_content = db.Column(db.Text)
    status = db.Column(db.String(32), nullable=False, default="pending", index=True)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)
    completed_at = db.Column(db.DateTime)

    user = db.relationship("User", back_populates="analyses")
    results = db.relationship("AnalysisResult", back_populates="analysis", cascade="all, delete-orphan")

    def to_dict(self, include_results=False):
        data = {
            "id": self.id,
            "content_type": self.content_type,
            "original_filename": self.original_filename,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }
        if include_results:
            data["results"] = [result.to_dict() for result in self.results]
        return data
