from datetime import datetime, timezone

from app.extensions import db


class AnalysisResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.Integer, db.ForeignKey("analysis.id"), nullable=False, index=True)
    module_name = db.Column(db.String(80), nullable=False, index=True)
    model_name = db.Column(db.String(160), nullable=False)
    model_version = db.Column(db.String(80), nullable=False, default="1.0")
    predicted_labels = db.Column(db.JSON, nullable=False, default=list)
    confidence_scores = db.Column(db.JSON, nullable=False, default=dict)
    risk_category = db.Column(db.String(32), nullable=False, default="needs_review")
    explanation = db.Column(db.Text, nullable=False)
    availability_status = db.Column(db.String(32), nullable=False, default="available")
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc), nullable=False)

    analysis = db.relationship("Analysis", back_populates="results")

    def to_dict(self):
        return {
            "id": self.id,
            "module": self.module_name,
            "model": self.model_name,
            "model_version": self.model_version,
            "model_status": self.availability_status,
            "labels": self.predicted_labels,
            "confidence": self.confidence_scores,
            "risk_category": self.risk_category,
            "explanation": self.explanation,
        }
