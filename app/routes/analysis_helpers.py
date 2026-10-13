from datetime import datetime, timezone

from app.extensions import db
from app.models import Analysis, AnalysisResult


def persist_analysis(user, content_type, results, text_content=None, filename=None, storage_path=None):
    analysis = Analysis(
        user_id=user.id,
        content_type=content_type,
        original_filename=filename,
        safe_storage_path=storage_path,
        text_content=text_content,
        status="completed",
        completed_at=datetime.now(timezone.utc),
    )
    db.session.add(analysis)
    db.session.flush()
    for result in results:
        db.session.add(
            AnalysisResult(
                analysis_id=analysis.id,
                module_name=result["module_name"],
                model_name=result["model_name"],
                model_version=result["model_version"],
                predicted_labels=result["predicted_labels"],
                confidence_scores=result["confidence_scores"],
                risk_category=result["risk_category"],
                explanation=result["explanation"],
                availability_status=result["availability_status"],
            )
        )
    db.session.commit()
    return analysis


def response_for(analysis):
    return {
        "success": True,
        "analysis_id": analysis.id,
        "content_type": analysis.content_type,
        "status": analysis.status,
        "results": [result.to_dict() for result in analysis.results],
        "limitations": [
            "Model outputs are screening signals and review priorities, not verified proof of malicious intent or authenticity."
        ],
    }
