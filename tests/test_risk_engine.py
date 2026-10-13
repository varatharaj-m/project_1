from app.services.risk_engine import RiskEngine


def test_risk_engine_marks_uncertainty():
    result = RiskEngine().assess([{"module_name": "video_deepfake", "risk_category": "low", "availability_status": "unavailable"}])
    assert result["risk_category"] == "needs_review"
    assert result["warnings"]
