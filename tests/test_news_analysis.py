import pandas as pd

from app.services.fake_news_detector import FakeNewsDetector
from ml.training.train_fake_news_model import train_isot_model


def test_news_analysis_missing_evidence_notice(auth_client):
    response = auth_client.post("/api/v1/analyze/news", json={"text": "A shocking secret miracle claim"})
    result = response.get_json()["results"][0]
    assert response.status_code == 200
    assert result["module"] == "misinformation"
    assert "not fact verification" in result["explanation"]


def test_isot_model_distinguishes_fake_and_true_articles(app, tmp_path):
    fake_dataset = tmp_path / "Fake.csv"
    real_dataset = tmp_path / "True.csv"
    output = tmp_path / "fake_news_classifier.joblib"
    pd.DataFrame(
        [
            {"title": f"fabricated hoax fake claim {index}", "text": f"invented bogus story and fake scheme {index}"}
            for index in range(12)
        ]
    ).to_csv(fake_dataset, index=False)
    pd.DataFrame(
        [
            {"title": f"official verified report {index}", "text": f"published factual news from official sources {index}"}
            for index in range(12)
        ]
    ).to_csv(real_dataset, index=False)

    metadata = train_isot_model(fake_dataset, real_dataset, output)
    app.config["FAKE_NEWS_MODEL_PATH"] = str(output)
    detector = FakeNewsDetector()
    fake_result = detector.analyze("fabricated hoax fake claim, invented bogus story")
    real_result = detector.analyze("official verified report, published factual news from official sources")

    assert metadata["label_mapping"] == ["fake", "real"]
    assert fake_result["predicted_labels"] == ["likely_misleading"]
    assert real_result["predicted_labels"] == ["likely_not_misleading"]
    assert fake_result["confidence_scores"]["misinformation_score"] > real_result["confidence_scores"]["misinformation_score"]
