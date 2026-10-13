def test_news_analysis_missing_evidence_notice(auth_client):
    response = auth_client.post("/api/v1/analyze/news", json={"text": "A shocking secret miracle claim"})
    result = response.get_json()["results"][0]
    assert response.status_code == 200
    assert result["module"] == "misinformation"
    assert "not fact verification" in result["explanation"]
