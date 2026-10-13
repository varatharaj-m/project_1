def test_text_analysis_allows_local_workspace(client):
    response = client.post("/api/v1/analyze/text", json={"text": "hello"})
    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_text_analysis_valid(auth_client):
    response = auth_client.post("/api/v1/analyze/text", json={"text": "You are kind but that insult is stupid"})
    data = response.get_json()
    assert response.status_code == 200
    assert data["success"] is True
    assert data["results"][0]["module"] == "cyberbullying"


def test_text_analysis_empty(auth_client):
    response = auth_client.post("/api/v1/analyze/text", json={"text": ""})
    assert response.status_code == 400


def test_text_analysis_flags_appearance_mockery(auth_client):
    response = auth_client.post("/api/v1/analyze/text", json={"text": "how are you, what is on your face it is so funny"})
    result = response.get_json()["results"][0]
    assert response.status_code == 200
    assert "appearance_based_bullying" in result["labels"]
    assert result["risk_category"] == "medium"
