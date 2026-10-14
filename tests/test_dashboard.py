def test_dashboard_stats(auth_client):
    auth_client.post("/api/v1/analyze/text", json={"text": "hello"})
    response = auth_client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    assert response.get_json()["total_analyses"] == 1


def test_frontend_pages_render(client):
    for path in ["/", "/dashboard", "/text", "/news", "/media", "/history", "/reports"]:
        response = client.get(path)
        assert response.status_code == 200
        assert b"ShieldVision" in response.data

    settings = client.get("/settings")
    assert settings.status_code == 302
    assert settings.headers["Location"] == "/dashboard"


def test_dashboard_stats_without_login(client):
    response = client.get("/api/v1/dashboard/stats")
    assert response.status_code == 200
    assert response.get_json()["success"] is True


def test_dashboard_openai_api_key_can_be_set_temporarily(auth_client):
    response = auth_client.post("/api/v1/dashboard/settings", json={"api_key": "sk-test-temp", "persist": False})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is True
    assert payload["persisted"] is False

    check = auth_client.get("/api/v1/dashboard/settings")
    assert check.status_code == 200
    assert check.get_json()["openai_api_key_configured"] is True


def test_health_reports_model_states(client):
    response = client.get("/api/v1/health")
    data = response.get_json()
    assert response.status_code == 200
    assert set(data["models"]) >= {"text", "news", "image", "video", "audio_transcriber", "audio_deepfake"}
