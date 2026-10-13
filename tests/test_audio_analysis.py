import io


def test_audio_unavailable_models(auth_client):
    response = auth_client.post("/api/v1/analyze/audio", data={"file": (io.BytesIO(b"RIFFxxxxWAVEfmt "), "tiny.wav")}, content_type="multipart/form-data")
    assert response.status_code == 200
    statuses = {item["model_status"] for item in response.get_json()["results"]}
    assert "unavailable" in statuses
