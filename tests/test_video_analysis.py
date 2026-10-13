import io


def test_invalid_video_file(auth_client):
    response = auth_client.post("/api/v1/analyze/video", data={"file": (io.BytesIO(b"not-video"), "bad.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 400
