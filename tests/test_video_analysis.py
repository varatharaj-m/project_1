import io

from app.services.video_detector import VideoDetector


def test_invalid_video_file(auth_client):
    response = auth_client.post("/api/v1/analyze/video", data={"file": (io.BytesIO(b"not-video"), "bad.mp4")}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_video_samples_frames_and_returns_review_only_candidate(app, tmp_path, monkeypatch):
    import cv2
    import numpy as np

    class FakeCapture:
        def __init__(self, _path):
            self.current_frame = 0

        def isOpened(self):
            return True

        def get(self, property_id):
            return {
                cv2.CAP_PROP_FRAME_COUNT: 5,
                cv2.CAP_PROP_FPS: 25,
                cv2.CAP_PROP_FRAME_WIDTH: 32,
                cv2.CAP_PROP_FRAME_HEIGHT: 24,
            }[property_id]

        def set(self, _property_id, value):
            self.current_frame = int(value)
            return True

        def read(self):
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            pass

    class FakeFrameDetector:
        model_backend = "keras"

        def __init__(self, _path, register_state=True):
            pass

        def analyze(self, _path):
            return {
                "predicted_labels": ["needs_review"],
                "confidence_scores": {"real": 0.2, "fake": 0.8, "candidate_label": "fake"},
            }

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr("cv2.VideoCapture", FakeCapture)
    monkeypatch.setattr("app.services.video_detector.ImageDetector", FakeFrameDetector)
    app.config["VIDEO_MAX_FRAMES"] = 3

    with app.app_context():
        result = VideoDetector().analyze(video_path)

    assert result["availability_status"] == "available"
    assert result["predicted_labels"] == ["needs_review"]
    assert result["confidence_scores"]["analyzed_frames"] == 3
    assert result["confidence_scores"]["candidate_label"] == "fake"
