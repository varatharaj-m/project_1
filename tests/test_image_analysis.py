import json
import io
import sys
from types import SimpleNamespace

from PIL import Image
import torch

from app.services.image_model import IMAGE_CLASSES, ImageAuthenticityCNN
from app.services.image_detector import ImageDetector


def test_image_analysis_unavailable_model(auth_client):
    image = Image.new("RGB", (8, 8), "blue")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    response = auth_client.post("/api/v1/analyze/image", data={"file": (buffer, "test.png")}, content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.get_json()["results"][0]["model_status"] == "unavailable"


def test_image_analysis_bad_extension(auth_client):
    response = auth_client.post("/api/v1/analyze/image", data={"file": (io.BytesIO(b"x"), "bad.exe")}, content_type="multipart/form-data")
    assert response.status_code == 400


def test_image_analysis_runs_configured_classifier(auth_client, app, tmp_path):
    checkpoint = tmp_path / "openfake_image.pt"
    torch.save(
        {
            "state_dict": ImageAuthenticityCNN().state_dict(),
            "image_size": 32,
            "classes": list(IMAGE_CLASSES),
            "dataset_license": "CC-BY-NC-4.0",
        },
        checkpoint,
    )
    app.config["IMAGE_MODEL_PATH"] = str(checkpoint)
    app.config["ALLOW_NONCOMMERCIAL_MODELS"] = True
    image = Image.new("RGB", (32, 32), "gray")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = auth_client.post(
        "/api/v1/analyze/image",
        data={"file": (buffer, "test.png")},
        content_type="multipart/form-data",
    )

    result = response.get_json()["results"][0]
    assert response.status_code == 200
    assert result["model_status"] == "available"
    assert result["model"] == "openfake-image-cnn"
    assert set(result["confidence"]) >= {"real", "fake"}


def test_openfake_classifier_requires_noncommercial_opt_in(auth_client, app, tmp_path):
    checkpoint = tmp_path / "openfake_image.pt"
    torch.save(
        {
            "state_dict": ImageAuthenticityCNN().state_dict(),
            "image_size": 32,
            "classes": list(IMAGE_CLASSES),
            "dataset_license": "CC-BY-NC-4.0",
        },
        checkpoint,
    )
    app.config["IMAGE_MODEL_PATH"] = str(checkpoint)
    app.config["ALLOW_NONCOMMERCIAL_MODELS"] = False
    image = Image.new("RGB", (32, 32), "gray")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    response = auth_client.post(
        "/api/v1/analyze/image",
        data={"file": (buffer, "test.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.get_json()["results"][0]["model_status"] == "unavailable"


def test_external_keras_adapter_runs_inference(app, tmp_path, monkeypatch):
    class FakeKerasModel:
        input_shape = (None, 224, 224, 3)
        output_shape = (None, 1)

        def predict(self, images, verbose=0):
            assert images.shape == (1, 224, 224, 3)
            assert float(images.min()) >= 0.0
            assert float(images.max()) <= 1.0
            return [[0.8]]

    fake_tensorflow = SimpleNamespace(
        keras=SimpleNamespace(models=SimpleNamespace(load_model=lambda *_args, **_kwargs: FakeKerasModel()))
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tensorflow)
    checkpoint = tmp_path / "deepfake_model.h5"
    checkpoint.write_bytes(b"test checkpoint")
    adapter = tmp_path / "adapter.json"
    adapter.write_text(
        json.dumps({"checkpoint": checkpoint.name, "image_size": 224, "real_score_threshold": 0.5}),
        encoding="utf-8",
    )
    app.config["IMAGE_MODEL_PATH"] = str(adapter)
    image_path = tmp_path / "test.png"
    Image.new("RGBA", (40, 32), "white").save(image_path)

    with app.app_context():
        result = ImageDetector().analyze(image_path)

    assert result["availability_status"] == "available"
    assert result["predicted_labels"] == ["needs_review"]
    assert result["risk_category"] == "needs_review"
    assert result["confidence_scores"]["candidate_label"] == "real"
    assert result["confidence_scores"]["real"] == 0.8
    assert result["confidence_scores"]["fake"] == 0.2

    adapter.write_text(
        json.dumps({"checkpoint": checkpoint.name, "image_size": 224, "validated": True}),
        encoding="utf-8",
    )
    with app.app_context():
        validated_result = ImageDetector().analyze(image_path)
    assert validated_result["predicted_labels"] == ["real"]
