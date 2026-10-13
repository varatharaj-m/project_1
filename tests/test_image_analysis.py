import io

from PIL import Image
import torch

from app.services.image_model import IMAGE_CLASSES, ImageAuthenticityCNN


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
