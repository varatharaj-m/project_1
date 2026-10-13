from pathlib import Path

from flask import current_app
from PIL import Image, UnidentifiedImageError

from .file_validation import ValidationError, validate_image_signature
from .image_model import IMAGE_CLASSES, ImageAuthenticityCNN, image_to_tensor
from .model_registry import ModelState, registry


class ImageDetector:
    module_name = "image_manipulation"

    def __init__(self):
        model_path = current_app.config.get("IMAGE_MODEL_PATH")
        self.model = None
        self.image_size = 128
        self.model_name = "openfake-image-cnn"
        status = "unavailable"
        detail = "No OpenFake image classifier weights configured."
        if model_path and Path(model_path).exists() and Path(model_path).suffix.lower() not in {".pt", ".pth"}:
            self.model_name = "external-image-deepfake-adapter"
            status = "available"
            detail = "External image/deepfake source adapter configured. Native checkpoint inference is not connected."
        elif model_path and Path(model_path).is_file():
            try:
                import torch

                checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
                if checkpoint.get("dataset_license") == "CC-BY-NC-4.0" and not current_app.config.get(
                    "ALLOW_NONCOMMERCIAL_MODELS", False
                ):
                    raise ValueError("OpenFake-derived weights require ALLOW_NONCOMMERCIAL_MODELS=true.")
                self.image_size = int(checkpoint["image_size"])
                self.model = ImageAuthenticityCNN()
                self.model.load_state_dict(checkpoint["state_dict"])
                self.model.eval()
                status = "available"
                detail = "Configured OpenFake image classifier loaded."
            except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
                detail = f"Configured image classifier could not be loaded: {error}"
        registry.set("image", ModelState(name=self.model_name, status=status, detail=detail))

    def analyze(self, path):
        try:
            signature = validate_image_signature(path)
            with Image.open(path) as img:
                width, height = img.size
                if width * height > 40_000_000:
                    raise ValidationError("Image dimensions are too large.")
                preprocessing = {"signature": signature, "width": width, "height": height, "mode": img.mode}
                if self.model:
                    import torch

                    image_tensor = image_to_tensor(img, self.image_size).unsqueeze(0)
                    with torch.inference_mode():
                        probabilities = self.model(image_tensor).softmax(dim=1)[0].tolist()
                    fake_score = float(probabilities[1])
                    predicted = IMAGE_CLASSES[int(fake_score >= 0.5)]
                    risk = "high" if fake_score >= 0.8 else "medium" if fake_score >= 0.5 else "low"
                    return self._result(
                        "available",
                        [predicted],
                        {"real": round(float(probabilities[0]), 4), "fake": round(fake_score, 4)},
                        risk,
                        "Experimental image classifier trained on a bounded OpenFake sample. Scores are not proof of authenticity and may not generalize to new generators or image domains.",
                        preprocessing,
                    )
        except (ValidationError, UnidentifiedImageError) as exc:
            return self._result("invalid_input", [], {}, "needs_review", str(exc), {})

        if registry.get("image").status != "available":
            return self._result(
                "unavailable",
                ["analysis_not_performed"],
                {},
                "needs_review",
                "No dedicated image manipulation/deepfake model weights are installed. The image was decoded safely, but authenticity was not determined.",
                preprocessing,
            )
        return self._result(
            "available",
            ["requires_external_model_inference"],
            {},
            "needs_review",
            "Adapter is configured; connect project-specific model inference in ImageDetector before production use.",
            preprocessing,
        )

    def _result(self, status, labels, confidence, risk, explanation, preprocessing):
        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "openfake-cnn-1.0",
            "predicted_labels": labels,
            "confidence_scores": {**confidence, "preprocessing": preprocessing},
            "risk_category": risk,
            "availability_status": status,
            "explanation": explanation,
        }
