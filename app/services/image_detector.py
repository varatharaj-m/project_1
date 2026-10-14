from .model_loading import load_classifier, load_keras_model
import json
from pathlib import Path

from flask import current_app
import numpy as np
from PIL import Image, UnidentifiedImageError

from .file_validation import ValidationError, validate_image_signature
from .image_model import IMAGE_CLASSES, ImageAuthenticityCNN, image_to_tensor
from .model_registry import ModelState, registry


class ImageDetector:
    module_name = "image_manipulation"

    def __init__(self, model_path=None, register_state=True):
        configured_path = model_path if model_path is not None else current_app.config.get("IMAGE_MODEL_PATH")
        self.model = None
        self.model_backend = None
        self.image_size = 128
        self.real_score_threshold = 0.5
        self.model_validated = False
        self.model_name = "openfake-image-cnn"
        self.model_version = "openfake-cnn-1.0"
        status = "unavailable"
        detail = "No OpenFake image classifier weights configured."

        model_path = Path(configured_path).resolve() if configured_path else None
        adapter = {}
        if model_path and model_path.suffix.lower() == ".json":
            try:
                adapter = json.loads(model_path.read_text(encoding="utf-8"))
                checkpoint = adapter.get("checkpoint")
                if not checkpoint:
                    raise ValueError("Image adapter does not specify a model checkpoint.")
                model_path = (model_path.parent / checkpoint).resolve()
                self.model_name = adapter.get("model_name", "external-image-deepfake-model")
                self.model_version = adapter.get("model_version", "external-image-1.0")
                self.image_size = int(adapter.get("image_size", 224))
                self.real_score_threshold = float(adapter.get("real_score_threshold", 0.5))
                self.model_validated = bool(adapter.get("validated", False))
            except (OSError, TypeError, ValueError) as error:
                model_path = None
                detail = f"Image adapter configuration is invalid: {error}"

        if model_path:
            if not model_path.is_file():
                detail = f"Configured image checkpoint not found: {model_path}"
            elif model_path.suffix.lower() in {".pt", ".pth"}:
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
                    self.model_backend = "torch"
                    self.model_name = "openfake-image-cnn"
                    self.model_version = "openfake-cnn-1.0"
                    status = "available"
                    detail = "Configured OpenFake image classifier loaded."
                except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
                    detail = f"Configured image classifier could not be loaded: {error}"
            elif model_path.suffix.lower() in {".h5", ".hdf5"}:
                try:
                    keras_model = load_keras_model(model_path)
                    input_shape = keras_model.input_shape
                    output_shape = keras_model.output_shape
                    if isinstance(input_shape, list) or len(input_shape) != 4 or input_shape[-1] != 3:
                        raise ValueError("Expected one image input with three RGB channels.")
                    if isinstance(output_shape, list) or len(output_shape) != 2 or output_shape[-1] != 1:
                        raise ValueError("Expected one sigmoid probability output for the real class.")
                    if input_shape[1] and int(input_shape[1]) != self.image_size:
                        raise ValueError(f"Adapter image_size must be {input_shape[1]} for this checkpoint.")
                    self.model = keras_model
                    self.model_backend = "keras"
                    status = "available"
                    detail = (
                        "Configured external Keras image classifier loaded."
                        if self.model_validated
                        else "External image model loaded; candidate labels require review until independently validated."
                    )
                except ImportError:
                    detail = "This HDF5 image checkpoint requires TensorFlow; install requirements-image.txt."
                except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
                    detail = f"Configured Keras image classifier could not be loaded: {error}"
            else:
                detail = "Unsupported image checkpoint. Configure a .pt, .pth, .h5, or .hdf5 model file."
        if register_state:
            registry.set("image", ModelState(name=self.model_name, status=status, detail=detail))

    def analyze(self, path):
        try:
            signature = validate_image_signature(path)
            with Image.open(path) as img:
                width, height = img.size
                if width * height > 40_000_000:
                    raise ValidationError("Image dimensions are too large.")
                preprocessing = {"signature": signature, "width": width, "height": height, "mode": img.mode}
                if self.model_backend == "torch":
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
                if self.model_backend == "keras":
                    image = img.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
                    image_array = np.asarray(image, dtype=np.float32) / 255.0
                    try:
                        real_score = float(np.asarray(self.model.predict(image_array[None, ...], verbose=0)).reshape(-1)[0])
                    except Exception as error:
                        return self._result(
                            "unavailable",
                            ["analysis_not_performed"],
                            {},
                            "needs_review",
                            f"External image inference failed: {error}",
                            preprocessing,
                        )
                    if not np.isfinite(real_score) or not 0.0 <= real_score <= 1.0:
                        return self._result(
                            "unavailable",
                            ["analysis_not_performed"],
                            {},
                            "needs_review",
                            "External image model did not return a valid real-class probability.",
                            preprocessing,
                        )
                    fake_score = 1.0 - real_score
                    predicted = "real" if real_score >= self.real_score_threshold else "fake"
                    risk = "high" if fake_score >= 0.8 else "medium" if fake_score >= 0.5 else "low"
                    if not self.model_validated:
                        return self._result(
                            "available",
                            ["needs_review"],
                            {
                                "real": round(real_score, 4),
                                "fake": round(fake_score, 4),
                                "candidate_label": predicted,
                            },
                            "needs_review",
                            f"Unvalidated external model candidate: {predicted}. Its scores are shown for review only, not as an authenticity classification.",
                            preprocessing,
                        )
                    return self._result(
                        "available",
                        [predicted],
                        {"real": round(real_score, 4), "fake": round(fake_score, 4)},
                        risk,
                        "External Keras image classifier output. This is a model score, not proof of authenticity; validate it on representative images before relying on it.",
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
            "model_version": self.model_version,
            "predicted_labels": labels,
            "confidence_scores": {**confidence, "preprocessing": preprocessing},
            "risk_category": risk,
            "availability_status": status,
            "explanation": explanation,
        }
