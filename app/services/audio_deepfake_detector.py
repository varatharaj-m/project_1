from pathlib import Path

from flask import current_app

from .model_registry import ModelState, registry


class AudioDeepfakeDetector:
    module_name = "synthetic_audio"

    def __init__(self):
        model_path = current_app.config.get("AUDIO_MODEL_PATH")
        status = "available" if model_path and Path(model_path).exists() else "unavailable"
        registry.set("audio_deepfake", ModelState(name="external-synthetic-audio-adapter", status=status, detail="Set AUDIO_MODEL_PATH for synthetic-speech detection."))

    def analyze(self, path):
        if not Path(path).exists():
            status, explanation = "invalid_input", "Audio file was not found."
        elif registry.get("audio_deepfake").status != "available":
            status, explanation = "unavailable", "No dedicated synthetic-speech model weights are installed. Authenticity analysis was not performed."
        else:
            status, explanation = "available", "Adapter is ready for configured synthetic-audio model inference."
        return {
            "module_name": self.module_name,
            "model_name": "external-synthetic-audio-adapter",
            "model_version": "adapter-1.0",
            "predicted_labels": ["analysis_not_performed"] if status != "available" else ["requires_external_model_inference"],
            "confidence_scores": {},
            "risk_category": "needs_review",
            "availability_status": status,
            "explanation": explanation,
        }
