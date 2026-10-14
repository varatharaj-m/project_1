from pathlib import Path

from flask import current_app

from .model_loading import load_classifier
from .model_registry import ModelState, registry


MISINFO_CUES = {"shocking", "secret", "they don't want you to know", "miracle", "hoax", "100% guaranteed"}
TRUST_CUES = {"according to", "reported by", "study", "official", "published", "data"}


class FakeNewsDetector:
    module_name = "misinformation"

    def __init__(self):
        self.model_path = current_app.config.get("FAKE_NEWS_MODEL_PATH", "")
        self.trained_model = None
        self.model_name = current_app.config.get("FAKE_NEWS_MODEL_NAME", "tfidf-logreg-baseline")
        detail = "Transparent lexical baseline active."
        if self.model_path:
            try:
                if not Path(self.model_path).is_file():
                    raise FileNotFoundError(self.model_path)
                self.trained_model = load_classifier(self.model_path)
                self.model_name = "trained-isot-tfidf-logistic-regression"
                detail = "ISOT dataset-trained news classifier active."
            except (OSError, ValueError, ImportError, EOFError) as error:
                detail = f"Trained news model unavailable; lexical baseline active ({error})."
        registry.set("news", ModelState(name=self.model_name, status="available", detail=detail))

    def analyze(self, text):
        if text is None or not text.strip():
            return self._invalid("Headline, article, or claim text is required.")
        max_chars = current_app.config.get("NEWS_MAX_CHARS", 12000)
        if len(text) > max_chars:
            return self._invalid(f"News text exceeds the configured {max_chars} character limit.")

        if self.trained_model:
            return self._analyze_with_model(text)

        lowered = text.lower()
        misinfo_hits = sum(1 for cue in MISINFO_CUES if cue in lowered)
        trust_hits = sum(1 for cue in TRUST_CUES if cue in lowered)
        score = max(0.05, min(0.95, 0.45 + 0.15 * misinfo_hits - 0.1 * trust_hits))
        label = "likely_misleading" if score >= 0.55 else "likely_not_misleading"
        risk = "high" if score >= 0.75 else "medium" if score >= 0.55 else "low"
        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "baseline-1.0",
            "predicted_labels": [label],
            "confidence_scores": {"misinformation_score": round(score, 3)},
            "risk_category": risk,
            "availability_status": "available",
            "explanation": (
                "Classifier output is not fact verification. No external evidence provider is configured, "
                "so sources were not retrieved and no citation claims are made."
            ),
        }

    def _analyze_with_model(self, text):
        model = self.trained_model["model"]
        probabilities = model.predict_proba([text])[0]
        classes = model.named_steps["classifier"].classes_
        label_scores = {str(label).strip().lower(): float(score) for label, score in zip(classes, probabilities)}
        fake_score = label_scores.get("fake", 0.0)
        is_misleading = fake_score >= 0.5
        risk = "high" if fake_score >= 0.75 else "medium" if fake_score >= 0.55 else "low"
        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "trained-1.0",
            "predicted_labels": ["likely_misleading" if is_misleading else "likely_not_misleading"],
            "confidence_scores": {
                "misinformation_score": round(fake_score, 3),
                "label_scores": {label: round(score, 3) for label, score in label_scores.items()},
            },
            "risk_category": risk,
            "availability_status": "available",
            "explanation": (
                "Dataset-trained ISOT fake-versus-true article classifier. This is not fact verification: "
                "the model can learn source and writing-style patterns and does not retrieve external evidence."
            ),
        }

    def _invalid(self, message):
        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "baseline-1.0",
            "predicted_labels": [],
            "confidence_scores": {},
            "risk_category": "needs_review",
            "availability_status": "invalid_input",
            "explanation": message,
        }
