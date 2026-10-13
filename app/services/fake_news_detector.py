from flask import current_app

from .model_registry import ModelState, registry


MISINFO_CUES = {"shocking", "secret", "they don't want you to know", "miracle", "hoax", "100% guaranteed"}
TRUST_CUES = {"according to", "reported by", "study", "official", "published", "data"}


class FakeNewsDetector:
    module_name = "misinformation"

    def __init__(self):
        self.model_name = current_app.config.get("FAKE_NEWS_MODEL_NAME", "tfidf-logreg-baseline")
        registry.set("news", ModelState(name=self.model_name, status="available", detail="Transparent lexical baseline active."))

    def analyze(self, text):
        if text is None or not text.strip():
            return self._invalid("Headline, article, or claim text is required.")
        max_chars = current_app.config.get("NEWS_MAX_CHARS", 12000)
        if len(text) > max_chars:
            return self._invalid(f"News text exceeds the configured {max_chars} character limit.")

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
