import re
from pathlib import Path

from flask import current_app

from .model_loading import load_classifier
from .model_registry import ModelState, registry


HARMFUL_TERMS = {
    "hate": "hate_speech",
    "idiot": "offensive_language",
    "stupid": "offensive_language",
    "kill": "threatening_language",
    "die": "self_or_other_harm_language",
    "loser": "bullying",
}

POSITIVE_WORDS = {"good", "safe", "kind", "help", "happy", "support"}
NEGATIVE_WORDS = {"bad", "angry", "hate", "awful", "sad", "stupid", "kill"}
APPEARANCE_TARGETS = {"body", "face", "hair", "skin", "weight"}
MOCKING_TERMS = {"funny", "ugly", "weird", "gross", "disgusting"}
NON_HARMFUL_LABELS = {
    "0",
    "0.0",
    "clean",
    "non bullying",
    "non toxic",
    "neutral",
    "no bullying",
    "no cyberbullying",
    "no toxic",
    "not cyberbullying",
    "not bullying",
    "not cyberbullying",
    "not toxic",
    "normal",
}
LABEL_ALIASES = {
    "0": "no_harmful_language_detected",
    "0.0": "no_harmful_language_detected",
    "1": "possibly_offensive_or_hateful",
    "1.0": "possibly_offensive_or_hateful",
    "2": "hate_speech",
    "2.0": "hate_speech",
}


class TextDetector:
    module_name = "cyberbullying"

    def __init__(self):
        self.model_path = current_app.config.get("TEXT_MODEL_PATH", "")
        self.trained_model = None
        self.sentiment_model = None
        self.model_name = current_app.config.get("TEXT_MODEL_NAME", "lightweight-keyword-baseline")
        detail = "Rule-based lightweight baseline active."
        if self.model_path:
            try:
                if not Path(self.model_path).is_file():
                    raise FileNotFoundError(self.model_path)
                self.trained_model = load_classifier(self.model_path)
                self.model_name = "trained-tfidf-logistic-regression"
                detail = "Configured trained text classifier active."
            except (OSError, ValueError, ImportError, EOFError) as error:
                detail = f"Trained text model unavailable; keyword baseline active ({error})."
        sentiment_path = current_app.config.get("SENTIMENT_MODEL_PATH", "")
        if sentiment_path and Path(sentiment_path).is_file():
            try:
                self.sentiment_model = load_classifier(sentiment_path)
                detail += " Separate dataset-trained sentiment signal active."
            except (OSError, ValueError, ImportError, EOFError) as error:
                detail += f" Sentiment model unavailable ({error})."
        registry.set("text", ModelState(name=self.model_name, status="available", detail=detail))

    def analyze(self, text):
        if text is None or not text.strip():
            return self._invalid("Text is required.")
        max_chars = current_app.config.get("TEXT_MAX_CHARS", 5000)
        if len(text) > max_chars:
            return self._invalid(f"Text exceeds the configured {max_chars} character limit.")

        if self.trained_model:
            result = self._analyze_with_model(text)
        else:
            result = self._analyze_with_baseline(text)
        if self.sentiment_model:
            self._add_sentiment_signal(text, result)
        return result

    def _analyze_with_baseline(self, text):
        lowered = text.lower()
        found = sorted({label for term, label in HARMFUL_TERMS.items() if re.search(rf"\b{re.escape(term)}\b", lowered)})
        appearance_mockery = self._has_appearance_mockery(lowered)
        if appearance_mockery:
            found.append("appearance_based_bullying")
        neg = sum(1 for word in NEGATIVE_WORDS if re.search(rf"\b{word}\b", lowered))
        pos = sum(1 for word in POSITIVE_WORDS if re.search(rf"\b{word}\b", lowered))
        toxicity = min(0.95, 0.15 + 0.2 * len(found) + 0.08 * neg + (0.1 if appearance_mockery else 0))
        sentiment = "negative" if neg > pos else "positive" if pos > neg else "neutral"
        emotion = "anger" if any(label in found for label in ["hate_speech", "threatening_language"]) else "uncertain"
        risk = "high" if toxicity >= 0.7 else "medium" if toxicity >= 0.4 else "low"
        labels = found or ["no_harmful_language_detected"]

        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "baseline-1.0",
            "predicted_labels": labels,
            "confidence_scores": {"toxicity": round(toxicity, 3), "sentiment": sentiment, "emotion": emotion},
            "risk_category": risk,
            "availability_status": "available",
            "explanation": (
                "Lightweight keyword baseline. This is a screening signal, not a judgment about a person "
                "and not proof of intent. Install a validated transformer model for production use."
            ),
        }

    def _add_sentiment_signal(self, text, result):
        model = self.sentiment_model["model"]
        probabilities = model.predict_proba([text])[0]
        classes = model.named_steps["classifier"].classes_
        scores = {str(label).strip().lower(): float(score) for label, score in zip(classes, probabilities)}
        sentiment = max(scores, key=scores.get)
        result["confidence_scores"]["sentiment"] = sentiment
        result["confidence_scores"]["sentiment_confidence"] = round(scores[sentiment], 3)
        result["explanation"] += (
            " Sentiment is reported separately and does not determine the harmful-language label."
        )

    def _analyze_with_model(self, text):
        model = self.trained_model["model"]
        mode = self.trained_model["mode"]
        labels = self.trained_model["labels"]
        probabilities = model.predict_proba([text])
        appearance_mockery = self._has_appearance_mockery(text.lower())

        if mode == "multilabel":
            label_scores = {label: float(score) for label, score in zip(labels, probabilities[0])}
            found = [label for label, score in label_scores.items() if score >= self.trained_model["threshold"]]
            toxicity = max(label_scores.values(), default=0.0)
        else:
            classes = model.named_steps["classifier"].classes_
            label_scores = {str(label): float(score) for label, score in zip(classes, probabilities[0])}
            selected = max(label_scores, key=label_scores.get)
            if self._is_non_harmful(selected):
                found = []
                toxicity = max(
                    (score for label, score in label_scores.items() if not self._is_non_harmful(label)),
                    default=0.0,
                )
            else:
                found = [self._display_label(selected)]
                toxicity = label_scores[selected]

        if appearance_mockery and "appearance_based_bullying" not in found:
            found.append("appearance_based_bullying")
            toxicity = max(toxicity, 0.55)

        risk = "high" if toxicity >= 0.75 else "medium" if toxicity >= 0.5 else "low"
        return {
            "module_name": self.module_name,
            "model_name": self.model_name,
            "model_version": "trained-1.0",
            "predicted_labels": found or ["no_harmful_language_detected"],
            "confidence_scores": {
                "toxicity": round(toxicity, 3),
                "label_scores": {label: round(score, 3) for label, score in label_scores.items()},
            },
            "risk_category": risk,
            "availability_status": "available",
            "explanation": (
                "Dataset-trained text classifier. This is a screening signal, not a judgment about a person "
                "or proof of intent; performance depends on the supplied dataset and evaluation split."
            ),
        }

    @staticmethod
    def _is_non_harmful(label):
        normalized = re.sub(r"[_-]+", " ", str(label).strip().lower())
        return normalized in NON_HARMFUL_LABELS

    @staticmethod
    def _display_label(label):
        normalized = str(label).strip()
        return LABEL_ALIASES.get(normalized, normalized)

    @staticmethod
    def _has_appearance_mockery(text):
        targets_person = re.search(r"\b(you|your|u|ur)\b", text)
        mentions_appearance = any(re.search(rf"\b{re.escape(word)}\b", text) for word in APPEARANCE_TARGETS)
        uses_mocking_language = any(re.search(rf"\b{re.escape(word)}\b", text) for word in MOCKING_TERMS)
        return bool(targets_person and mentions_appearance and uses_mocking_language)

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
