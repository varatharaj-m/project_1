import pandas as pd

from app.services.text_detector import TextDetector
from ml.training.train_text_model import train_model


def test_trained_classifier_is_used_by_text_detector(app, tmp_path):
    dataset = tmp_path / "comments.csv"
    output = tmp_path / "model_weights" / "text_classifier.joblib"
    rows = [
        {"text": f"kind supportive message number {index}", "label": "not bullying"}
        for index in range(12)
    ] + [
        {"text": f"cruel insulting message number {index}", "label": "bullying"}
        for index in range(12)
    ]
    pd.DataFrame(rows).to_csv(dataset, index=False)

    train_model(dataset, output, text_column="text", label_column="label")
    app.config["TEXT_MODEL_PATH"] = str(output)

    result = TextDetector().analyze("cruel insulting message")

    assert result["model_name"] == "trained-tfidf-logistic-regression"
    assert result["predicted_labels"] == ["bullying"]
    assert result["confidence_scores"]["toxicity"] > 0.5


def test_multilabel_training_supports_jigsaw_schema(app, tmp_path):
    dataset = tmp_path / "jigsaw.csv"
    output = tmp_path / "model_weights" / "jigsaw.joblib"
    rows = [
        {"comment_text": f"kind supportive comment {index}", "toxic": 0, "insult": 0}
        for index in range(12)
    ] + [
        {"comment_text": f"cruel insulting comment {index}", "toxic": 1, "insult": 1}
        for index in range(12)
    ]
    pd.DataFrame(rows).to_csv(dataset, index=False)

    metadata = train_model(
        dataset,
        output,
        text_column="comment_text",
        label_columns=["toxic", "insult"],
    )
    app.config["TEXT_MODEL_PATH"] = str(output)

    result = TextDetector().analyze("cruel insulting comment")

    assert metadata["mode"] == "multilabel"
    assert set(result["predicted_labels"]) == {"toxic", "insult"}
