import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_TEXT_COLUMNS = ("text", "tweet", "comment_text", "comment")
DEFAULT_LABEL_COLUMNS = ("toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate")
DEFAULT_LABEL_COLUMN = ("label", "cyberbullying_type", "class")


def train_model(
    dataset_path,
    output_path,
    text_column=None,
    label_column=None,
    label_columns=None,
    seed=42,
    test_size=0.2,
    dataset_version=None,
):
    dataset_path = Path(dataset_path)
    if not dataset_path.is_file():
        raise ValueError(f"Dataset not found: {dataset_path}")
    if label_column and label_columns:
        raise ValueError("Choose either --label-column or --label-columns, not both.")

    frame = _read_dataset(dataset_path)
    text_column = text_column or next((name for name in DEFAULT_TEXT_COLUMNS if name in frame.columns), None)
    if not text_column or text_column not in frame.columns:
        raise ValueError(f"Text column not found. Available columns: {', '.join(frame.columns)}")

    if label_columns:
        selected_labels = list(label_columns)
        mode = "multilabel"
    elif label_column:
        selected_labels = [label_column]
        mode = "multiclass"
    elif all(name in frame.columns for name in DEFAULT_LABEL_COLUMNS):
        selected_labels = list(DEFAULT_LABEL_COLUMNS)
        mode = "multilabel"
    else:
        selected = next((name for name in DEFAULT_LABEL_COLUMN if name in frame.columns), None)
        if not selected:
            raise ValueError(
                "Could not infer label columns. Supply --label-column or --label-columns. "
                f"Available columns: {', '.join(frame.columns)}"
            )
        selected_labels = [selected]
        mode = "multiclass"

    missing = [name for name in selected_labels if name not in frame.columns]
    if missing:
        raise ValueError(f"Label column(s) not found: {', '.join(missing)}")

    frame = frame.dropna(subset=[text_column, *selected_labels]).copy()
    texts = frame[text_column].astype(str).str.strip()
    nonempty = texts.ne("")
    frame = frame.loc[nonempty]
    texts = texts.loc[nonempty]
    if len(frame) < 4:
        raise ValueError("At least four non-empty labeled rows are required.")

    if mode == "multilabel":
        target_frame = frame[selected_labels].apply(pd.to_numeric, errors="coerce")
        if target_frame.isna().any().any() or not target_frame.isin((0, 1)).all().all():
            raise ValueError("Multi-label columns must contain only binary 0/1 values.")
        targets = target_frame.astype(int)
        if any(targets[column].nunique() < 2 for column in selected_labels):
            raise ValueError("Each multi-label column must contain both 0 and 1 examples.")
    else:
        targets = frame[selected_labels[0]].astype(str)
        if targets.nunique() < 2:
            raise ValueError("The label column must contain at least two classes.")

    stratify = targets if mode == "multiclass" and targets.value_counts().min() >= 2 else None
    train_texts, valid_texts, train_targets, valid_targets = train_test_split(
        texts,
        targets,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    model = _new_pipeline(mode)
    model.fit(train_texts, train_targets)
    predictions = model.predict(valid_texts)
    if mode == "multilabel":
        metrics = {
            "f1_micro": float(f1_score(valid_targets, predictions, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(valid_targets, predictions, average="macro", zero_division=0)),
        }
        labels = selected_labels
    else:
        metrics = {
            "accuracy": float(accuracy_score(valid_targets, predictions)),
            "f1_macro": float(f1_score(valid_targets, predictions, average="macro", zero_division=0)),
        }
        labels = sorted(targets.unique().tolist())

    model.fit(texts, targets)
    artifact = {"model": model, "mode": mode, "labels": labels, "threshold": 0.5}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(artifact, output_path)
    metadata = {
        "model": "tfidf-logistic-regression",
        "dataset": str(dataset_path),
        "dataset_version": dataset_version or dataset_path.stem,
        "text_column": text_column,
        "label_columns": selected_labels,
        "mode": mode,
        "label_mapping": labels,
        "split_seed": seed,
        "test_size": test_size,
        "metrics": metrics,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _new_pipeline(mode):
    classifier = LogisticRegression(max_iter=1000, class_weight="balanced")
    if mode == "multilabel":
        classifier = OneVsRestClassifier(classifier)
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000, sublinear_tf=True)),
            ("classifier", classifier),
        ]
    )


def _read_dataset(dataset_path):
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(dataset_path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(dataset_path)
    raise ValueError("Unsupported dataset format. Use CSV or Parquet.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="CSV from an approved dataset source.")
    parser.add_argument("--text-column", help="Text column; inferred for common dataset schemas.")
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument("--label-column", help="Single multiclass target column.")
    labels.add_argument("--label-columns", nargs="+", help="Binary target columns for multilabel data.")
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "model_weights" / "text_classifier.joblib"),
        help="Model artifact output path.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--dataset-version", help="Dataset release or collection identifier for metadata.")
    args = parser.parse_args()
    try:
        metadata = train_model(
            args.dataset,
            args.output,
            text_column=args.text_column,
            label_column=args.label_column,
            label_columns=args.label_columns,
            seed=args.seed,
            test_size=args.test_size,
            dataset_version=args.dataset_version,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
