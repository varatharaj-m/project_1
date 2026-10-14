import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ml.training.train_fake_news_model import train_isot_model
from ml.training.train_text_model import train_model


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT_DIR / "datasets" / "raw" / "user_uploads"
WEIGHTS_DIR = ROOT_DIR / "model_weights"


def train_uploaded_models(data_dir=DEFAULT_DATA_DIR, include_sentiment=False):
    data_dir = Path(data_dir)
    cyberbullying_data = data_dir / "cyberbullying_tweets.csv"
    fake_news_data = data_dir / "Fake.csv"
    real_news_data = data_dir / "True.csv"
    required = [cyberbullying_data, fake_news_data, real_news_data]
    missing = [str(path) for path in required if not path.is_file()]
    train_path = data_dir / "train-00000-of-00001.parquet"
    test_path = data_dir / "test-00000-of-00001.parquet"
    if include_sentiment and (not train_path.is_file() or not test_path.is_file()):
        missing.extend(str(path) for path in (train_path, test_path) if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Missing training inputs: {', '.join(missing)}")

    results = {
        "cyberbullying": train_model(
            cyberbullying_data,
            WEIGHTS_DIR / "text_classifier.joblib",
            text_column="tweet_text",
            label_column="cyberbullying_type",
            dataset_version="cyberbullying-tweets",
        ),
        "fake_news": train_isot_model(
            fake_news_data,
            real_news_data,
            WEIGHTS_DIR / "fake_news_classifier.joblib",
        ),
    }

    if include_sentiment:
        sentiment_metadata = train_model(
            train_path,
            WEIGHTS_DIR / "sentiment_classifier.joblib",
            text_column="text",
            label_column="annotation",
            dataset_version="uploaded-sentiment-train-split",
        )
        test_data = pd.read_parquet(test_path).dropna(subset=["text", "annotation"])
        sentiment_model = train_model_artifact(WEIGHTS_DIR / "sentiment_classifier.joblib")
        predictions = sentiment_model.predict(test_data["text"].astype(str))
        results["sentiment"] = {
            "training": sentiment_metadata,
            "held_out_test": {
                "rows": len(test_data),
                "accuracy": float(accuracy_score(test_data["annotation"], predictions)),
                "f1_macro": float(f1_score(test_data["annotation"], predictions, average="macro")),
            },
        }
    return results


def train_model_artifact(path):
    import joblib

    return joblib.load(path)["model"]


def main():
    parser = argparse.ArgumentParser(description="Train local models from prepared, git-ignored datasets.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--include-sentiment",
        action="store_true",
        help="Also train the optional sentiment model and evaluate its untouched test split.",
    )
    args = parser.parse_args()
    results = train_uploaded_models(args.data_dir, include_sentiment=args.include_sentiment)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()