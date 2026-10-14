import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from ml.training.train_text_model import train_model


ROOT_DIR = Path(__file__).resolve().parents[2]


def train_isot_model(fake_dataset, real_dataset, output_path, seed=42, test_size=0.2):
    rows = []
    for dataset_path, label in ((fake_dataset, "fake"), (real_dataset, "real")):
        frame = pd.read_csv(dataset_path)
        missing = {"title", "text"} - set(frame.columns)
        if missing:
            raise ValueError(f"{dataset_path} is missing required columns: {', '.join(sorted(missing))}")
        title = frame["title"].fillna("").astype(str).str.strip()
        article = frame["text"].fillna("").astype(str).str.strip()
        text = (title + "\n\n" + article).str.strip()
        rows.extend({"text": value, "label": label} for value in text if value)

    if not rows:
        raise ValueError("The supplied ISOT files contain no non-empty articles.")

    output_path = Path(output_path)
    with tempfile.TemporaryDirectory() as temporary_directory:
        dataset_path = Path(temporary_directory) / "isot_fake_news.csv"
        pd.DataFrame(rows).to_csv(dataset_path, index=False)
        metadata = train_model(
            dataset_path,
            output_path,
            text_column="text",
            label_column="label",
            seed=seed,
            test_size=test_size,
            dataset_version="ISOT Fake.csv + True.csv",
        )

    metadata["dataset"] = f"{Path(fake_dataset).name} (fake) + {Path(real_dataset).name} (real)"
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fake-dataset",
        default=str(ROOT_DIR / "datasets" / "raw" / "user_uploads" / "Fake.csv"),
    )
    parser.add_argument(
        "--real-dataset",
        default=str(ROOT_DIR / "datasets" / "raw" / "user_uploads" / "True.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT_DIR / "model_weights" / "fake_news_classifier.joblib"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    metadata = train_isot_model(
        args.fake_dataset,
        args.real_dataset,
        args.output,
        seed=args.seed,
        test_size=args.test_size,
    )
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
