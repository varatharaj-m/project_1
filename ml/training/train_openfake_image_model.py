import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from app.services.image_model import (
    IMAGE_CLASSES,
    OPENFAKE_LICENSE,
    OPENFAKE_REVISION,
    ImageAuthenticityCNN,
    image_to_tensor,
)


DATASET_ID = "ComplexDataLab/OpenFake"
ROOT_DIR = Path(__file__).resolve().parents[2]


def load_balanced_split(config, split, samples_per_class, image_size, seed, revision):
    stream = load_dataset(DATASET_ID, config, split=split, streaming=True, revision=revision)
    stream = stream.shuffle(seed=seed, buffer_size=min(128, max(32, samples_per_class // 2)))
    images = {"real": [], "fake": []}

    for row in stream:
        label = str(row.get("label", "")).lower()
        if label not in images or len(images[label]) >= samples_per_class:
            continue
        try:
            images[label].append(image_to_tensor(row["image"], image_size))
        except (OSError, ValueError, TypeError):
            continue
        if all(len(samples) >= samples_per_class for samples in images.values()):
            break

    if any(not samples for samples in images.values()):
        counts = {label: len(samples) for label, samples in images.items()}
        raise RuntimeError(f"Could not collect both labels from {config}/{split}: {counts}")

    tensors = torch.stack(images["real"] + images["fake"])
    labels = torch.tensor([0] * len(images["real"]) + [1] * len(images["fake"]), dtype=torch.long)
    order = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    return TensorDataset(tensors[order], labels[order]), {label: len(samples) for label, samples in images.items()}


def evaluate(model, dataset, batch_size, device):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    expected = []
    predicted = []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            expected.extend(labels.tolist())
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision_macro": float(precision_score(expected, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(expected, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
        "samples": len(expected),
    }


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    train_data, train_counts = load_balanced_split(
        "core", "train", args.samples_per_class, args.image_size, args.seed, args.revision
    )
    validation_data, validation_counts = load_balanced_split(
        "core", "validation", args.validation_per_class, args.image_size, args.seed + 1, args.revision
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    model = ImageAuthenticityCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics = evaluate(model, validation_data, args.batch_size, device)
        average_loss = total_loss / len(train_data)
        print(json.dumps({"epoch": epoch + 1, "train_loss": round(average_loss, 4), **metrics}))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "dataset": DATASET_ID,
        "dataset_revision": args.revision,
        "dataset_license": OPENFAKE_LICENSE,
        "usage_restriction": "Non-commercial use only; the OpenFake dataset is CC BY-NC 4.0.",
        "attribution": "Livernoche et al., OpenFake: An Open Dataset and Platform Toward Real-World Deepfake Detection (2025), https://arxiv.org/abs/2509.09495",
        "config": "core",
        "training_split": "train",
        "validation_split": "validation",
        "classes": list(IMAGE_CLASSES),
        "train_samples_per_class": train_counts,
        "validation_samples_per_class": validation_counts,
        "image_size": args.image_size,
        "seed": args.seed,
        "epochs": args.epochs,
        "metrics": metrics,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.evaluate_ood:
        metadata["evaluation_metrics"] = {}
        for config, split, count in (
            ("core", "test", args.evaluation_per_class),
            ("reddit", "test", args.evaluation_per_class),
        ):
            evaluation_data, counts = load_balanced_split(
                config, split, count, args.image_size, args.seed + 2, args.revision
            )
            evaluation_metrics = evaluate(model, evaluation_data, args.batch_size, device)
            metadata["evaluation_metrics"][f"{config}/{split}"] = {"counts": counts, **evaluation_metrics}
            print(json.dumps({"evaluation": f"{config}/{split}", "counts": counts, **evaluation_metrics}))

    model = model.cpu().eval()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "image_size": args.image_size,
            "classes": list(IMAGE_CLASSES),
            "dataset_license": OPENFAKE_LICENSE,
        },
        output,
    )
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"model_path": str(output), "metadata_path": str(output.with_suffix('.json'))}))
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Train a bounded OpenFake image-authenticity classifier.")
    parser.add_argument("--output", default=str(ROOT_DIR / "model_weights" / "openfake_image.pt"))
    parser.add_argument("--revision", default=OPENFAKE_REVISION)
    parser.add_argument("--samples-per-class", type=int, default=256)
    parser.add_argument("--validation-per-class", type=int, default=64)
    parser.add_argument("--evaluation-per-class", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--evaluate-ood", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if min(args.samples_per_class, args.validation_per_class, args.evaluation_per_class) < 1:
        parser.error("sample counts must be positive")
    if args.image_size < 32 or args.batch_size < 1 or args.epochs < 1:
        parser.error("image size must be at least 32, and batch size/epochs must be positive")
    train(args)


if __name__ == "__main__":
    main()