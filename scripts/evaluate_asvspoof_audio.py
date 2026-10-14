import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_ID = "SpeechAntiSpoofingBenchmarks/ASVspoof2019_LA"
DATASET_LICENSE = "ODC-By"
DEFAULT_ADAPTER = ROOT_DIR / "model_weights" / "adapters" / "audio_deepfake_adapter.json"


def calculate_metrics(labels, spoof_scores):
    labels = np.asarray(labels, dtype=np.int8)
    spoof_scores = np.asarray(spoof_scores, dtype=np.float64)
    predicted = (spoof_scores >= 0.5).astype(np.int8)
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, spoof_scores)
    false_negative_rate = 1.0 - true_positive_rate
    eer_index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    eer = float((false_positive_rate[eer_index] + false_negative_rate[eer_index]) / 2.0)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predicted)),
        "f1_spoof": float(f1_score(labels, predicted, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, spoof_scores)),
        "eer_approx": eer,
        "confusion_matrix_labels": ["bonafide", "spoof"],
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def load_audio_model(model_path=None):
    checkpoint_path = Path(model_path or DEFAULT_ADAPTER).resolve()
    settings = {
        "sample_rate": 22050,
        "duration_seconds": 3,
        "mfcc_features": 40,
        "expected_frames": 130,
        "real_class_index": 0,
        "fake_class_index": 1,
        "model_name": "external-shieldvision-audio-cnn",
    }
    if checkpoint_path.suffix.lower() == ".json":
        adapter = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        settings.update({key: adapter[key] for key in settings if key in adapter})
        checkpoint = adapter.get("checkpoint")
        if not checkpoint:
            raise ValueError(f"Audio adapter {checkpoint_path} has no checkpoint path.")
        checkpoint_path = (checkpoint_path.parent / checkpoint).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Audio checkpoint not found: {checkpoint_path}. Prepare local model files as described in model_weights/README.md."
        )

    import tensorflow as tf

    model = tf.keras.models.load_model(str(checkpoint_path), compile=False)
    input_shape = model.input_shape
    output_shape = model.output_shape
    expected_input = (
        settings["mfcc_features"],
        settings["expected_frames"],
        1,
    )
    if isinstance(input_shape, list) or tuple(input_shape[1:]) != expected_input:
        raise ValueError(f"Audio checkpoint input shape {input_shape} does not match (*, {expected_input}).")
    if isinstance(output_shape, list) or len(output_shape) != 2 or output_shape[-1] != 2:
        raise ValueError(f"Audio checkpoint must output two classes; got {output_shape}.")
    return model, settings


def prepare_mfcc(audio_bytes, settings):
    import librosa

    samples, source_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    target_rate = int(settings["sample_rate"])
    if source_rate != target_rate:
        samples = librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate)
    sample_count = int(target_rate * float(settings["duration_seconds"]))
    if len(samples) < sample_count:
        samples = np.pad(samples, (0, sample_count - len(samples)))
    else:
        samples = samples[:sample_count]
    mfcc = librosa.feature.mfcc(y=samples, sr=target_rate, n_mfcc=int(settings["mfcc_features"]))
    expected_frames = int(settings["expected_frames"])
    if mfcc.shape[1] < expected_frames:
        mfcc = np.pad(mfcc, ((0, 0), (0, expected_frames - mfcc.shape[1])))
    return mfcc[:, :expected_frames, None].astype(np.float32)


def _spoof_probability(model, features, settings):
    scores = np.asarray(model.predict(features[None, ...], verbose=0), dtype=np.float64).reshape(-1)
    if scores.size != 2 or not np.isfinite(scores).all():
        raise ValueError("Audio model returned invalid class scores.")
    if (scores < 0).any() or (scores > 1).any() or not np.isclose(scores.sum(), 1.0, atol=0.05):
        scores = np.exp(scores - scores.max())
        scores /= scores.sum()
    fake_index = int(settings["fake_class_index"])
    if fake_index not in (0, 1):
        raise ValueError("fake_class_index must be 0 or 1.")
    return float(scores[fake_index])


def evaluate_dataset(limit=512, model_path=None, shuffle_buffer=4096):
    from datasets import Audio, load_dataset

    dataset = load_dataset(DATASET_ID, split="test", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))
    dataset = dataset.shuffle(seed=42, buffer_size=shuffle_buffer)
    model, settings = load_audio_model(model_path)

    labels = []
    spoof_scores = []
    for row in dataset:
        audio_bytes = row["audio"].get("bytes")
        if not audio_bytes:
            raise ValueError(f"No audio bytes for benchmark record {row.get('path', 'unknown')}.")
        features = prepare_mfcc(audio_bytes, settings)
        labels.append(int(row["label"]))
        spoof_scores.append(_spoof_probability(model, features, settings))
        if limit and len(labels) >= limit:
            break

    if not labels:
        raise RuntimeError("No benchmark examples were evaluated.")
    return {
        "dataset": DATASET_ID,
        "license": DATASET_LICENSE,
        "split": "test",
        "evaluated_examples": len(labels),
        "shuffle_seed": 42,
        "shuffle_buffer": shuffle_buffer,
        "label_mapping": {"0": "bonafide", "1": "spoof"},
        "metrics": calculate_metrics(labels, spoof_scores),
        "note": "Streaming evaluation only. No examples were used for training; the EER is the nearest-point approximation on this evaluated sample.",
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the local audio model on the ASVspoof 2019 LA test split.")
    parser.add_argument("--limit", type=int, default=512, help="Maximum examples; use 0 to stream the full split.")
    parser.add_argument("--shuffle-buffer", type=int, default=4096, help="Streaming shuffle buffer size for representative bounded samples.")
    parser.add_argument("--model", type=Path, help="Optional adapter JSON or audio checkpoint path.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "model_weights" / "asvspoof2019_la_evaluation.json",
    )
    args = parser.parse_args()
    if args.limit < 0 or args.shuffle_buffer < 1:
        parser.error("--limit must be zero or positive and --shuffle-buffer must be positive.")
    try:
        report = evaluate_dataset(
            limit=args.limit,
            model_path=args.model,
            shuffle_buffer=args.shuffle_buffer,
        )
    except (OSError, ImportError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()