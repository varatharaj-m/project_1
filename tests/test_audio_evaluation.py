from io import BytesIO

import numpy as np
import soundfile as sf

from scripts.evaluate_asvspoof_audio import calculate_metrics, prepare_mfcc


def test_asvspoof_metrics_respect_bonafide_and_spoof_mapping():
    report = calculate_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])

    assert report["accuracy"] == 1.0
    assert report["balanced_accuracy"] == 1.0
    assert report["f1_spoof"] == 1.0
    assert report["roc_auc"] == 1.0
    assert report["eer_approx"] == 0.0
    assert report["confusion_matrix"] == [[2, 0], [0, 2]]


def test_asvspoof_flac_preprocessing_matches_audio_model_shape():
    audio_file = BytesIO()
    sf.write(audio_file, np.zeros(16000, dtype=np.float32), 16000, format="FLAC")
    features = prepare_mfcc(
        audio_file.getvalue(),
        {
            "sample_rate": 22050,
            "duration_seconds": 3,
            "mfcc_features": 40,
            "expected_frames": 130,
        },
    )

    assert features.shape == (40, 130, 1)
    assert np.isfinite(features).all()