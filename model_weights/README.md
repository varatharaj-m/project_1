# Model Catalog

This directory contains model configuration and local artifacts. Small JSON adapter descriptions are tracked in Git; trained/downloaded weights, datasets, and evaluation output are ignored. A fresh clone therefore needs local data/checkpoints before the corresponding services can perform inference.

## Models and Services

| Service | Implementation | Local artifact/config | GitHub status |
|---|---|---|---|
| Cyberbullying text | TF-IDF + logistic regression, trained from six-class cyberbullying tweets; keyword fallback when weights are absent | `text_classifier.joblib`, configured by `TEXT_MODEL_PATH` | Training code is included; generated weights are local only |
| Fake-news screening | ISOT Fake/True TF-IDF + logistic regression; lexical fallback without weights | `fake_news_classifier.joblib`, configured by `FAKE_NEWS_MODEL_PATH` | Training code is included; this is not external fact verification |
| Sentiment | Optional separate binary classifier | `sentiment_classifier.joblib`, configured by `SENTIMENT_MODEL_PATH` | Disabled by default; the supplied test split scored below its majority-class baseline |
| Image screening | OpenFake CNN (`.pt`) or external Keras image checkpoint (`.h5`) | `adapters/image_deepfake_adapter.json`; local checkpoint selected by `IMAGE_MODEL_PATH` | Adapter is tracked; external checkpoint is not. External model is review-only until validated |
| Video frame screening | Samples up to `VIDEO_MAX_FRAMES` frames and runs the image model on each | `adapters/video_deepfake_adapter.json`; local visual checkpoint selected by `VIDEO_MODEL_PATH` | No video-specific classifier is included; frame candidates always require review |
| Speech transcription | OpenAI Whisper transcription | `WHISPER_MODEL_NAME` (for example, `base`) | Whisper weights download/cache locally on first use; current audio route does not forward the transcript to cyberbullying screening |
| Synthetic-audio detection | Adapter placeholder; current detector does not execute the archived HDF5 checkpoint | `adapters/audio_deepfake_adapter.json`, path set by `AUDIO_MODEL_PATH` | Inference is not implemented; status is not proof of audio authenticity |
| Evidence retrieval | No external evidence provider configured | No model artifact | Fact verification is unavailable; news classifier labels are screening signals only |

## Adapter Metadata

The JSON files in `adapters/` document source, expected inputs, checkpoint location, and validation state. They do not contain weights. Checkpoint paths are local and ignored by Git. Do not change `validated` to `true` just to remove a review badge; set it only after evaluation on a representative held-out dataset.

The uploaded ShieldVision image checkpoint has no license metadata and no image-training/validation data in its archive. Keep it local until usage and redistribution rights are confirmed. It has produced a false-fake result on a known real photo.

The separately trainable OpenFake CNN uses data under **CC BY-NC 4.0**. The checkpoint inherits non-commercial restrictions and must not be used commercially. Loading it requires `ALLOW_NONCOMMERCIAL_MODELS=true` and `IMAGE_MODEL_PATH=model_weights/openfake_image.pt`.

## Setup

Install the optional TensorFlow backend for HDF5 image checkpoints:

```powershell
pip install -r requirements-image.txt
```

See [ml/README.md](../ml/README.md) for training/evaluation commands and [datasets/README.md](../datasets/README.md) for dataset provenance and terms.
