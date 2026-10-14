# ML Pipeline

The text training entry point supports single-label cyberbullying CSVs (such as `tweet` + `cyberbullying_type`) and multi-label toxicity CSVs (such as Jigsaw `comment_text` plus its binary label columns). See [datasets/README.md](../datasets/README.md) for source and terms notes and exact commands.

Training writes a TF-IDF/logistic-regression artifact and adjacent JSON metadata under `model_weights/`, including dataset identifier, label mapping, split seed, and held-out metrics. Set `TEXT_MODEL_PATH` to enable it in the Flask text-analysis route. With no trained artifact configured, the app keeps using its explicitly named lightweight baseline.

## Uploaded datasets

After preparing the local inputs as described in [datasets/README.md](../datasets/README.md), retrain the text and news classifiers with one command:

```powershell
python -m scripts.train_uploaded_models
```

The command writes `text_classifier.joblib` and `fake_news_classifier.joblib` under ignored `model_weights/`, with adjacent metadata JSON files, and prints held-out metrics.

The Parquet `annotation` column contains sentiment labels, not bullying labels. It can be trained as a separate experimental signal:

```powershell
python -m ml.training.train_text_model --dataset datasets/raw/user_uploads/train-00000-of-00001.parquet --text-column text --label-column annotation --output model_weights/sentiment_classifier.joblib --dataset-version uploaded-sentiment
```

The 200-row Parquet test split is reserved for independent evaluation. The generated sentiment model scored 53.5% accuracy on that split, below its 59% majority-class baseline, so the app does not load it by default. Set `SENTIMENT_MODEL_PATH=model_weights/sentiment_classifier.joblib` only when experimenting with better sentiment data. Sentiment is not a proxy for bullying or toxicity.

To prepare the separate HDF5 image model, pass its source archive as `--image-model-archive` to the preparation command. Install `requirements-image.txt` to enable TensorFlow inference. The image adapter reads `model_weights/adapters/image_deepfake_adapter.json`, which points to the local checkpoint. The supplied archive has no license file and no independent validation data; confirm rights and evaluate it on representative images before relying on or redistributing the model.

## Audio Anti-Spoof Evaluation

The Hugging Face ASVspoof 2019 LA dataset contains only an evaluation split, so it must not be used for training. Evaluate the local HDF5 checkpoint directly (independent of the app's audio-deepfake adapter, which does not currently perform inference) on a reproducibly shuffled stream:

```powershell
python -m scripts.evaluate_asvspoof_audio --limit 512
```

The benchmark is about 4.39 GB. Use `--limit 0` to process its full 71,237 examples; results are written to the ignored `model_weights/asvspoof2019_la_evaluation.json`. The evaluator uses `Audio(decode=False)` and `soundfile`, avoiding the optional `torchcodec` decoder. This benchmark is external validation; it does not automatically mark the audio checkpoint as production-validated.

The referenced multimodal repositories do not provide ready-to-use, cleared video/audio data and weights in their GitHub trees. The existing media adapters remain unavailable until compatible, licensed model artifacts are supplied; the text classifier does not claim to detect deepfakes or video hate speech.
The image training entry point `python -m ml.training.train_openfake_image_model` streams a bounded sample from OpenFake v2, trains the CNN used by the image-analysis endpoint, and saves validation metadata. OpenFake is CC BY-NC 4.0, so this model is strictly non-commercial. The script can optionally evaluate the held-out OOD and Reddit splits independently. See [datasets/README.md](../datasets/README.md) for attribution, limits, and configuration.
