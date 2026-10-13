# ML Pipeline

The text training entry point supports single-label cyberbullying CSVs (such as `tweet` + `cyberbullying_type`) and multi-label toxicity CSVs (such as Jigsaw `comment_text` plus its binary label columns). See [datasets/README.md](../datasets/README.md) for source and terms notes and exact commands.

Training writes a TF-IDF/logistic-regression artifact and adjacent JSON metadata under `model_weights/`, including dataset identifier, label mapping, split seed, and held-out metrics. Set `TEXT_MODEL_PATH` to enable it in the Flask text-analysis route. With no trained artifact configured, the app keeps using its explicitly named lightweight baseline.

The referenced multimodal repositories do not provide ready-to-use, cleared video/audio data and weights in their GitHub trees. The existing media adapters remain unavailable until compatible, licensed model artifacts are supplied; the text classifier does not claim to detect deepfakes or video hate speech.
The image training entry point `python -m ml.training.train_openfake_image_model` streams a bounded sample from OpenFake v2, trains the CNN used by the image-analysis endpoint, and saves validation metadata. OpenFake is CC BY-NC 4.0, so this model is strictly non-commercial. The script can optionally evaluate the held-out OOD and Reddit splits independently. See [datasets/README.md](../datasets/README.md) for attribution, limits, and configuration.
