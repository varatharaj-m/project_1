# Model Weights

Downloaded or trained weights belong here and are not committed by default.

Configure paths with:

- `IMAGE_MODEL_PATH`
- `VIDEO_MODEL_PATH`
- `AUDIO_MODEL_PATH`
- `WHISPER_MODEL_NAME`

The application reports unavailable statuses when dedicated model weights are absent.

The OpenFake image classifier is released under the dataset's CC BY-NC 4.0 terms. Its checkpoint and metadata must remain attributed and must not be used commercially. Loading it requires `ALLOW_NONCOMMERCIAL_MODELS=true`; keep that setting false for commercial deployments.
