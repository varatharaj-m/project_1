from dataclasses import dataclass, asdict


@dataclass
class ModelState:
    name: str
    status: str
    version: str = "1.0"
    device: str = "cpu"
    detail: str = ""


class ModelRegistry:
    def __init__(self):
        self._states = {}

    def set(self, key, state):
        self._states[key] = state

    def get(self, key):
        return self._states.get(key, ModelState(name=key, status="unavailable", detail="Model not initialized."))

    def health(self):
        return {key: asdict(value) for key, value in self._states.items()}


registry = ModelRegistry()


def refresh_model_registry():
    from .audio_deepfake_detector import AudioDeepfakeDetector
    from .audio_transcriber import AudioTranscriber
    from .fake_news_detector import FakeNewsDetector
    from .image_detector import ImageDetector
    from .text_detector import TextDetector
    from .video_detector import VideoDetector

    TextDetector()
    FakeNewsDetector()
    ImageDetector()
    VideoDetector()
    AudioTranscriber()
    AudioDeepfakeDetector()
