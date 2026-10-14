from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _load_classifier(model_path, modified_at):
    import joblib

    return joblib.load(model_path)


def load_classifier(model_path):
    path = Path(model_path).resolve()
    return _load_classifier(str(path), path.stat().st_mtime_ns)

@lru_cache(maxsize=4)
def _load_keras_model(model_path, modified_at):
    import tensorflow as tf

    return tf.keras.models.load_model(model_path, compile=False)

def load_keras_model(model_path):
    path = Path(model_path).resolve()
    return _load_keras_model(str(path), path.stat().st_mtime_ns)