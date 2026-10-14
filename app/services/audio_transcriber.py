import subprocess
from pathlib import Path
from functools import lru_cache

from flask import current_app
import numpy as np

from .model_registry import ModelState, registry


class AudioTranscriber:
    module_name = "audio_transcription"

    def __init__(self):
        model_name = current_app.config.get("WHISPER_MODEL_NAME")
        status = "available" if model_name else "unavailable"
        registry.set("audio_transcriber", ModelState(name=model_name or "whisper-not-configured", status=status, detail="Set WHISPER_MODEL_NAME to enable transcription."))

    def analyze(self, path):
        if not Path(path).exists():
            return self._result("invalid_input", [], {}, "needs_review", "Audio file was not found.")
        if registry.get("audio_transcriber").status != "available":
            return self._result(
                "unavailable",
                ["transcription_not_performed"],
                {},
                "needs_review",
                "Whisper is not configured in this environment, so speech transcription was not performed.",
            )
        try:
            model_name = registry.get("audio_transcriber").name
            transcript = _transcribe_audio(model_name, str(path))
        except (OSError, RuntimeError, ImportError, ValueError) as error:
            return self._result(
                "unavailable",
                ["transcription_failed"],
                {},
                "needs_review",
                f"Whisper could not transcribe this audio in the current environment: {error}",
            )
        labels = ["speech_transcribed"] if transcript else ["no_speech_detected"]
        return self._result(
            "available",
            labels,
            {"transcript": transcript},
            "needs_review",
            "Whisper transcription completed. Review the transcript before using it for moderation decisions.",
        )

    def _result(self, status, labels, confidence, risk, explanation):
        return {
            "module_name": self.module_name,
            "model_name": registry.get("audio_transcriber").name,
            "model_version": "adapter-1.0",
            "predicted_labels": labels,
            "confidence_scores": confidence,
            "risk_category": risk,
            "availability_status": status,
            "explanation": explanation,
        }


@lru_cache(maxsize=2)
def _load_whisper_model(model_name):
    import whisper

    return whisper.load_model(model_name)


def _transcribe_audio(model_name, path):
    model = _load_whisper_model(model_name)
    import imageio_ffmpeg

    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-nostdin",
        "-threads",
        "0",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-",
    ]
    try:
        decoded = subprocess.run(command, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as error:
        detail = error.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"FFmpeg could not decode this audio: {detail or error}") from error
    if not decoded:
        raise ValueError("FFmpeg returned no audio samples.")
    audio = np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768.0
    result = model.transcribe(audio, fp16=False)
    return (result.get("text") or "").strip()
