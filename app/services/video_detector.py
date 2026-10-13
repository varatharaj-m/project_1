from pathlib import Path

from flask import current_app

from .model_registry import ModelState, registry


class VideoDetector:
    module_name = "video_deepfake"

    def __init__(self):
        model_path = current_app.config.get("VIDEO_MODEL_PATH")
        status = "available" if model_path and Path(model_path).exists() else "unavailable"
        detail = "External video deepfake model configured." if status == "available" else "No video deepfake weights configured."
        registry.set("video", ModelState(name="external-video-deepfake-adapter", status=status, detail=detail))

    def analyze(self, path):
        import cv2

        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return self._result("invalid_input", [], {}, "needs_review", "Video could not be decoded.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = round(frame_count / fps, 2) if fps else None
        capture.release()
        meta = {"frame_count": frame_count, "fps": fps, "width": width, "height": height, "duration_seconds": duration}
        if frame_count <= 0 or width <= 0 or height <= 0:
            return self._result("invalid_input", [], {"video": meta}, "needs_review", "Video metadata is unreadable.")
        if registry.get("video").status != "available":
            return self._result(
                "unavailable",
                ["analysis_not_performed"],
                {"video": meta, "analyzed_frames": 0, "per_frame_results": []},
                "needs_review",
                "No dedicated video deepfake model weights are installed. Metadata was read, but frame-level authenticity analysis was not performed.",
            )
        return self._result("available", ["requires_external_model_inference"], {"video": meta}, "needs_review", "Adapter ready for configured model inference.")

    def _result(self, status, labels, confidence, risk, explanation):
        return {
            "module_name": self.module_name,
            "model_name": "external-video-deepfake-adapter",
            "model_version": "adapter-1.0",
            "predicted_labels": labels,
            "confidence_scores": confidence,
            "risk_category": risk,
            "availability_status": status,
            "explanation": explanation,
        }
