import tempfile
from pathlib import Path

from flask import current_app
import numpy as np
from PIL import Image

from .image_detector import ImageDetector
from .model_registry import ModelState, registry


class VideoDetector:
    module_name = "video_deepfake"

    def __init__(self):
        model_path = current_app.config.get("VIDEO_MODEL_PATH") or current_app.config.get("IMAGE_MODEL_PATH")
        self.frame_detector = ImageDetector(model_path, register_state=False)
        self.max_frames = max(1, int(current_app.config.get("VIDEO_MAX_FRAMES", 8)))
        status = "available" if self.frame_detector.model_backend else "unavailable"
        detail = (
            "Video frames will be screened by the configured image model; aggregate results require video-specific validation."
            if status == "available"
            else "No compatible visual model is available for video frame screening."
        )
        registry.set("video", ModelState(name="external-video-deepfake-adapter", status=status, detail=detail))

    def analyze(self, path):
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return self._result("invalid_input", [], {}, "needs_review", "Video could not be decoded.")
        try:
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration = round(frame_count / fps, 2) if fps else None
            meta = {
                "frame_count": frame_count,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_seconds": duration,
            }
            if frame_count <= 0 or width <= 0 or height <= 0:
                return self._result("invalid_input", [], {"video": meta}, "needs_review", "Video metadata is unreadable.")
            if registry.get("video").status != "available":
                return self._result(
                    "unavailable",
                    ["analysis_not_performed"],
                    {"video": meta, "analyzed_frames": 0, "per_frame_results": []},
                    "needs_review",
                    registry.get("video").detail,
                )

            frame_indices = np.linspace(0, frame_count - 1, min(self.max_frames, frame_count), dtype=int)
            frame_results = []
            with tempfile.TemporaryDirectory() as temporary_directory:
                for frame_index in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                    success, frame = capture.read()
                    if not success:
                        continue
                    image_path = Path(temporary_directory) / f"frame_{frame_index}.png"
                    Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(image_path)
                    image_result = self.frame_detector.analyze(image_path)
                    confidence = image_result.get("confidence_scores", {})
                    if "real" in confidence and "fake" in confidence:
                        frame_results.append(
                            {
                                "frame_index": int(frame_index),
                                "real": confidence["real"],
                                "fake": confidence["fake"],
                                "candidate_label": confidence.get("candidate_label")
                                or (image_result.get("predicted_labels") or [None])[0],
                            }
                        )
            if not frame_results:
                return self._result(
                    "unavailable",
                    ["analysis_not_performed"],
                    {"video": meta, "analyzed_frames": 0, "per_frame_results": []},
                    "needs_review",
                    "The configured frame model did not return usable real/fake scores.",
                )

            mean_real = float(np.mean([frame["real"] for frame in frame_results]))
            mean_fake = float(np.mean([frame["fake"] for frame in frame_results]))
            candidate = "real" if mean_real >= mean_fake else "fake"
            return self._result(
                "available",
                ["needs_review"],
                {
                    "video": meta,
                    "analyzed_frames": len(frame_results),
                    "real": round(mean_real, 4),
                    "fake": round(mean_fake, 4),
                    "candidate_label": candidate,
                    "per_frame_results": frame_results,
                },
                "needs_review",
                f"Frame-screening candidate: {candidate}. The image checkpoint is not validated for video authenticity, so this result requires human review.",
            )
        except (OSError, RuntimeError, ValueError) as error:
            return self._result("unavailable", ["analysis_not_performed"], {}, "needs_review", f"Video frame analysis failed: {error}")
        finally:
            capture.release()

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
