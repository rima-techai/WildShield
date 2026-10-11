from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from wildshield.perception.labels import project_label
from wildshield.types import BoundingBox, Detection


class YoloDetector:
    """Thin adapter around Ultralytics YOLO with project-specific output."""

    def __init__(self, model_path: str, confidence: float = 0.4, device: str | None = None) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Ultralytics is not installed. Run: uv sync --extra dev") from exc

        self._model = YOLO(model_path)
        self._confidence = confidence
        self._device = device

    def detect(self, frame: Any) -> tuple[list[Detection], float]:
        predictions = self._model.predict(
            source=frame,
            conf=self._confidence,
            device=self._device,
            verbose=False,
        )
        if not predictions:
            return [], 0.0

        result = predictions[0]
        names: Mapping[int, str] = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            model_label = str(names[class_id])
            x1, y1, x2, y2 = (round(value) for value in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    label=project_label(model_label),
                    model_label=model_label,
                    confidence=float(box.conf.item()),
                    box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    class_id=class_id,
                )
            )

        inference_ms = float(result.speed.get("inference", 0.0))
        return detections, inference_ms

