from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


FRAME_WIDTH = 32
FRAME_HEIGHT = 24


@dataclass(frozen=True, slots=True)
class ThermalTarget:
    x1: int
    y1: int
    x2: int
    y2: int
    pixel_count: int
    baseline_c: float
    peak_c: float
    mean_c: float

    def as_dict(self) -> dict[str, object]:
        return {
            "label": "warm_target",
            "box": {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2},
            "hot_pixel_count": self.pixel_count,
            "baseline_c": round(self.baseline_c, 2),
            "mean_c": round(self.mean_c, 2),
            "peak_c": round(self.peak_c, 2),
        }


class WarmTargetDetector:
    def __init__(self, threshold_delta_c: float = 3.0, min_pixels: int = 4, persistence: int = 2) -> None:
        if threshold_delta_c <= 0:
            raise ValueError("threshold delta must be positive")
        if min_pixels < 1 or persistence < 1:
            raise ValueError("min pixels and persistence must be at least 1")
        self.threshold_delta_c = threshold_delta_c
        self.min_pixels = min_pixels
        self.persistence = persistence
        self._streak = 0

    def detect(self, temperatures: list[float] | np.ndarray) -> tuple[ThermalTarget | None, bool]:
        values = np.asarray(temperatures, dtype=np.float32)
        if values.size != FRAME_WIDTH * FRAME_HEIGHT:
            raise ValueError(f"expected 768 temperatures, got {values.size}")
        thermal = values.reshape((FRAME_HEIGHT, FRAME_WIDTH))
        baseline = float(np.percentile(thermal, 25.0))
        mask = (thermal >= baseline + self.threshold_delta_c).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        candidate: ThermalTarget | None = None
        if component_count > 1:
            component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area >= self.min_pixels:
                x = int(stats[component, cv2.CC_STAT_LEFT])
                y = int(stats[component, cv2.CC_STAT_TOP])
                width = int(stats[component, cv2.CC_STAT_WIDTH])
                height = int(stats[component, cv2.CC_STAT_HEIGHT])
                component_values = thermal[labels == component]
                candidate = ThermalTarget(
                    x1=x,
                    y1=y,
                    x2=x + width,
                    y2=y + height,
                    pixel_count=area,
                    baseline_c=baseline,
                    peak_c=float(component_values.max()),
                    mean_c=float(component_values.mean()),
                )

        self._streak = self._streak + 1 if candidate is not None else 0
        return candidate, candidate is not None and self._streak >= self.persistence
