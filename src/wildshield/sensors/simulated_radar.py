from __future__ import annotations

import time
from collections.abc import Callable

from wildshield.types import Detection, RadarObservation


class SimulatedRadar:
    """Deterministic radar substitute for UI and risk-engine development."""

    def __init__(
        self,
        start_range_m: float = 5.0,
        approach_speed_mps: float = 0.6,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if start_range_m <= 0 or approach_speed_mps < 0:
            raise ValueError("simulated radar range must be positive and speed non-negative")
        self.start_range_m = start_range_m
        self.approach_speed_mps = approach_speed_mps
        self._clock = clock
        self._started_at: dict[int, float] = {}

    def reset(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._started_at.clear()
        else:
            self._started_at.pop(track_id, None)

    def observe(self, detection: Detection, frame_width: int) -> RadarObservation:
        if detection.track_id is None:
            raise ValueError("simulated radar requires a tracked detection")
        now = self._clock()
        started = self._started_at.setdefault(detection.track_id, now)
        range_m = max(0.0, self.start_range_m - self.approach_speed_mps * (now - started))
        center_x = (detection.box.x1 + detection.box.x2) / 2
        azimuth_deg = ((center_x / max(frame_width, 1)) - 0.5) * 60.0
        return RadarObservation(
            track_id=detection.track_id,
            range_m=round(range_m, 2),
            radial_velocity_mps=-self.approach_speed_mps,
            azimuth_deg=round(azimuth_deg, 1),
        )
