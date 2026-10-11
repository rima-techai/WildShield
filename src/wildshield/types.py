from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Detection:
    label: str
    model_label: str
    confidence: float
    box: BoundingBox
    class_id: int
    track_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.as_dict()
        return result


@dataclass(frozen=True, slots=True)
class RadarObservation:
    track_id: int
    range_m: float
    radial_velocity_mps: float
    azimuth_deg: float
    simulated: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
