from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from wildshield.types import RadarObservation


class RiskLevel(StrEnum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    DANGER = "DANGER"


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    level: RiskLevel
    reason: str
    time_to_path_s: float | None


class RiskEvaluator:
    def __init__(self, warning_range_m: float = 3.0, danger_range_m: float = 1.0) -> None:
        if not 0 < danger_range_m < warning_range_m:
            raise ValueError("risk ranges must satisfy 0 < danger < warning")
        self.warning_range_m = warning_range_m
        self.danger_range_m = danger_range_m

    def evaluate(self, radar: RadarObservation) -> RiskAssessment:
        speed_toward = max(0.0, -radar.radial_velocity_mps)
        time_to_path = radar.range_m / speed_toward if speed_toward else None
        if radar.range_m <= self.danger_range_m:
            return RiskAssessment(RiskLevel.DANGER, "target is at the danger boundary", time_to_path)
        if radar.range_m <= self.warning_range_m and speed_toward > 0:
            return RiskAssessment(RiskLevel.WARNING, "target is approaching the roadway", time_to_path)
        return RiskAssessment(RiskLevel.SAFE, "target is outside the warning boundary", time_to_path)
