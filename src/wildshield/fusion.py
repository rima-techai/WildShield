from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from wildshield.types import Detection, RadarObservation


class FusionLevel(StrEnum):
    SAFE = "SAFE"
    MONITOR = "MONITOR"
    WARNING = "WARNING"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True, slots=True)
class FusionDecision:
    level: FusionLevel
    label: str
    basis: str
    radar_simulated: bool

    def as_dict(self) -> dict[str, str | bool]:
        result = asdict(self)
        result["level"] = str(self.level)
        return result


def evaluate_fusion(
    rgb_detections: list[Detection],
    thermal_confirmed: bool,
    radar: RadarObservation | None,
) -> FusionDecision:
    """Combine evidence while degrading gracefully when RGB fails."""
    rgb_detected = bool(rgb_detections)
    approaching = radar is not None and radar.radial_velocity_mps < 0
    simulated = bool(radar is not None and radar.simulated)

    if rgb_detected and thermal_confirmed:
        return FusionDecision(FusionLevel.CONFIRMED, "CONFIRMED ANIMAL", "RGB classification and thermal confirmation agree", simulated)
    if thermal_confirmed and approaching:
        return FusionDecision(FusionLevel.WARNING, "PROBABLE ANIMAL - WARNING", "thermal target with radar motion toward the roadway", simulated)
    if thermal_confirmed:
        return FusionDecision(FusionLevel.MONITOR, "THERMAL TARGET - MONITOR", "thermal target without approaching radar motion", simulated)
    if rgb_detected and approaching:
        return FusionDecision(FusionLevel.WARNING, "POSSIBLE OBJECT - WARNING", "RGB target with radar motion toward the roadway", simulated)
    if rgb_detected:
        return FusionDecision(FusionLevel.MONITOR, "RGB TARGET - MONITOR", "RGB target without thermal or approaching radar confirmation", simulated)
    return FusionDecision(FusionLevel.SAFE, "NO TARGET", "no sensor target evidence", simulated)
