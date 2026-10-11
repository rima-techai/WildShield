from wildshield.fusion import FusionLevel, evaluate_fusion
from wildshield.types import BoundingBox, Detection, RadarObservation


def rgb_detection() -> Detection:
    return Detection("animal_model", "teddy bear", 0.8, BoundingBox(1, 2, 20, 30), 77)


def radar(velocity: float, simulated: bool = False) -> RadarObservation:
    return RadarObservation(1, 2.0, velocity, 0.0, simulated)


def test_rgb_and_thermal_confirm_animal():
    decision = evaluate_fusion([rgb_detection()], True, None)
    assert decision.level is FusionLevel.CONFIRMED


def test_thermal_and_approaching_radar_warn_when_rgb_fails():
    decision = evaluate_fusion([], True, radar(-0.5))
    assert decision.level is FusionLevel.WARNING
    assert decision.label == "PROBABLE ANIMAL - WARNING"


def test_thermal_without_approaching_radar_is_monitored():
    assert evaluate_fusion([], True, None).level is FusionLevel.MONITOR
    assert evaluate_fusion([], True, radar(0.2)).level is FusionLevel.MONITOR


def test_rgb_and_approaching_radar_warn_without_thermal():
    assert evaluate_fusion([rgb_detection()], False, radar(-0.5)).level is FusionLevel.WARNING


def test_rgb_alone_is_monitored():
    assert evaluate_fusion([rgb_detection()], False, None).level is FusionLevel.MONITOR


def test_no_evidence_is_safe():
    assert evaluate_fusion([], False, None).level is FusionLevel.SAFE


def test_simulated_radar_is_disclosed_in_decision():
    decision = evaluate_fusion([], True, radar(-0.5, simulated=True))
    assert decision.radar_simulated is True
    assert decision.as_dict()["radar_simulated"] is True
