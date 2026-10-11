from wildshield.risk.evaluator import RiskEvaluator, RiskLevel
from wildshield.types import RadarObservation


def observation(distance: float, velocity: float = -1.0) -> RadarObservation:
    return RadarObservation(1, distance, velocity, 0.0)


def test_three_risk_zones() -> None:
    evaluator = RiskEvaluator(warning_range_m=3.0, danger_range_m=1.0)
    assert evaluator.evaluate(observation(4.0)).level == RiskLevel.SAFE
    assert evaluator.evaluate(observation(2.0)).level == RiskLevel.WARNING
    assert evaluator.evaluate(observation(0.5)).level == RiskLevel.DANGER


def test_stationary_target_outside_danger_is_safe() -> None:
    evaluator = RiskEvaluator()
    assert evaluator.evaluate(observation(2.0, 0.0)).level == RiskLevel.SAFE
