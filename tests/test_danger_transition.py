from wildshield.risk.evaluator import RiskLevel
from wildshield.ui.alert import DangerTransitionGate


def test_alerts_only_when_entering_danger():
    gate = DangerTransitionGate()
    levels = [None, RiskLevel.SAFE, RiskLevel.WARNING, RiskLevel.DANGER, RiskLevel.DANGER, RiskLevel.DANGER]
    assert [gate.update(level) for level in levels] == [False, False, False, True, False, False]


def test_rearms_after_danger_clears():
    gate = DangerTransitionGate()
    assert gate.update(RiskLevel.DANGER) is True
    assert gate.update(None) is False
    assert gate.update(RiskLevel.DANGER) is True
