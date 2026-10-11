from wildshield.types import BoundingBox, Detection
from wildshield.ui.alert import DetectionAlert


def cup(confidence: float = 0.8) -> Detection:
    return Detection("cup", "cup", confidence, BoundingBox(1, 2, 3, 4), 41)


def test_alerts_on_detection_and_respects_cooldown() -> None:
    messages: list[str] = []
    times = iter([10.0, 11.0, 12.1])
    alert = DetectionAlert(2.0, emit=messages.append, clock=lambda: next(times))

    assert alert.update([cup()]) is True
    assert alert.update([cup()]) is False
    assert alert.update([cup(0.9)]) is True
    assert messages == ["ALERT: CUP detected (80%)", "ALERT: CUP detected (90%)"]


def test_does_not_alert_without_detection() -> None:
    alert = DetectionAlert(2.0, emit=lambda _: None)
    assert alert.update([]) is False
