import numpy as np
import pytest

from wildshield.perception.thermal_detector import WarmTargetDetector


def frame_with_warm_region():
    frame = np.full((24, 32), 22.0, dtype=np.float32)
    frame[8:13, 10:16] = 31.0
    return frame.ravel()


def test_warm_target_requires_persistence():
    detector = WarmTargetDetector(threshold_delta_c=3.0, min_pixels=4, persistence=2)
    first, first_confirmed = detector.detect(frame_with_warm_region())
    second, second_confirmed = detector.detect(frame_with_warm_region())
    assert first is not None
    assert first_confirmed is False
    assert second_confirmed is True
    assert second is not None
    assert second.peak_c == pytest.approx(31.0)
    assert second.x1 <= 10 < second.x2
    assert second.y1 <= 8 < second.y2


def test_uniform_background_has_no_target():
    detector = WarmTargetDetector()
    target, confirmed = detector.detect(np.full(768, 22.0, dtype=np.float32))
    assert target is None
    assert confirmed is False


def test_small_hot_noise_is_rejected():
    frame = np.full(768, 22.0, dtype=np.float32)
    frame[100] = 40.0
    detector = WarmTargetDetector(min_pixels=4, persistence=1)
    target, confirmed = detector.detect(frame)
    assert target is None
    assert confirmed is False


def test_invalid_frame_size_is_rejected():
    with pytest.raises(ValueError, match="expected 768"):
        WarmTargetDetector().detect([20.0] * 20)
