import numpy as np
import pytest

from wildshield.sensors.thermal_view import colorize_frame


def test_colorize_frame_returns_scaled_bgr_image_and_hotspot():
    temperatures = np.linspace(20.0, 30.0, 768, dtype=np.float32)
    image, low, high, hottest = colorize_frame(temperatures, scale=4)
    assert image.shape == (24 * 4, 32 * 4, 3)
    assert low < high
    assert hottest == (31, 23)


def test_colorize_frame_handles_uniform_temperature():
    image, low, high, hottest = colorize_frame([25.0] * 768, scale=2)
    assert image.shape == (48, 64, 3)
    assert high > low
    assert hottest == (0, 0)


def test_colorize_frame_rejects_wrong_frame_size():
    with pytest.raises(ValueError, match="expected 768"):
        colorize_frame([20.0] * 10)


def test_colorize_frame_rejects_invalid_scale():
    with pytest.raises(ValueError, match="scale"):
        colorize_frame([20.0] * 768, scale=0)
