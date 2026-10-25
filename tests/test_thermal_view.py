import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from wildshield.perception.thermal_detector import ThermalTarget
from wildshield.sensors import thermal_view
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


def test_confirmed_warm_target_triggers_audio_without_simulated_radar(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeSensor:
        def getFrame(self, temperatures):
            temperatures[:] = [25.0] * 768

    class FakeDetector:
        def __init__(self, *args):
            pass

        def detect(self, temperatures):
            return ThermalTarget(10, 8, 15, 13, 25, 25.0, 31.0, 29.0), True

    board = ModuleType("board")
    board.I2C = lambda: object()
    mlx = ModuleType("adafruit_mlx90640")
    mlx.MLX90640 = lambda i2c: FakeSensor()
    mlx.RefreshRate = SimpleNamespace(REFRESH_1_HZ=1, REFRESH_2_HZ=2, REFRESH_4_HZ=4)
    monkeypatch.setitem(sys.modules, "board", board)
    monkeypatch.setitem(sys.modules, "adafruit_mlx90640", mlx)
    monkeypatch.setenv("BLINKA_MCP2221", "1")
    monkeypatch.setattr(thermal_view, "WarmTargetDetector", FakeDetector)
    audio_messages: list[str] = []
    monkeypatch.setattr(thermal_view, "AudioEmitter", lambda sound: audio_messages.append)

    args = SimpleNamespace(
        scale=2,
        refresh_rate=2,
        warm_threshold=3.0,
        min_hot_pixels=4,
        persistence=2,
        sim_range=5.0,
        sim_speed=0.6,
        warning_range=3.0,
        danger_range=1.0,
        simulate_radar=False,
        alert_audio=True,
        alert_sound=None,
        alert_cooldown=3.0,
        no_display=True,
        no_log=True,
        max_frames=1,
        duration_seconds=None,
        snapshot=tmp_path / "thermal.png",
        save_final=True,
        log=tmp_path / "thermal.jsonl",
    )

    assert thermal_view.run(args) == 0
    assert len(audio_messages) == 1
    assert "warm_target" in audio_messages[0].lower()
    assert args.snapshot.is_file()
