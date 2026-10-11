from wildshield.sensors.simulated_radar import SimulatedRadar
from wildshield.types import BoundingBox, Detection


def test_simulated_radar_approaches_over_time() -> None:
    times = iter([10.0, 12.0])
    radar = SimulatedRadar(5.0, 0.5, clock=lambda: next(times))
    target = Detection("cup", "cup", 0.8, BoundingBox(40, 0, 60, 20), 41, track_id=1)
    assert radar.observe(target, 100).range_m == 5.0
    observation = radar.observe(target, 100)
    assert observation.range_m == 4.0
    assert observation.radial_velocity_mps == -0.5
    assert observation.azimuth_deg == 0.0


def test_simulated_radar_reset_restarts_range():
    times = iter([10.0, 12.0, 20.0])
    radar = SimulatedRadar(5.0, 0.5, clock=lambda: next(times))
    target = Detection("other", "warm_target", 0.8, BoundingBox(40, 0, 60, 20), -1, track_id=1)
    radar.observe(target, 100)
    assert radar.observe(target, 100).range_m == 4.0
    radar.reset(1)
    assert radar.observe(target, 100).range_m == 5.0
