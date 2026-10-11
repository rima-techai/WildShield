from wildshield.perception.tracker import CentroidTracker
from wildshield.types import BoundingBox, Detection


def detection(x: int) -> Detection:
    return Detection("cup", "cup", 0.8, BoundingBox(x, 0, x + 20, 20), 41)


def test_tracker_preserves_id_for_nearby_detection() -> None:
    tracker = CentroidTracker(max_distance_px=50)
    first = tracker.update([detection(10)])[0]
    second = tracker.update([detection(20)])[0]
    assert first.track_id == second.track_id == 1


def test_tracker_creates_new_id_for_distant_detection() -> None:
    tracker = CentroidTracker(max_distance_px=10)
    assert tracker.update([detection(0)])[0].track_id == 1
    assert tracker.update([detection(100)])[0].track_id == 2
