from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

from wildshield.types import Detection


@dataclass(slots=True)
class _TrackState:
    center: tuple[float, float]
    missed_frames: int = 0


class CentroidTracker:
    """Small nearest-centroid tracker suitable for a controlled one-target demo."""

    def __init__(self, max_distance_px: float = 200.0, max_missed_frames: int = 15) -> None:
        self.max_distance_px = max_distance_px
        self.max_missed_frames = max_missed_frames
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    @staticmethod
    def _center(detection: Detection) -> tuple[float, float]:
        box = detection.box
        return ((box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2)

    def update(self, detections: list[Detection]) -> list[Detection]:
        unmatched_tracks = set(self._tracks)
        tracked: list[Detection] = []

        for detection in detections:
            center = self._center(detection)
            candidates = [
                (hypot(center[0] - state.center[0], center[1] - state.center[1]), track_id)
                for track_id, state in self._tracks.items()
                if track_id in unmatched_tracks
            ]
            distance, track_id = min(candidates, default=(float("inf"), -1))
            if distance > self.max_distance_px:
                track_id = self._next_id
                self._next_id += 1
                self._tracks[track_id] = _TrackState(center)
            else:
                self._tracks[track_id].center = center
                self._tracks[track_id].missed_frames = 0
                unmatched_tracks.remove(track_id)
            tracked.append(replace(detection, track_id=track_id))

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed_frames += 1
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if state.missed_frames <= self.max_missed_frames
        }
        return tracked
