from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TrialMetadata:
    trial_id: str
    target_present: bool
    condition: str
    motion: str
    target_label: str
    notes: str = ""


class TrialRecorder:
    def __init__(self, metadata: TrialMetadata) -> None:
        self.metadata = metadata
        self.total_frames = 0
        self.detection_frames = 0
        self.first_detection_frame: int | None = None
        self.risk_counts: Counter[str] = Counter()

    def observe(self, frame_number: int, detections: list[Any], risk_level: str | None) -> None:
        self.total_frames += 1
        if detections:
            self.detection_frames += 1
            if self.first_detection_frame is None:
                self.first_detection_frame = frame_number
        if risk_level is not None:
            self.risk_counts[risk_level] += 1

    def summary(self) -> dict[str, Any]:
        rate = self.detection_frames / self.total_frames if self.total_frames else 0.0
        return {
            "schema_version": 1,
            **asdict(self.metadata),
            "total_frames": self.total_frames,
            "detection_frames": self.detection_frames,
            "detection_rate": round(rate, 4),
            "false_alarm": bool(not self.metadata.target_present and self.detection_frames > 0),
            "missed_target": bool(self.metadata.target_present and self.detection_frames == 0),
            "first_detection_frame": self.first_detection_frame,
            "risk_counts": dict(self.risk_counts),
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.summary(), indent=2) + "\n", encoding="utf-8")
