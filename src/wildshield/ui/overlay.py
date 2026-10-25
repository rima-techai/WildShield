from __future__ import annotations

from datetime import datetime
from typing import Any

from wildshield.types import Detection
from wildshield.risk.evaluator import RiskAssessment, RiskLevel
from wildshield.types import RadarObservation


COLORS = {
    "deer": (0, 165, 255),
    "elephant": (150, 100, 220),
    "armadillo": (120, 150, 190),
    "dog": (255, 180, 0),
    "cow": (180, 105, 255),
    "person": (0, 220, 255),
    "cup": (80, 220, 80),
    "animal_model": (60, 210, 255),
    "other": (180, 180, 180),
}


def draw_overlay(frame: Any, detections: list[Detection], fps: float, timestamp: datetime) -> Any:
    import cv2

    for detection in detections:
        box = detection.box
        color = COLORS[detection.label]
        cv2.rectangle(frame, (box.x1, box.y1), (box.x2, box.y2), color, 2)
        track = f" #{detection.track_id}" if detection.track_id is not None else ""
        text = f"{detection.label.upper()}{track} {detection.confidence:.0%}"
        cv2.putText(frame, text, (box.x1, max(20, box.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    status = f"WildShield RGB | {fps:.1f} FPS | {timestamp:%Y-%m-%d %H:%M:%S}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(frame, status, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
    return frame


def draw_risk_panel(frame: Any, radar: RadarObservation, risk: RiskAssessment) -> Any:
    import cv2

    colors = {
        RiskLevel.SAFE: (40, 190, 40),
        RiskLevel.WARNING: (0, 190, 255),
        RiskLevel.DANGER: (30, 30, 235),
    }
    color = colors[risk.level]
    x1, y1, x2, y2 = max(0, frame.shape[1] - 430), 46, frame.shape[1] - 12, 180
    cv2.rectangle(frame, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    lines = [
        "SIMULATED RADAR",
        f"Track: #{radar.track_id}   Range: {radar.range_m:.2f} m",
        f"Velocity: {radar.radial_velocity_mps:.2f} m/s   Angle: {radar.azimuth_deg:.1f} deg",
        f"RISK: {risk.level}",
    ]
    for index, line in enumerate(lines):
        scale = 0.72 if index == 3 else 0.55
        line_color = color if index == 3 else (235, 235, 235)
        cv2.putText(frame, line, (x1 + 12, y1 + 26 + index * 29), cv2.FONT_HERSHEY_SIMPLEX, scale, line_color, 2)
    return frame
