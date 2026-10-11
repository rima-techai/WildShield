from __future__ import annotations

from pathlib import Path
from typing import Any


def normalize_source(source: str) -> int | str:
    """Convert numeric camera IDs while preserving file and stream paths."""
    stripped = source.strip()
    return int(stripped) if stripped.isdecimal() else stripped


class VideoSource:
    def __init__(self, source: str) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is not installed. Run: uv sync --extra dev") from exc

        self.cv2 = cv2
        self.source = normalize_source(source)
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video source: {source}")

    @property
    def fps(self) -> float:
        value = float(self.capture.get(self.cv2.CAP_PROP_FPS))
        return value if 1.0 <= value <= 240.0 else 30.0

    def read(self) -> tuple[bool, Any]:
        return self.capture.read()

    def close(self) -> None:
        self.capture.release()


class VideoSink:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]) -> None:
        import cv2

        path.parent.mkdir(parents=True, exist_ok=True)
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), codec, fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create output video: {path}")

    def write(self, frame: Any) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

