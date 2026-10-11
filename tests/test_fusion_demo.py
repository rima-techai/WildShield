from pathlib import Path

import numpy as np

from wildshield.fusion_demo import build_parser, compose_dashboard, fusion_status, unique_snapshot_path
from wildshield.types import BoundingBox, Detection


def detection():
    return Detection("person", "person", 0.9, BoundingBox(1, 2, 20, 30), 0)


def test_fusion_statuses_are_explicit():
    assert fusion_status([detection()], True) == "RGB + THERMAL TEMPORAL CONFIRMATION"
    assert fusion_status([], True) == "THERMAL TARGET | RGB CLASS UNAVAILABLE"
    assert fusion_status([detection()], False) == "RGB TARGET | NO THERMAL CONFIRMATION"
    assert fusion_status([], False) == "NO TARGET"


def test_dashboard_has_two_panels_and_footer():
    left = np.zeros((480, 640, 3), dtype=np.uint8)
    right = np.zeros((480, 640, 3), dtype=np.uint8)
    dashboard = compose_dashboard(left, right, "NO TARGET")
    assert dashboard.shape == (564, 1280, 3)


def test_unique_snapshot_path_preserves_new_path(tmp_path: Path):
    path = tmp_path / "snapshot.png"
    assert unique_snapshot_path(path) == path
    path.write_bytes(b"existing")
    unique = unique_snapshot_path(path)
    assert unique != path
    assert unique.suffix == ".png"


def test_fusion_parser_accepts_video_output(tmp_path: Path):
    output = tmp_path / "demo.mp4"
    args = build_parser().parse_args(["--video-output", str(output)])
    assert args.video_output == output
    assert args.video_fps == 1.0
