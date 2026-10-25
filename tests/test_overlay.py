from datetime import datetime

import numpy as np

from wildshield.types import BoundingBox, Detection
from wildshield.ui.overlay import draw_overlay


def test_overlay_draws_armadillo_detection() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    detection = Detection(
        "armadillo",
        "armadillo",
        0.85,
        BoundingBox(10, 10, 80, 80),
        0,
    )

    annotated = draw_overlay(frame, [detection], 30.0, datetime.now())

    assert annotated.shape == frame.shape
    assert np.any(annotated[80, 50] != 0)
