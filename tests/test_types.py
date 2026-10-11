from wildshield.types import BoundingBox, Detection


def test_detection_serialization() -> None:
    detection = Detection(
        label="other",
        model_label="horse",
        confidence=0.75,
        box=BoundingBox(1, 2, 30, 40),
        class_id=17,
    )
    assert detection.as_dict() == {
        "label": "other",
        "model_label": "horse",
        "confidence": 0.75,
        "box": {"x1": 1, "y1": 2, "x2": 30, "y2": 40},
        "class_id": 17,
        "track_id": None,
    }
