import sys
from types import ModuleType, SimpleNamespace

import pytest

from wildshield.perception.detector import YoloDetector


def test_world_detector_sets_prompt_classes(monkeypatch) -> None:
    model = SimpleNamespace(
        set_classes=lambda classes: setattr(model, "classes", classes),
        predict=lambda **kwargs: [],
    )
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLOWorld = lambda model_path: model
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    detector = YoloDetector("yolov8s-world.pt", world_classes=[" deer ", ""])

    assert model.classes == ["deer"]
    assert detector.detect(object()) == ([], 0.0)


def test_world_detector_rejects_empty_prompt_list() -> None:
    with pytest.raises(ValueError, match="at least one"):
        YoloDetector("yolov8s-world.pt", world_classes=[" ", ""])


def test_armadillo_prompt_includes_model_synonym(monkeypatch) -> None:
    model = SimpleNamespace(set_classes=lambda classes: setattr(model, "classes", classes))
    ultralytics = ModuleType("ultralytics")
    ultralytics.YOLOWorld = lambda model_path: model
    monkeypatch.setitem(sys.modules, "ultralytics", ultralytics)

    YoloDetector("yolov8s-world.pt", world_classes=["armadillo"])

    assert model.classes == ["nine-banded armadillo"]
