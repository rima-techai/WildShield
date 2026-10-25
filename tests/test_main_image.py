import cv2
import numpy as np

from wildshield import main


def test_run_saves_annotated_still_and_passes_world_prompts(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "deer.jpg"
    output_path = tmp_path / "annotated.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8))
    detector_options = {}

    class StubDetector:
        def __init__(self, model_path, **kwargs):
            detector_options.update(kwargs)

        def detect(self, frame):
            return [], 1.0

    monkeypatch.setattr(main, "YoloDetector", StubDetector)
    monkeypatch.setattr(main, "draw_overlay", lambda frame, *args: frame)
    args = main.build_parser().parse_args(
        [
            "--source",
            str(image_path),
            "--world-classes",
            "deer",
            "--image-output",
            str(output_path),
            "--log",
            str(tmp_path / "detections.jsonl"),
            "--no-display",
        ]
    )

    assert main.run(args) == 0
    assert detector_options["world_classes"] == ["deer"]
    assert cv2.imread(str(output_path)).shape == (12, 16, 3)
