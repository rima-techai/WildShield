import argparse
import json
from pathlib import Path

import pytest

from wildshield.experiments.unified_trial import build_parser, detection_for_row, runner_args, summarize, validate


def args(**changes):
    values=dict(mode="fusion",trial_id="test-01",condition="dark",target_present=True,distance_m=3.0,motion="stationary",duration_seconds=10.0,notes="",classes="person",confidence=0.6,warm_threshold=3.0,model="yolo11n.pt",world_classes=None,image_output=None)
    values.update(changes)
    return argparse.Namespace(**values)


def test_validation_requires_empty_metadata_for_absent_trial():
    with pytest.raises(ValueError,match="motion=empty"):
        validate(args(target_present=False,motion="stationary",distance_m=0))
    with pytest.raises(ValueError,match="distance 0"):
        validate(args(target_present=False,motion="empty",distance_m=2))


def test_rgb_trial_accepts_world_prompts_and_image_output(tmp_path: Path):
    parsed = build_parser().parse_args(
        [
            "--mode", "rgb",
            "--trial-id", "deer-test",
            "--condition", "bright",
            "--target-present",
            "--distance-m", "1",
            "--motion", "stationary",
            "--source", "deer1.jpg",
            "--model", "yolov8s-world.pt",
            "--world-classes", "deer",
            "--classes", "deer",
            "--image-output", str(tmp_path / "annotated.jpg"),
        ]
    )
    validate(parsed)
    mode_args = runner_args(parsed, tmp_path / "trial.jsonl", tmp_path / "snapshot.png")
    assert mode_args.world_classes == "deer"
    assert mode_args.image_output == tmp_path / "annotated.jpg"


def test_world_prompts_are_restricted_to_rgb_trials():
    with pytest.raises(ValueError, match="only supported for RGB"):
        validate(args(mode="fusion", world_classes="deer"))


def test_mode_specific_detection():
    assert detection_for_row("rgb",{"detections":[{"label":"person"}]})
    assert detection_for_row("thermal",{"detected":True})
    assert detection_for_row("fusion",{"fusion_status":"RGB + THERMAL TEMPORAL CONFIRMATION"})


def test_summary_records_ground_truth_and_rates(tmp_path: Path):
    path=tmp_path/"trial.jsonl"
    rows=[
        {"frame":1,"captured_at":"2026-09-13T12:00:00+00:00","fusion_status":"NO TARGET","rgb_inference_ms":20,"risk":None},
        {"frame":2,"captured_at":"2026-09-13T12:00:01+00:00","fusion_status":"RGB + THERMAL TEMPORAL CONFIRMATION","rgb_inference_ms":30,"risk":{"level":"SAFE"},"thermal_detection":{"peak_c":35,"baseline_c":28}},
    ]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))
    result=summarize(path,args())
    assert result["ground_truth"]["condition"]=="dark"
    assert result["results"]["detection_rate"]==0.5
    assert result["results"]["first_detection_frame"]==2
    assert result["results"]["median_inference_ms"]==25.0
    assert result["configuration"]["rgb_world_classes"] is None
