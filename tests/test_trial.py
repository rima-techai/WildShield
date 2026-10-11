import json

from wildshield.experiments.trial import TrialMetadata, TrialRecorder


def test_positive_trial_metrics(tmp_path) -> None:
    trial = TrialRecorder(TrialMetadata("bright-01", True, "bright", "approaching", "cup"))
    trial.observe(1, [], "SAFE")
    trial.observe(2, [object()], "WARNING")
    trial.observe(3, [object()], "DANGER")
    path = tmp_path / "trial.summary.json"
    trial.write(path)
    result = json.loads(path.read_text())
    assert result["detection_rate"] == 0.6667
    assert result["first_detection_frame"] == 2
    assert result["false_alarm"] is False
    assert result["missed_target"] is False
    assert result["risk_counts"] == {"SAFE": 1, "WARNING": 1, "DANGER": 1}


def test_negative_trial_marks_false_alarm() -> None:
    trial = TrialRecorder(TrialMetadata("empty-01", False, "dark", "empty", "cup"))
    trial.observe(1, [object()], None)
    assert trial.summary()["false_alarm"] is True
