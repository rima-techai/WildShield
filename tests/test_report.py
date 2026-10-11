import csv
import json

from wildshield.experiments.report import load_summaries, write_csv


def test_report_loads_summaries_and_writes_csv(tmp_path) -> None:
    record = {
        "trial_id": "one", "target_present": True, "condition": "bright",
        "motion": "stationary", "target_label": "cup", "total_frames": 10,
        "detection_frames": 8, "detection_rate": 0.8, "false_alarm": False,
        "missed_target": False, "first_detection_frame": 2, "notes": "",
    }
    (tmp_path / "one.summary.json").write_text(json.dumps(record))
    records = load_summaries(tmp_path)
    output = tmp_path / "results.csv"
    write_csv(records, output)
    with output.open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["trial_id"] == "one"
    assert rows[0]["detection_rate"] == "0.8"
