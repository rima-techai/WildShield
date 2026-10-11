import json

from wildshield.io.logger import JsonlLogger


def test_jsonl_logger_writes_one_record_per_line(tmp_path) -> None:
    path = tmp_path / "nested" / "trial.jsonl"
    with JsonlLogger(path) as logger:
        logger.write({"frame": 1, "detections": []})
        logger.write({"frame": 2, "detections": [{"label": "dog"}]})

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records == [
        {"frame": 1, "detections": []},
        {"frame": 2, "detections": [{"label": "dog"}]},
    ]

