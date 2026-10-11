from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_summaries(input_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(input_dir.rglob("*.summary.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "trial_id", "target_present", "condition", "motion", "target_label",
        "total_frames", "detection_frames", "detection_rate", "false_alarm",
        "missed_target", "first_detection_frame", "notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def write_chart(records: list[dict[str, Any]], path: Path) -> None:
    cache_dir = path.parent / ".matplotlib-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive: dict[str, list[float]] = defaultdict(list)
    negative: dict[str, list[float]] = defaultdict(list)
    for record in records:
        group = positive if record["target_present"] else negative
        group[str(record["condition"])].append(float(record["detection_rate"]))

    conditions = sorted(set(positive) | set(negative))
    positive_means = [sum(positive[c]) / len(positive[c]) if positive[c] else 0 for c in conditions]
    negative_means = [sum(negative[c]) / len(negative[c]) if negative[c] else 0 for c in conditions]
    positions = list(range(len(conditions)))
    width = 0.36

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar([x - width / 2 for x in positions], positive_means, width, label="Target present")
    axis.bar([x + width / 2 for x in positions], negative_means, width, label="Target absent")
    axis.set_title("WildShield RGB Detection by Lighting Condition")
    axis.set_ylabel("Fraction of frames with detection")
    axis.set_ylim(0, 1)
    axis.set_xticks(positions, conditions)
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate WildShield trial summaries")
    parser.add_argument("--input", type=Path, default=Path("outputs/trials"))
    parser.add_argument("--output", type=Path, default=Path("outputs/report"))
    return parser


def run(args: argparse.Namespace) -> int:
    records = load_summaries(args.input)
    if not records:
        print(f"No *.summary.json files found in {args.input}")
        return 1
    csv_path = args.output / "trial_results.csv"
    chart_path = args.output / "detection_by_condition.png"
    write_csv(records, csv_path)
    write_chart(records, chart_path)
    print(f"Trials: {len(records)}")
    print(f"CSV: {csv_path}")
    print(f"Chart: {chart_path}")
    return 0


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))
