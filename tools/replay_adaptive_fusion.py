"""Replay recorded trial logs through adaptive fusion rule version 2."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from wildshield.fusion import FusionLevel, evaluate_fusion
from wildshield.types import BoundingBox, Detection, RadarObservation


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "outputs" / "trials"
OUT = ROOT / "outputs" / "competition"


def detection(item: dict) -> Detection:
    box = item["box"]
    return Detection(item["label"], item["model_label"], item["confidence"], BoundingBox(box["x1"], box["y1"], box["x2"], box["y2"]), item["class_id"], item.get("track_id"))


def radar(item: dict | None) -> RadarObservation | None:
    return RadarObservation(**item) if item is not None else None


def main() -> None:
    manifest = json.loads((OUT / "animal_video_manifest.json").read_text())
    results = []
    decisions = defaultdict(int)
    for trial_id in manifest["accepted_trials"]:
        summary = json.loads((TRIALS / f"{trial_id}.summary.json").read_text())
        rows = [json.loads(line) for line in (TRIALS / f"{trial_id}.jsonl").read_text().splitlines() if line.strip()]
        adaptive = []
        for row in rows:
            decision = evaluate_fusion([detection(x) for x in row["rgb_detections"]], row["thermal_confirmed"], radar(row.get("radar")))
            adaptive.append(decision)
            decisions[decision.label] += 1
        legacy_frames = sum(row["fusion_status"] == "RGB + THERMAL TEMPORAL CONFIRMATION" for row in rows)
        actionable_frames = sum(d.level in (FusionLevel.WARNING, FusionLevel.CONFIRMED) for d in adaptive)
        results.append({
            "trial_id": trial_id,
            "lighting": summary["ground_truth"]["condition"],
            "target_present": summary["ground_truth"]["target_present"],
            "frames": len(rows),
            "legacy_confirmed_frames": legacy_frames,
            "legacy_rate": legacy_frames / len(rows),
            "adaptive_actionable_frames": actionable_frames,
            "adaptive_rate": actionable_frames / len(rows),
        })

    fields = list(results[0])
    with (OUT / "adaptive_fusion_replay.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    grouped = defaultdict(list)
    for row in results:
        grouped[(row["lighting"], row["target_present"])].append(row)
    lines = []
    for lighting in ("bright", "dim", "dark"):
        for present in (True, False):
            group = grouped[(lighting, present)]
            legacy = sum(r["legacy_confirmed_frames"] for r in group) / sum(r["frames"] for r in group)
            adaptive = sum(r["adaptive_actionable_frames"] for r in group) / sum(r["frames"] for r in group)
            lines.append(f"| {lighting.title()} | {'Present' if present else 'Absent'} | {100*legacy:.1f}% | {100*adaptive:.1f}% |")

    report = f"""# Adaptive Fusion Rule Retrospective Replay

## Status

This is a software replay of rule version 2 against the 18 accepted recorded trials. It is not a new physical experiment. The stored radar observations were simulated, so adaptive radar-supported warnings are hypothetical and must not be reported as measured radar performance.

## Rule

- RGB plus thermal: confirmed animal.
- Thermal plus approaching radar: probable animal warning.
- Thermal without approaching radar: monitor.
- RGB plus approaching radar: possible-object warning.
- RGB alone: monitor.
- No evidence: safe.

## Replay results

"Actionable" means either `CONFIRMED` or `WARNING`.

| Lighting | Target | Original AND-rule rate | Adaptive actionable rate |
|---|---|---:|---:|
{chr(10).join(lines)}

The adaptive rule restores dark-condition warnings because confirmed thermal targets are paired with the prototype's simulated approaching-radar observations. This demonstrates the intended software behavior only. A real radar sensor and a new preregistered validation experiment are required before claiming real sensor-fusion performance.

Decision counts across all recorded frames: `{dict(decisions)}`.
"""
    (OUT / "adaptive_fusion_replay.md").write_text(report)
    print(f"Replayed {len(results)} trials and {sum(r['frames'] for r in results)} frames")


if __name__ == "__main__":
    main()
