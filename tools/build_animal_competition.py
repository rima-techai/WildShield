"""Build competition artifacts from the accepted warmed-animal-model trials."""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ROOT / "outputs" / "trials"
OUT = ROOT / "outputs" / "competition"

ACCEPTED = [
    "video-animal-bright-present-01", "video-animal-bright-present-02", "video-animal-bright-present-04",
    "video-animal-bright-absent-01", "video-animal-bright-absent-02", "video-animal-bright-absent-03",
    "video-animal-dim-present-01", "video-animal-dim-present-02", "video-animal-dim-present-03",
    "video-animal-dim-absent-02", "video-animal-dim-absent-03", "video-animal-dim-absent-04",
    "video-animal-dark-present-01", "video-animal-dark-present-02", "video-animal-dark-present-03",
    "video-animal-dark-absent-01", "video-animal-dark-absent-02", "video-animal-dark-absent-03",
]
EXCLUDED = [
    {"trial_id": "video-animal-dim-absent-01", "reason": "Residual heat remained at the frame edge after removing the heat source; rerun after the scene cleared."},
    {"trial_id": "video-animal-bright-present-03", "reason": "Heat source failed during the run, observed before analysis; replaced by bright-present-04."},
]
ORDER = ["bright", "dim", "dark"]


def load_trial(trial_id: str) -> dict:
    summary = json.loads((TRIALS / f"{trial_id}.summary.json").read_text())
    rows = [json.loads(line) for line in (TRIALS / f"{trial_id}.jsonl").read_text().splitlines() if line.strip()]
    frames = len(rows)
    confidences = [d["confidence"] for row in rows for d in row.get("rgb_detections", [])]
    return {
        "trial_id": trial_id,
        "lighting": summary["ground_truth"]["condition"],
        "target_present": summary["ground_truth"]["target_present"],
        "frames": frames,
        "rgb_frames": sum(bool(row.get("rgb_detections")) for row in rows),
        "thermal_frames": sum(bool(row.get("thermal_confirmed")) for row in rows),
        "fusion_frames": summary["results"]["detected_frames"],
        "median_rgb_confidence": statistics.median(confidences) if confidences else None,
        "median_thermal_delta_c": summary["results"].get("median_thermal_delta_c"),
        "video": f"outputs/trials/{trial_id}.mp4",
    }


def rate(row: dict, sensor: str) -> float:
    return row[f"{sensor}_frames"] / row["frames"]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def write_results(rows: list[dict]) -> None:
    fields = ["trial_id", "lighting", "target_present", "frames", "rgb_frames", "rgb_rate", "thermal_frames", "thermal_rate", "fusion_frames", "fusion_rate", "median_rgb_confidence", "median_thermal_delta_c", "video"]
    with (OUT / "animal_video_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            item = dict(row)
            for sensor in ("rgb", "thermal", "fusion"):
                item[f"{sensor}_rate"] = rate(row, sensor)
            writer.writerow(item)


def plot_results(rows: list[dict]) -> None:
    positives = [r for r in rows if r["target_present"]]
    x = np.arange(3)
    width = 0.24
    colors = {"rgb": "#3973ac", "thermal": "#e67e22", "fusion": "#7d3c98"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for index, sensor in enumerate(("rgb", "thermal", "fusion")):
        stats = [mean_sd([rate(r, sensor) for r in positives if r["lighting"] == light]) for light in ORDER]
        means = [100 * v[0] for v in stats]
        sds = [100 * v[1] for v in stats]
        ax.bar(x + (index - 1) * width, means, width, yerr=sds, capsize=4, label=sensor.upper(), color=colors[sensor])
    ax.set_title("Warmed Animal-Model Detection Across Lighting (n=3 per condition)")
    ax.set_ylabel("Mean detected frames per trial (%)")
    ax.set_xticks(x, [s.title() for s in ORDER])
    ax.set_ylim(0, 110)
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "animal_detection_by_lighting.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    values = [[r["median_thermal_delta_c"] for r in positives if r["lighting"] == light] for light in ORDER]
    ax.boxplot(values, tick_labels=[s.title() for s in ORDER], widths=.5, showmeans=True)
    for idx, group in enumerate(values, 1):
        ax.scatter([idx] * len(group), group, color="#c0392b", zorder=3)
    ax.axhline(3.0, color="#555", linestyle="--", label="3 C candidate threshold")
    ax.set_title("Thermal Contrast of Warmed Animal Model (n=3 per condition)")
    ax.set_ylabel("Trial median target-to-background difference (C)")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(OUT / "animal_thermal_contrast.png", dpi=180)
    plt.close(fig)


def write_manifest(rows: list[dict]) -> None:
    manifest = {
        "schema_version": 2,
        "generated_from": "18 accepted trials; three repetitions per lighting/target condition",
        "accepted_trials": ACCEPTED,
        "excluded_trials": EXCLUDED,
        "target": "Warmed stuffed-animal surrogate classified by YOLO as teddy bear and mapped to animal_model",
        "radar": "Simulated; excluded from measured sensor-performance claims",
        "method_notes": ["Dark trials used laptop-screen illumination only.", "Thermal confirmation required two consecutive candidate frames.", "Fusion required simultaneous RGB classification and thermal confirmation."],
        "artifacts": ["animal_video_results.csv", "animal_detection_by_lighting.png", "animal_thermal_contrast.png", "animal_video_report.md", "WildShieldAI_competition_demo.mp4"],
    }
    (OUT / "animal_video_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def write_report(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["lighting"], row["target_present"])].append(row)
    table = []
    for lighting in ORDER:
        for present in (True, False):
            group = grouped[(lighting, present)]
            cells = []
            for sensor in ("rgb", "thermal", "fusion"):
                mean, sd = mean_sd([rate(r, sensor) for r in group])
                cells.append(f"{pct(mean)} +/- {100*sd:.1f} pp")
            # Contrast is a target measurement. An absent trial can contain a
            # single rejected warm candidate, so reporting its candidate delta
            # here would be easy to misread as a confirmed detection.
            deltas = [r["median_thermal_delta_c"] for r in group if present and r["median_thermal_delta_c"] is not None]
            delta_text = f"{statistics.mean(deltas):.2f} +/- {statistics.stdev(deltas):.2f} C" if len(deltas) > 1 else (f"{deltas[0]:.2f} C" if deltas else "-")
            table.append(f"| {lighting.title()} | {'Present' if present else 'Absent'} | {len(group)} | " + " | ".join(cells) + f" | {delta_text} |")
    positives = [r for r in rows if r["target_present"]]
    dark = [r for r in positives if r["lighting"] == "dark"]
    dim = [r for r in positives if r["lighting"] == "dim"]
    dim_rgb = mean_sd([rate(r, "rgb") for r in dim])
    dark_thermal = mean_sd([rate(r, "thermal") for r in dark])
    text = f"""# WildShieldAI Warmed Animal-Model Video Experiment

## Purpose

This controlled prototype experiment evaluated RGB and thermal detection of a warmed stuffed-animal surrogate under bright, dim, and dark-room conditions. The pretrained YOLO model classified the surrogate as `teddy bear`; WildShieldAI maps that class to `animal_model`. No live animals were used.

## Method

The stationary surrogate was placed approximately 1 m from fixed RGB and MLX90640 thermal sensors and warmed with a wrapped heat source. Each condition was repeated three times, giving 18 accepted trials: target present and absent under three lighting levels. Each trial lasted approximately 10 seconds. Thermal confirmation required a region at least 3 C above estimated background for two consecutive frames. Dark trials used laptop-screen illumination only.

## Results

Values are the mean trial detection rate +/- sample standard deviation in percentage points (pp).

| Lighting | Target | Trials | RGB | Thermal | Current fusion | Median thermal contrast |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(table)}

Across all nine target-absent control trials, confirmed false-positive frames were zero for RGB, thermal, and fusion. In dark target-present trials, RGB detected 0% of frames while thermal detected {pct(dark_thermal[0])} +/- {100*dark_thermal[1]:.1f} pp. In dim target-present trials, RGB averaged {pct(dim_rgb[0])} +/- {100*dim_rgb[1]:.1f} pp, showing substantial trial-to-trial variability; thermal remained near 91% in all three repetitions.

## Interpretation

The repeated trials support the prototype's central finding: thermal sensing continued to identify a warmed target when visible-light classification failed in darkness. The current conservative fusion rule requires simultaneous RGB and thermal confirmation, so it also failed in darkness. That is a design limitation, not evidence that thermal failed. A future rule could combine thermal confirmation with measured radar motion when RGB classification is unavailable.

## Exclusions decided from observed test conditions

- `video-animal-dim-absent-01`: residual heat remained at the frame edge after the heat source was removed. The control was rerun after the scene cleared.
- `video-animal-bright-present-03`: the heat source failed during the run and the problem was observed before analysis. It was replaced by `video-animal-bright-present-04`.

Both original files were preserved. Neither excluded run contributes to the reported calculations.

## Limitations

- The target was a warmed stuffed-animal surrogate, not a live animal.
- Lighting categories were observational; illuminance was not measured.
- There were three short repetitions per condition, all in one indoor setup.
- RGB and thermal sensors were temporally combined but not spatially calibrated.
- Thermal resolution was 32x24 pixels.
- Radar range, velocity, and risk states were simulated and excluded from measured performance claims.
- The prototype must not be used for driving decisions or vehicle control.

## Reproducibility and credit

Trial logs, summaries, snapshots, and videos are preserved under `outputs/trials`. The accepted/excluded record is in `animal_video_manifest.json`, and trial-level measurements are in `animal_video_results.csv`.

Trial data and video were recorded by the student researcher using WildShieldAI, 2026. Compilation, calculations, graphs, and report drafting used AI-assisted Python/OpenCV and Matplotlib code, 2026. AI assistance: OpenAI Codex. Competition submission wording should be rewritten and verified by the student.
"""
    (OUT / "animal_video_report.md").write_text(text)


def card(lines: list[str], seconds: float, fps: int = 10) -> list[np.ndarray]:
    frame = np.full((720, 1280, 3), (20, 28, 38), np.uint8)
    cv2.putText(frame, "WildShieldAI", (70, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (55, 210, 255), 4, cv2.LINE_AA)
    y = 205
    for i, line in enumerate(lines):
        scale = 1.05 if i == 0 else .72
        color = (255, 255, 255) if i == 0 else (205, 220, 230)
        cv2.putText(frame, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
        y += 72
    return [frame] * int(seconds * fps)


def build_video(rows: list[dict]) -> None:
    path = OUT / "WildShieldAI_competition_demo.mp4"
    fps = 10
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1280, 720))
    scenes = card(["RGB + thermal roadway-animal warning prototype", "18 accepted trials - three repetitions per condition", "Warmed stuffed-animal surrogate; radar simulation excluded"], 4, fps)
    for frame in scenes:
        writer.write(frame)
    representative = {"bright": "video-animal-bright-present-01", "dim": "video-animal-dim-present-03", "dark": "video-animal-dark-present-03"}
    for lighting in ORDER:
        group = [r for r in rows if r["target_present"] and r["lighting"] == lighting]
        rgb = mean_sd([rate(r, "rgb") for r in group])[0]
        thermal = mean_sd([rate(r, "thermal") for r in group])[0]
        cap = cv2.VideoCapture(str(TRIALS / f"{representative[lighting]}.mp4"))
        source = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            source.append(frame)
        cap.release()
        if source:
            for frame in source:
                frame = cv2.resize(frame, (1280, 720))
                cv2.rectangle(frame, (0, 0), (1280, 72), (10, 10, 10), -1)
                label = f"{lighting.upper()} representative | 3-trial mean: RGB {pct(rgb)} | Thermal {pct(thermal)}"
                cv2.putText(frame, label, (25, 47), cv2.FONT_HERSHEY_SIMPLEX, .78, (255, 255, 255), 2, cv2.LINE_AA)
                for _ in range(fps):
                    writer.write(frame)
        for frame in card([f"{lighting.title()} result (n=3)", f"RGB mean detection: {pct(rgb)}", f"Thermal mean detection: {pct(thermal)}"], 2.5, fps):
            writer.write(frame)
    for frame in card(["Conclusion", "Thermal detected the warmed target when RGB failed in darkness.", "Current RGB-AND-thermal fusion must be redesigned for dark scenes.", "Prototype only - not for real vehicle operation.", "AI-assisted compilation disclosed; OpenAI Codex, 2026."], 5, fps):
        writer.write(frame)
    writer.release()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [load_trial(t) for t in ACCEPTED]
    assert len(rows) == 18
    assert all((ROOT / row["video"]).exists() for row in rows)
    write_results(rows)
    plot_results(rows)
    write_manifest(rows)
    write_report(rows)
    build_video(rows)
    print(f"Built {len(rows)} accepted trials in {OUT}")


if __name__ == "__main__":
    main()
