# Household-Object Experiment

## Setup

Use the laptop webcam, a coffee cup, tape, and the laptop speaker. Keep the camera fixed. Mark three positions on a desk or floor for `SAFE`, `WARNING`, and `DANGER`. The displayed radar values remain software simulation and must not be described as measured distance.

Keep each trial 10 seconds (about 300 frames at 30 FPS) and use the same camera angle. Change only the condition being tested.

## Minimum trial matrix

Run three repetitions of each condition:

| Lighting | Target | Motion | Repetitions |
|---|---|---|---:|
| Bright | Cup | Approaching | 3 |
| Dim | Cup | Approaching | 3 |
| Dark | Cup | Approaching | 3 |
| Bright | None | Empty | 3 |
| Dim | None | Empty | 3 |
| Dark | None | Empty | 3 |

That produces 18 trials. Do not stand in the frame during empty trials.

## Positive trial command

Change the trial ID and condition for each run:

```bash
uv run wildshield-rgb \
  --source 0 --classes cup --confidence 0.25 \
  --simulate-radar --danger-alert-only --alert-audio \
  --trial-id bright-cup-01 --target-present \
  --condition bright --motion approaching \
  --max-frames 300 \
  --log outputs/trials/bright-cup-01.jsonl
```

Hold the cup outside the frame, start the command, then move the cup steadily into view and toward the marked road.

## Empty trial command

```bash
uv run wildshield-rgb \
  --source 0 --classes cup --confidence 0.25 \
  --trial-id bright-empty-01 --target-absent \
  --condition bright --motion empty \
  --max-frames 300 \
  --log outputs/trials/bright-empty-01.jsonl
```

## Generate results

After completing the trials:

```bash
uv run wildshield-report --input outputs/trials --output outputs/report
```

This creates `trial_results.csv` and `detection_by_condition.png`. A good experiment also records unusual events in `--trial-notes`, preserves the JSONL logs, and does not delete failed trials.
