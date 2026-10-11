# WildShieldAI

WildShieldAI is an educational prototype for detecting potential animal hazards near a model roadway. This first milestone provides RGB-camera detection using a pretrained YOLO model.

It is not a vehicle safety system. Do not use it to make driving decisions or control a vehicle.

## RGB baseline

The program supports:

- A webcam (`--source 0`) or video/image file.
- YOLO detection with project labels: `deer`, `dog`, `cow`, `person`, `cup`, `animal_model`, and `other`. The controlled `animal_model` label maps YOLO's `teddy bear` class to the stuffed-animal surrogate used in experiments.
- Bounding boxes, confidence, FPS, and timestamps.
- JSON Lines logs for later experiments.
- Optional annotated video output.
- Headless operation for prerecorded experiments.

## Setup

Python 3.12 is recommended.

```bash
uv sync --extra dev
```

The first real run downloads the small pretrained `yolo11n.pt` weights if they are not already cached.

## Run

Webcam with a display:

```bash
uv run wildshield-rgb --source 0
```

For the first physical test, point the webcam at a coffee cup and filter the display to cups only:

```bash
uv run wildshield-rgb --source 0 --classes cup --confidence 0.25 --alert
```

Move the cup around the frame, partly hide it, and vary the room lighting. A green `CUP` box should follow it, while detections are saved to `outputs/detections.jsonl`.

`--alert` prints an alert and sends the terminal bell when a matching object is detected. It alerts at most once every two seconds by default. Change that interval with `--alert-cooldown 5`. Terminal sound must be enabled to hear the bell; alert text always appears.

For speaker audio without terminal alert messages on macOS, use:

```bash
uv run wildshield-rgb --source 0 --classes cup --confidence 0.25 --alert-audio
```

macOS uses its built-in Ping sound. To select another audio file, add `--alert-sound /absolute/path/to/sound.wav`. On Raspberry Pi/Linux, install `aplay` or `paplay` and provide a WAV file with `--alert-sound`.

## No-hardware simulated radar demonstration

This mode tracks a detected cup and supplies a clearly labeled simulated range and approach speed to the risk engine:

```bash
uv run wildshield-rgb \
  --source 0 \
  --classes cup \
  --confidence 0.25 \
  --simulate-radar \
  --sim-range 5 \
  --sim-speed 0.6 \
  --warning-range 3 \
  --danger-range 1 \
  --alert-audio \
  --danger-alert-only
```

Keep a cup visible. The simulated target starts at 5 m and approaches at 0.6 m/s, producing `SAFE`, `WARNING`, and `DANGER` states. The speaker sounds only at `DANGER`. The dashboard and logs explicitly mark the radar data as simulated; these values are not measurements from the camera.

Video file with saved annotations and logs:

```bash
uv run wildshield-rgb \
  --source data/samples/night_test.mp4 \
  --output outputs/night_test_annotated.mp4 \
  --log outputs/night_test.jsonl
```

Headless experiment:

```bash
uv run wildshield-rgb --source input.mp4 --no-display --log outputs/trial.jsonl
```

Useful options:

```text
--model yolo11n.pt       model name or local weights path
--confidence 0.40        minimum confidence
--device cpu             inference device (cpu, mps, 0, ...)
--classes person,deer    show only selected project labels
--max-frames 300         stop after a fixed number of frames
```

Press `q` or Escape to stop the display. Each JSONL record includes the source, frame number, timestamp, image size, inference time, and detections.

## Test

```bash
uv run pytest
```

## Thermal hardware check (MCP2221A + MLX90640)

Connect the MLX90640 to the MCP2221A with the STEMMA QT cable, then connect the MCP2221A to the Mac with a USB data cable. Install the optional dependencies:

```bash
uv sync --extra thermal --extra dev
```

Run the hardware check with MCP2221 mode enabled:

```bash
BLINKA_MCP2221=1 uv run wildshield-thermal-check
```

A successful check finds I2C address `0x33`, reads all 768 temperatures in the 32 by 24 frame, and prints the minimum, center, and maximum temperatures.

## Live thermal heatmap

After the hardware check succeeds, open the live thermal viewer:

```bash
BLINKA_MCP2221=1 uv run wildshield-thermal
```

The viewer enlarges the 32 by 24 sensor frame, applies an inferno color map, marks the hottest pixel, and shows the hottest, center, and display-range temperatures. Press `s` to save `outputs/thermal_snapshot.png`; press `q` or Escape to quit.

For a five-frame headless hardware test that saves the last frame:

```bash
BLINKA_MCP2221=1 uv run wildshield-thermal --no-display --max-frames 5
```

The viewer also detects the largest connected warm region after it persists for two frames, draws a green `WARM TARGET` box, and logs detections plus raw temperatures to `outputs/thermal_detections.jsonl`.

Useful tuning options:

```text
--warm-threshold 3.0  degrees C above the estimated background
--min-hot-pixels 4    smallest connected region to accept
--persistence 2       consecutive frames required before confirmation
--no-log              disable the thermal JSONL log
```

Use the default 2 Hz with the MCP2221A. The 4 Hz mode may produce I2C retry failures on this USB bridge.

### Thermal + simulated radar danger demonstration

Keep a warm target in view while the clearly labeled simulated range approaches the configured boundaries:

```bash
BLINKA_MCP2221=1 uv run wildshield-thermal \
  --simulate-radar --sim-range 5 --sim-speed 0.6 \
  --warning-range 3 --danger-range 1 \
  --alert-audio
```

The simulated range starts only after a persistent warm target is confirmed. If the target leaves the thermal frame, the simulated approach resets. Audio plays only when the simulated risk reaches `DANGER`. These values are not physical distance measurements.

## Combined RGB + thermal dashboard

Point the Mac webcam and MLX90640 toward the same controlled scene, then run:

```bash
BLINKA_MCP2221=1 uv run wildshield-fusion --alert-audio --trial-id demo-01
```

The dashboard shows RGB and thermal views side by side, labels whether both sensors detected a target during the same processing cycle, displays sensor status and the trial ID, and uses the thermal target to drive the clearly labeled simulated radar and risk states. The audio alert plays once when the risk enters `DANGER` and re-arms only after the danger condition clears. This is temporal confirmation only; the sensors are not spatially calibrated and the simulated range is not a measurement. Press `s` repeatedly to save uniquely timestamped dashboard snapshots.

The dashboard now uses adaptive fusion rule version 2. RGB plus thermal confirms an animal; thermal plus approaching radar produces a probable-animal warning; a single sensor without approaching radar produces a monitor state. Every JSONL row stores both the original raw `fusion_status` and the versioned `fusion_decision`. A decision based on prototype radar data is explicitly marked `radar_simulated: true` and must not be presented as measured radar performance.

Replay the accepted recorded experiments through the adaptive rule without changing the original evidence:

```bash
uv run python tools/replay_adaptive_fusion.py
```

The replay writes `outputs/competition/adaptive_fusion_replay.md` and `adaptive_fusion_replay.csv`. This is a retrospective software analysis, not a new physical experiment.

For a person-only RGB demonstration, keep the default `--classes person,dog,cow,deer`. Use `--classes person` to remove other supported labels.

## Unified controlled experiment command

Use `wildshield-trial` for official practice and experiment runs. It requires ground-truth metadata and produces a JSONL log, summary JSON, and (for thermal/fusion) final snapshot with the same trial ID.

```bash
BLINKA_MCP2221=1 uv run wildshield-trial \
  --mode fusion \
  --trial-id fusion-bright-present-01 \
  --condition bright \
  --target-present \
  --distance-m 3 \
  --motion stationary \
  --duration-seconds 10 \
  --alert-audio
```

For an empty trial, use `--target-absent --distance-m 0 --motion empty`. Valid modes are `rgb`, `thermal`, and `fusion`. The distance is physical ground truth entered by the operator; simulated radar values remain clearly separate.

## Known baseline limitation

The pretrained COCO model recognizes person, dog, cow, and several other animals, but **does not contain a deer class**. The software correctly maps a model class named `deer` if a future fine-tuned model supplies one; with the default weights, deer may be missed or mislabeled. This limitation should be reported rather than hidden.
