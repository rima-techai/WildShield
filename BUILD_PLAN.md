# WildShieldAI Prototype Build Plan

## 1. Project objective

Build a tabletop prototype that detects an animal near a model roadway at night, estimates whether it is moving toward the road, assigns a risk level, and warns the operator.

The prototype demonstrates this pipeline:

`detect -> track -> fuse sensors -> predict path -> warn`

It is an educational experiment, not a vehicle safety system. It will not control steering, braking, or any real vehicle.

## 2. Definition of success

The final demonstration is successful when it can:

1. Detect a person or animal target in RGB and thermal imagery.
2. Accept radar distance and relative-velocity observations.
3. associate a radar observation with an image detection in a controlled scene.
4. classify the situation as `SAFE`, `WARNING`, or `DANGER`.
5. show the detection, measurements, and reason for the risk decision.
6. activate an audible alert for a danger event.
7. save trial results for comparison and graphing.

Target prototype metrics (adjust after pilot tests):

- At least 20 trials per sensor configuration and lighting condition.
- At least 80% target detection in the controlled nighttime test area.
- Fewer than 10% false alarms in controlled negative trials.
- Risk-state response within 500 ms on the chosen computer.
- Repeatable `SAFE -> WARNING -> DANGER` demonstration in at least 8 of 10 runs.

These are project goals, not safety claims.

## 3. Recommended system

### Compute

- Preferred: Jetson Orin Nano if one is available.
- Lower-cost option: Raspberry Pi 5, preferably 8 GB.
- Development option: laptop first, then move the program to embedded hardware.

### Sensors and outputs

- USB RGB camera for the baseline.
- FLIR Lepton module or another radiometric/USB thermal camera.
- TI mmWave evaluation board with documented point-cloud or target output.
- USB speaker, powered speaker, or GPIO buzzer.
- Tripod or rigid sensor mount.
- Toy vehicle, animal/person models, marked tabletop roadway, and measuring tape.

Do not purchase the radar until its computer interface is confirmed. The exact TI board determines its SDK, supported operating system, serial ports, firmware, and output format.

## 4. Software architecture

Keep each hardware component behind a replaceable interface so recorded and simulated data can be used during development.

```text
RGB camera ----> image source ----> detector ----> tracker ----+
Thermal camera -> image source ----> detector ----> tracker ----+--> fusion
Radar board ---> radar adapter ------------------------------->+       |
                                                                       v
                                                               risk engine
                                                                  |     |
                                                                  v     v
                                                             dashboard alert
                                                                  |
                                                                  v
                                                              trial logger
```

Suggested modules:

```text
wildshield/
  config.py
  main.py
  sensors/
    image_source.py
    rgb_camera.py
    thermal_camera.py
    radar_source.py
    simulated_radar.py
  perception/
    detector.py
    tracker.py
    fusion.py
  risk/
    geometry.py
    evaluator.py
  ui/
    dashboard.py
    alert.py
  logging/
    trial_logger.py
tests/
data/
  samples/
  trials/
configs/
```

Use a common observation structure containing timestamp, sensor, class, confidence, position or bounding box, range, velocity, and track ID. Timestamp every observation so fusion is explicit rather than based on whichever result arrived last.

## 5. Risk model for the prototype

Define a simple coordinate system:

- `x`: position across the road.
- `y`: distance along or away from the sensor.
- A configurable rectangle represents the vehicle path.

Use rules that can be explained during judging:

- `SAFE`: target is outside the warning boundary and not approaching it.
- `WARNING`: target is approaching the roadway or its estimated time to enter the path is below a configurable threshold.
- `DANGER`: target is already in the path or is predicted to enter it very soon.

Initial tabletop thresholds should be configuration values, not hard-coded logic. For example:

```yaml
warning_distance_m: 1.5
danger_distance_m: 0.5
warning_time_to_path_s: 3.0
danger_time_to_path_s: 1.5
minimum_detection_confidence: 0.40
```

Use real measured tabletop distances. Do not pretend scaled measurements prove full-road performance.

## 6. Milestones

### Milestone 0 — Freeze the setup (half day)

- Choose the compute device.
- Identify exact camera and radar models.
- Mark a fixed road, target path, sensor location, and lighting positions.
- Record the bill of materials and sensor documentation.

Done when the hardware choices and physical test dimensions are written down.

### Milestone 1 — RGB baseline (2–4 days)

- Set up Python, OpenCV, and a pretrained Ultralytics YOLO model.
- Read live video and recorded video through the same image-source interface.
- Draw boxes, labels, confidence, frame rate, and timestamp.
- Map YOLO classes into `deer`, `dog`, `cow`, `person`, and `other`.
- Save detections to CSV or JSON Lines.

Done when a recorded sample and live camera feed both produce annotated output and a machine-readable log.

### Milestone 2 — Tracking and simulated radar (2–3 days)

- Add a simple tracker and stable track IDs.
- Create a simulated radar source that replays timestamped range and velocity data.
- Implement the risk rules and three-state display.
- Add an audible alert with a mute switch and cooldown.

Done when a scripted tabletop scenario reliably changes `SAFE -> WARNING -> DANGER` without physical radar.

### Milestone 3 — Thermal integration (3–7 days)

- Capture raw thermal frames and preserve temperature/intensity information where supported.
- Normalize thermal imagery for display separately from detector input.
- Start with person/animal-vs-background detection; exact species is a stretch goal.
- If ordinary pretrained YOLO performs poorly, collect a small project-specific dataset and fine-tune a small model using transfer learning.
- Log thermal detections in the same format as RGB detections.

Done when RGB-only and thermal-only trials can be run using the same test script and metrics.

### Milestone 4 — Physical radar integration (4–10 days)

- Install the vendor SDK and confirm the board produces data before connecting it to the main program.
- Write an adapter for range, azimuth if available, radial velocity, signal strength, and timestamp.
- Calibrate radar coordinates to the camera/tabletop coordinates.
- Associate radar targets and image tracks using time, direction, and expected position gates.
- Display an explicit `unmatched` state when fusion is uncertain.

Done when a moving target produces repeatable range/velocity readings and is associated with the correct visual track in the controlled single-target scene.

### Milestone 5 — Experiment and evaluation (3–5 days)

Run the same scene under:

- RGB only: light and dark.
- Thermal only: light and dark.
- Thermal + radar: light and dark.

Include positive trials, stationary targets, targets moving away, empty-road negative trials, and at least one distractor. Randomize trial order where practical.

Record ground truth before reviewing predictions. Compute:

- Detection rate/recall.
- False alarms per trial.
- First detection distance.
- Warning lead time.
- Risk-class accuracy.
- Processing latency and frame rate.

Done when a script can summarize trial logs and generate the final tables and graphs.

### Milestone 6 — Final demonstration (2 days)

- Rehearse one normal run and one failure/uncertainty case.
- Display live sensor status so disconnected hardware is obvious.
- Keep recorded inputs as a backup demonstration.
- Add a limitations slide and explain that radar radial velocity is not complete 2D motion by itself.

Done when another person can run the demo from a one-page checklist.

## 7. Experiment data format

Each trial should store:

- Trial ID and timestamp.
- Sensor configuration.
- Light level or named lighting condition.
- Target type and true movement direction.
- Ground-truth distance markers and path-entry time.
- Detection class, confidence, and first-detection time.
- Radar range, angle, velocity, and signal strength when available.
- Predicted risk state and transition times.
- False alarm, missed detection, and operator notes.

Save raw recordings when storage permits, but keep derived CSV/JSON logs small enough to audit by hand.

## 8. Key risks and fallbacks

| Risk | First response | Demonstration fallback |
|---|---|---|
| Thermal model cannot identify species | Detect warm moving target or person/animal | State species recognition as future work |
| Radar SDK is difficult to integrate | Validate vendor output independently | Replay recorded or simulated radar data |
| Raspberry Pi inference is slow | Use a nano model and lower resolution | Run on laptop or Jetson |
| Sensor association is unreliable | Restrict initial demo to one target | Display sensor outputs side by side |
| Hardware delivery is late | Develop against video and simulators | Present recorded inputs with reproducible logs |
| Small models do not resemble training data | Use a person as a repeatable warm target | Test animal imagery separately and document limits |

## 9. Scope control

Required:

- RGB baseline.
- Thermal detection.
- Radar range/motion input or a clearly labeled replay if physical integration is blocked.
- Three risk states.
- Dashboard, alert, logging, and controlled experiment.

Stretch goals:

- Multi-object fusion.
- Fine-tuned thermal detector.
- Better trajectory filtering.
- Web dashboard.
- Enclosure and polished mounting.

Explicitly out of scope:

- Testing in live traffic.
- Automatic braking or steering.
- Claims of automotive reliability.
- Custom radar electronics.
- Perfect species recognition.

## 10. Immediate next actions

1. Inventory available compute hardware and sensors, including exact model numbers.
2. Decide the tabletop dimensions and mark the sensor, road, and target paths.
3. Collect three short RGB clips: target approaching, moving away, and no target.
4. Create the Python project skeleton and implement the RGB baseline.
5. In parallel with software work, confirm the thermal and radar connection requirements from their exact manuals.

The first software release should use RGB video plus simulated radar. That provides an end-to-end system early, after which each real sensor can replace its simulator without rewriting the risk engine or dashboard.
