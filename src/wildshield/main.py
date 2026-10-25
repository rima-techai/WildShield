from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from wildshield.io.logger import JsonlLogger
from wildshield.experiments.trial import TrialMetadata, TrialRecorder
from wildshield.io.video import VideoSink, VideoSource
from wildshield.perception.detector import YoloDetector
from wildshield.perception.labels import parse_label_filter
from wildshield.perception.tracker import CentroidTracker
from wildshield.risk.evaluator import RiskEvaluator, RiskLevel
from wildshield.sensors.simulated_radar import SimulatedRadar
from wildshield.ui.alert import AudioEmitter, DetectionAlert
from wildshield.ui.overlay import draw_overlay, draw_risk_panel

IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WildShieldAI RGB animal-detection baseline")
    parser.add_argument("--source", default="0", help="Camera number, video path, image path, or stream URL")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model name or weights path")
    parser.add_argument(
        "--world-classes",
        default=None,
        help="Comma-separated YOLO-World class prompts, for example deer",
    )
    parser.add_argument("--confidence", type=float, default=0.4, help="Minimum detection confidence [0, 1]")
    parser.add_argument("--device", default=None, help="Inference device such as cpu, mps, or 0")
    parser.add_argument("--classes", default=None, help="Comma-separated project labels to display")
    parser.add_argument("--log", type=Path, default=Path("outputs/detections.jsonl"))
    parser.add_argument("--output", type=Path, default=None, help="Optional annotated MP4 path")
    parser.add_argument("--image-output", type=Path, default=None, help="Save an annotated still image")
    parser.add_argument("--no-display", action="store_true", help="Disable the preview window")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames")
    parser.add_argument("--duration-seconds", type=float, default=None, help="Stop after this many elapsed seconds")
    parser.add_argument("--alert", action="store_true", help="Alert when a filtered object is detected")
    parser.add_argument("--alert-audio", action="store_true", help="Play speaker audio instead of terminal text")
    parser.add_argument("--alert-sound", type=Path, default=None, help="Audio file used by --alert-audio")
    parser.add_argument("--alert-cooldown", type=float, default=2.0, help="Minimum seconds between alerts")
    parser.add_argument("--simulate-radar", action="store_true", help="Add clearly labeled simulated radar and risk")
    parser.add_argument("--sim-range", type=float, default=5.0, help="Simulated starting range in meters")
    parser.add_argument("--sim-speed", type=float, default=0.6, help="Simulated approach speed in m/s")
    parser.add_argument("--warning-range", type=float, default=3.0, help="Warning boundary in meters")
    parser.add_argument("--danger-range", type=float, default=1.0, help="Danger boundary in meters")
    parser.add_argument("--danger-alert-only", action="store_true", help="Alert only when simulated risk is DANGER")
    parser.add_argument("--trial-id", default=None, help="Experiment trial identifier")
    ground_truth = parser.add_mutually_exclusive_group()
    ground_truth.add_argument("--target-present", action="store_true", dest="target_present")
    ground_truth.add_argument("--target-absent", action="store_false", dest="target_present")
    parser.set_defaults(target_present=None)
    parser.add_argument("--condition", choices=("bright", "dim", "dark"), default="bright")
    parser.add_argument("--motion", choices=("approaching", "away", "stationary", "empty"), default="stationary")
    parser.add_argument("--trial-notes", default="")
    return parser


def run(args: argparse.Namespace) -> int:
    try:
        import cv2

        label_filter = parse_label_filter(args.classes)
        world_classes = None
        if args.world_classes is not None:
            world_classes = [name.strip() for name in args.world_classes.split(",") if name.strip()]
            if not world_classes:
                raise ValueError("--world-classes must contain at least one class name")
        if args.image_output is not None:
            image_source = Path(args.source)
            if image_source.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("--image-output requires --source to be a still image file")
            if args.image_output.suffix.lower() not in IMAGE_SUFFIXES:
                raise ValueError("--image-output must use an image extension such as .jpg or .png")
            if image_source.resolve() == args.image_output.resolve():
                raise ValueError("--image-output must not overwrite the source image")
        if args.trial_id and args.target_present is None:
            raise ValueError("a trial requires --target-present or --target-absent")
        if args.alert_cooldown < 0:
            raise ValueError("alert cooldown must be zero or greater")
        audio_emitter = AudioEmitter(args.alert_sound) if args.alert_audio else None
        radar = SimulatedRadar(args.sim_range, args.sim_speed) if args.simulate_radar else None
        risk_evaluator = RiskEvaluator(args.warning_range, args.danger_range) if args.simulate_radar else None
        detector = YoloDetector(
            args.model,
            confidence=args.confidence,
            device=args.device,
            world_classes=world_classes,
        )
        source = VideoSource(args.source)
    except (RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    sink: VideoSink | None = None
    logger = JsonlLogger(args.log)
    frame_number = 0
    started: float | None = None
    alert_enabled = args.alert or args.alert_audio
    alert = DetectionAlert(args.alert_cooldown, emit=audio_emitter) if alert_enabled else None
    tracker = CentroidTracker()
    trial = None
    if args.trial_id:
        target_label = next(iter(label_filter)) if label_filter and len(label_filter) == 1 else "any"
        trial = TrialRecorder(
            TrialMetadata(args.trial_id, args.target_present, args.condition, args.motion, target_label, args.trial_notes)
        )

    try:
        while (args.max_frames is None or frame_number < args.max_frames) and (args.duration_seconds is None or started is None or time.perf_counter() - started < args.duration_seconds):
            ok, frame = source.read()
            if not ok:
                break
            frame_number += 1
            detections, inference_ms = detector.detect(frame)
            if started is None:
                # Exclude one-time model warm-up from the observation duration.
                started = time.perf_counter()
            captured_at = datetime.now(timezone.utc)
            if label_filter is not None:
                detections = [item for item in detections if item.label in label_filter]
            detections = tracker.update(detections)

            radar_observation = None
            risk = None
            if radar is not None and detections:
                best = max(detections, key=lambda item: item.confidence)
                radar_observation = radar.observe(best, int(frame.shape[1]))
                risk = risk_evaluator.evaluate(radar_observation)

            should_alert = detections and (
                not args.danger_alert_only or (risk is not None and risk.level == RiskLevel.DANGER)
            )
            if alert is not None and should_alert:
                alert.update(detections)

            elapsed = max(time.perf_counter() - started, 1e-9)
            measured_fps = frame_number / elapsed
            annotated = draw_overlay(frame.copy(), detections, measured_fps, captured_at.astimezone())
            if radar_observation is not None and risk is not None:
                annotated = draw_risk_panel(annotated, radar_observation, risk)

            if args.image_output is not None and frame_number == 1:
                args.image_output.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(args.image_output), annotated):
                    raise RuntimeError(f"Could not save annotated image: {args.image_output}")

            if sink is None and args.output is not None:
                height, width = annotated.shape[:2]
                sink = VideoSink(args.output, source.fps, (width, height))
            if sink is not None:
                sink.write(annotated)

            logger.write(
                {
                    "schema_version": 1,
                    "source": str(args.source),
                    "frame": frame_number,
                    "captured_at": captured_at.isoformat(),
                    "width": int(frame.shape[1]),
                    "height": int(frame.shape[0]),
                    "inference_ms": round(inference_ms, 3),
                    "detections": [item.as_dict() for item in detections],
                    "radar": radar_observation.as_dict() if radar_observation else None,
                    "risk": (
                        {
                            "level": str(risk.level),
                            "reason": risk.reason,
                            "time_to_path_s": risk.time_to_path_s,
                        }
                        if risk
                        else None
                    ),
                }
            )
            if trial is not None:
                trial.observe(frame_number, detections, str(risk.level) if risk else None)

            if not args.no_display:
                cv2.imshow("WildShieldAI RGB Baseline", annotated)
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
    finally:
        source.close()
        logger.close()
        if sink is not None:
            sink.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if trial is not None:
        summary_path = args.log.with_suffix(".summary.json")
        trial.write(summary_path)
        print(f"Trial summary: {summary_path}")

    print(f"Processed {frame_number} frame(s). Log: {args.log}")
    if args.output is not None:
        print(f"Annotated video: {args.output}")
    if args.image_output is not None and frame_number:
        print(f"Annotated image: {args.image_output}")
    return 0 if frame_number else 1


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
