from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from wildshield.fusion import FusionDecision, evaluate_fusion

from wildshield.io.logger import JsonlLogger
from wildshield.io.video import VideoSink, VideoSource
from wildshield.perception.detector import YoloDetector
from wildshield.perception.labels import parse_label_filter
from wildshield.perception.thermal_detector import FRAME_HEIGHT, FRAME_WIDTH, ThermalTarget, WarmTargetDetector
from wildshield.risk.evaluator import RiskEvaluator, RiskLevel
from wildshield.sensors.simulated_radar import SimulatedRadar
from wildshield.sensors.thermal_view import annotate_frame, colorize_frame
from wildshield.types import BoundingBox, Detection
from wildshield.ui.alert import AudioEmitter, DangerTransitionGate, DetectionAlert
from wildshield.ui.overlay import draw_overlay, draw_risk_panel


DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480


def fusion_status(rgb_detections: list[Detection], thermal_confirmed: bool) -> str:
    if rgb_detections and thermal_confirmed:
        return "RGB + THERMAL TEMPORAL CONFIRMATION"
    if thermal_confirmed:
        return "THERMAL TARGET | RGB CLASS UNAVAILABLE"
    if rgb_detections:
        return "RGB TARGET | NO THERMAL CONFIRMATION"
    return "NO TARGET"


def unique_snapshot_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return path.with_name(f"{path.stem}-{stamp}{path.suffix}")


def compose_dashboard(
    rgb_image: np.ndarray,
    thermal_image: np.ndarray,
    status: str,
    trial_id: str | None = None,
    decision: FusionDecision | None = None,
) -> np.ndarray:
    rgb = cv2.resize(rgb_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)
    thermal = cv2.resize(thermal_image, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_NEAREST)
    body = np.hstack((rgb, thermal))
    footer = np.full((84, body.shape[1], 3), (20, 20, 20), dtype=np.uint8)
    display_status = decision.label if decision is not None else status
    simulated = " | SIMULATED RADAR" if decision is not None and decision.radar_simulated else ""
    color = (50, 220, 50) if display_status != "NO TARGET" else (220, 220, 220)
    trial = f" | TRIAL: {trial_id}" if trial_id else ""
    cv2.putText(footer, f"ADAPTIVE FUSION: {display_status}{simulated}{trial}", (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    cv2.putText(footer, "RGB: CONNECTED | THERMAL: CONNECTED | RADAR/RANGE: SIMULATED - NOT MEASURED", (18, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 190, 255), 2, cv2.LINE_AA)
    cv2.putText(footer, "Temporal prototype only - sensors are not spatially calibrated | Q/ESC quit | S snapshot", (18, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 190, 190), 1, cv2.LINE_AA)
    return np.vstack((body, footer))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="WildShieldAI RGB + thermal prototype dashboard")
    parser.add_argument("--source", default="0", help="RGB camera number or video source")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--classes", default="person,dog,cow,deer", help="RGB project labels")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--device", default=None)
    parser.add_argument("--refresh-rate", type=int, choices=(1, 2), default=2)
    parser.add_argument("--warm-threshold", type=float, default=3.0)
    parser.add_argument("--min-hot-pixels", type=int, default=4)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--sim-range", type=float, default=5.0)
    parser.add_argument("--sim-speed", type=float, default=0.6)
    parser.add_argument("--warning-range", type=float, default=3.0)
    parser.add_argument("--danger-range", type=float, default=1.0)
    parser.add_argument("--alert-audio", action="store_true")
    parser.add_argument("--alert-sound", type=Path, default=None)
    parser.add_argument("--alert-cooldown", type=float, default=3.0)
    parser.add_argument("--trial-id", default=None, help="identifier displayed and stored with each run")
    parser.add_argument("--log", type=Path, default=Path("outputs/fusion_detections.jsonl"))
    parser.add_argument("--snapshot", type=Path, default=Path("outputs/fusion_snapshot.png"))
    parser.add_argument("--video-output", type=Path, default=None, help="Optional dashboard MP4 path")
    parser.add_argument("--video-fps", type=float, default=1.0, help="Dashboard recording frame rate")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None, help="stop after this many elapsed seconds")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--save-final", action="store_true", help="save the final displayed frame when the run ends")
    return parser


def thermal_detection_as_visual(target: ThermalTarget, threshold: float) -> Detection:
    delta = max(0.0, target.peak_c - target.baseline_c)
    return Detection(
        label="other",
        model_label="warm_target",
        confidence=min(1.0, delta / max(threshold * 2.0, 0.1)),
        box=BoundingBox(target.x1, target.y1, target.x2, target.y2),
        class_id=-1,
        track_id=1,
    )


def run(args: argparse.Namespace) -> int:
    if os.environ.get("BLINKA_MCP2221") != "1":
        print("error: run with BLINKA_MCP2221=1", file=sys.stderr)
        return 2
    source = None
    logger = None
    try:
        import board
        import adafruit_mlx90640
        label_filter = parse_label_filter(args.classes)
        source = VideoSource(args.source)
        rgb_detector = YoloDetector(args.model, args.confidence, args.device)
        i2c = board.I2C()
        thermal_sensor = adafruit_mlx90640.MLX90640(i2c)
        rates = {1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ, 2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ}
        thermal_sensor.refresh_rate = rates[args.refresh_rate]
        thermal_detector = WarmTargetDetector(args.warm_threshold, args.min_hot_pixels, args.persistence)
        radar = SimulatedRadar(args.sim_range, args.sim_speed)
        risk_evaluator = RiskEvaluator(args.warning_range, args.danger_range)
        audio = AudioEmitter(args.alert_sound) if args.alert_audio else None
        alert = DetectionAlert(args.alert_cooldown, emit=audio) if audio is not None else None
        danger_gate = DangerTransitionGate()
        logger = JsonlLogger(args.log)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        if source is not None:
            source.close()
        print(f"error: could not initialize fusion dashboard: {exc}", file=sys.stderr)
        return 1

    temperatures = [0.0] * (FRAME_WIDTH * FRAME_HEIGHT)
    frame_number = 0
    failed_thermal_reads = 0
    started: float | None = None
    last_dashboard: np.ndarray | None = None
    video_sink: VideoSink | None = None
    print("Fusion dashboard started at 2 Hz. Click the dashboard before q/s; Control+C also stops it.")
    try:
        while (args.max_frames is None or frame_number < args.max_frames) and (args.duration_seconds is None or started is None or time.perf_counter() - started < args.duration_seconds):
            ok, rgb_frame = source.read()
            if not ok:
                break
            try:
                thermal_sensor.getFrame(temperatures)
            except ValueError:
                failed_thermal_reads += 1
                if failed_thermal_reads >= 20:
                    raise RuntimeError("too many incomplete thermal-frame reads")
                time.sleep(0.05)
                continue
            failed_thermal_reads = 0
            frame_number += 1
            rgb_detections, inference_ms = rgb_detector.detect(rgb_frame)
            if started is None:
                # Exclude one-time model warm-up after both sensors are available.
                started = time.perf_counter()
            captured_at = datetime.now(timezone.utc)
            if label_filter is not None:
                rgb_detections = [item for item in rgb_detections if item.label in label_filter]
            target, thermal_confirmed = thermal_detector.detect(temperatures)
            status = fusion_status(rgb_detections, thermal_confirmed)

            radar_observation = None
            risk = None
            thermal_visual_detection = None
            if target is not None and thermal_confirmed:
                thermal_visual_detection = thermal_detection_as_visual(target, args.warm_threshold)
                radar_observation = radar.observe(thermal_visual_detection, FRAME_WIDTH)
                risk = risk_evaluator.evaluate(radar_observation)
            else:
                radar.reset(1)
            decision = evaluate_fusion(rgb_detections, thermal_confirmed, radar_observation)
            entered_danger = danger_gate.update(risk.level if risk is not None else None)
            if alert is not None and entered_danger and thermal_visual_detection is not None:
                alert.update([thermal_visual_detection])

            fps = frame_number / max(time.monotonic() - started, 1e-9)
            rgb_annotated = draw_overlay(rgb_frame.copy(), rgb_detections, fps, captured_at.astimezone())
            thermal_image, low, high, hottest = colorize_frame(temperatures, 20)
            thermal_annotated = annotate_frame(thermal_image, temperatures, low, high, hottest, 20, fps, target, thermal_confirmed)
            if radar_observation is not None and risk is not None:
                thermal_annotated = draw_risk_panel(thermal_annotated, radar_observation, risk)
            last_dashboard = compose_dashboard(rgb_annotated, thermal_annotated, status, args.trial_id, decision)
            video_output = getattr(args, "video_output", None)
            if video_output is not None:
                if video_sink is None:
                    video_sink = VideoSink(video_output, getattr(args, "video_fps", 1.0), (last_dashboard.shape[1], last_dashboard.shape[0]))
                video_sink.write(last_dashboard)

            logger.write({
                "schema_version": 1,
                "mode": "rgb_thermal_temporal_prototype",
                "trial_id": args.trial_id,
                "frame": frame_number,
                "captured_at": captured_at.isoformat(),
                "fusion_status": status,
                "fusion_rule_version": 2,
                "fusion_decision": decision.as_dict(),
                "rgb_inference_ms": round(inference_ms, 3),
                "rgb_detections": [item.as_dict() for item in rgb_detections],
                "thermal_confirmed": thermal_confirmed,
                "thermal_detection": target.as_dict() if target is not None else None,
                "radar": radar_observation.as_dict() if radar_observation is not None else None,
                "risk": ({"level": str(risk.level), "reason": risk.reason, "time_to_path_s": risk.time_to_path_s} if risk is not None else None),
                "temperatures_c": [round(float(value), 2) for value in temperatures],
            })

            if not args.no_display:
                cv2.imshow("WildShieldAI RGB + Thermal", last_dashboard)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    path = unique_snapshot_path(args.snapshot)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(path), last_dashboard)
                    print(f"Snapshot saved: {path}")
    except KeyboardInterrupt:
        print("Stopping fusion dashboard.")
    except (OSError, RuntimeError) as exc:
        print(f"error: fusion dashboard stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        source.close()
        logger.close()
        if video_sink is not None:
            video_sink.close()
        if not args.no_display:
            cv2.destroyAllWindows()

    if (args.no_display or args.save_final) and last_dashboard is not None:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.snapshot), last_dashboard)
        print(f"Snapshot saved: {args.snapshot}")
    print(f"Processed {frame_number} fused frame(s). Log: {args.log}")
    if getattr(args, "video_output", None) is not None:
        print(f"Dashboard video: {args.video_output}")
    return 0 if frame_number else 1


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
