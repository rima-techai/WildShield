from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from wildshield.io.logger import JsonlLogger
from wildshield.perception.thermal_detector import FRAME_HEIGHT, FRAME_WIDTH, ThermalTarget, WarmTargetDetector
from wildshield.risk.evaluator import RiskEvaluator, RiskLevel
from wildshield.sensors.simulated_radar import SimulatedRadar
from wildshield.types import BoundingBox, Detection
from wildshield.ui.alert import AudioEmitter, DetectionAlert
from wildshield.ui.overlay import draw_risk_panel


def colorize_frame(
    temperatures: list[float] | np.ndarray,
    scale: int = 20,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> tuple[np.ndarray, float, float, tuple[int, int]]:
    values = np.asarray(temperatures, dtype=np.float32)
    if values.size != FRAME_WIDTH * FRAME_HEIGHT:
        raise ValueError(f"expected 768 temperatures, got {values.size}")
    if scale < 1:
        raise ValueError("scale must be at least 1")
    thermal = values.reshape((FRAME_HEIGHT, FRAME_WIDTH))
    low = float(np.percentile(thermal, low_percentile))
    high = float(np.percentile(thermal, high_percentile))
    if high <= low:
        high = low + 0.1
    normalized = np.clip((thermal - low) / (high - low), 0.0, 1.0)
    gray = np.round(normalized * 255).astype(np.uint8)
    enlarged = cv2.resize(gray, (FRAME_WIDTH * scale, FRAME_HEIGHT * scale), interpolation=cv2.INTER_CUBIC)
    colored = cv2.applyColorMap(enlarged, cv2.COLORMAP_INFERNO)
    hot_y, hot_x = divmod(int(np.argmax(thermal)), FRAME_WIDTH)
    return colored, low, high, (hot_x, hot_y)


def annotate_frame(
    image: np.ndarray,
    temperatures: list[float] | np.ndarray,
    low: float,
    high: float,
    hot_position: tuple[int, int],
    scale: int,
    fps: float,
    target: ThermalTarget | None = None,
    confirmed: bool = False,
) -> np.ndarray:
    values = np.asarray(temperatures, dtype=np.float32).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    hot_x, hot_y = hot_position
    hottest = float(values[hot_y, hot_x])
    center = float(values[FRAME_HEIGHT // 2, FRAME_WIDTH // 2])
    marker = (int((hot_x + 0.5) * scale), int((hot_y + 0.5) * scale))
    cv2.drawMarker(image, marker, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)

    if target is not None and confirmed:
        p1 = (target.x1 * scale, target.y1 * scale)
        p2 = (target.x2 * scale - 1, target.y2 * scale - 1)
        cv2.rectangle(image, p1, p2, (0, 255, 0), 2)
        label_x = max(8, min(p1[0], image.shape[1] - 230))
        label_y = min(image.shape[0] - 52, max(86, p1[1] + 24))
        cv2.putText(image, f"WARM TARGET {target.peak_c:.1f} C", (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 2, cv2.LINE_AA)

    status = "DETECTED" if confirmed else "SCANNING"
    status_color = (0, 255, 0) if confirmed else (255, 255, 255)
    cv2.putText(image, f"{status} | HOT {hottest:.1f} C", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(image, f"CENTER {center:.1f} C", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"DISPLAY {low:.1f}-{high:.1f} C", (12, image.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"{fps:.1f} FPS | CLICK WINDOW + Q TO QUIT | S TO SAVE", (12, image.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Live MLX90640 thermal heatmap and warm-target detector")
    parser.add_argument("--scale", type=int, default=20, help="display pixels per thermal sensor pixel")
    parser.add_argument("--refresh-rate", type=int, choices=(1, 2, 4), default=2, help="sensor refresh rate in Hz")
    parser.add_argument("--warm-threshold", type=float, default=3.0, help="degrees C above background baseline")
    parser.add_argument("--min-hot-pixels", type=int, default=4, help="minimum connected warm pixels")
    parser.add_argument("--persistence", type=int, default=2, help="consecutive frames required for detection")
    parser.add_argument("--snapshot", type=Path, default=Path("outputs/thermal_snapshot.png"))
    parser.add_argument("--log", type=Path, default=Path("outputs/thermal_detections.jsonl"))
    parser.add_argument("--no-log", action="store_true", help="disable thermal JSONL logging")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after this many successful frames")
    parser.add_argument("--duration-seconds", type=float, default=None, help="stop after this many elapsed seconds")
    parser.add_argument("--no-display", action="store_true", help="read frames without opening a window")
    parser.add_argument("--save-final", action="store_true", help="save the final displayed frame when the run ends")
    parser.add_argument("--simulate-radar", action="store_true", help="add clearly labeled simulated range and risk")
    parser.add_argument("--sim-range", type=float, default=5.0, help="simulated starting range in meters")
    parser.add_argument("--sim-speed", type=float, default=0.6, help="simulated approach speed in m/s")
    parser.add_argument("--warning-range", type=float, default=3.0, help="warning boundary in meters")
    parser.add_argument("--danger-range", type=float, default=1.0, help="danger boundary in meters")
    parser.add_argument("--alert-audio", action="store_true", help="play audio when a warm target is confirmed")
    parser.add_argument("--alert-sound", type=Path, default=None, help="custom audio file")
    parser.add_argument("--alert-cooldown", type=float, default=3.0, help="seconds between danger alerts")
    return parser


def run(args: argparse.Namespace) -> int:
    if os.environ.get("BLINKA_MCP2221") != "1":
        print("error: run with BLINKA_MCP2221=1", file=sys.stderr)
        return 2
    if args.scale < 1:
        print("error: --scale must be at least 1", file=sys.stderr)
        return 2
    try:
        import board
        import adafruit_mlx90640
        i2c = board.I2C()
        sensor = adafruit_mlx90640.MLX90640(i2c)
        rates = {1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ, 2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ, 4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ}
        sensor.refresh_rate = rates[args.refresh_rate]
        detector = WarmTargetDetector(args.warm_threshold, args.min_hot_pixels, args.persistence)
        radar = SimulatedRadar(args.sim_range, args.sim_speed) if args.simulate_radar else None
        risk_evaluator = RiskEvaluator(args.warning_range, args.danger_range) if radar is not None else None
        audio = AudioEmitter(args.alert_sound) if args.alert_audio else None
        alert = DetectionAlert(args.alert_cooldown, emit=audio) if audio is not None else None
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"error: could not initialize thermal system: {exc}", file=sys.stderr)
        return 1

    temperatures = [0.0] * (FRAME_WIDTH * FRAME_HEIGHT)
    frames = 0
    failed_reads = 0
    started: float | None = None
    last_image: np.ndarray | None = None
    logger = None if args.no_log else JsonlLogger(args.log)
    print(f"Thermal detection started at {args.refresh_rate} Hz. Click the image window before using q or s; Control+C also stops it.")
    try:
        while (args.max_frames is None or frames < args.max_frames) and (args.duration_seconds is None or started is None or time.monotonic() - started < args.duration_seconds):
            try:
                sensor.getFrame(temperatures)
            except ValueError:
                failed_reads += 1
                if failed_reads >= 20:
                    raise RuntimeError("too many incomplete thermal-frame reads")
                time.sleep(0.05)
                continue
            failed_reads = 0
            frames += 1
            captured_at = datetime.now(timezone.utc)
            if started is None:
                # Begin timing once the sensor has delivered its first usable frame.
                started = time.monotonic()
            candidate, confirmed = detector.detect(temperatures)
            synthetic_detection = None
            radar_observation = None
            risk = None
            if candidate is not None and confirmed:
                temperature_delta = max(0.0, candidate.peak_c - candidate.baseline_c)
                confidence = min(1.0, temperature_delta / max(args.warm_threshold * 2.0, 0.1))
                synthetic_detection = Detection(
                    label="other",
                    model_label="warm_target",
                    confidence=confidence,
                    box=BoundingBox(candidate.x1, candidate.y1, candidate.x2, candidate.y2),
                    class_id=-1,
                    track_id=1,
                )
                if radar is not None and risk_evaluator is not None:
                    radar_observation = radar.observe(synthetic_detection, FRAME_WIDTH)
                    risk = risk_evaluator.evaluate(radar_observation)
            elif radar is not None:
                radar.reset(1)
            if alert is not None and confirmed and synthetic_detection is not None:
                alert.update([synthetic_detection])

            fps = frames / max(time.monotonic() - started, 1e-9)
            image, low, high, hottest = colorize_frame(temperatures, args.scale)
            last_image = annotate_frame(image, temperatures, low, high, hottest, args.scale, fps, candidate, confirmed)
            if radar_observation is not None and risk is not None:
                last_image = draw_risk_panel(last_image, radar_observation, risk)
            if logger is not None:
                logger.write({
                    "schema_version": 1,
                    "sensor": "mlx90640",
                    "frame": frames,
                    "captured_at": captured_at.isoformat(),
                    "width": FRAME_WIDTH,
                    "height": FRAME_HEIGHT,
                    "detected": confirmed,
                    "detection": candidate.as_dict() if candidate is not None else None,
                    "radar": radar_observation.as_dict() if radar_observation is not None else None,
                    "risk": ({"level": str(risk.level), "reason": risk.reason, "time_to_path_s": risk.time_to_path_s} if risk is not None else None),
                    "temperatures_c": [round(float(value), 2) for value in temperatures],
                })
            if not args.no_display:
                cv2.imshow("WildShieldAI Thermal", last_image)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("s"), ord("S")):
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(args.snapshot), last_image)
                    print(f"Snapshot saved: {args.snapshot}")
    except KeyboardInterrupt:
        print("Stopping thermal stream.")
    except (OSError, RuntimeError) as exc:
        print(f"error: thermal stream stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        if logger is not None:
            logger.close()
        if not args.no_display:
            cv2.destroyAllWindows()
    if last_image is not None and (args.no_display or args.save_final):
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.snapshot), last_image)
        print(f"Snapshot saved: {args.snapshot}")
    print(f"Processed {frames} thermal frame(s)." + (f" Log: {args.log}" if logger is not None else ""))
    return 0 if frames else 1


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
