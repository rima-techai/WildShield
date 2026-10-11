from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a controlled WildShieldAI experiment trial")
    parser.add_argument("--mode", choices=("rgb", "thermal", "fusion"), required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--condition", choices=("bright", "dim", "dark"), required=True)
    presence = parser.add_mutually_exclusive_group(required=True)
    presence.add_argument("--target-present", action="store_true", dest="target_present")
    presence.add_argument("--target-absent", action="store_false", dest="target_present")
    parser.add_argument("--distance-m", type=float, required=True, help="physically measured target distance")
    parser.add_argument("--motion", choices=("approaching", "away", "stationary", "empty"), required=True)
    parser.add_argument("--duration-seconds", type=float, default=10.0)
    parser.add_argument("--notes", default="")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--classes", default="person")
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--device", default=None)
    parser.add_argument("--warm-threshold", type=float, default=3.0)
    parser.add_argument("--min-hot-pixels", type=int, default=4)
    parser.add_argument("--persistence", type=int, default=2)
    parser.add_argument("--alert-audio", action="store_true")
    parser.add_argument("--alert-sound", type=Path, default=None)
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/trials"))
    return parser


def validate(args: argparse.Namespace) -> None:
    if args.distance_m < 0:
        raise ValueError("distance must be zero or greater")
    if args.duration_seconds <= 0:
        raise ValueError("duration must be positive")
    if args.target_present and args.motion == "empty":
        raise ValueError("target-present trials cannot use motion=empty")
    if not args.target_present and args.motion != "empty":
        raise ValueError("target-absent trials must use motion=empty")
    if not args.target_present and args.distance_m != 0:
        raise ValueError("target-absent trials must use distance 0")


def trial_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base=args.output_dir/args.trial_id
    return base.with_suffix(".jsonl"), base.with_suffix(".summary.json"), base.with_suffix(".png")


def detection_for_row(mode: str, row: dict[str, Any]) -> bool:
    if mode == "rgb":
        return bool(row.get("detections"))
    if mode == "thermal":
        return bool(row.get("detected"))
    return row.get("fusion_status") == "RGB + THERMAL TEMPORAL CONFIRMATION"


def summarize(log_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    rows=[json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    detected=[detection_for_row(args.mode,row) for row in rows]
    first=next((row.get("frame") for row,yes in zip(rows,detected) if yes),None)
    timestamps=[datetime.fromisoformat(row["captured_at"]) for row in rows if row.get("captured_at")]
    elapsed=(timestamps[-1]-timestamps[0]).total_seconds() if len(timestamps)>1 else 0.0
    inference=[float(row.get("inference_ms",row.get("rgb_inference_ms",0.0))) for row in rows]
    fusion=Counter(row.get("fusion_status") for row in rows if row.get("fusion_status"))
    risks=Counter((row.get("risk") or {}).get("level") for row in rows if row.get("risk"))
    thermal_deltas=[]
    for row in rows:
        item=row.get("thermal_detection") or row.get("detection") if args.mode != "rgb" else None
        if isinstance(item,dict) and "peak_c" in item and "baseline_c" in item:
            thermal_deltas.append(float(item["peak_c"])-float(item["baseline_c"]))
    detection_count=sum(detected)
    return {
        "schema_version":1,
        "trial_id":args.trial_id,
        "mode":args.mode,
        "ground_truth":{
            "condition":args.condition,
            "target_present":args.target_present,
            "distance_m":args.distance_m,
            "motion":args.motion,
            "notes":args.notes,
        },
        "configuration":{
            "duration_seconds_requested":args.duration_seconds,
            "rgb_classes":args.classes if args.mode in ("rgb","fusion") else None,
            "rgb_confidence":args.confidence if args.mode in ("rgb","fusion") else None,
            "warm_threshold_c":args.warm_threshold if args.mode in ("thermal","fusion") else None,
            "simulated_radar_used":args.mode in ("thermal","fusion"),
        },
        "results":{
            "frames":len(rows),
            "elapsed_seconds":round(elapsed,3),
            "detected_frames":detection_count,
            "detection_rate":round(detection_count/len(rows),4) if rows else 0.0,
            "detected_any":detection_count>0,
            "first_detection_frame":first,
            "false_positive":not args.target_present and detection_count>0,
            "missed_target":args.target_present and detection_count==0,
            "median_inference_ms":round(statistics.median(inference),3) if inference else None,
            "fusion_status_counts":dict(fusion),
            "risk_state_counts":dict(risks),
            "median_thermal_delta_c":round(statistics.median(thermal_deltas),3) if thermal_deltas else None,
        },
        "files":{"log":str(log_path)},
    }


def runner_args(args: argparse.Namespace, log: Path, snapshot: Path) -> SimpleNamespace:
    common=dict(source=args.source,model=args.model,confidence=args.confidence,device=args.device,classes=args.classes,log=log,no_display=args.no_display,max_frames=None,duration_seconds=args.duration_seconds,alert_audio=args.alert_audio,alert_sound=args.alert_sound,alert_cooldown=3.0)
    if args.mode == "rgb":
        return SimpleNamespace(**common,output=None,alert=False,simulate_radar=False,sim_range=5.0,sim_speed=0.6,warning_range=3.0,danger_range=1.0,danger_alert_only=False,trial_id=None,target_present=None,condition=args.condition,motion=args.motion,trial_notes=args.notes)
    thermal=dict(refresh_rate=2,warm_threshold=args.warm_threshold,min_hot_pixels=args.min_hot_pixels,persistence=args.persistence,snapshot=snapshot,no_log=False,save_final=True,simulate_radar=True,sim_range=5.0,sim_speed=0.6,warning_range=3.0,danger_range=1.0)
    if args.mode == "thermal":
        return SimpleNamespace(scale=20,**thermal,**{k:v for k,v in common.items() if k not in ("source","model","confidence","device","classes")})
    return SimpleNamespace(**common,**thermal,trial_id=args.trial_id)


def run(args: argparse.Namespace) -> int:
    try:
        validate(args)
    except ValueError as exc:
        print(f"error: {exc}",file=sys.stderr)
        return 2
    log_path,summary_path,snapshot_path=trial_paths(args)
    video_path=snapshot_path.with_suffix(".mp4")
    if args.mode == "rgb":
        from wildshield.main import run as mode_run
    elif args.mode == "thermal":
        from wildshield.sensors.thermal_view import run as mode_run
    else:
        from wildshield.fusion_demo import run as mode_run
    mode_args=runner_args(args,log_path,snapshot_path)
    if args.mode == "fusion":
        mode_args.video_output=video_path
        mode_args.video_fps=1.0
    result=mode_run(mode_args)
    if not log_path.exists():
        return result
    summary=summarize(log_path,args)
    if snapshot_path.exists():
        summary["files"]["snapshot"]=str(snapshot_path)
    if video_path.exists():
        summary["files"]["video"]=str(video_path)
    summary_path.write_text(json.dumps(summary,indent=2)+"\n")
    print(f"Trial summary: {summary_path}")
    return result


def main() -> None:
    raise SystemExit(run(build_parser().parse_args()))


if __name__ == "__main__":
    main()
