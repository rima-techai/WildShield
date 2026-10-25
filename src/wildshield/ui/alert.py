from __future__ import annotations

import platform
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from wildshield.types import Detection


class DetectionAlert:
    """Rate-limited alert for frames containing a matching detection."""

    def __init__(
        self,
        cooldown_seconds: float = 2.0,
        emit: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("alert cooldown must be zero or greater")

        self.cooldown_seconds = cooldown_seconds
        self._emit = emit or self._terminal_alert
        self._clock = clock
        self._last_alert_at: float | None = None

    @staticmethod
    def _terminal_alert(message: str) -> None:
        """Print the alert and terminal bell."""
        print(f"\a{message}", file=sys.stderr, flush=True)

    def update(self, detections: list[Detection]) -> bool:
        """Emit an alert if detections exist and cooldown has elapsed."""
        if not detections:
            return False

        now = self._clock()

        if (
            self._last_alert_at is not None
            and now - self._last_alert_at < self.cooldown_seconds
        ):
            return False

        best = max(detections, key=lambda item: item.confidence)
        model_label = best.model_label.strip().lower()
        prompt_label = f"; {model_label}" if model_label != best.label else ""
        self._emit(
            f"ALERT: {best.label.upper()} detected "
            f"({best.confidence:.0%}{prompt_label})"
        )
        self._last_alert_at = now
        return True


class AudioEmitter:
    """Play category-specific warning sounds without blocking video."""

    MACOS_DEFAULT = Path("/System/Library/Sounds/Ping.aiff")
    ANIMAL_LABELS = frozenset(
        {
            "armadillo",
            "bear",
            "cat",
            "cow",
            "deer",
            "dog",
            "elephant",
            "fox",
            "goat",
            "horse",
            "lion",
            "moose",
            "rabbit",
            "sheep",
            "tiger",
            "wolf",
            "zebra",
        }
    )

    def __init__(self, sound_path: Path | None = None, on_play: Callable[[Path], None] | None = None) -> None:
        self.on_play = on_play
        self.sound_path = sound_path or self._default_sound()

        if not self.sound_path.is_file():
            raise ValueError(
                f"Alert sound does not exist: {self.sound_path}"
            )

        self.player = self._find_player()

        # Resolve the existing project sound directory:
        # project/assets/sound/
        # This assumes alert.py is in project/src/wildshield/ui/.
        project_root = Path(__file__).resolve().parents[3]
        sound_dir = project_root / "assets" / "sounds"

        self.sound_paths = {
            "animal": sound_dir / "animal_warning.aiff",
            "person": sound_dir / "person_warning.aiff",
            "risk_warning": sound_dir / "risk_warning.aiff",
            "risk_danger": sound_dir / "risk_danger.aiff",
            "danger": sound_dir / "danger_warning.aiff",
            "thermal": sound_dir / "thermal_warning.aiff",
        }

        for category, path in self.sound_paths.items():
            if not path.is_file():
                print(
                    f"Warning: {category} sound not found: {path}. "
                    "The default sound will be used.",
                    file=sys.stderr,
                )

    def _default_sound(self) -> Path:
        if platform.system() == "Darwin":
            return self.MACOS_DEFAULT

        raise ValueError(
            "Specify a default alert sound using --alert-sound."
        )

    def _find_player(self) -> str:
        candidates = (
            ["afplay"]
            if platform.system() == "Darwin"
            else ["paplay", "ffplay", "aplay"]
        )

        for candidate in candidates:
            executable = shutil.which(candidate)
            if executable:
                return executable

        raise ValueError(
            "No supported audio player found "
            "(expected afplay on macOS)"
        )

    def __call__(self, message: str) -> None:
        """Choose the warning sound based on the alert message."""
        text = message.lower()

        if "simulated radar" in text and "danger" in text:
            category = "risk_danger"
        elif "simulated radar" in text and "warning" in text:
            category = "risk_warning"
        elif re.search(r"\b(person|human)\b", text):
            category = "person"
        elif "thermal" in text or "temperature" in text or "warm_target" in text:
            category = "thermal"
        elif "danger" in text or "dangerous" in text:
            category = "danger"
        elif "animal" in text or any(
            re.search(rf"\b{re.escape(label)}\b", text)
            for label in self.ANIMAL_LABELS
        ):
            category = "animal"
        else:
            category = None

        sound = self.sound_paths.get(category, self.sound_path)

        if not sound.is_file():
            sound = self.sound_path

        if self.on_play is not None:
            self.on_play(sound)

        try:
            if Path(self.player).name == "ffplay":
                command = [
                    self.player,
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "error",
                    str(sound),
                ]
            else:
                command = [self.player, str(sound)]

            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            print(
                f"Unable to play alert sound: {exc}",
                file=sys.stderr,
            )


class DangerTransitionGate:
    """Emit once when risk enters DANGER; re-arm after it leaves DANGER."""

    def __init__(self) -> None:
        self._was_danger = False

    def update(self, level: object | None) -> bool:
        is_danger = (
            str(level).upper() == "DANGER"
            if level is not None
            else False
        )
        entered = is_danger and not self._was_danger
        self._was_danger = is_danger
        return entered

class RiskAudioAlert:
    """Play once on entry to WARNING/DANGER, independently of RGB detection."""

    def __init__(self, emit: Callable[[str], None]) -> None:
        self.emit = emit
        self.previous_level: str | None = None

    def update(self, level: object | None) -> bool:
        current = str(level).upper() if level is not None else None
        changed = current != self.previous_level
        self.previous_level = current
        if changed and current in {"WARNING", "DANGER"}:
            message = ("DANGER: simulated radar danger boundary reached"
                       if current == "DANGER"
                       else "THERMAL WARNING: simulated radar warning boundary reached")
            self.emit(message)
            return True
        return False
