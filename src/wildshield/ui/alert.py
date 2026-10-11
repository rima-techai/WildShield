from __future__ import annotations

import sys
import time
import platform
import shutil
import subprocess
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
        # The bell is audible when terminal sound is enabled; the text is the fallback.
        print(f"\a{message}", file=sys.stderr, flush=True)

    def update(self, detections: list[Detection]) -> bool:
        if not detections:
            return False
        now = self._clock()
        if self._last_alert_at is not None and now - self._last_alert_at < self.cooldown_seconds:
            return False

        best = max(detections, key=lambda item: item.confidence)
        self._emit(f"ALERT: {best.label.upper()} detected ({best.confidence:.0%})")
        self._last_alert_at = now
        return True


class AudioEmitter:
    """Play a short sound without blocking the video processing loop."""

    MACOS_DEFAULT = Path("/System/Library/Sounds/Ping.aiff")

    def __init__(self, sound_path: Path | None = None) -> None:
        self.sound_path = sound_path or self._default_sound()
        if not self.sound_path.is_file():
            raise ValueError(f"alert sound does not exist: {self.sound_path}")
        self.player = self._find_player()

    def _default_sound(self) -> Path:
        if platform.system() == "Darwin":
            return self.MACOS_DEFAULT
        raise ValueError("this platform needs --alert-sound pointing to a WAV file")

    def _find_player(self) -> str:
        candidates = ["afplay"] if platform.system() == "Darwin" else ["aplay", "paplay"]
        for candidate in candidates:
            executable = shutil.which(candidate)
            if executable:
                return executable
        raise ValueError("no supported audio player found (expected afplay, aplay, or paplay)")

    def __call__(self, _: str) -> None:
        subprocess.Popen(
            [self.player, str(self.sound_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class DangerTransitionGate:
    """Emit once when risk enters DANGER; re-arm after it leaves DANGER."""

    def __init__(self) -> None:
        self._was_danger = False

    def update(self, level: object | None) -> bool:
        is_danger = str(level) == "DANGER" if level is not None else False
        entered = is_danger and not self._was_danger
        self._was_danger = is_danger
        return entered
