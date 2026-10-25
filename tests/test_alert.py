from wildshield.types import BoundingBox, Detection
from wildshield.ui.alert import AudioEmitter, DetectionAlert

from pathlib import Path


def cup(confidence: float = 0.8) -> Detection:
    return Detection("cup", "cup", confidence, BoundingBox(1, 2, 3, 4), 41)


def test_alerts_on_detection_and_respects_cooldown() -> None:
    messages: list[str] = []
    times = iter([10.0, 11.0, 12.1])
    alert = DetectionAlert(2.0, emit=messages.append, clock=lambda: next(times))

    assert alert.update([cup()]) is True
    assert alert.update([cup()]) is False
    assert alert.update([cup(0.9)]) is True
    assert messages == ["ALERT: CUP detected (80%)", "ALERT: CUP detected (90%)"]


def test_does_not_alert_without_detection() -> None:
    alert = DetectionAlert(2.0, emit=lambda _: None)
    assert alert.update([]) is False


def test_alert_includes_open_vocabulary_model_label() -> None:
    messages: list[str] = []
    elephant = Detection("other", "elephant", 0.85, BoundingBox(1, 2, 3, 4), 0)
    alert = DetectionAlert(emit=messages.append)

    assert alert.update([elephant]) is True
    assert messages == ["ALERT: OTHER detected (85%; elephant)"]


def test_alert_selects_animal_sound_for_armadillo(
    tmp_path: Path, monkeypatch
) -> None:
    default = tmp_path / "default.aiff"
    animal = tmp_path / "animal_warning.aiff"
    default.touch()
    animal.touch()
    monkeypatch.setattr(AudioEmitter, "_find_player", lambda self: "afplay")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        "wildshield.ui.alert.subprocess.Popen",
        lambda command, **kwargs: commands.append(command),
    )
    emitter = AudioEmitter(sound_path=default)
    emitter.sound_paths = {"animal": animal}

    emitter("ALERT: ARMADILLO detected (85%)")

    assert Path(commands[0][-1]) == animal


def test_audio_emitter_selects_category_sounds(
    tmp_path: Path, monkeypatch
) -> None:
    default = tmp_path / "default.aiff"
    animal = tmp_path / "animal_warning.aiff"
    danger = tmp_path / "danger_warning.aiff"
    thermal = tmp_path / "thermal_warning.aiff"

    for path in (default, animal, danger, thermal):
        path.touch()

    # Avoid depending on the operating system audio player.
    monkeypatch. setattr(
        AudioEmitter, "_find_player", lambda self: "afplay"
    )

    commands: list[list[str]] = []

    monkeypatch. setattr(
        "wildshield.ui.alert.subprocess.Popen",
        lambda command, **kwargs: commands.append(command),
    )

    emitter = AudioEmitter(sound_path=default)
    emitter.sound_paths = {
        "animal": animal,
        "danger": danger,
        "thermal": thermal,
    }

    emitter("ALERT: DEER detected (92%)")
    emitter("ALERT: ELEPHANT detected (85%)")
    emitter("ALERT: ARMADILLO detected (85%)")
    emitter("ALERT: OTHER detected (85%; elephant)")
    emitter("ALERT: DANGER detected")
    emitter("ALERT: THERMAL detected")
    emitter("ALERT: OTHER detected (50%; warm_target)")

    assert [Path(command[-1]).name for command in commands] == [
        "animal_warning.aiff",
        "animal_warning.aiff",
        "animal_warning.aiff",
        "animal_warning.aiff",
        "danger_warning.aiff",
        "thermal_warning.aiff",
        "thermal_warning.aiff",
    ]