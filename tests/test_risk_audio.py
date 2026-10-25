from unittest.mock import patch
from wildshield.ui.alert import AudioEmitter, RiskAudioAlert
from wildshield.risk.evaluator import RiskLevel

def test_risk_audio_transitions_without_rgb_and_rearms():
    messages = []
    alert = RiskAudioAlert(messages.append)
    levels = [None, RiskLevel.SAFE, RiskLevel.WARNING, RiskLevel.WARNING, RiskLevel.DANGER, RiskLevel.DANGER, None, RiskLevel.DANGER]
    assert [alert.update(level) for level in levels] == [False, False, True, False, True, False, False, True]
    assert messages[0].startswith("THERMAL WARNING")
    assert messages[1].startswith("DANGER")

def test_uploaded_warning_and_danger_files_selected():
    with patch("wildshield.ui.alert.AudioEmitter._find_player", return_value="/usr/bin/afplay"):
        emitter = AudioEmitter()
    with patch("wildshield.ui.alert.subprocess.Popen") as play:
        emitter("THERMAL WARNING: simulated radar warning boundary reached")
        assert play.call_args.args[0][-1].endswith("risk_warning.aiff")
        emitter("DANGER: simulated radar danger boundary reached")
        assert play.call_args.args[0][-1].endswith("risk_danger.aiff")


def test_person_detection_uses_person_sound():
    with patch("wildshield.ui.alert.AudioEmitter._find_player", return_value="/usr/bin/afplay"):
        emitter = AudioEmitter()
    with patch("wildshield.ui.alert.subprocess.Popen") as play:
        emitter("ALERT: PERSON detected (95%)")
        assert play.call_args.args[0][-1].endswith("person_warning.aiff")
        emitter("ALERT: ANIMAL_MODEL detected (95%; teddy bear)")
        assert play.call_args.args[0][-1].endswith("animal_warning.aiff")
