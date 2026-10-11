from pathlib import Path
from unittest.mock import patch

from wildshield.ui.alert import AudioEmitter


def test_audio_emitter_starts_nonblocking_player(tmp_path: Path) -> None:
    sound = tmp_path / "alert.wav"
    sound.write_bytes(b"test")
    with patch.object(AudioEmitter, "_find_player", return_value="/usr/bin/fake-player"):
        emitter = AudioEmitter(sound)
    with patch("wildshield.ui.alert.subprocess.Popen") as popen:
        emitter("ignored message")
    args = popen.call_args.args[0]
    assert args == ["/usr/bin/fake-player", str(sound)]
