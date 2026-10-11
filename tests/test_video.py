from wildshield.io.video import normalize_source


def test_numeric_source_is_camera_index() -> None:
    assert normalize_source("0") == 0
    assert normalize_source(" 12 ") == 12


def test_non_numeric_source_is_preserved() -> None:
    assert normalize_source("video.mp4") == "video.mp4"
    assert normalize_source("rtsp://camera/1") == "rtsp://camera/1"
