import cv2
import numpy as np

from wildshield.io.video import VideoSource, normalize_source


def test_numeric_source_is_camera_index() -> None:
    assert normalize_source("0") == 0
    assert normalize_source(" 12 ") == 12


def test_non_numeric_source_is_preserved() -> None:
    assert normalize_source("video.mp4") == "video.mp4"
    assert normalize_source("rtsp://camera/1") == "rtsp://camera/1"


def test_video_source_reads_a_still_image(tmp_path) -> None:
    image_path = tmp_path / "sample.jpg"
    assert cv2.imwrite(str(image_path), np.zeros((12, 16, 3), dtype=np.uint8))

    source = VideoSource(str(image_path))
    try:
        ok, frame = source.read()
        assert ok
        assert frame.shape == (12, 16, 3)
        assert source.read()[0] is False
    finally:
        source.close()
