from wildshield.main import build_parser


def test_rgb_parser_has_optional_duration() -> None:
    args = build_parser().parse_args(["--source", "0", "--classes", "dog"])
    assert args.duration_seconds is None


def test_rgb_parser_accepts_duration() -> None:
    args = build_parser().parse_args(["--duration-seconds", "10"])
    assert args.duration_seconds == 10.0
