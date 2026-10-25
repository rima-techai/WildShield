import pytest

from wildshield.perception.labels import parse_label_filter, project_label


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("deer", "deer"),
        ("elephant", "elephant"),
        ("armadillo", "armadillo"),
        ("nine-banded armadillo", "armadillo"),
        ("DOG", "dog"),
        ("cow", "cow"),
        ("person", "person"),
        ("cup", "cup"),
        ("teddy bear", "animal_model"),
        ("horse", "other"),
        ("traffic light", "other"),
    ],
)
def test_project_label(source: str, expected: str) -> None:
    assert project_label(source) == expected


def test_parse_label_filter() -> None:
    assert parse_label_filter("person, cup") == frozenset({"person", "cup"})
    assert parse_label_filter("armadillo") == frozenset({"armadillo"})
    assert parse_label_filter("animal_model") == frozenset({"animal_model"})
    assert parse_label_filter(None) is None


def test_parse_label_filter_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="Unknown project label"):
        parse_label_filter("moose")
