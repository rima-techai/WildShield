from __future__ import annotations


PROJECT_LABELS = frozenset({"deer", "dog", "cow", "person", "cup", "animal_model", "other", "elephant", "armadillo"})


def project_label(model_label: str) -> str:
    """Map a model label into the controlled labels used by the prototype."""
    normalized = model_label.strip().lower()
    if normalized == "teddy bear":
        return "animal_model"
    if normalized == "nine-banded armadillo":
        return "armadillo"
    if normalized in {"deer", "dog", "cow", "person", "cup", "elephant", "armadillo"}:
        return normalized
    return "other"


def parse_label_filter(raw: str | None) -> frozenset[str] | None:
    if raw is None:
        return None
    labels = frozenset(part.strip().lower() for part in raw.split(",") if part.strip())
    invalid = labels - PROJECT_LABELS
    if invalid:
        choices = ", ".join(sorted(PROJECT_LABELS))
        raise ValueError(f"Unknown project label(s): {', '.join(sorted(invalid))}. Choose from {choices}.")
    return labels
