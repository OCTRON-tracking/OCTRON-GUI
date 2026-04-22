"""
Tests for pure utility functions in octron/tools/train.py.
No GPU, no model weights, and no heavy dependencies required — only the
yolo_models.yaml config file that ships with the package.

Covered
-------
_normalise_model_name(model, models_yaml_path)
    Exact match returns the canonical key unchanged
    All four models in yolo_models.yaml resolve correctly
    Lowercase input matched case-insensitively → canonical casing returned
    All-uppercase input matched case-insensitively → canonical casing returned
    Mixed-case input matched case-insensitively → canonical casing returned
    Unknown model name not in YAML → input returned as-is
    Accepts enum-like objects with a .value attribute (e.g. typer enums)
"""

from pathlib import Path
from octron.tools.train import _normalise_model_name

MODELS_YAML = Path(__file__).parent.parent / "octron" / "yolo_octron" / "yolo_models.yaml"


# ---------------------------------------------------------------------------
# _normalise_model_name
# ---------------------------------------------------------------------------

def test_normalise_exact_match():
    assert _normalise_model_name("YOLO26m", MODELS_YAML) == "YOLO26m"


def test_normalise_exact_match_all_models():
    for name in ("YOLO11m", "YOLO11l", "YOLO26m", "YOLO26l"):
        assert _normalise_model_name(name, MODELS_YAML) == name


def test_normalise_lowercase_input():
    assert _normalise_model_name("yolo26m", MODELS_YAML) == "YOLO26m"


def test_normalise_uppercase_input():
    assert _normalise_model_name("YOLO11M", MODELS_YAML) == "YOLO11m"


def test_normalise_mixed_case_input():
    assert _normalise_model_name("Yolo26L", MODELS_YAML) == "YOLO26l"


def test_normalise_unknown_model_returns_input():
    assert _normalise_model_name("nonexistent_model", MODELS_YAML) == "nonexistent_model"


def test_normalise_accepts_enum_like_object():
    """Objects with a .value attribute (e.g. typer enums) should be handled."""
    class FakeEnum:
        value = "yolo26l"
    assert _normalise_model_name(FakeEnum(), MODELS_YAML) == "YOLO26l"
