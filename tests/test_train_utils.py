"""
Tests for model-name resolution.

Resolution now lives in core ``YOLO_octron.resolve_model_name`` (case-insensitive
match against the model catalog), shared by the CLI and the GUI, replacing the
old CLI-only ``_normalise_model_name``.

The catalog keys are read from the ``yolo_models.yaml`` that ships with the
package; a ``YOLO_octron`` instance is built via ``__new__`` (bypassing
``__init__``) so no weights are downloaded or loaded.
"""

from pathlib import Path

import yaml

from octron.yolo_octron.yolo_octron import YOLO_octron

MODELS_YAML = Path(__file__).parent.parent / "octron" / "yolo_octron" / "yolo_models.yaml"


def _resolver():
    """A YOLO_octron (no __init__) whose models_dict keys are the catalog names."""
    with open(MODELS_YAML) as f:
        keys = list((yaml.safe_load(f) or {}).keys())
    obj = YOLO_octron.__new__(YOLO_octron)
    obj.models_dict = {k: {} for k in keys}
    return obj, keys


# ---------------------------------------------------------------------------
# resolve_model_name
# ---------------------------------------------------------------------------

def test_resolve_exact_match():
    obj, _ = _resolver()
    assert obj.resolve_model_name("YOLO26m") == "YOLO26m"


def test_resolve_exact_match_all_models():
    obj, keys = _resolver()
    for name in keys:
        assert obj.resolve_model_name(name) == name


def test_resolve_lowercase_input():
    obj, _ = _resolver()
    assert obj.resolve_model_name("yolo26m") == "YOLO26m"


def test_resolve_uppercase_input():
    obj, _ = _resolver()
    assert obj.resolve_model_name("YOLO11M") == "YOLO11m"


def test_resolve_mixed_case_input():
    obj, _ = _resolver()
    assert obj.resolve_model_name("Yolo26L") == "YOLO26l"


def test_resolve_unknown_model_returns_none():
    obj, _ = _resolver()
    assert obj.resolve_model_name("nonexistent_model") is None


def test_resolve_accepts_enum_like_object():
    """Objects with a .value attribute (e.g. typer enums) are handled."""
    obj, _ = _resolver()

    class FakeEnum:
        value = "yolo26l"

    assert obj.resolve_model_name(FakeEnum()) == "YOLO26l"
