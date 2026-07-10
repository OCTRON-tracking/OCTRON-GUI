"""Tests for model-name resolution.

Resolution now lives in core ``YOLO_octron.resolve_model_name`` (case-insensitive
match against the model catalog), shared by the CLI and the GUI, replacing the
old CLI-only ``_normalise_model_name``.

The catalog keys are read from the ``yolo_models.yaml`` that ships with the
package; a ``YOLO_octron`` instance is built via ``__new__`` (bypassing
``__init__``) so no weights are downloaded or loaded.
"""

from pathlib import Path

import pytest
import yaml

from octron.yolo_octron.yolo_octron import YOLO_octron

MODELS_YAML = (
    Path(__file__).parent.parent
    / "octron"
    / "yolo_octron"
    / "yolo_models.yaml"
)


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


# ---------------------------------------------------------------------------
# Training-state helpers: config_path / _resolve_batch_size / resolve_resume_state
#
# These live in core so the CLI and GUI share one implementation. __init__ is
# bypassed via __new__; only the attributes the helpers touch are set.
# ---------------------------------------------------------------------------


def _make_yolo(tmp_path):
    """Build a minimal YOLO_octron instance without running __init__."""
    obj = YOLO_octron.__new__(YOLO_octron)
    obj._project_path = tmp_path
    obj.training_path = tmp_path / "model"
    obj.data_path = obj.training_path / "training_data"
    obj._config_path = None
    obj.model = None
    (obj.training_path / "training" / "weights").mkdir(
        parents=True, exist_ok=True
    )
    return obj


def _weights_dir(obj):
    return obj.training_path / "training" / "weights"


def _write_checkpoint(path, epoch, imgsz=640, include_imgsz=True):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    train_args = {"imgsz": imgsz} if include_imgsz else {}
    torch.save({"epoch": epoch, "train_args": train_args}, path)


# config_path property -------------------------------------------------------


def test_config_path_derives_from_data_path(tmp_path):
    obj = _make_yolo(tmp_path)
    assert obj.config_path == obj.data_path / "yolo_config.yaml"


def test_config_path_explicit_override_and_reset(tmp_path):
    obj = _make_yolo(tmp_path)
    custom = tmp_path / "elsewhere" / "cfg.yaml"
    obj.config_path = custom
    assert obj.config_path == custom
    # Resetting to None falls back to the derived path.
    obj.config_path = None
    assert obj.config_path == obj.data_path / "yolo_config.yaml"


def test_config_path_none_without_data_path(tmp_path):
    obj = _make_yolo(tmp_path)
    obj.data_path = None
    assert obj.config_path is None


# _resolve_batch_size (cpu / mps) -------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_resolve_batch_size_non_cuda_returns_minus_one(tmp_path, device):
    obj = _make_yolo(tmp_path)
    # No CUDA-only AutoBatch profiling for cpu/mps; returns the -1 sentinel.
    assert obj._resolve_batch_size(640, device) == -1


# resolve_resume_state ------------------------------------------------------


def test_resume_state_fresh_when_nothing(tmp_path):
    obj = _make_yolo(tmp_path)
    state = obj.resolve_resume_state(resume=False, overwrite=False)
    assert state["action"] == "fresh"
    assert state["checkpoint"] is None


def test_resume_state_completed_when_best_exists(tmp_path):
    obj = _make_yolo(tmp_path)
    (_weights_dir(obj) / "best.pt").touch()
    state = obj.resolve_resume_state(resume=False, overwrite=False)
    assert state["action"] == "completed"


def test_resume_state_overwrite_wins_over_existing(tmp_path):
    obj = _make_yolo(tmp_path)
    (_weights_dir(obj) / "best.pt").touch()
    (_weights_dir(obj) / "last.pt").touch()
    state = obj.resolve_resume_state(resume=True, overwrite=True)
    assert state["action"] == "fresh"


def test_resume_state_strict_resume_for_interrupted_run(tmp_path):
    pytest.importorskip("torch")
    obj = _make_yolo(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    _write_checkpoint(last_pt, epoch=7, imgsz=640)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "resume"
    assert state["checkpoint"] == last_pt
    assert state["imgsz"] == 640


def test_resume_state_init_from_completed_checkpoint(tmp_path):
    pytest.importorskip("torch")
    obj = _make_yolo(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    # ultralytics writes epoch == -1 for a completed run.
    _write_checkpoint(last_pt, epoch=-1, imgsz=832)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "init_from_checkpoint"
    assert state["imgsz"] == 832


def test_resume_state_resume_without_checkpoint_is_fresh(tmp_path):
    obj = _make_yolo(tmp_path)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "fresh"


def test_resume_state_handles_imgsz_as_list(tmp_path):
    pytest.importorskip("torch")
    obj = _make_yolo(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    _write_checkpoint(last_pt, epoch=3, imgsz=[768])
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "resume"
    assert state["imgsz"] == 768


# ---------------------------------------------------------------------------
# collect_labels / find_files_with_depth_limit — missing-folder safety
#
# Loading a video then returning to the project tab before generating any data
# used to crash: the per-video subfolder (project/<hash>) doesn't exist yet, so
# scanning it raised FileNotFoundError. It should yield no labels instead.
# These early-return before any heavy (napari/zarr) imports, so they stay light.
# ---------------------------------------------------------------------------


def test_collect_labels_missing_subfolder_returns_empty(tmp_path):
    from octron.yolo_octron.helpers.training import collect_labels

    # Project dir exists; the per-video subfolder has not been created yet.
    assert collect_labels(tmp_path, subfolder="acb207d1") == {}


def test_find_files_with_depth_limit_missing_base_returns_empty(tmp_path):
    from octron.yolo_octron.helpers.training import find_files_with_depth_limit

    missing = tmp_path / "does_not_exist"
    assert find_files_with_depth_limit(missing, "object_organizer.json") == []


def test_resume_state_missing_imgsz_errors(tmp_path):
    pytest.importorskip("torch")
    obj = _make_yolo(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    _write_checkpoint(last_pt, epoch=5, include_imgsz=False)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "error"
    assert "image size" in state["message"].lower()
