"""Tests for model-name resolution.

Resolution now lives in core ``AnalysisOctron.resolve_model_name``
(case-insensitive
match against the model catalog), shared by the CLI and the GUI, replacing the
old CLI-only ``_normalise_model_name``.

The catalog keys are read from the ``analysis_models.yaml`` that ships with the
package; a ``AnalysisOctron`` instance is built via ``__new__`` (bypassing
``__init__``) so no weights are downloaded or loaded.
"""

from pathlib import Path

import pytest
import yaml

from octron.analysis_octron import analysis_octron as analysis_octron_module
from octron.analysis_octron.analysis_octron import AnalysisOctron

MODELS_YAML = (
    Path(analysis_octron_module.__file__).parent / "analysis_models.yaml"
)


def _resolver():
    """Build a AnalysisOctron (no __init__) with catalog-name model keys."""
    with open(MODELS_YAML) as f:
        keys = list((yaml.safe_load(f) or {}).keys())
    obj = AnalysisOctron.__new__(AnalysisOctron)
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


def test_resolve_rtdetr_keys():
    """RT-DETR catalog keys resolve case-insensitively like YOLO ones."""
    obj, _ = _resolver()
    assert obj.resolve_model_name("rtdetr-l") == "RTDETR-l"
    assert obj.resolve_model_name("RTDETR-X") == "RTDETR-x"


# ---------------------------------------------------------------------------
# supports_task / load_model capability guard
#
# Capability is derived from which catalog variants are non-empty; a
# detection-only model (e.g. RT-DETR, empty model_path_seg) must not be
# usable for segmentation. __init__ is bypassed via __new__ so no weights
# are downloaded or loaded.
# ---------------------------------------------------------------------------


def _catalog():
    """AnalysisOctron (no __init__) with a mixed-capability catalog."""
    obj = AnalysisOctron.__new__(AnalysisOctron)
    obj.models_dict = {
        "YOLO26m": {
            "name": "YOLO26m",
            "model_path_seg": "yolo26m-seg.pt",
            "model_path_detect": "yolo26m.pt",
        },
        "RTDETR-l": {
            "name": "RT-DETR-l",
            "model_path_seg": "",
            "model_path_detect": "rtdetr-l.pt",
        },
    }
    return obj


def test_supports_task_detect_only_model():
    obj = _catalog()
    assert obj.supports_task("rtdetr-l", "detect") is True
    assert obj.supports_task("rtdetr-l", "segment") is False


def test_supports_task_dual_capability_model():
    obj = _catalog()
    assert obj.supports_task("yolo26m", "detect") is True
    assert obj.supports_task("yolo26m", "segment") is True


def test_supports_task_unknown_name_defers_true():
    """Non-catalog names (e.g. a file path) defer to the loader."""
    obj = _catalog()
    assert obj.supports_task("/some/trained/best.pt", "segment") is True


def test_load_model_rejects_unsupported_task():
    """Segmentation on a detect-only catalog model raises ValueError.

    The guard fires before any ultralytics/torch import, so this stays a
    light unit test.
    """
    obj = _catalog()
    obj.training_path = None  # skip the analysis_settings block
    with pytest.raises(ValueError, match="does not support segmentation"):
        obj.load_model("rtdetr-l", train_mode="segment")


# ---------------------------------------------------------------------------
# Training-state helpers:
# config_path / _resolve_batch_size / resolve_resume_state
#
# These live in core so the CLI and GUI share one implementation. __init__ is
# bypassed via __new__; only the attributes the helpers touch are set.
# ---------------------------------------------------------------------------


def _make_analysis(tmp_path):
    """Build a minimal AnalysisOctron instance without running __init__."""
    obj = AnalysisOctron.__new__(AnalysisOctron)
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
    obj = _make_analysis(tmp_path)
    assert obj.config_path == obj.data_path / "ultralytics_config.yaml"


def test_config_path_explicit_override_and_reset(tmp_path):
    obj = _make_analysis(tmp_path)
    custom = tmp_path / "elsewhere" / "cfg.yaml"
    obj.config_path = custom
    assert obj.config_path == custom
    # Resetting to None falls back to the derived path.
    obj.config_path = None
    assert obj.config_path == obj.data_path / "ultralytics_config.yaml"


def test_config_path_none_without_data_path(tmp_path):
    obj = _make_analysis(tmp_path)
    obj.data_path = None
    assert obj.config_path is None


# _resolve_batch_size (cpu / mps) -------------------------------------------


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_resolve_batch_size_non_cuda_returns_minus_one(tmp_path, device):
    obj = _make_analysis(tmp_path)
    # No CUDA-only AutoBatch profiling for cpu/mps; returns the -1 sentinel.
    assert obj._resolve_batch_size(640, device) == -1


# resolve_resume_state ------------------------------------------------------


def test_resume_state_fresh_when_nothing(tmp_path):
    obj = _make_analysis(tmp_path)
    state = obj.resolve_resume_state(resume=False, overwrite=False)
    assert state["action"] == "fresh"
    assert state["checkpoint"] is None


def test_resume_state_completed_when_best_exists(tmp_path):
    obj = _make_analysis(tmp_path)
    (_weights_dir(obj) / "best.pt").touch()
    state = obj.resolve_resume_state(resume=False, overwrite=False)
    assert state["action"] == "completed"


def test_resume_state_overwrite_wins_over_existing(tmp_path):
    obj = _make_analysis(tmp_path)
    (_weights_dir(obj) / "best.pt").touch()
    (_weights_dir(obj) / "last.pt").touch()
    state = obj.resolve_resume_state(resume=True, overwrite=True)
    assert state["action"] == "fresh"


def test_resume_state_strict_resume_for_interrupted_run(tmp_path):
    pytest.importorskip("torch")
    obj = _make_analysis(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    _write_checkpoint(last_pt, epoch=7, imgsz=640)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "resume"
    assert state["checkpoint"] == last_pt
    assert state["imgsz"] == 640


def test_resume_state_init_from_completed_checkpoint(tmp_path):
    pytest.importorskip("torch")
    obj = _make_analysis(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    # ultralytics writes epoch == -1 for a completed run.
    _write_checkpoint(last_pt, epoch=-1, imgsz=832)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "init_from_checkpoint"
    assert state["imgsz"] == 832


def test_resume_state_resume_without_checkpoint_is_fresh(tmp_path):
    obj = _make_analysis(tmp_path)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "fresh"


def test_resume_state_handles_imgsz_as_list(tmp_path):
    pytest.importorskip("torch")
    obj = _make_analysis(tmp_path)
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
# These early-return before any heavy (napari/zarr) imports, so they
# stay light.
# ---------------------------------------------------------------------------


def test_collect_labels_missing_subfolder_returns_empty(tmp_path):
    from octron.analysis_octron.helpers.training import collect_labels

    # Project dir exists; the per-video subfolder has not been created yet.
    assert collect_labels(tmp_path, subfolder="acb207d1") == {}


def test_find_files_with_depth_limit_missing_base_returns_empty(tmp_path):
    from octron.analysis_octron.helpers.training import (
        find_files_with_depth_limit,
    )

    missing = tmp_path / "does_not_exist"
    assert find_files_with_depth_limit(missing, "object_organizer.json") == []


def test_resume_state_missing_imgsz_errors(tmp_path):
    pytest.importorskip("torch")
    obj = _make_analysis(tmp_path)
    last_pt = _weights_dir(obj) / "last.pt"
    _write_checkpoint(last_pt, epoch=5, include_imgsz=False)
    state = obj.resolve_resume_state(resume=True, overwrite=False)
    assert state["action"] == "error"
    assert "image size" in state["message"].lower()


# ---------------------------------------------------------------------------
# _patch_ultralytics_mlflow_artifacts
#
# ultralytics' MLflow ``on_train_end`` copies the whole weights dir plus
# every result plot/CSV into the MLflow artifact store, duplicating files
# already written to the training folder. The patch swaps it for one that
# only closes the run, keeping the per-epoch metric callbacks (curves).
# ---------------------------------------------------------------------------


def test_patch_mlflow_artifacts_removes_duplication(monkeypatch):
    """on_train_end is swapped for a no-artifact run-closer; metrics stay."""
    import inspect
    from collections import defaultdict

    m = pytest.importorskip("ultralytics.utils.callbacks.mlflow")

    # Isolate module-global state; monkeypatch restores it on teardown so
    # the process-wide patch cannot leak into other tests. Build a fresh
    # callbacks dict (populated as it is when the integration is enabled)
    # so the test does not depend on the ultralytics mlflow setting.
    cbs = {
        "on_train_epoch_end": m.on_train_epoch_end,
        "on_fit_epoch_end": m.on_fit_epoch_end,
        "on_train_end": m.on_train_end,
    }
    monkeypatch.setattr(m, "callbacks", cbs)
    monkeypatch.setattr(m, "on_train_end", m.on_train_end)
    monkeypatch.setattr(m, "_octron_no_artifact_patch", False, raising=False)

    # Precondition: the stock callback copies artifacts.
    assert "log_artifact" in inspect.getsource(cbs["on_train_end"])

    AnalysisOctron._patch_ultralytics_mlflow_artifacts()

    end_fn = m.callbacks["on_train_end"]
    src = inspect.getsource(end_fn)
    assert "log_artifact" not in src  # no weight/plot duplication
    assert "end_run" in src  # but the run is still closed
    # Metric callbacks (the dashboard curves) must be untouched.
    epoch_src = inspect.getsource(m.callbacks["on_train_epoch_end"])
    fit_src = inspect.getsource(m.callbacks["on_fit_epoch_end"])
    assert "log_metrics" in epoch_src
    assert "log_metrics" in fit_src

    # A trainer's add_integration_callbacks would register the no-artifact
    # function (it copies module callbacks into the trainer at init).
    inst = defaultdict(list)
    for k, v in m.callbacks.items():
        if v not in inst[k]:
            inst[k].append(v)
    assert inst["on_train_end"] == [end_fn]

    # Idempotent: a second call keeps the already-patched function.
    AnalysisOctron._patch_ultralytics_mlflow_artifacts()
    assert m.callbacks["on_train_end"] is end_fn


# ---------------------------------------------------------------------------
# load_predictions — empty results (zero detections/tracks for a video)
#
# Regression: ticking "View results" for a video with zero detections used
# to raise ValueError deep inside AnalysisResults, crashing the batch
# prediction worker. load_predictions() must now detect the empty-results
# case and simply yield nothing.
# ---------------------------------------------------------------------------


def test_load_predictions_empty_results_yields_nothing(tmp_path):
    obj = AnalysisOctron.__new__(AnalysisOctron)
    results_dir = tmp_path / "clip_ByteTrack"
    results_dir.mkdir()
    predictions = obj.load_predictions(save_dir=results_dir, open_viewer=False)
    assert list(predictions) == []
