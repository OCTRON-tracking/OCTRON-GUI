"""
Tests for pure utility functions in octron/tools/predict.py.
No heavy dependencies (torch, cv2, YOLO) required.

Covered
-------
_is_network_path()
    UNC paths with forward slashes (//server/share) → True
    UNC paths with backslashes (\\\\server\\share) → True
    Mixed-slash UNC paths → True
    Local absolute Unix paths (/home/user/...) → False
    Windows drive paths (C:\\...) → False
    Relative paths → False
    Single-slash paths (/tmp/...) → False
    Accepts pathlib.Path objects for both network and local paths

Input validation (run_predict entry checks fire before heavy imports)
---------------------------------------------------------------------
run_predict(videos=..., model_path=...)
    Empty videos directory (no .mp4 inside) → ValueError naming the dir
    Single str path coerced to list (no validation error)
    Single Path coerced to list (no validation error)
    model_path is a directory without best.pt → FileNotFoundError
    model_path directory containing weights/best.pt is resolved
"""

from pathlib import Path
import pytest
from octron.tools.predict import _is_network_path, run_predict


# ---------------------------------------------------------------------------
# _is_network_path
# ---------------------------------------------------------------------------

def test_is_network_path_unc_forward_slashes():
    assert _is_network_path("//server/share") is True


def test_is_network_path_unc_backslashes():
    assert _is_network_path("\\\\server\\share") is True


def test_is_network_path_unc_mixed_slashes():
    assert _is_network_path("\\\\server/share/subdir") is True


def test_is_network_path_local_unix():
    assert _is_network_path("/home/user/data") is False


def test_is_network_path_windows_drive():
    assert _is_network_path("C:\\Users\\data") is False


def test_is_network_path_relative():
    assert _is_network_path("data/videos") is False


def test_is_network_path_single_slash():
    assert _is_network_path("/tmp/videos") is False


def test_is_network_path_accepts_path_object():
    assert _is_network_path(Path("//server/share/data")) is True


def test_is_network_path_local_path_object():
    assert _is_network_path(Path("/home/user/data")) is False


# ---------------------------------------------------------------------------
# run_predict — videos and model_path validation
#
# These tests rely on the validation block at the top of run_predict firing
# before any heavy imports (loguru/yolo_octron) or model loading.
# ---------------------------------------------------------------------------

def test_run_predict_empty_video_directory_raises(tmp_path):
    """A directory with no .mp4 files should be rejected with a clear message."""
    empty_dir = tmp_path / "no_videos"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No .mp4 files found"):
        run_predict(videos=[empty_dir], model_path=tmp_path / "fake.pt")


def test_run_predict_empty_video_directory_via_str(tmp_path):
    """Same check should apply when the directory is passed as a single string."""
    empty_dir = tmp_path / "no_videos"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No .mp4 files found"):
        run_predict(videos=str(empty_dir), model_path=tmp_path / "fake.pt")


def test_run_predict_model_dir_without_best_pt_raises(tmp_path):
    """model_path pointing at a directory with no recognised best.pt should error."""
    # Need a real .mp4 so we get past video validation and into model_path checks
    (tmp_path / "clip.mp4").write_bytes(b"")
    model_dir = tmp_path / "trained_run"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="no best.pt"):
        run_predict(
            videos=[tmp_path / "clip.mp4"],
            model_path=model_dir,
        )


def test_run_predict_model_dir_with_weights_best_pt_resolves(tmp_path):
    """If weights/best.pt exists inside the model directory, validation passes.

    We only want to confirm that the model_path resolution accepts this layout —
    any later error (the .pt is empty, no torch model can load) is fine.
    """
    (tmp_path / "clip.mp4").write_bytes(b"")
    model_dir = tmp_path / "trained_run"
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "weights" / "best.pt").write_bytes(b"")

    # Should NOT raise FileNotFoundError("no best.pt") — any later failure is OK.
    with pytest.raises(Exception) as exc:
        run_predict(
            videos=[tmp_path / "clip.mp4"],
            model_path=model_dir,
        )
    assert "no best.pt" not in str(exc.value)
