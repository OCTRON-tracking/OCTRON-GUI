"""
Tests for run_predict entry validation in octron/tools/predict.py.

These checks (video discovery + model_path resolution) fire before any heavy
imports (loguru / yolo_octron) or model loading, so no torch/cv2/YOLO needed.

The network-share helper moved to octron.yolo_octron.helpers.cache_io and is
covered by tests/test_cache_io.py.
"""

import pytest

from octron.tools.predict import run_predict


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
    # Need a real .mp4 so we get past video validation and into model_path checks.
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

    We only confirm model_path resolution accepts this layout — any later error
    (the .pt is empty, no torch model can load) is fine.
    """
    (tmp_path / "clip.mp4").write_bytes(b"")
    model_dir = tmp_path / "trained_run"
    (model_dir / "weights").mkdir(parents=True)
    (model_dir / "weights" / "best.pt").write_bytes(b"")

    with pytest.raises(Exception) as exc:
        run_predict(
            videos=[tmp_path / "clip.mp4"],
            model_path=model_dir,
        )
    assert "no best.pt" not in str(exc.value)
