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


# ---------------------------------------------------------------------------
# device handling
#
# These stub run_predict's lazy heavy imports (YOLO_octron / auto_device) so the
# device-normalisation logic can be exercised without torch/cv2/YOLO.
# ---------------------------------------------------------------------------

def _stub_predict_batch(monkeypatch):
    """Replace the lazy imports inside run_predict with light fakes.

    Returns the dict that captures the kwargs forwarded to predict_batch.
    """
    import sys
    import types

    captured = {}

    class _FakeYOLO:
        def predict_batch(self, **kwargs):
            captured.update(kwargs)
            return iter(())  # no progress events

    fake_yolo = types.ModuleType("octron.yolo_octron.yolo_octron")
    fake_yolo.YOLO_octron = _FakeYOLO
    monkeypatch.setitem(sys.modules, "octron.yolo_octron.yolo_octron", fake_yolo)

    fake_gpu = types.ModuleType("octron.test_gpu")
    fake_gpu.auto_device = lambda: "cpu"
    monkeypatch.setitem(sys.modules, "octron.test_gpu", fake_gpu)
    return captured


def test_run_predict_unwraps_device_enum(monkeypatch, tmp_path):
    """A Device enum must reach predict_batch as a plain string.

    boxmot's select_device() does str(device).lower(); a (str, Enum) member
    stringifies to "Device.mps" -> "device.mps", which it misreads as a CUDA
    device. run_predict must forward the bare value ("mps") instead.
    """
    captured = _stub_predict_batch(monkeypatch)
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "model.pt").write_bytes(b"")

    from octron.cli import Device
    run_predict(
        videos=[tmp_path / "clip.mp4"],
        model_path=tmp_path / "model.pt",
        tracker_name="bytetrack",
        device=Device.mps,
    )
    assert captured["device"] == "mps"
    assert type(captured["device"]) is str  # plain str, not the Device enum


def test_run_predict_auto_device_is_resolved(monkeypatch, tmp_path):
    """device='auto' (or Device.auto) must be resolved via auto_device()."""
    captured = _stub_predict_batch(monkeypatch)  # auto_device() -> "cpu"
    (tmp_path / "clip.mp4").write_bytes(b"")
    (tmp_path / "model.pt").write_bytes(b"")

    from octron.cli import Device
    run_predict(
        videos=[tmp_path / "clip.mp4"],
        model_path=tmp_path / "model.pt",
        tracker_name="bytetrack",
        device=Device.auto,
    )
    assert captured["device"] == "cpu"
