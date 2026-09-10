"""Tests for the model catalog checks and checkpoint introspection.

These cover the RT-DETR support additions:

* ``check_yolo_models`` must tolerate catalog entries that ship only one
  task variant (empty ``model_path_seg`` or ``model_path_detect``),
  downloading only the non-empty variants and rejecting entries with
  neither.
* ``get_model_info`` must expose a normalized ``model_type`` derived from
  the stored model class name (``rtdetr`` vs ``yolo``), which is how the
  loader picks the ultralytics class.

Downloads and torch checkpoint reads are monkeypatched, so no network or
real weights are needed.
"""

from pathlib import Path

import pytest
import yaml

from octron.yolo_octron.helpers import yolo_checks
from octron.yolo_octron.yolo_octron import YOLO_octron


def _write_catalog(tmp_path, catalog):
    yaml_path = tmp_path / "models.yaml"
    yaml_path.write_text(yaml.safe_dump(catalog))
    return yaml_path


def test_check_yolo_models_skips_empty_variants(tmp_path, monkeypatch):
    """Only non-empty variants are downloaded; empty ones are skipped."""
    catalog = {
        "RTDETR-l": {
            "name": "RT-DETR-l",
            "model_path_seg": "",  # detection-only
            "model_path_detect": "rtdetr-l.pt",
        },
        "YOLO26m": {
            "name": "YOLO26m",
            "model_path_seg": "yolo26m-seg.pt",
            "model_path_detect": "yolo26m.pt",
        },
    }
    yaml_path = _write_catalog(tmp_path, catalog)
    cache = tmp_path / "cache"
    cache.mkdir()

    from octron import config

    monkeypatch.setattr(config, "get_yolo_models_dir", lambda: cache)
    monkeypatch.setattr(
        yolo_checks, "check_url_availability", lambda url: True
    )
    downloaded = []
    monkeypatch.setattr(
        yolo_checks,
        "download_yolo_model",
        lambda url, fpath, overwrite=False: downloaded.append(
            Path(fpath).name
        ),
    )

    result = yolo_checks.check_yolo_models(
        YOLO_BASE_URL="http://example/base",
        models_yaml_path=yaml_path,
        force_download=True,
    )

    assert set(result) == {"RTDETR-l", "YOLO26m"}
    # RT-DETR contributes only its detection weight; the empty seg
    # variant is never attempted.
    assert sorted(downloaded) == sorted(
        ["rtdetr-l.pt", "yolo26m-seg.pt", "yolo26m.pt"]
    )


def test_check_yolo_models_requires_one_variant(tmp_path, monkeypatch):
    """An entry with neither variant is rejected."""
    catalog = {
        "BROKEN": {
            "name": "Broken",
            "model_path_seg": "",
            "model_path_detect": "",
        }
    }
    yaml_path = _write_catalog(tmp_path, catalog)

    from octron import config

    monkeypatch.setattr(config, "get_yolo_models_dir", lambda: tmp_path)
    monkeypatch.setattr(
        yolo_checks, "check_url_availability", lambda url: True
    )
    monkeypatch.setattr(
        yolo_checks, "download_yolo_model", lambda *a, **k: None
    )

    with pytest.raises(AssertionError, match="neither a segmentation"):
        yolo_checks.check_yolo_models(
            YOLO_BASE_URL="http://example/base",
            models_yaml_path=yaml_path,
        )


def test_get_model_info_model_type_rtdetr_vs_yolo(monkeypatch):
    """model_type is derived from the stored model class name."""
    torch = pytest.importorskip("torch")

    class RTDETRDetectionModel:
        names = {0: "a"}

    class DetectionModel:
        names = {0: "a"}

    def fake_load(path, map_location=None, weights_only=False):
        model = (
            RTDETRDetectionModel()
            if "rtdetr" in str(path).lower()
            else DetectionModel()
        )
        return {"train_args": {"task": "detect"}, "model": model}

    monkeypatch.setattr(torch, "load", fake_load)

    info_rtdetr = YOLO_octron.get_model_info("some/rtdetr-l.pt")
    assert info_rtdetr["model_type"] == "rtdetr"

    info_yolo = YOLO_octron.get_model_info("some/yolo11m.pt")
    assert info_yolo["model_type"] == "yolo"


def test_get_model_info_model_type_defaults_yolo_on_read_error(monkeypatch):
    """A checkpoint that cannot be read falls back to model_type 'yolo'."""
    torch = pytest.importorskip("torch")

    def boom(path, map_location=None, weights_only=False):
        raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr(torch, "load", boom)

    info = YOLO_octron.get_model_info("some/whatever.pt")
    assert info["model_type"] == "yolo"
    assert info["task"] is None
