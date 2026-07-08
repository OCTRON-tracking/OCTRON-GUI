"""Tests for the OCTRON user-config loader (octron/config.py).

The loader is intentionally lightweight (stdlib + yaml + loguru), so these
tests need no project, model, or napari import.  ``OCTRON_CONFIG_PATH`` is
pointed at a temp file so the real ``~/.octron/config.yaml`` is never touched.
"""

from pathlib import Path

import pytest

from octron import config


@pytest.fixture
def cfg_path(tmp_path, monkeypatch):
    p = tmp_path / "config.yaml"
    monkeypatch.setenv("OCTRON_CONFIG_PATH", str(p))
    return p


# ---------------------------------------------------------------------------
# Location
# ---------------------------------------------------------------------------


def test_config_path_respects_env(cfg_path):
    assert config.config_path() == cfg_path


# ---------------------------------------------------------------------------
# Defaults / load
# ---------------------------------------------------------------------------


def test_defaults_when_no_file(cfg_path):
    assert not cfg_path.exists()
    assert config.get_value("prediction_cache_dir") is None
    assert config.get_prediction_cache_dir() is None


def test_unknown_key_in_file_is_ignored(cfg_path):
    cfg_path.write_text("prediction_cache_dir: /tmp/x\nbogus_key: 123\n")
    merged = config.load()
    assert merged["prediction_cache_dir"] == "/tmp/x"
    assert "bogus_key" not in merged


def test_malformed_file_falls_back_to_defaults(cfg_path):
    # Invalid YAML (mapping value where a scalar is expected) must not crash load().
    cfg_path.write_text("a: b: c\n")
    merged = config.load()
    assert merged["prediction_cache_dir"] is None


# ---------------------------------------------------------------------------
# set_value / round-trip
# ---------------------------------------------------------------------------


def test_set_and_get_roundtrip(cfg_path, tmp_path):
    cache = tmp_path / "scratch"
    written = config.set_value("prediction_cache_dir", str(cache))
    assert written == cfg_path
    assert cfg_path.exists()
    assert config.get_value("prediction_cache_dir") == str(cache)
    assert config.get_prediction_cache_dir() == cache


def test_empty_string_means_caching_off(cfg_path):
    config.set_value("prediction_cache_dir", "   ")
    assert config.get_value("prediction_cache_dir") is None
    assert config.get_prediction_cache_dir() is None


def test_tilde_is_expanded_on_read(cfg_path):
    config.set_value("prediction_cache_dir", "~/octron_scratch")
    assert config.get_prediction_cache_dir() == (
        Path.home() / "octron_scratch"
    )


def test_set_value_preserves_unrelated_file_content(cfg_path):
    # A forward-compat key written by a newer OCTRON must survive a set_value.
    cfg_path.write_text("prediction_cache_dir: /tmp/a\nfuture_key: keep_me\n")
    config.set_value("prediction_cache_dir", "/tmp/b")
    raw = cfg_path.read_text()
    assert "future_key: keep_me" in raw
    assert config.get_value("prediction_cache_dir") == "/tmp/b"


# ---------------------------------------------------------------------------
# Unknown keys
# ---------------------------------------------------------------------------


def test_get_value_rejects_unknown_key(cfg_path):
    with pytest.raises(KeyError):
        config.get_value("nope")


def test_set_value_rejects_unknown_key(cfg_path):
    with pytest.raises(KeyError):
        config.set_value("nope", 1)


# ---------------------------------------------------------------------------
# Schema introspection (drives the CLI / settings dialog)
# ---------------------------------------------------------------------------


def test_specs_includes_prediction_cache_dir(cfg_path):
    by_key = {s.key: s for s in config.specs()}
    assert "prediction_cache_dir" in by_key
    spec = by_key["prediction_cache_dir"]
    assert spec.kind == "dir"
    assert spec.default is None
    assert spec.description  # non-empty, for the dialog/CLI


# ---------------------------------------------------------------------------
# Model cache dir (downloaded weights / checkpoints)
# ---------------------------------------------------------------------------


def test_model_cache_dir_defaults_to_user_cache(cfg_path):
    # Unset -> per-user cache dir for "octron" (does NOT create it).
    d = config.get_model_cache_dir()
    assert isinstance(d, Path)
    assert d.name == "octron"


def test_model_cache_dir_honors_config_and_tilde(cfg_path):
    config.set_value("model_cache_dir", "~/octron_models")
    assert config.get_model_cache_dir() == (Path.home() / "octron_models")


def test_yolo_models_and_sam_checkpoints_dirs_created(cfg_path, tmp_path):
    base = tmp_path / "mcache"
    config.set_value("model_cache_dir", str(base))
    models = config.get_yolo_models_dir()
    ckpts = config.get_sam_checkpoints_dir()
    assert models == base / "models" and models.is_dir()
    assert ckpts == base / "checkpoints" and ckpts.is_dir()


def test_reid_weights_dir_created(cfg_path, tmp_path):
    # boxmot ReID weights live alongside the YOLO/SAM downloads, not the CWD.
    base = tmp_path / "mcache"
    config.set_value("model_cache_dir", str(base))
    reid = config.get_reid_weights_dir()
    assert reid == base / "reid" and reid.is_dir()


def test_specs_includes_model_cache_dir(cfg_path):
    by_key = {s.key: s for s in config.specs()}
    assert "model_cache_dir" in by_key
    assert by_key["model_cache_dir"].kind == "dir"
    assert by_key["model_cache_dir"].default is None
