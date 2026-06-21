"""
CLI smoke tests — verifies every subcommand and flag via --help.

No heavy dependencies (torch, cv2, ultralytics) are imported; all tests run
quickly in CI with no GPU and no model files required.

Covered subcommands and flags
------------------------------
gui         --help
gpu-test    --help; gpu-test runs (skipped if torch DLLs unavailable)
split       --help: --mode, --train, --val, --seed, --dry-run
train       --help: --model, --mode, --device, --epochs, --imagesz,
                    --save-period, --overwrite, --resume, --no-split,
                    --train, --val, --seed
predict     --help: --model, --tracker, --tracker-config, --device,
                    --conf-thresh, --iou-thresh, --skip-frames,
                    --one-object-per-label, --opening-radius, --overwrite,
                    --detailed, --buffer-size, --output-dir, --local-cache-dir
render      --help: --video, --output, --preset, --start, --end, --alpha,
                    --masks/--no-masks, --boxes/--no-boxes,
                    --labels/--no-labels, --tracklets,
                    --tracklet-size, --tracklet-smooth-sigma,
                    --tracklet-interpolate, --track-ids, --min-observations,
                    --min-confidence, --bbox-sizes
transcode   --help: --output, --crf, --fps, --no-audio, --overwrite
gif         --help (GUI launcher; body not invoked by --help)

auto_device returns 'cuda', 'mps', or 'cpu' (skipped if torch unavailable)
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner
from octron.cli import app

# Wide terminal so Rich/typer never truncates long flag names in help output
runner = CliRunner(env={"COLUMNS": "200", "NO_COLOR": "1"})


# ---------------------------------------------------------------------------
# Root help
# ---------------------------------------------------------------------------

def test_root_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'OCTRON' in result.output


# ---------------------------------------------------------------------------
# gui
# ---------------------------------------------------------------------------

def test_gui_help():
    result = runner.invoke(app, ['gui', '--help'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# gpu-test
# ---------------------------------------------------------------------------

def test_gpu_test_help():
    result = runner.invoke(app, ['gpu-test', '--help'])
    assert result.exit_code == 0


def test_gpu_test_runs():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    result = runner.invoke(app, ['gpu-test'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

def test_split_help():
    result = runner.invoke(app, ['split', '--help'])
    assert result.exit_code == 0
    assert '--mode' in result.output
    assert '--train' in result.output
    assert '--val' in result.output
    assert '--seed' in result.output
    assert '--dry-run' in result.output


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

def test_train_help():
    result = runner.invoke(app, ['train', '--help'])
    assert result.exit_code == 0
    assert '--model' in result.output
    assert '--mode' in result.output
    assert '--device' in result.output
    assert '--epochs' in result.output
    assert '--imagesz' in result.output
    assert '--save-period' in result.output
    assert '--overwrite' in result.output
    assert '--resume' in result.output
    assert '--no-split' in result.output
    assert '--train' in result.output
    assert '--val' in result.output
    assert '--seed' in result.output


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------

def test_predict_help():
    result = runner.invoke(app, ['predict', '--help'])
    assert result.exit_code == 0
    assert '--model' in result.output
    assert '--tracker' in result.output
    assert '--tracker-config' in result.output
    assert '--device' in result.output
    assert '--conf-thresh' in result.output
    assert '--iou-thresh' in result.output
    assert '--skip-frames' in result.output
    assert '--one-object-per-label' in result.output
    assert '--opening-radius' in result.output
    assert '--overwrite' in result.output
    assert '--detailed' in result.output
    assert '--buffer-size' in result.output
    assert '--output-dir' in result.output
    assert '--local-cache-dir' in result.output


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------

def test_render_help():
    result = runner.invoke(app, ['render', '--help'])
    assert result.exit_code == 0
    assert '--video' in result.output
    assert '--output' in result.output
    assert '--preset' in result.output
    assert '--start' in result.output
    assert '--end' in result.output
    assert '--alpha' in result.output
    assert '--masks' in result.output
    assert '--no-masks' in result.output
    assert '--boxes' in result.output
    assert '--no-boxes' in result.output
    assert '--labels' in result.output
    assert '--no-labels' in result.output
    assert '--tracklets' in result.output
    assert '--tracklet-size' in result.output
    assert '--tracklet-smooth-sigma' in result.output
    assert '--tracklet-interpolate' in result.output
    assert '--track-ids' in result.output
    assert '--min-observations' in result.output
    assert '--min-confidence' in result.output
    assert '--bbox-sizes' in result.output
    assert '--debug' in result.output


# ---------------------------------------------------------------------------
# transcode
# ---------------------------------------------------------------------------

def test_transcode_help():
    result = runner.invoke(app, ['transcode', '--help'])
    assert result.exit_code == 0
    assert '--output' in result.output
    assert '--crf' in result.output
    assert '--fps' in result.output
    assert '--no-audio' in result.output
    assert '--overwrite' in result.output


# ---------------------------------------------------------------------------
# gif
# ---------------------------------------------------------------------------

def test_gif_help():
    # --help must not import/launch the Qt GUI (body lazy-imports mp4_to_gif).
    result = runner.invoke(app, ['gif', '--help'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# auto_device
# ---------------------------------------------------------------------------

def test_auto_device_returns_valid():
    try:
        import torch  # noqa: F401
    except OSError:
        pytest.skip("torch DLLs could not be loaded in this environment")
    from octron.test_gpu import auto_device
    device = auto_device()
    assert device in ('cuda', 'mps', 'cpu')


# ---------------------------------------------------------------------------
# Dynamic enum building (cli._sanitize_identifier / _enum_from_yaml)
# ---------------------------------------------------------------------------

def test_sanitize_identifier_handles_invalid_names():
    from octron.cli import _sanitize_identifier
    assert _sanitize_identifier("bot-sort") == "bot_sort"
    assert _sanitize_identifier("bot.sort") == "bot_sort"
    # Leading digit / spaces are not valid identifiers on their own.
    assert _sanitize_identifier("3d sort").isidentifier()
    assert _sanitize_identifier("3d sort").startswith("_")
    assert _sanitize_identifier("") == "_"


def test_enum_from_yaml_missing_file_uses_fallback():
    from octron.cli import _enum_from_yaml
    enum = _enum_from_yaml("Tmp", Path("/no/such/catalog.yaml"), fallback="bytetrack")
    assert enum("bytetrack").value == "bytetrack"


def test_enum_from_yaml_sanitizes_member_names(tmp_path):
    from octron.cli import _enum_from_yaml
    cat = tmp_path / "cat.yaml"
    cat.write_text("3d-sort:\n  available: true\nbyte.track:\n  available: true\n")
    enum = _enum_from_yaml("Tmp2", cat, available_only=True, fallback="bytetrack")
    # Values preserve the lower-cased catalog keys; members are valid identifiers.
    values = {m.value for m in enum}
    assert "3d-sort" in values
    assert "byte.track" in values
    assert all(m.name.isidentifier() for m in enum)
