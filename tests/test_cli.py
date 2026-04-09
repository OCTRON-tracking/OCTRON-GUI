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
                    --labels/--no-labels, --tracklets, --tracklet-overlay,
                    --tracklet-size, --tracklet-mask-centroids,
                    --tracklet-smooth-cutoff, --tracklet-smooth-order,
                    --tracklet-interpolate, --track-ids, --min-observations,
                    --bbox-sizes
transcode   --help: --output, --crf, --overwrite

auto_device returns 'cuda', 'mps', or 'cpu' (skipped if torch unavailable)
"""

import pytest
from typer.testing import CliRunner
from octron.cli import app

# Wide terminal so Rich/typer never truncates long flag names in help output
runner = CliRunner(env={"COLUMNS": "200"})


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
    assert '--tracklet-mask-centroids' in result.output
    assert '--tracklet-smooth-cutoff' in result.output
    assert '--tracklet-smooth-order' in result.output
    assert '--tracklet-interpolate' in result.output
    assert '--track-ids' in result.output
    assert '--min-observations' in result.output
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
    assert '--overwrite' in result.output


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
