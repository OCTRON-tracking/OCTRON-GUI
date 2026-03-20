"""
Basic CLI smoke tests.

These tests verify that every subcommand loads and responds to --help without
importing torch or any other heavy dependency.  They run quickly in CI with no
GPU and no model files required.
"""

import pytest
from typer.testing import CliRunner
from octron.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# --help for every subcommand
# ---------------------------------------------------------------------------

def test_root_help():
    result = runner.invoke(app, ['--help'])
    assert result.exit_code == 0
    assert 'OCTRON' in result.output


def test_gui_help():
    result = runner.invoke(app, ['gui', '--help'])
    assert result.exit_code == 0


def test_gpu_test_help():
    result = runner.invoke(app, ['gpu-test', '--help'])
    assert result.exit_code == 0


def test_train_help():
    result = runner.invoke(app, ['train', '--help'])
    assert result.exit_code == 0
    assert '--model' in result.output
    assert '--device' in result.output
    assert '--epochs' in result.output


def test_predict_help():
    result = runner.invoke(app, ['predict', '--help'])
    assert result.exit_code == 0
    assert '--model' in result.output
    assert '--tracker' in result.output
    assert '--device' in result.output


def test_render_help():
    result = runner.invoke(app, ['render', '--help'])
    assert result.exit_code == 0
    assert '--preset' in result.output
    assert '--tracklets' in result.output
    assert '--video' in result.output


def test_bbox_sizes_help():
    result = runner.invoke(app, ['bbox-sizes', '--help'])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# auto_device returns a valid value on any platform
# ---------------------------------------------------------------------------

def test_auto_device_returns_valid():
    from octron.test_gpu import auto_device
    device = auto_device()
    assert device in ('cuda', 'mps', 'cpu')


# ---------------------------------------------------------------------------
# gpu-test command runs without crashing (CPU is fine)
# ---------------------------------------------------------------------------

def test_gpu_test_runs():
    result = runner.invoke(app, ['gpu-test'])
    assert result.exit_code == 0
