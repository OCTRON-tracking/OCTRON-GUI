from pathlib import Path

import pytest
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
from huggingface_hub import hf_hub_download

from octron.cotracker_octron.helpers.build_cotracker import build_cotracker
from octron.cotracker_octron.helpers.cotracker_checks import (
    check_cotracker_models,
    download_cotracker_file,
)


@pytest.fixture(scope="session")
def cotracker_checkpoint(tmp_path_factory):
    """Return the path to a cotracker downloaded model.

    The scope of the fixture is "session", so that the model
    is downloaded once per test session only.
    """
    mock_octron_dir = tmp_path_factory.mktemp("mock_octron")
    cotracker_ckpt_dir = mock_octron_dir / "cotracker_octron" / "checkpoints"
    cotracker_ckpt_dir.mkdir(parents=True)

    return Path(
        hf_hub_download(
            repo_id="facebook/cotracker3",
            filename="scaled_online.pth",
            local_dir=str(cotracker_ckpt_dir),
            local_dir_use_symlinks=False,
        )
    )


def test_build_cotracker(cotracker_checkpoint):
    """Test that the CoTracker model can be built from a checkpoint path."""
    model, device = build_cotracker(cotracker_checkpoint)

    assert isinstance(model, CoTrackerOnlinePredictor)
    assert isinstance(device, torch.device)
    assert model.image_size is None  # no prescribed image input size


def mock_hf_download(repo_id, filename, local_dir, **kwargs):
    """Mock HuggingFace download via hf_hub_download"""
    path = Path(local_dir) / filename
    path.touch()
    return str(path)


@pytest.mark.parametrize(
    "file_exists, expected_messages",
    [
        (False, ["Downloading foo.pth from ", "Saved foo.pth to "]),
        (True, ["exists. Skipping download."]),
    ],
)
def test_download_cotracker_file(
    tmp_path, monkeypatch, capsys, file_exists, expected_messages
):
    """Test download log messages, with and without a pre-existing file."""
    # Create foo.pth file if file_exists is True
    filename = "foo.pth"
    if file_exists:
        (tmp_path / filename).touch()

    # Monkeypatch hf_hub_download function
    monkeypatch.setattr(
        "octron.cotracker_octron.helpers.cotracker_checks.hf_hub_download",
        mock_hf_download,
    )

    # Call the download function and capture print output
    local_path = download_cotracker_file(filename, tmp_path)
    captured = capsys.readouterr()

    # Assert file exists
    assert local_path.exists()
    assert local_path.name == filename

    # Check log messages
    assert all(msg in captured.out for msg in expected_messages)


def test_check_cotracker_models(cotracker_checkpoint, monkeypatch):
    """Test that check_cotracker_models returns the expected dict."""
    # Monkeypatch hf_hub_download function
    monkeypatch.setattr(
        "octron.cotracker_octron.helpers.cotracker_checks.hf_hub_download",
        mock_hf_download,
    )

    # Run function to check models exist locally
    result = check_cotracker_models(checkpoints_dir=cotracker_checkpoint.parent)

    # Check checkpoint exists
    expected_ckpt_path = cotracker_checkpoint.parent / "scaled_online.pth"
    assert expected_ckpt_path.exists()

    # Check returned dict
    assert result == {
        "cotracker": {
            "name": "Cotracker3",
            "checkpoint_path": "checkpoints/scaled_online.pth",
            # path should be relative to "cotracker_octron"
        }
    }
