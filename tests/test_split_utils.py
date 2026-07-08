"""Tests for train/val split-fraction validation.

The fraction guard now lives in core ``YOLO_octron._validate_split_fractions``
(the single source of truth, also enforced by ``prepare_split`` for the GUI and
programmatic callers).  ``run_split`` calls the same guard up front, so the CLI
still fails before any model, label, or geometry work.
"""

import numpy as np
import pytest

from octron.tools.split import run_split
from octron.yolo_octron.yolo_octron import YOLO_octron

# ---------------------------------------------------------------------------
# Core guard: YOLO_octron._validate_split_fractions
# ---------------------------------------------------------------------------


def test_validate_rejects_zero_train_fraction():
    with pytest.raises(ValueError, match="training_fraction"):
        YOLO_octron._validate_split_fractions(0.0, 0.15)


def test_validate_rejects_negative_train_fraction():
    with pytest.raises(ValueError, match="training_fraction"):
        YOLO_octron._validate_split_fractions(-0.1, 0.15)


def test_validate_rejects_train_fraction_one():
    with pytest.raises(ValueError, match="training_fraction"):
        YOLO_octron._validate_split_fractions(1.0, 0.0)


def test_validate_rejects_negative_val_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        YOLO_octron._validate_split_fractions(0.7, -0.05)


def test_validate_rejects_val_fraction_one():
    with pytest.raises(ValueError, match="validation_fraction"):
        YOLO_octron._validate_split_fractions(0.5, 1.0)


def test_validate_rejects_sum_equal_to_one():
    """Train + val == 1 leaves no test split, which is invalid."""
    with pytest.raises(ValueError, match="must be < 1"):
        YOLO_octron._validate_split_fractions(0.7, 0.3)


def test_validate_rejects_sum_greater_than_one():
    with pytest.raises(ValueError, match="must be < 1"):
        YOLO_octron._validate_split_fractions(0.7, 0.4)


def test_validate_accepts_valid_fractions():
    """Valid fractions return None (no exception)."""
    assert YOLO_octron._validate_split_fractions(0.7, 0.15) is None


# ---------------------------------------------------------------------------
# CLI wiring: run_split calls the guard up front
# ---------------------------------------------------------------------------


def test_run_split_rejects_invalid_fractions():
    """run_split delegates to the core guard before touching the project."""
    with pytest.raises(ValueError, match="must be < 1"):
        run_split(
            project_path="/nope_for_test", train_fraction=0.7, val_fraction=0.4
        )


def test_run_split_accepts_valid_fractions():
    """Valid fractions pass the guard; later failure (missing project) is OK."""
    with pytest.raises(Exception) as exc:
        run_split(
            project_path="/nope_for_test",
            train_fraction=0.7,
            val_fraction=0.15,
        )
    # Must not have failed at the fraction guard.
    assert "fraction" not in str(exc.value).lower()


# ---------------------------------------------------------------------------
# prepare_split threads the seed through to train_test_val
# ---------------------------------------------------------------------------


def _split_with_seed(frames, seed):
    """Run prepare_split on a one-label fixture; return the (train, val, test) split."""
    obj = YOLO_octron.__new__(YOLO_octron)
    obj.label_dict = {
        "sub": {
            "video": None,
            "video_file_path": None,
            0: {"label": "a", "frames": np.array(frames)},
        }
    }
    obj.prepare_split(
        training_fraction=0.6, validation_fraction=0.2, random_seed=seed
    )
    s = obj.label_dict["sub"][0]["frames_split"]
    return tuple(s["train"]), tuple(s["val"]), tuple(s["test"])


def test_prepare_split_is_reproducible_with_seed():
    frames = list(range(30))
    assert _split_with_seed(frames, 123) == _split_with_seed(frames, 123)


def test_prepare_split_seed_changes_partition():
    """A different seed must actually change the split (regression: seed was ignored)."""
    frames = list(range(30))
    assert _split_with_seed(frames, 1) != _split_with_seed(frames, 2)
