"""
Tests for input validation in octron/tools/split.py.

run_split's fraction checks fire at the top of the function, before
any heavy imports (yolo_octron) or disk I/O, so these tests are fast
and don't need a real OCTRON project on disk.

Covered
-------
run_split(train_fraction, val_fraction)
    train_fraction <= 0           → ValueError
    train_fraction >= 1           → ValueError
    val_fraction < 0              → ValueError
    val_fraction >= 1             → ValueError
    train + val == 1              → ValueError (test split needs > 0)
    train + val > 1               → ValueError
    Valid fractions (0.7 + 0.15)  → pass entry checks (later failure is OK)
"""

import pytest

from octron.tools.split import run_split


def test_run_split_rejects_zero_train_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        run_split(project_path="/nope_for_test", train_fraction=0.0, val_fraction=0.15)


def test_run_split_rejects_negative_train_fraction():
    with pytest.raises(ValueError, match="train_fraction"):
        run_split(project_path="/nope_for_test", train_fraction=-0.1, val_fraction=0.15)


def test_run_split_rejects_train_fraction_one():
    with pytest.raises(ValueError, match="train_fraction"):
        run_split(project_path="/nope_for_test", train_fraction=1.0, val_fraction=0.0)


def test_run_split_rejects_negative_val_fraction():
    with pytest.raises(ValueError, match="val_fraction"):
        run_split(project_path="/nope_for_test", train_fraction=0.7, val_fraction=-0.05)


def test_run_split_rejects_val_fraction_one():
    with pytest.raises(ValueError, match="val_fraction"):
        run_split(project_path="/nope_for_test", train_fraction=0.5, val_fraction=1.0)


def test_run_split_rejects_sum_equal_to_one():
    """train + val == 1 leaves no test split, which is invalid."""
    with pytest.raises(ValueError, match="must be < 1"):
        run_split(project_path="/nope_for_test", train_fraction=0.7, val_fraction=0.3)


def test_run_split_rejects_sum_greater_than_one():
    with pytest.raises(ValueError, match="must be < 1"):
        run_split(project_path="/nope_for_test", train_fraction=0.7, val_fraction=0.4)


def test_run_split_accepts_valid_fractions():
    """Valid fractions should pass entry checks; later failure (no project) is OK."""
    with pytest.raises(Exception) as exc:
        run_split(project_path="/nope_for_test", train_fraction=0.7, val_fraction=0.15)
    msg = str(exc.value)
    assert "train_fraction" not in msg
    assert "val_fraction" not in msg
