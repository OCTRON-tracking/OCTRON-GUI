"""Tests for train/val split-fraction validation.

The fraction guard now lives in core ``YOLO_octron._validate_split_fractions``
(the single source of truth, also enforced by ``prepare_split`` for the GUI and
programmatic callers).  ``run_split`` calls the same guard up front, so the CLI
still fails before any model, label, or geometry work.
"""

import numpy as np
import pytest

from octron.tools.split import (
    _build_frame_to_split,
    _num_frames_for,
    _print_split_timelines,
    _segment_episodes,
    _timeline_bins,
    run_split,
)
from octron.yolo_octron.helpers.training import train_test_val
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
    """Valid fractions pass the guard; a later missing-project failure
    is OK.
    """
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
    """Run prepare_split on a one-label fixture and return the split."""
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
    frames = list(range(300))
    assert _split_with_seed(frames, 123) == _split_with_seed(frames, 123)


def test_prepare_split_seed_changes_partition():
    """A different seed must change the split (regression: seed was
    ignored).
    """
    frames = list(range(300))
    assert _split_with_seed(frames, 1) != _split_with_seed(frames, 2)


# ---------------------------------------------------------------------------
# train_test_val: contiguous-block split behaviour
# ---------------------------------------------------------------------------


def test_ttv_partitions_disjoint_subset_nonempty():
    frames = np.arange(500)
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    train = {int(x) for x in s["train"]}
    val = {int(x) for x in s["val"]}
    test = {int(x) for x in s["test"]}
    assert train and val and test
    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert (train | val | test).issubset(set(frames.tolist()))


def test_ttv_no_adjacent_train_val_leakage():
    # With the default 1-frame buffer, no train frame may sit immediately
    # next to a val frame (that would be a near-duplicate across the split).
    frames = np.arange(500)
    s = train_test_val(
        frames, 0.7, 0.15, random_seed=0, block_size=20, buffer=1
    )
    train = {int(x) for x in s["train"]}
    val = {int(x) for x in s["val"]}
    for v in val:
        assert (v - 1) not in train
        assert (v + 1) not in train


def test_ttv_keeps_runs_together():
    # Contiguous-block assignment keeps adjacent frames on the same side:
    # split crossings scale with the number of blocks, not with n (as a
    # per-frame random split would).
    frames = np.arange(600)
    s = train_test_val(frames, 0.7, 0.15, random_seed=0, buffer=0)
    label = {}
    for name in ("train", "val", "test"):
        for f in s[name]:
            label[int(f)] = name
    crossings = sum(
        1
        for f in range(599)
        if f in label and (f + 1) in label and label[f] != label[f + 1]
    )
    assert crossings <= 40


def test_ttv_reproducible_and_seed_sensitive():
    frames = np.arange(500)
    a = [x.tolist() for x in train_test_val(frames, 0.7, 0.15, 1).values()]
    b = [x.tolist() for x in train_test_val(frames, 0.7, 0.15, 1).values()]
    c = [x.tolist() for x in train_test_val(frames, 0.7, 0.15, 2).values()]
    assert a == b
    assert a != c


def test_ttv_minimum_three_frames():
    s = train_test_val(np.array([0, 1, 2]), 0.6, 0.2, random_seed=0)
    assert len(s["train"]) == 1
    assert len(s["val"]) == 1
    assert len(s["test"]) == 1


def test_ttv_episodes_are_balanced_across_splits():
    # Two annotation bursts in one video separated by a large gap: each
    # episode must appear in every split (not dominated by the denser one).
    ep1 = np.arange(0, 100)
    ep2 = np.arange(5000, 5040)
    frames = np.concatenate([ep1, ep2])
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    ep1_set = set(ep1.tolist())
    ep2_set = set(ep2.tolist())
    for split in ("train", "val", "test"):
        present = {int(x) for x in s[split]}
        assert present & ep1_set, f"episode 1 missing from {split}"
        assert present & ep2_set, f"episode 2 missing from {split}"


def test_ttv_tiny_episode_goes_to_train_only():
    main = np.arange(0, 100)
    tiny = np.array([9000, 9001])  # 2 frames: too small to split 3 ways
    frames = np.concatenate([main, tiny])
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    train = {int(x) for x in s["train"]}
    val = {int(x) for x in s["val"]}
    test = {int(x) for x in s["test"]}
    assert {9000, 9001} <= train
    assert not ({9000, 9001} & (val | test))


# ---------------------------------------------------------------------------
# Terminal split-timeline visualization helpers
# ---------------------------------------------------------------------------


class _FakeMask:
    """Minimal stand-in for a zarr mask array (only ``.shape`` is used)."""

    def __init__(self, num_frames):
        self.shape = (num_frames, 4, 4)


def _labels_with_split(num_frames=200):
    """One-label subfolder dict shaped like collect_labels output."""
    return {
        "video": None,
        "video_file_path": None,
        0: {
            "label": "a",
            "masks": [_FakeMask(num_frames)],
            "frames": np.arange(0, 60),
            "frames_split": {
                "train": np.arange(0, 40),
                "val": np.arange(40, 50),
                "test": np.arange(50, 60),
            },
        },
    }


def test_build_frame_to_split_aggregates_and_skips_meta():
    mapping = _build_frame_to_split(_labels_with_split())
    assert mapping[0] == "train"
    assert mapping[45] == "val"
    assert mapping[59] == "test"
    # Meta keys must never appear as frames.
    assert all(isinstance(k, int) for k in mapping)
    assert len(mapping) == 60


def test_num_frames_prefers_mask_shape():
    labels = _labels_with_split(num_frames=321)
    assert _num_frames_for(labels, _build_frame_to_split(labels)) == 321


def test_num_frames_falls_back_to_max_index():
    labels = {0: {"label": "a", "frames_split": {"train": [3, 7]}}}
    assert _num_frames_for(labels, _build_frame_to_split(labels)) == 8


def test_timeline_bins_dominant_and_empty():
    # Frames only in the first half -> back half must be unannotated (None).
    frame_to_split = {i: "train" for i in range(50)}
    bins = _timeline_bins(frame_to_split, num_frames=100, width=10)
    assert len(bins) == 10
    assert bins[0] == "train"
    assert bins[-1] is None


def _labels_two_episodes(num_frames=10000):
    """Two annotation bursts separated by a large unannotated gap."""
    ep1 = np.arange(0, 100)
    ep2 = np.arange(5000, 5050)
    frames = np.concatenate([ep1, ep2])
    n = len(frames)
    return {
        "video": None,
        "video_file_path": None,
        0: {
            "label": "a",
            "masks": [_FakeMask(num_frames)],
            "frames": frames,
            "frames_split": {
                "train": frames[: int(n * 0.7)],
                "val": frames[int(n * 0.7) : int(n * 0.85)],
                "test": frames[int(n * 0.85) :],
            },
        },
    }


def test_segment_episodes_splits_on_large_gap():
    assert _segment_episodes([0, 1, 2, 100, 101], gap=10) == [
        [0, 1, 2],
        [100, 101],
    ]


def test_segment_episodes_single_when_dense():
    assert _segment_episodes([0, 3, 6, 9], gap=10) == [[0, 3, 6, 9]]


def test_render_timeline_smoke(capsys):
    _print_split_timelines({"proj/sub": _labels_with_split()})
    out = capsys.readouterr().out
    assert "Timeline: sub" in out
    assert "200 frames, 60 annotated, 1 episode(s)" in out
    assert "train" in out and "val" in out and "test" in out
    assert "unannotated" in out


def test_render_timeline_compresses_gaps(capsys):
    # Two far-apart episodes: the empty middle must collapse to an
    # ellipsis and the header must report both episodes.
    _print_split_timelines({"proj/sub": _labels_two_episodes()})
    out = capsys.readouterr().out
    assert "2 episode(s)" in out
    assert "\u2026" in out  # elided gap marker


def test_render_timeline_noop_without_split():
    # No frames_split -> nothing to render, and no exception.
    _print_split_timelines({"proj/sub": {0: {"label": "a"}}})
