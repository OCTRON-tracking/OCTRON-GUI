"""Tests for train/val split-fraction validation.

The fraction guard now lives in core
``AnalysisOctron._validate_split_fractions``
(the single source of truth, also enforced by ``prepare_split`` for the GUI
and programmatic callers).  ``run_split`` calls the same guard up front, so
the CLI still fails before any model, label, or geometry work.
"""

import numpy as np
import pytest

from octron.analysis_octron.analysis_octron import AnalysisOctron
from octron.analysis_octron.helpers.split_report import (
    _annotated_frame_count,
    _build_frame_to_split,
    _num_frames_for,
    _segment_episodes,
    _timeline_bins,
    build_split_report,
    render_split_report,
)
from octron.analysis_octron.helpers.training import train_test_val
from octron.tools.split import run_split

# ---------------------------------------------------------------------------
# Core guard: AnalysisOctron._validate_split_fractions
# ---------------------------------------------------------------------------


def test_validate_rejects_zero_train_fraction():
    with pytest.raises(ValueError, match="training_fraction"):
        AnalysisOctron._validate_split_fractions(0.0, 0.15)


def test_validate_rejects_negative_train_fraction():
    with pytest.raises(ValueError, match="training_fraction"):
        AnalysisOctron._validate_split_fractions(-0.1, 0.15)


def test_validate_rejects_train_fraction_one():
    with pytest.raises(ValueError, match="training_fraction"):
        AnalysisOctron._validate_split_fractions(1.0, 0.0)


def test_validate_rejects_negative_val_fraction():
    with pytest.raises(ValueError, match="validation_fraction"):
        AnalysisOctron._validate_split_fractions(0.7, -0.05)


def test_validate_rejects_val_fraction_one():
    with pytest.raises(ValueError, match="validation_fraction"):
        AnalysisOctron._validate_split_fractions(0.5, 1.0)


def test_validate_rejects_sum_equal_to_one():
    """Train + val == 1 leaves no test split, which is invalid."""
    with pytest.raises(ValueError, match="must be < 1"):
        AnalysisOctron._validate_split_fractions(0.7, 0.3)


def test_validate_rejects_sum_greater_than_one():
    with pytest.raises(ValueError, match="must be < 1"):
        AnalysisOctron._validate_split_fractions(0.7, 0.4)


def test_validate_accepts_valid_fractions():
    """Valid fractions return None (no exception)."""
    assert AnalysisOctron._validate_split_fractions(0.7, 0.15) is None


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


def test_run_split_threads_prune_and_watershed(monkeypatch):
    """run_split forwards prune/watershed to prepare_labels/enable_watershed.

    Regression for the CLI/GUI parity gap (#90): run_split used to call
    prepare_labels() with no args (always pruning) and never set
    enable_watershed, so headless runs silently differed from the GUI.
    """
    recorded = {}
    instances = []

    class _FakeYolo:
        def __init__(self, **kwargs):
            self.train_mode = None
            self.enable_watershed = None
            self.label_dict = {}
            instances.append(self)

        @staticmethod
        def _validate_split_fractions(train, val):
            return None

        def prepare_labels(self, prune_empty_labels=True, **kwargs):
            recorded["prune"] = prune_empty_labels

        def prepare_geometry(self):
            return iter(())

        def prepare_split(self, **kwargs):
            pass

        def summarize_split(self):
            return []

    monkeypatch.setattr(
        "octron.analysis_octron.analysis_octron.AnalysisOctron", _FakeYolo
    )
    run_split(
        project_path="/nope_for_test",
        train_fraction=0.7,
        val_fraction=0.15,
        seed=0,
        prune=True,
        watershed=True,
        dry_run=True,
    )
    assert recorded["prune"] is True
    assert instances[0].enable_watershed is True


# ---------------------------------------------------------------------------
# prepare_split threads the seed through to train_test_val
# ---------------------------------------------------------------------------


def _split_with_seed(frames, seed):
    """Run prepare_split on a one-label fixture and return the split."""
    obj = AnalysisOctron.__new__(AnalysisOctron)
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


def _assigned_count_with_buffer(frames, buffer):
    """Run prepare_split on a one-label fixture; count assigned frames."""
    obj = AnalysisOctron.__new__(AnalysisOctron)
    obj.label_dict = {
        "sub": {
            "video": None,
            "video_file_path": None,
            0: {"label": "a", "frames": np.array(frames)},
        }
    }
    obj.prepare_split(
        training_fraction=0.6,
        validation_fraction=0.2,
        random_seed=0,
        buffer=buffer,
    )
    s = obj.label_dict["sub"][0]["frames_split"]
    return sum(len(s[k]) for k in ("train", "val", "test"))


def test_prepare_split_forwards_buffer():
    """prepare_split threads buffer through to train_test_val.

    buffer=0 keeps every frame (a single contiguous episode assigns all
    frames to some split); buffer>0 drops frames at block boundaries, so
    the assigned set shrinks. Regression: prepare_split used to ignore
    buffer and always use train_test_val's default.
    """
    frames = list(range(200))
    assert _assigned_count_with_buffer(frames, 0) == 200
    assert _assigned_count_with_buffer(frames, 1) < 200


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


def test_ttv_global_proportions_multi_episode():
    # Several episodes must not inflate val/test: the realized split should
    # track the requested 70/15/15 globally (regression: the old
    # per-episode allocation forced >=1 val + >=1 test block per episode,
    # skewing short/many-episode videos toward ~55/22/22 or worse).
    eps = [np.arange(i * 10000, i * 10000 + 300) for i in range(4)]
    frames = np.concatenate(eps)
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    tot = sum(len(s[k]) for k in ("train", "val", "test"))
    frac = {k: len(s[k]) / tot for k in ("train", "val", "test")}
    assert abs(frac["train"] - 0.70) < 0.06, frac["train"]
    assert abs(frac["val"] - 0.15) < 0.06, frac["val"]
    assert abs(frac["test"] - 0.15) < 0.06, frac["test"]


def test_ttv_valtest_spread_across_timeline():
    # Stratified selection keeps val and test spread across the whole
    # timeline rather than clustered in one region.
    frames = np.arange(2000)
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    for name in ("val", "test"):
        arr = np.sort(np.asarray(s[name], dtype=int))
        assert arr[-1] - arr[0] > 0.6 * 2000, f"{name} span {arr[-1] - arr[0]}"


def test_ttv_tiny_episode_stays_together():
    # A tiny (near-duplicate) episode is one atomic block, so both its
    # frames land in the SAME split -- never divided across train/val/test.
    main = np.arange(0, 400)
    tiny = np.array([9000, 9001])
    frames = np.concatenate([main, tiny])
    s = train_test_val(frames, 0.7, 0.15, random_seed=0)
    holders = {
        name
        for name in ("train", "val", "test")
        if {9000, 9001} & {int(x) for x in s[name]}
    }
    assert len(holders) == 1
    only = holders.pop()
    assert {9000, 9001} <= {int(x) for x in s[only]}


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
    render_split_report(
        build_split_report({"proj/sub": _labels_with_split()}), 88
    )
    out = capsys.readouterr().out
    assert "Timeline: sub" in out
    assert "200 frames, 60 assigned, 0 buffered, 1 episode(s)" in out
    assert "train" in out and "val" in out and "test" in out
    assert "unannotated" in out


def test_render_timeline_compresses_gaps(capsys):
    # Two far-apart episodes: the empty middle must collapse to an
    # ellipsis and the header must report both episodes.
    render_split_report(
        build_split_report({"proj/sub": _labels_two_episodes()}), 88
    )
    out = capsys.readouterr().out
    assert "2 episode(s)" in out
    assert "\u2026" in out  # elided gap marker


def test_annotated_frame_count_unions_labels():
    labels = {
        "video": None,
        0: {"label": "a", "frames": np.array([0, 1, 2])},
        1: {"label": "b", "frames": np.array([2, 3])},
    }
    assert _annotated_frame_count(labels) == 4  # {0, 1, 2, 3}


def test_render_timeline_reports_buffered(capsys):
    # 10 annotated frames but only 9 assigned to a split -> 1 buffered.
    labels = {
        "video": None,
        "video_file_path": None,
        0: {
            "label": "a",
            "masks": [_FakeMask(50)],
            "frames": np.arange(0, 10),
            "frames_split": {
                "train": np.array([0, 1, 2, 3, 4, 5]),
                "val": np.array([6, 7]),
                "test": np.array([8]),
            },
        },
    }
    render_split_report(build_split_report({"proj/sub": labels}), 88)
    out = capsys.readouterr().out
    assert "9 assigned, 1 buffered" in out


def test_render_timeline_noop_without_split():
    # No frames_split -> no timeline rendered, and no exception.
    render_split_report(
        build_split_report({"proj/sub": {0: {"label": "a"}}}), 88
    )


def test_build_split_report_structure():
    report = build_split_report({"proj/sub": _labels_with_split()})
    assert len(report) == 1
    sub = report[0]
    assert sub["name"] == "sub"
    assert sub["rows"] == [("a", 40, 10, 10, 60)]
    tl = sub["timeline"]
    assert tl["num_frames"] == 200
    assert tl["assigned"] == 60
    assert tl["buffered"] == 0
    assert tl["n_episodes"] == 1


def test_summarize_split_via_core_matches_build():
    # AnalysisOctron.summarize_split() is a thin wrapper over
    # build_split_report.
    obj = AnalysisOctron.__new__(AnalysisOctron)
    obj.label_dict = {"proj/sub": _labels_with_split()}
    report = obj.summarize_split()
    assert report[0]["rows"] == [("a", 40, 10, 10, 60)]
    assert report[0]["timeline"]["assigned"] == 60
