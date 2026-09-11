"""Tests for empty-label pruning and how it interacts with the split.

Pruning happens in two stages, both gated by ``prune_empty_labels``:

1. ``collect_labels`` intersects each label's annotated-frame set via
   ``find_common_frames`` (keep only frames where *all* labels are
   annotated).
2. ``prepare_polygons``/``prepare_bboxes`` drop frames whose geometry came
   out empty via ``prune_frames_by_geometry`` (cross-label intersection
   when pruning, else per label).

The pruned per-label ``frames`` then feed ``prepare_split``, which decides
the split once per frame over the union of all labels' frames -- so a
frame shared by several labels lands in the same split for every label
(no cross-label leakage) regardless of pruning; pruning only controls
which frames are in the pool.
"""

import numpy as np
import pytest

from octron.analysis_octron.analysis_octron import AnalysisOctron
from octron.analysis_octron.helpers.training import (
    find_common_frames,
    prune_frames_by_geometry,
)

# Any non-empty geometry value marks a frame as "valid" (a polygon/bbox
# was produced); an empty list marks it invalid and prunable.
_GEOM = [[(0, 0), (1, 1), (2, 0)]]


# ---------------------------------------------------------------------------
# Stage 1 primitive: find_common_frames (cross-label frame intersection)
# ---------------------------------------------------------------------------


def test_find_common_frames_intersection():
    a = np.arange(0, 100)
    b = np.arange(50, 150)
    assert list(find_common_frames([a, b])) == list(range(50, 100))


def test_find_common_frames_single_array_unchanged():
    a = np.array([3, 1, 2])
    assert list(find_common_frames([a])) == [3, 1, 2]


def test_find_common_frames_disjoint_is_empty():
    a = np.array([0, 1, 2])
    b = np.array([5, 6, 7])
    assert len(find_common_frames([a, b])) == 0


# ---------------------------------------------------------------------------
# Stage 2: prune_frames_by_geometry (drop frames with empty polygons/bboxes)
# ---------------------------------------------------------------------------


def _labels_with_geometry(key):
    """Two-label subfolder where each label is invalid on a different frame.

    Valid A = {0, 1, 3} (frame 2 empty); valid B = {0, 2, 3} (frame 1
    empty). Cross-label intersection is {0, 3}.
    """
    return {
        "video": None,
        "video_file_path": None,
        0: {
            "label": "a",
            "frames": np.array([0, 1, 2, 3]),
            key: {0: _GEOM, 1: _GEOM, 2: [], 3: _GEOM},
        },
        1: {
            "label": "b",
            "frames": np.array([0, 1, 2, 3]),
            key: {0: _GEOM, 1: [], 2: _GEOM, 3: _GEOM},
        },
    }


@pytest.mark.parametrize("key", ["polygons", "bboxes"])
def test_prune_by_geometry_cross_label_intersection(key):
    labels = _labels_with_geometry(key)
    prune_frames_by_geometry(labels, True, key)
    # Both labels collapse to the common valid set {0, 3}.
    assert list(labels[0]["frames"]) == [0, 3]
    assert list(labels[1]["frames"]) == [0, 3]


@pytest.mark.parametrize("key", ["polygons", "bboxes"])
def test_prune_by_geometry_per_label(key):
    labels = _labels_with_geometry(key)
    prune_frames_by_geometry(labels, False, key)
    # Without pruning each label keeps its own valid frames.
    assert list(labels[0]["frames"]) == [0, 1, 3]
    assert list(labels[1]["frames"]) == [0, 2, 3]


def test_prune_by_geometry_noop_when_all_valid():
    labels = {
        "video": None,
        "video_file_path": None,
        0: {
            "label": "a",
            "frames": np.array([0, 1]),
            "polygons": {0: _GEOM, 1: _GEOM},
        },
    }
    prune_frames_by_geometry(labels, True, "polygons")
    assert list(labels[0]["frames"]) == [0, 1]


# ---------------------------------------------------------------------------
# Interaction: pruning + split fractions
# ---------------------------------------------------------------------------


def _split_labels(label_dict):
    """Run prepare_split on a prebuilt label_dict and return it."""
    obj = AnalysisOctron.__new__(AnalysisOctron)
    obj.label_dict = label_dict
    obj.prepare_split(
        training_fraction=0.7, validation_fraction=0.15, random_seed=0
    )
    return obj.label_dict


def test_prune_reduces_pool_and_unifies_splits():
    # Label A annotated on 0-799, label B only on 0-399. Stage-1 pruning
    # intersects to 0-399, so BOTH labels split the SAME 400-frame pool:
    # identical, leakage-free per-label splits, and the fractions apply to
    # the reduced pool (~70/15/15).
    common = find_common_frames([np.arange(0, 800), np.arange(0, 400)])
    assert list(common) == list(range(400))

    ld = _split_labels(
        {
            "sub": {
                "video": None,
                "video_file_path": None,
                0: {"label": "a", "frames": common},
                1: {"label": "b", "frames": common},
            }
        }
    )
    sa = ld["sub"][0]["frames_split"]
    sb = ld["sub"][1]["frames_split"]
    # Same frame -> same split for every label (no cross-label leakage).
    for k in ("train", "val", "test"):
        assert list(map(int, sa[k])) == list(map(int, sb[k]))
    # Drawn only from the pruned 0-399 pool, and ~70/15/15 of it.
    assigned = {int(x) for k in ("train", "val", "test") for x in sa[k]}
    assert assigned <= set(range(400))
    tot = len(assigned)
    assert abs(len(sa["train"]) / tot - 0.70) < 0.06
    assert abs(len(sa["val"]) / tot - 0.15) < 0.06
    assert abs(len(sa["test"]) / tot - 0.15) < 0.06


def _frame_to_split_map(split_dict):
    """Map each assigned frame -> its split name for one label."""
    mapping = {}
    for name in ("train", "val", "test"):
        for f in split_dict[name]:
            mapping[int(f)] = name
    return mapping


def test_no_prune_shared_frames_get_one_split():
    # Without pruning, labels keep their own (different) frames, but the
    # split is decided per FRAME over their union: a frame shared by both
    # labels lands in the SAME split for each (no cross-label leakage),
    # and each label's split stays within its own frames.
    ld = _split_labels(
        {
            "sub": {
                "video": None,
                "video_file_path": None,
                0: {"label": "a", "frames": np.arange(0, 100)},
                1: {"label": "b", "frames": np.arange(50, 150)},
            }
        }
    )
    ma = _frame_to_split_map(ld["sub"][0]["frames_split"])
    mb = _frame_to_split_map(ld["sub"][1]["frames_split"])
    # Each label's assigned frames stay within its own annotated frames.
    assert set(ma) <= set(range(0, 100))
    assert set(mb) <= set(range(50, 150))
    # Frames shared by both labels get the same split in both.
    shared = set(ma) & set(mb)
    assert shared
    for f in shared:
        assert ma[f] == mb[f]
