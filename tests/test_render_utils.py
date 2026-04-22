"""
Tests for pure utility functions in octron/tools/render.py.

Uses synthetic pos_lookup dicts ({track_id: {frame_idx: pd.Series}})
matching the production structure, so no video files, zarr stores, or
OpenCV are required.

Covered
-------
interpolate_tracklet_gaps(pos_lookup, max_gap)
    Gap smaller than max_gap → filled with interpolated frames
    Gap larger than max_gap → left untouched
    Gap exactly equal to max_gap → filled
    max_gap=0 → no-op (interpolation disabled)
    No gaps in track → nothing added
    Single-frame track → unchanged
    Interpolated midpoint values lie between the anchor endpoints
    Leading gap (before first known frame) → not filled
    Trailing gap (after last known frame) → not filled
    Multiple gaps in one track: short filled, long skipped independently
    Returns the same dict object (in-place modification)
    Multiple tracks are interpolated independently of each other
"""

import numpy as np
import pandas as pd
import pytest

from octron.tools.render import interpolate_tracklet_gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pos_lookup(frame_positions, tid=0):
    """Build a pos_lookup dict for one track from {frame_idx: (x, y)}."""
    return {
        tid: {
            f: pd.Series({"pos_x": float(x), "pos_y": float(y)})
            for f, (x, y) in frame_positions.items()
        }
    }


def _frames(pos_lookup, tid=0):
    return sorted(pos_lookup[tid].keys())


# ===========================================================================
# interpolate_tracklet_gaps
# ===========================================================================

def test_interpolate_fills_gap_within_max():
    """A gap smaller than max_gap should be fully filled."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 5: (10.0, 10.0)})
    interpolate_tracklet_gaps(lookup, max_gap=5)
    assert set(lookup[0].keys()) == {0, 1, 2, 3, 4, 5}


def test_interpolate_skips_gap_exceeding_max():
    """A gap wider than max_gap should not be touched."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 11: (10.0, 10.0)})
    interpolate_tracklet_gaps(lookup, max_gap=5)
    assert set(lookup[0].keys()) == {0, 11}


def test_interpolate_gap_exactly_at_max_is_filled():
    """A gap exactly equal to max_gap should be filled."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 6: (60.0, 0.0)})
    interpolate_tracklet_gaps(lookup, max_gap=5)
    assert set(lookup[0].keys()) == {0, 1, 2, 3, 4, 5, 6}


def test_interpolate_max_gap_zero_is_noop():
    """max_gap=0 should disable interpolation entirely."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 3: (30.0, 30.0)})
    interpolate_tracklet_gaps(lookup, max_gap=0)
    assert set(lookup[0].keys()) == {0, 3}


def test_interpolate_no_gaps_unchanged():
    """Consecutive frames have no gaps — nothing should be added."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 1: (1.0, 1.0), 2: (2.0, 2.0)})
    interpolate_tracklet_gaps(lookup, max_gap=5)
    assert set(lookup[0].keys()) == {0, 1, 2}


def test_interpolate_single_frame_unchanged():
    """A single-frame track has nothing to interpolate."""
    lookup = _make_pos_lookup({5: (100.0, 200.0)})
    interpolate_tracklet_gaps(lookup, max_gap=10)
    assert set(lookup[0].keys()) == {5}
    assert lookup[0][5]["pos_x"] == 100.0
    assert lookup[0][5]["pos_y"] == 200.0


def test_interpolate_values_lie_between_endpoints():
    """With two anchor points the spline is linear: midpoint ≈ midpoint of endpoints."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 10: (100.0, 200.0)})
    interpolate_tracklet_gaps(lookup, max_gap=15)
    mid = lookup[0][5]
    assert abs(mid["pos_x"] - 50.0) < 1.0
    assert abs(mid["pos_y"] - 100.0) < 1.0


def test_interpolate_does_not_fill_leading_gap():
    """Frames before the first known position are not created."""
    lookup = _make_pos_lookup({10: (0.0, 0.0), 20: (100.0, 0.0)})
    interpolate_tracklet_gaps(lookup, max_gap=20)
    assert min(lookup[0].keys()) == 10


def test_interpolate_does_not_fill_trailing_gap():
    """Frames after the last known position are not created."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 10: (100.0, 0.0)})
    interpolate_tracklet_gaps(lookup, max_gap=20)
    assert max(lookup[0].keys()) == 10


def test_interpolate_multiple_gaps_mixed():
    """Short gaps filled, long gaps skipped, within the same track."""
    # gap of 2 at frames 5-8, gap of 10 at frames 10-21
    lookup = _make_pos_lookup({
        0: (0.0, 0.0),
        5: (50.0, 0.0),
        8: (80.0, 0.0),
        10: (100.0, 0.0),
        21: (210.0, 0.0),
    })
    interpolate_tracklet_gaps(lookup, max_gap=3)
    filled = set(lookup[0].keys())
    # Gap 5→8 (gap=2, ≤3): frames 6,7 filled
    assert 6 in filled
    assert 7 in filled
    # Gap 10→21 (gap=10, >3): not filled
    assert 11 not in filled
    assert 20 not in filled


def test_interpolate_returns_same_dict():
    """The function modifies pos_lookup in-place and returns the same object."""
    lookup = _make_pos_lookup({0: (0.0, 0.0), 5: (50.0, 50.0)})
    result = interpolate_tracklet_gaps(lookup, max_gap=10)
    assert result is lookup


def test_interpolate_multiple_tracks_independent():
    """Each track is interpolated independently."""
    lookup = {
        1: {0: pd.Series({"pos_x": 0.0, "pos_y": 0.0}),
            5: pd.Series({"pos_x": 50.0, "pos_y": 0.0})},
        2: {0: pd.Series({"pos_x": 0.0, "pos_y": 0.0}),
            20: pd.Series({"pos_x": 200.0, "pos_y": 0.0})},
    }
    interpolate_tracklet_gaps(lookup, max_gap=6)
    # Track 1: gap of 4 → filled
    assert 2 in lookup[1]
    # Track 2: gap of 19 > 6 → not filled
    assert 1 not in lookup[2]
