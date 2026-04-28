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

Input validation
----------------
_validate_render_args(preset, min_confidence, alpha)
    Unknown preset → ValueError naming valid options
    min_confidence outside [0, 1] → ValueError
    alpha outside [0, 1] → ValueError
    Boundary values (0.0 and 1.0) accepted
_coerce_track_ids(track_ids)
    None → None
    Comma-separated string → list[int]
    Single int → single-element list
    Iterable of ints → list[int]
    Malformed string / mixed types / float → ValueError
run_tracklets(offset=...) [contract enforcement]
    Wrong arity / non-numeric / non-iterable offset → ValueError before any I/O
    Valid (int, int) / (float, float) / list passes the entry check
"""

import numpy as np
import pandas as pd
import pytest

from octron.tools.render import (
    _coerce_track_ids,
    _validate_render_args,
    interpolate_tracklet_gaps,
    run_tracklets,
)


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


# ---------------------------------------------------------------------------
# _validate_render_args
# ---------------------------------------------------------------------------

def test_validate_render_args_rejects_unknown_preset():
    with pytest.raises(ValueError, match="preset must be one of"):
        _validate_render_args("ultra", min_confidence=0.5, alpha=0.4)


def test_validate_render_args_rejects_min_confidence_above_one():
    with pytest.raises(ValueError, match="min_confidence"):
        _validate_render_args("draft", min_confidence=1.5, alpha=0.4)


def test_validate_render_args_rejects_min_confidence_below_zero():
    with pytest.raises(ValueError, match="min_confidence"):
        _validate_render_args("draft", min_confidence=-0.1, alpha=0.4)


def test_validate_render_args_rejects_alpha_above_one():
    with pytest.raises(ValueError, match="alpha"):
        _validate_render_args("draft", min_confidence=0.5, alpha=2.0)


def test_validate_render_args_rejects_alpha_below_zero():
    with pytest.raises(ValueError, match="alpha"):
        _validate_render_args("draft", min_confidence=0.5, alpha=-0.5)


def test_validate_render_args_accepts_boundary_values():
    """0.0 and 1.0 are inclusive; all three presets are valid."""
    for preset in ("preview", "draft", "final"):
        _validate_render_args(preset, min_confidence=0.0, alpha=0.0)
        _validate_render_args(preset, min_confidence=1.0, alpha=1.0)


# ---------------------------------------------------------------------------
# _coerce_track_ids
# ---------------------------------------------------------------------------

def test_coerce_track_ids_none():
    assert _coerce_track_ids(None) is None


def test_coerce_track_ids_string():
    assert _coerce_track_ids("1,3,5") == [1, 3, 5]


def test_coerce_track_ids_string_with_whitespace():
    assert _coerce_track_ids(" 1 , 3 , 5 ") == [1, 3, 5]


def test_coerce_track_ids_string_with_trailing_comma():
    """Trailing/empty fields are dropped (matches CLI behaviour)."""
    assert _coerce_track_ids("1,3,") == [1, 3]


def test_coerce_track_ids_single_int():
    assert _coerce_track_ids(7) == [7]


def test_coerce_track_ids_list_of_ints():
    assert _coerce_track_ids([1, 2, 3]) == [1, 2, 3]


def test_coerce_track_ids_tuple_of_ints():
    assert _coerce_track_ids((1, 2, 3)) == [1, 2, 3]


def test_coerce_track_ids_rejects_malformed_string():
    with pytest.raises(ValueError, match="track_ids string"):
        _coerce_track_ids("foo")


def test_coerce_track_ids_rejects_mixed_iterable():
    with pytest.raises(ValueError, match="track_ids"):
        _coerce_track_ids([1, "bad"])


def test_coerce_track_ids_rejects_float():
    with pytest.raises(ValueError, match="track_ids"):
        _coerce_track_ids(1.5)


# ---------------------------------------------------------------------------
# run_tracklets — offset contract enforcement
# ---------------------------------------------------------------------------
#
# These tests rely on offset validation firing BEFORE any I/O or heavy
# imports inside run_tracklets, so a non-existent predictions_path is fine.

@pytest.mark.parametrize("bad_offset", [
    "20,-30",       # string, not a tuple
    (1,),           # wrong arity
    (1, 2, 3),      # wrong arity
    ("x", "y"),     # non-numeric
    None,           # not iterable in the (a, b) sense
])
def test_run_tracklets_rejects_bad_offset(bad_offset):
    with pytest.raises(ValueError, match="offset"):
        run_tracklets(predictions_path="/nope_for_test", offset=bad_offset)


@pytest.mark.parametrize("good_offset", [
    (0, 0),
    (1, -2),
    (1.0, 2.0),     # floats coerced to int
    [3, 4],         # list also accepted
])
def test_run_tracklets_accepts_valid_offset(good_offset):
    """Offset validation passes; later code fails because the path doesn't exist.

    We only want to confirm the entry-check doesn't reject these — any later
    failure (FileNotFoundError, etc.) is fine.
    """
    with pytest.raises(Exception) as exc:
        run_tracklets(predictions_path="/nope_for_test", offset=good_offset)
    # Make sure it did NOT fail at the offset check
    assert "offset" not in str(exc.value).lower()
