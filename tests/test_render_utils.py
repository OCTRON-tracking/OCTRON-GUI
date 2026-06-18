"""
Tests for pure utility functions in octron/tools/render.py.

Covered
-------
_validate_render_args(preset, min_confidence, alpha)
    Unknown preset / out-of-range min_confidence / out-of-range alpha -> ValueError
    Boundary values (0.0 and 1.0) accepted
_coerce_track_ids(track_ids)
    None / comma string / single int / iterable of ints -> list[int] or None
    Malformed string / mixed types / float -> ValueError
run_tracklets(offset=...) [contract enforcement]
    Wrong arity / non-numeric / non-iterable offset -> ValueError before any I/O
    Valid (int, int) / (float, float) / list passes the entry check

Tracklet centroid smoothing and gap interpolation now live in core
``YOLO_results.get_tracking_data`` (shared by the CLI render path and the
napari GUI) and are covered in tests/test_smooth_tracklets.py.
"""

import pytest

from octron.tools.render import (
    _coerce_track_ids,
    _validate_render_args,
    run_tracklets,
)


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
