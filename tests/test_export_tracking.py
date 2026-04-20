"""
Tests for octron/tools/export_tracking.py.

No video files, zarr archives, or GPU are required — all tests use synthetic
in-memory data or temporary CSV files.

Column contract
---------------
All tuple-valued columns are resolved to scalars in the output.  Original
column names (pos_x, pos_y, area, …) are kept — no renaming.

  pos_x / pos_y : scalar centroid per chosen method
  area          : largest segment's area ("largest") OR sum of all segments
                  ("weighted" / "mask_com")
  orientation   : always from the largest segment (circular quantity)
  all other regionprops : resolved by chosen method

Covered
-------
_parse_tuple_or_scalar
    scalar int/float returned as single-element tuple
    tuple-string parsed correctly
    NaN / unparseable value returned as (nan,)

_resolve_xy_largest
    single-segment rows: returns the scalar position unchanged
    multi-segment rows: returns position of the largest segment

_resolve_xy_weighted
    single-segment: result equals the scalar position
    multi-segment with equal areas: result is the simple mean
    multi-segment: result is between the two endpoints
    zero-area segments: no divide-by-zero

_resolve_prop_largest / _resolve_prop_weighted
    scalar passthrough
    multi-segment: correct value selected / weighted

_sum_segments_area
    scalar rows: result equals the scalar
    tuple rows: result equals the arithmetic sum

_build_header / _read_csv_metadata
    round-trip: written header parsed back to same dict

_export_tracking_from_data
    output file(s) created
    pos_x / pos_y columns present and numeric
    area column present and numeric
    no centroid_x / centroid_y / segments_x / segments_y columns in output
    "largest": single-segment pos_x == raw input
    "largest": multi-segment pos_x == x of biggest segment
    "largest": area == area of biggest segment
    "weighted": pos_x between segment endpoints
    "weighted": area == sum of segment areas
    orientation always resolved via largest regardless of method
    other regionprops resolved via chosen method
    combined=False → one file per track
    combined=True  → single all_tracks.csv
    mask_com with zarr_root=None falls back gracefully
    header metadata preserved in output

export_tracking (public)
    reads CSV, writes new CSV with correct column names
    "largest" / "weighted" methods forwarded correctly
    combined flag forwarded correctly
    raises FileNotFoundError on missing CSVs
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from octron.tools.export_tracking import (
    _build_header,
    _export_tracking_from_data,
    _parse_tuple_or_scalar,
    _read_csv_metadata,
    _resolve_prop_largest,
    _resolve_prop_weighted,
    _resolve_xy_largest,
    _resolve_xy_weighted,
    _sum_segments_area,
    export_tracking,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _scalar_arr(*vals):
    return np.array(vals, dtype=object)


def _tup(t):
    """Produce a tuple-string exactly as predict_batch writes it."""
    return str(t)


def _minimal_data(n=4, n_tracks=1):
    """
    Return the positional args needed by _export_tracking_from_data
    (all single-segment scalar rows, no tuples).
    """
    tids   = list(range(n_tracks))
    labels = {tid: "animal" for tid in tids}
    fc     = {tid: np.arange(n) for tid in tids}
    fi     = {tid: np.arange(n) for tid in tids}
    conf   = {tid: np.full(n, 0.9) for tid in tids}
    sx     = {tid: np.arange(n, dtype=float) for tid in tids}
    sy     = {tid: np.arange(n, dtype=float) for tid in tids}
    sa     = {tid: np.full(n, 100.0) for tid in tids}
    bbox   = {tid: {
        "bbox_x_min": np.zeros(n), "bbox_x_max": np.ones(n),
        "bbox_y_min": np.zeros(n), "bbox_y_max": np.ones(n),
        "bbox_area":  np.ones(n),  "bbox_aspect_ratio": np.ones(n),
    } for tid in tids}
    rp     = {tid: {} for tid in tids}
    meta   = {
        "video_name": "test.mp4", "frame_count": "100",
        "frame_count_analyzed": str(n), "video_height": "480",
        "video_width": "640",   "created_at": "2026-01-01 00:00:00",
    }
    return tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta


def _read_output(tmp_path, filename="animal_track_0.csv"):
    return pd.read_csv(tmp_path / filename, skiprows=7)


# ===========================================================================
# _parse_tuple_or_scalar
# ===========================================================================

def test_parse_scalar_int():
    assert _parse_tuple_or_scalar(5) == (5.0,)

def test_parse_scalar_float():
    assert _parse_tuple_or_scalar(3.14) == pytest.approx((3.14,))

def test_parse_two_element_tuple_string():
    result = _parse_tuple_or_scalar("(10.5, 20.3)")
    assert len(result) == 2
    assert result[0] == pytest.approx(10.5)
    assert result[1] == pytest.approx(20.3)

def test_parse_single_element_tuple_string():
    assert _parse_tuple_or_scalar("(42.0,)") == pytest.approx((42.0,))

def test_parse_nan_string():
    result = _parse_tuple_or_scalar("NaN")
    assert len(result) == 1 and math.isnan(result[0])

def test_parse_unparseable_string():
    result = _parse_tuple_or_scalar("garbage")
    assert len(result) == 1 and math.isnan(result[0])


# ===========================================================================
# _resolve_xy_largest
# ===========================================================================

def test_largest_single_segment_passthrough():
    xs = _scalar_arr(10.0, 20.0)
    ys = _scalar_arr( 5.0, 15.0)
    sa = _scalar_arr(100.0, 200.0)
    cx, cy = _resolve_xy_largest(xs, ys, sa)
    assert cx == pytest.approx([10.0, 20.0])
    assert cy == pytest.approx([ 5.0, 15.0])

def test_largest_picks_first_when_biggest():
    cx, _ = _resolve_xy_largest(
        _scalar_arr(_tup((10.0, 90.0))),
        _scalar_arr(_tup(( 5.0, 50.0))),
        _scalar_arr(_tup((200.0, 50.0))),
    )
    assert cx[0] == pytest.approx(10.0)

def test_largest_picks_second_when_biggest():
    cx, cy = _resolve_xy_largest(
        _scalar_arr(_tup((10.0, 90.0))),
        _scalar_arr(_tup(( 5.0, 50.0))),
        _scalar_arr(_tup(( 50.0, 300.0))),
    )
    assert cx[0] == pytest.approx(90.0)
    assert cy[0] == pytest.approx(50.0)


# ===========================================================================
# _resolve_xy_weighted
# ===========================================================================

def test_weighted_single_segment_passthrough():
    cx, cy = _resolve_xy_weighted(
        _scalar_arr(10.0), _scalar_arr(5.0), _scalar_arr(100.0)
    )
    assert cx[0] == pytest.approx(10.0)
    assert cy[0] == pytest.approx(5.0)

def test_weighted_equal_areas_gives_mean():
    cx, cy = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0, 200.0))),
        _scalar_arr(_tup((50.0, 50.0))),
    )
    assert cx[0] == pytest.approx(50.0)
    assert cy[0] == pytest.approx(100.0)

def test_weighted_result_between_endpoints():
    cx, _ = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0,   0.0))),
        _scalar_arr(_tup((75.0, 25.0))),
    )
    assert 0.0 < cx[0] < 100.0

def test_weighted_zero_area_no_error():
    cx, _ = _resolve_xy_weighted(
        _scalar_arr(_tup((0.0, 100.0))),
        _scalar_arr(_tup((0.0,   0.0))),
        _scalar_arr(_tup((0.0,   0.0))),
    )
    assert not math.isnan(cx[0])


# ===========================================================================
# _resolve_prop_largest / _resolve_prop_weighted
# ===========================================================================

def test_resolve_prop_largest_scalar_passthrough():
    out = _resolve_prop_largest(_scalar_arr(0.8, 0.5), _scalar_arr(100.0, 200.0))
    assert out == pytest.approx([0.8, 0.5])

def test_resolve_prop_largest_picks_correct_index():
    # areas (50, 300) → index 1 is largest
    out = _resolve_prop_largest(
        _scalar_arr(_tup((0.8, 0.4))),
        _scalar_arr(_tup((50.0, 300.0))),
    )
    assert out[0] == pytest.approx(0.4)

def test_resolve_prop_weighted_scalar_passthrough():
    out = _resolve_prop_weighted(_scalar_arr(0.6, 0.9), _scalar_arr(100.0, 200.0))
    assert out == pytest.approx([0.6, 0.9])

def test_resolve_prop_weighted_equal_areas_is_mean():
    out = _resolve_prop_weighted(
        _scalar_arr(_tup((0.0, 1.0))),
        _scalar_arr(_tup((50.0, 50.0))),
    )
    assert out[0] == pytest.approx(0.5)


# ===========================================================================
# _sum_segments_area
# ===========================================================================

def test_sum_area_scalar():
    out = _sum_segments_area(_scalar_arr(100.0, 200.0))
    assert out == pytest.approx([100.0, 200.0])

def test_sum_area_tuples():
    out = _sum_segments_area(_scalar_arr(_tup((200.0, 50.0)), _tup((10.0, 10.0, 5.0))))
    assert out[0] == pytest.approx(250.0)
    assert out[1] == pytest.approx(25.0)


# ===========================================================================
# _build_header / _read_csv_metadata  round-trip
# ===========================================================================

def test_header_round_trip(tmp_path):
    meta = {
        "video_name": "myvideo.mp4", "frame_count": "500",
        "frame_count_analyzed": "250", "video_height": "720",
        "video_width": "1280", "created_at": "2026-01-01 12:00:00",
    }
    p = tmp_path / "track.csv"
    p.write_text(_build_header(meta) + "col1,col2\n1,2\n")
    recovered = _read_csv_metadata(p, n_header_lines=7)
    for key in meta:
        assert recovered[key] == meta[key]


# ===========================================================================
# _export_tracking_from_data — column contract
# ===========================================================================

def test_export_output_has_pos_x_pos_y_area(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    for col in ("pos_x", "pos_y", "area"):
        assert col in df.columns, f"Missing: {col}"

def test_export_no_centroid_or_segments_columns(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    for col in ("centroid_x", "centroid_y", "segments_x", "segments_y", "segments_area"):
        assert col not in df.columns, f"Unexpected column present: {col}"

def test_export_pos_columns_are_numeric(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert pd.api.types.is_float_dtype(df["pos_x"])
    assert pd.api.types.is_float_dtype(df["pos_y"])
    assert pd.api.types.is_float_dtype(df["area"])

def test_export_area_is_numeric(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(), centroid_method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert pd.api.types.is_float_dtype(df["area"])


# ===========================================================================
# _export_tracking_from_data — centroid_method="largest"
# ===========================================================================

def test_largest_single_segment_pos_equals_input(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=4)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    np.testing.assert_allclose(df["pos_x"].values, sx[0].astype(float))

def test_largest_multi_segment_pos_x_picks_biggest(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    # Second segment has larger area (300 vs 50) → pos_x should be 90
    sx[0] = np.array([_tup((10.0, 90.0))], dtype=object)
    sy[0] = np.array([_tup(( 5.0, 50.0))], dtype=object)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["pos_x"].iloc[0] == pytest.approx(90.0)

def test_largest_area_is_area_of_biggest_segment(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["area"].iloc[0] == pytest.approx(300.0)


# ===========================================================================
# _export_tracking_from_data — centroid_method="weighted"
# ===========================================================================

def test_weighted_pos_x_between_segments(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sx[0] = np.array([_tup((0.0, 100.0))], dtype=object)
    sy[0] = np.array([_tup((0.0,   0.0))], dtype=object)
    sa[0] = np.array([_tup((75.0, 25.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert 0.0 < df["pos_x"].iloc[0] < 100.0

def test_weighted_area_is_sum_of_segments(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
    sa[0] = np.array([_tup((200.0, 50.0))], dtype=object)
    sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="weighted",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert df["area"].iloc[0] == pytest.approx(250.0)


# ===========================================================================
# _export_tracking_from_data — regionprops resolution
# ===========================================================================

def test_regionprop_tuples_resolved_to_scalar(tmp_path):
    tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=2)
    sa[0] = np.array([_tup((200.0, 50.0)), 100.0], dtype=object)
    rp[0] = {"solidity": np.array([_tup((0.9, 0.4)), 0.8], dtype=object)}
    _export_tracking_from_data(tmp_path, tids, labels, fc, fi, conf, sx, sy, sa,
                                bbox, rp, meta, centroid_method="largest",
                                zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert "solidity" in df.columns
    assert pd.api.types.is_float_dtype(df["solidity"])
    # largest segment has area 200 (index 0) → solidity should be 0.9
    assert df["solidity"].iloc[0] == pytest.approx(0.9)

def test_orientation_always_uses_largest_regardless_of_method(tmp_path):
    """orientation is circular — must always come from the largest segment."""
    for method in ("largest", "weighted"):
        out = tmp_path / method
        out.mkdir()
        tids, labels, fc, fi, conf, sx, sy, sa, bbox, rp, meta = _minimal_data(n=1)
        sa[0] = np.array([_tup((50.0, 300.0))], dtype=object)
        sx[0] = np.array([_tup((0.0, 0.0))], dtype=object)
        sy[0] = np.array([_tup((0.0, 0.0))], dtype=object)
        # orientation values differ per segment
        rp[0] = {"orientation": np.array([_tup((1.0, 2.0))], dtype=object)}
        _export_tracking_from_data(out, tids, labels, fc, fi, conf, sx, sy, sa,
                                    bbox, rp, meta, centroid_method=method,
                                    zarr_root=None, combined=False)
        df = pd.read_csv(out / "animal_track_0.csv", skiprows=7)
        # second segment is largest (area 300 → index 1 → orientation 2.0)
        assert df["orientation"].iloc[0] == pytest.approx(2.0), \
            f"orientation wrong for method={method}"


# ===========================================================================
# _export_tracking_from_data — file-writing
# ===========================================================================

def test_export_per_track_files(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=2),
                                centroid_method="largest", zarr_root=None, combined=False)
    assert (tmp_path / "animal_track_0.csv").exists()
    assert (tmp_path / "animal_track_1.csv").exists()

def test_export_combined_file(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=2),
                                centroid_method="largest", zarr_root=None, combined=True)
    assert (tmp_path / "all_tracks.csv").exists()
    assert not (tmp_path / "animal_track_0.csv").exists()

def test_export_combined_contains_all_tracks(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(n_tracks=3),
                                centroid_method="largest", zarr_root=None, combined=True)
    df = pd.read_csv(tmp_path / "all_tracks.csv", skiprows=7)
    assert set(df["track_id"].unique()) == {0, 1, 2}

def test_export_mask_com_fallback_no_error(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(),
                                centroid_method="mask_com", zarr_root=None, combined=False)
    df = _read_output(tmp_path)
    assert "pos_x" in df.columns
    assert not df["pos_x"].isna().all()

def test_export_header_preserved(tmp_path):
    _export_tracking_from_data(tmp_path, *_minimal_data(),
                                centroid_method="largest", zarr_root=None, combined=False)
    meta = _read_csv_metadata(tmp_path / "animal_track_0.csv")
    assert meta["video_name"] == "test.mp4"
    assert meta["video_height"] == "480"


# ===========================================================================
# export_tracking  (public)
# ===========================================================================

def _write_prediction_csv(path, track_id, label="animal", n=4):
    """Write a CSV in the format produced by _export_tracking_from_data."""
    from octron.tools.export_tracking import _build_header
    meta = {
        "video_name": "vid.mp4", "frame_count": "100",
        "frame_count_analyzed": str(n), "video_height": "480",
        "video_width": "640",   "created_at": "2026-01-01 00:00:00",
    }
    df = pd.DataFrame({
        "frame_counter":    np.arange(n),
        "frame_idx":        np.arange(n),
        "track_id":         track_id,
        "label":            label,
        "confidence":       0.9,
        "pos_x":            np.arange(n, dtype=float),
        "pos_y":            np.arange(n, dtype=float),
        "bbox_x_min":       0.0, "bbox_x_max": 1.0,
        "bbox_y_min":       0.0, "bbox_y_max": 1.0,
        "bbox_area":        1.0, "bbox_aspect_ratio": 1.0,
        "area":             100.0,
    })
    with open(path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")


def test_public_export_creates_output(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    assert (tmp_path / "animal_track_1.csv").exists()

def test_public_export_output_has_pos_and_area(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    df = pd.read_csv(tmp_path / "animal_track_1.csv", skiprows=7)
    assert "pos_x" in df.columns
    assert "area" in df.columns

def test_public_export_no_spurious_columns(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    df = pd.read_csv(tmp_path / "animal_track_1.csv", skiprows=7)
    for col in ("centroid_x", "centroid_y", "segments_x", "segments_y", "segments_area"):
        assert col not in df.columns

def test_public_export_combined(tmp_path):
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    _write_prediction_csv(tmp_path / "animal_track_2.csv", track_id=2)
    export_tracking(tmp_path, output_dir=tmp_path, combined=True)
    assert (tmp_path / "all_tracks.csv").exists()

def test_public_export_region_properties_filter(tmp_path):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    # Append extra regionprop columns
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    df["solidity"]     = 0.8
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties=["eccentricity"], overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns
    assert "solidity" not in out.columns

def test_public_export_region_properties_none_strips_extras(tmp_path):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties="none", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" not in out.columns

def test_public_export_region_properties_all_keeps_extras(tmp_path):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    df["eccentricity"] = 0.5
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")

    export_tracking(tmp_path, output_dir=tmp_path, region_properties="all", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns

def _write_csv_with_extras(tmp_path, extras: dict):
    csv_path = tmp_path / "animal_track_1.csv"
    _write_prediction_csv(csv_path, track_id=1)
    df = pd.read_csv(csv_path, skiprows=7)
    for col, val in extras.items():
        df[col] = val
    meta = _read_csv_metadata(csv_path)
    from octron.tools.export_tracking import _build_header
    with open(csv_path, "w") as f:
        f.write(_build_header(meta))
        df.to_csv(f, index=False, lineterminator="\n")
    return csv_path

def test_public_export_region_properties_shape_alias(tmp_path):
    csv_path = _write_csv_with_extras(tmp_path, {"eccentricity": 0.5, "intensity_mean": 100.0})
    export_tracking(tmp_path, output_dir=tmp_path, region_properties="shape", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "eccentricity" in out.columns
    assert "intensity_mean" not in out.columns

def test_public_export_region_properties_intensity_alias(tmp_path):
    csv_path = _write_csv_with_extras(tmp_path, {"eccentricity": 0.5, "intensity_mean": 100.0})
    export_tracking(tmp_path, output_dir=tmp_path, region_properties="intensity", overwrite=True)
    out = pd.read_csv(csv_path, skiprows=7)
    assert "intensity_mean" in out.columns
    assert "eccentricity" not in out.columns

def test_public_export_raises_on_missing_csvs(tmp_path):
    with pytest.raises(FileNotFoundError):
        export_tracking(tmp_path / "nonexistent")


# ===========================================================================
# export_tracking — overwrite guard
# ===========================================================================

def test_overwrite_false_raises_when_file_exists(tmp_path):
    """Default overwrite=False should raise FileExistsError if output exists."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    # First export creates the file
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    # Second export without overwrite should fail
    with pytest.raises(FileExistsError):
        export_tracking(tmp_path, output_dir=tmp_path, overwrite=False)


def test_overwrite_true_replaces_existing_file(tmp_path):
    """overwrite=True should silently replace an existing output file."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)
    export_tracking(tmp_path, output_dir=tmp_path, overwrite=True)  # no error
    assert (tmp_path / "animal_track_1.csv").exists()


def test_overwrite_false_raises_for_combined(tmp_path):
    """overwrite=False should raise when all_tracks.csv already exists."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    export_tracking(tmp_path, output_dir=tmp_path, combined=True, overwrite=True)
    with pytest.raises(FileExistsError):
        export_tracking(tmp_path, output_dir=tmp_path, combined=True, overwrite=False)


def test_overwrite_false_no_error_when_output_dir_is_different(tmp_path):
    """No error when writing to a fresh directory even with overwrite=False."""
    _write_prediction_csv(tmp_path / "animal_track_1.csv", track_id=1)
    out = tmp_path / "fresh"
    out.mkdir()
    export_tracking(tmp_path, output_dir=out, overwrite=False)  # should not raise
    assert (out / "animal_track_1.csv").exists()
