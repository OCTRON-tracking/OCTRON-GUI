"""
Tests for tracklet centroid smoothing and gap interpolation.

Both now live in core ``YOLO_results.get_tracking_data`` (shared by the CLI
render path and the napari GUI), so they are exercised here through that single
core entry point rather than a render-local duplicate.

A lightweight ``YOLO_results`` is built via ``__new__`` (bypassing ``__init__``),
backed by one on-disk tracking CSV, so no video, zarr, or model is required.
"""

import numpy as np
import pandas as pd

from octron.yolo_octron.helpers.yolo_results import YOLO_results


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _results(tmp_path, frames, xs, ys, num_frames, header_lines=0):
    """Build a YOLO_results (no __init__) backed by one tracking CSV."""
    frames = list(frames)
    df = pd.DataFrame({
        "frame_idx": frames,
        "track_id": [1] * len(frames),
        "label": ["a"] * len(frames),
        "confidence": [0.9] * len(frames),
        "pos_x": [float(x) for x in xs],
        "pos_y": [float(y) for y in ys],
    })
    csv = tmp_path / "a_track_1.csv"
    df.to_csv(csv, index=False)

    obj = YOLO_results.__new__(YOLO_results)
    obj.csvs = [csv]
    obj.csv_header_lines = header_lines
    obj.num_frames = num_frames
    obj.verbose = False
    return obj


def _xy(results, *, sigma=0.0, interpolate=False, interpolate_limit=None, tid=1):
    td = results.get_tracking_data(
        interpolate=interpolate,
        interpolate_limit=interpolate_limit,
        sigma=sigma,
    )
    data = td[tid]["data"].sort_values("frame_idx")
    return (
        data["frame_idx"].to_numpy(),
        data["pos_x"].to_numpy(dtype=float),
        data["pos_y"].to_numpy(dtype=float),
    )


SIGMA = 3.0


# ===========================================================================
# Smoothing — get_tracking_data(sigma=...)
# ===========================================================================

def test_smooth_constant_trajectory_unchanged(tmp_path):
    n = 200
    res = _results(tmp_path, range(n), np.ones(n) * 100.0, np.ones(n) * 200.0, num_frames=n)
    _, xs, ys = _xy(res, sigma=SIGMA)
    np.testing.assert_allclose(xs, 100.0, atol=1e-6)
    np.testing.assert_allclose(ys, 200.0, atol=1e-6)


def test_smooth_reduces_noise(tmp_path):
    rng = np.random.default_rng(42)
    n = 500
    clean = 50.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / n)
    noisy = clean + rng.normal(0, 3, n)
    res = _results(tmp_path, range(n), noisy, noisy, num_frames=n)
    _, xs, _ = _xy(res, sigma=SIGMA)
    rmse_before = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_after = np.sqrt(np.mean((xs - clean) ** 2))
    assert rmse_after < rmse_before


def test_smooth_preserves_low_frequency(tmp_path):
    n = 500
    xs_in = 50.0 + 20.0 * np.sin(2 * np.pi * np.arange(n) / n)
    res = _results(tmp_path, range(n), xs_in, np.zeros(n), num_frames=n)
    _, xs, _ = _xy(res, sigma=SIGMA)
    mid = slice(50, 450)
    np.testing.assert_allclose(xs[mid], xs_in[mid], atol=0.5)


def test_smooth_sigma_zero_is_noop(tmp_path):
    rng = np.random.default_rng(1)
    n = 200
    xs_in = rng.normal(50, 10, n)
    ys_in = rng.normal(50, 10, n)
    res = _results(tmp_path, range(n), xs_in, ys_in, num_frames=n)
    _, xs, ys = _xy(res, sigma=0.0)
    np.testing.assert_allclose(xs, xs_in, atol=1e-9)
    np.testing.assert_allclose(ys, ys_in, atol=1e-9)


def test_smooth_larger_sigma_smooths_more(tmp_path):
    rng = np.random.default_rng(7)
    n = 500
    xs_in = rng.normal(0, 10, n)
    res = _results(tmp_path, range(n), xs_in, np.zeros(n), num_frames=n)
    _, x_small, _ = _xy(res, sigma=1.0)
    _, x_large, _ = _xy(res, sigma=6.0)
    assert x_large.std() < x_small.std()


def test_smooth_single_point_unchanged(tmp_path):
    res = _results(tmp_path, [5], [10.0], [5.0], num_frames=6)
    frames, xs, ys = _xy(res, sigma=SIGMA)
    assert list(frames) == [5]
    assert xs[0] == 10.0 and ys[0] == 5.0


def test_smooth_preserves_frame_indices(tmp_path):
    frames_in = list(range(10, 210))
    xs_in = np.linspace(0, 100, len(frames_in))
    res = _results(tmp_path, frames_in, xs_in, np.zeros(len(frames_in)), num_frames=210)
    frames, _, _ = _xy(res, sigma=SIGMA)
    assert list(frames) == frames_in


# ===========================================================================
# Interpolation — get_tracking_data(interpolate=True, interpolate_limit=...)
# ===========================================================================

def test_interp_fills_interior_gap(tmp_path):
    res = _results(tmp_path, [0, 5], [0.0, 100.0], [0.0, 200.0], num_frames=6)
    frames, xs, ys = _xy(res, interpolate=True)
    assert list(frames) == [0, 1, 2, 3, 4, 5]
    # linear fill: frame 3 -> 0 + 100*3/5 = 60, 0 + 200*3/5 = 120
    i3 = list(frames).index(3)
    assert abs(xs[i3] - 60.0) < 1e-6
    assert abs(ys[i3] - 120.0) < 1e-6


def test_interp_disabled_keeps_sparse(tmp_path):
    res = _results(tmp_path, [0, 5], [0.0, 100.0], [0.0, 200.0], num_frames=6)
    frames, _, _ = _xy(res, interpolate=False)
    assert list(frames) == [0, 5]


def test_interp_limit_partial_fill(tmp_path):
    # interior gap of 10 frames (1..10); limit=3 bridges only part of it,
    # so the gap is NOT fully filled (linear + consecutive-NaN limit semantics).
    res = _results(tmp_path, [0, 11], [0.0, 110.0], [0.0, 0.0], num_frames=12)
    frames, xs, _ = _xy(res, interpolate=True, interpolate_limit=3)
    frames = list(frames)
    assert frames[0] == 0 and frames[-1] == 11
    assert 1 in frames                 # forward-filled from the gap start
    assert 5 <= len(frames) < 12       # partial fill, not the full 12 frames
    i1 = frames.index(1)
    assert abs(xs[i1] - 10.0) < 1e-6   # linear: 110*1/11


def test_interp_does_not_fill_leading_gap(tmp_path):
    res = _results(tmp_path, [5, 10], [50.0, 100.0], [0.0, 0.0], num_frames=11)
    frames, _, _ = _xy(res, interpolate=True)
    assert min(frames) == 5
    assert list(frames) == [5, 6, 7, 8, 9, 10]


def test_interp_fills_trailing_gap_forward(tmp_path):
    # Core linear interpolation (default limit_direction='forward') forward-fills
    # the trailing gap — frames after the last detection up to num_frames — when
    # no limit is set.  The removed render-local helper left these empty; for the
    # CLI the trailing fill is bounded because --tracklet-interpolate always sets
    # a finite limit (see test_interp_limit_partial_fill).  The leading gap is
    # still never filled (see test_interp_does_not_fill_leading_gap).
    res = _results(tmp_path, [0, 5], [0.0, 50.0], [0.0, 0.0], num_frames=10)
    frames, _, _ = _xy(res, interpolate=True)
    assert list(frames) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_interp_then_smooth_contiguous(tmp_path):
    # interpolation densifies the gap; smoothing then runs on the dense series.
    res = _results(tmp_path, [0, 5], [0.0, 100.0], [0.0, 0.0], num_frames=6)
    frames, _, _ = _xy(res, interpolate=True, sigma=1.0)
    assert list(frames) == [0, 1, 2, 3, 4, 5]
