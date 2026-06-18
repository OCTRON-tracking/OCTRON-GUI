"""
Tests for gaussian_smooth_tracklet_positions (Gaussian centroid smoother).
"""

import numpy as np
import pandas as pd

from octron.tools.render import gaussian_smooth_tracklet_positions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pos_lookup(xs, ys, frame_offset=0):
    """Build a pos_lookup dict for a single track (tid=0) from x/y arrays."""
    return {
        0: {
            frame_offset + i: pd.Series({"pos_x": float(xs[i]), "pos_y": float(ys[i])})
            for i in range(len(xs))
        }
    }


def _extract_xy(pos_lookup, tid=0):
    frames = sorted(pos_lookup[tid].keys())
    xs = np.array([pos_lookup[tid][f]["pos_x"] for f in frames])
    ys = np.array([pos_lookup[tid][f]["pos_y"] for f in frames])
    return frames, xs, ys


# ===========================================================================
# gaussian_smooth_tracklet_positions
# ===========================================================================

SIGMA = 3.0


def test_gaussian_constant_trajectory_unchanged():
    """A constant trajectory should remain constant after Gaussian smoothing."""
    n = 200
    xs = np.ones(n) * 100.0
    ys = np.ones(n) * 200.0
    lookup = _make_pos_lookup(xs, ys)
    gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_allclose(xs_out, 100.0, atol=1e-6)
    np.testing.assert_allclose(ys_out, 200.0, atol=1e-6)


def test_gaussian_noise_is_reduced():
    """Smoothed signal should be closer to the clean signal than the noisy input."""
    rng = np.random.default_rng(42)
    n = 500
    # Clean signal: slow drift over the whole track
    clean = 50.0 + 10.0 * np.sin(2 * np.pi * np.arange(n) / n)
    # Noise: high-frequency, frame-to-frame jitter
    noise = rng.normal(0, 3, n)
    noisy = clean + noise

    lookup = _make_pos_lookup(noisy, noisy)
    gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    _, xs_out, _ = _extract_xy(lookup)

    rmse_before = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_after = np.sqrt(np.mean((xs_out - clean) ** 2))
    assert rmse_after < rmse_before, (
        f"Gaussian smoothing increased RMSE: {rmse_before:.3f} → {rmse_after:.3f}"
    )


def test_gaussian_low_frequency_signal_preserved():
    """A slowly varying signal should pass through largely unchanged."""
    n = 500
    # One full slow oscillation over the track — much slower than sigma
    xs = 50.0 + 20.0 * np.sin(2 * np.pi * np.arange(n) / n)
    ys = np.zeros(n)
    lookup = _make_pos_lookup(xs, ys)
    gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    # Ignore edge effects: check only the middle portion
    _, xs_out, _ = _extract_xy(lookup)
    mid = slice(50, 450)
    np.testing.assert_allclose(xs_out[mid], xs[mid], atol=0.5)


def test_gaussian_single_point_track_is_skipped():
    """A track with fewer than two points must be left untouched."""
    xs = np.array([10.0])
    ys = np.array([5.0])
    lookup = _make_pos_lookup(xs, ys)
    gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_array_equal(xs_out, xs)
    np.testing.assert_array_equal(ys_out, ys)


def test_gaussian_sigma_zero_is_noop():
    """sigma=0 should leave pos_lookup unchanged."""
    rng = np.random.default_rng(1)
    xs = rng.normal(50, 10, 200)
    ys = rng.normal(50, 10, 200)
    lookup = _make_pos_lookup(xs, ys)
    gaussian_smooth_tracklet_positions(lookup, sigma=0)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_array_equal(xs_out, xs)
    np.testing.assert_array_equal(ys_out, ys)


def test_gaussian_larger_sigma_smooths_more():
    """A larger sigma should attenuate high-frequency jitter more aggressively."""
    rng = np.random.default_rng(7)
    n = 500
    xs = rng.normal(0, 10, n)
    ys = np.zeros(n)

    lookup_small = _make_pos_lookup(xs, ys)
    lookup_large = _make_pos_lookup(xs, ys)
    gaussian_smooth_tracklet_positions(lookup_small, sigma=1.0)
    gaussian_smooth_tracklet_positions(lookup_large, sigma=6.0)

    _, x_small, _ = _extract_xy(lookup_small)
    _, x_large, _ = _extract_xy(lookup_large)
    assert x_large.std() < x_small.std(), (
        f"sigma=6 should attenuate more than sigma=1: {x_large.std():.4f} vs {x_small.std():.4f}"
    )


def test_gaussian_returns_same_dict():
    """The function should return the same dict object (in-place modification)."""
    xs = np.linspace(0, 10, 200)
    ys = np.linspace(0, 10, 200)
    lookup = _make_pos_lookup(xs, ys)
    result = gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    assert result is lookup


def test_gaussian_frame_indices_preserved():
    """Frame indices must not change after Gaussian smoothing."""
    frames_in = list(range(10, 210))
    xs = np.linspace(0, 100, len(frames_in))
    ys = np.zeros(len(frames_in))
    lookup = _make_pos_lookup(xs, ys, frame_offset=10)
    gaussian_smooth_tracklet_positions(lookup, sigma=SIGMA)
    assert sorted(lookup[0].keys()) == frames_in
