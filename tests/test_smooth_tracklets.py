"""
Tests for butterworth_smooth_tracklet_positions (Butterworth centroid smoother).
"""

import numpy as np
import pandas as pd

from octron.tools.render import butterworth_smooth_tracklet_positions


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
# butterworth_smooth_tracklet_positions
# ===========================================================================

FPS = 25.0


def test_butterworth_constant_trajectory_unchanged():
    """A constant trajectory should remain constant after Butterworth smoothing."""
    n = 200
    xs = np.ones(n) * 100.0
    ys = np.ones(n) * 200.0
    lookup = _make_pos_lookup(xs, ys)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS, cutoff_hz=2.0)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_allclose(xs_out, 100.0, atol=1e-6)
    np.testing.assert_allclose(ys_out, 200.0, atol=1e-6)


def test_butterworth_noise_is_reduced():
    """Smoothed signal should be closer to the clean signal than the noisy input."""
    rng = np.random.default_rng(42)
    n = 500
    t = np.arange(n) / FPS
    # Clean signal: slow drift at 0.5 Hz
    clean = 50.0 + 10.0 * np.sin(2 * np.pi * 0.5 * t)
    # Noise: high-frequency jitter at 8 Hz (well above 2 Hz cutoff)
    noise = 5.0 * np.sin(2 * np.pi * 8.0 * t) + rng.normal(0, 2, n)
    noisy = clean + noise

    lookup = _make_pos_lookup(noisy, noisy)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS, cutoff_hz=2.0)
    _, xs_out, _ = _extract_xy(lookup)

    rmse_before = np.sqrt(np.mean((noisy - clean) ** 2))
    rmse_after = np.sqrt(np.mean((xs_out - clean) ** 2))
    assert rmse_after < rmse_before, (
        f"Butterworth smoothing increased RMSE: {rmse_before:.3f} → {rmse_after:.3f}"
    )


def test_butterworth_low_frequency_signal_preserved():
    """A signal well below the cutoff should pass through largely unchanged."""
    n = 500
    t = np.arange(n) / FPS
    # 0.2 Hz sine — far below 2 Hz cutoff
    xs = 50.0 + 20.0 * np.sin(2 * np.pi * 0.2 * t)
    ys = np.zeros(n)
    lookup = _make_pos_lookup(xs, ys)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS, cutoff_hz=2.0)
    # Ignore edge effects: check only the middle portion
    _, xs_out, _ = _extract_xy(lookup)
    mid = slice(50, 450)
    np.testing.assert_allclose(xs_out[mid], xs[mid], atol=0.5)


def test_butterworth_track_too_short_is_skipped():
    """Tracks shorter than the minimum filtfilt length should be left untouched."""
    xs = np.array([10.0, 20.0, 30.0])
    ys = np.array([5.0, 15.0, 25.0])
    lookup = _make_pos_lookup(xs, ys)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS, cutoff_hz=2.0)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_array_equal(xs_out, xs)
    np.testing.assert_array_equal(ys_out, ys)


def test_butterworth_cutoff_zero_is_noop():
    """cutoff_hz=0 should leave pos_lookup unchanged."""
    rng = np.random.default_rng(1)
    xs = rng.normal(50, 10, 200)
    ys = rng.normal(50, 10, 200)
    lookup = _make_pos_lookup(xs, ys)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS, cutoff_hz=0)
    _, xs_out, ys_out = _extract_xy(lookup)
    np.testing.assert_array_equal(xs_out, xs)
    np.testing.assert_array_equal(ys_out, ys)


def test_butterworth_higher_order_steeper_rolloff():
    """Higher filter order should more aggressively attenuate near-cutoff frequencies."""
    n = 500
    t = np.arange(n) / FPS
    # Signal just above cutoff (3 Hz with cutoff at 2 Hz)
    xs = 10.0 * np.sin(2 * np.pi * 3.0 * t)
    ys = np.zeros(n)

    lookup2 = _make_pos_lookup(xs, ys)
    lookup8 = _make_pos_lookup(xs, ys)
    butterworth_smooth_tracklet_positions(lookup2, fps=FPS, cutoff_hz=2.0, order=2)
    butterworth_smooth_tracklet_positions(lookup8, fps=FPS, cutoff_hz=2.0, order=8)

    _, x2, _ = _extract_xy(lookup2)
    _, x8, _ = _extract_xy(lookup8)
    assert x8.std() < x2.std(), (
        f"order=8 should attenuate more than order=2: {x8.std():.4f} vs {x2.std():.4f}"
    )


def test_butterworth_returns_same_dict():
    """The function should return the same dict object (in-place modification)."""
    xs = np.linspace(0, 10, 200)
    ys = np.linspace(0, 10, 200)
    lookup = _make_pos_lookup(xs, ys)
    result = butterworth_smooth_tracklet_positions(lookup, fps=FPS)
    assert result is lookup


def test_butterworth_frame_indices_preserved():
    """Frame indices must not change after Butterworth smoothing."""
    frames_in = list(range(10, 210))
    xs = np.linspace(0, 100, len(frames_in))
    ys = np.zeros(len(frames_in))
    lookup = _make_pos_lookup(xs, ys, frame_offset=10)
    butterworth_smooth_tracklet_positions(lookup, fps=FPS)
    assert sorted(lookup[0].keys()) == frames_in
