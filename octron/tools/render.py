"""
OCTRON video rendering pipeline.

Reads prediction output (zarr masks + CSV tracks + metadata) produced by
``octron analyze`` and renders annotated videos.

Public functions
----------------
run_render      : Render an overlay video (masks, boxes, labels) from predictions.
run_tracklets   : Render one stabilised crop video per tracked animal.
report_bbox_sizes : Report bounding-box sizes to help choose tracklet crop size.
"""

from pathlib import Path

from loguru import logger

PRESETS = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_mask_centroids(zarr_root, track_ids, frame_start, frame_end, downsample=4):
    """
    Compute per-frame centre-of-mass centroids directly from zarr segmentation
    masks, as a more stable alternative to the bounding-box-derived positions
    stored in the tracking CSVs.

    Masks are downsampled before computing CoM for speed, then coordinates are
    scaled back to full resolution.  Processing is vectorised over the batch
    dimension — no Python loop over frames.

    Parameters
    ----------
    zarr_root : zarr.Group
        Root zarr group from a YOLO_results object.
    track_ids : iterable
        Track IDs to process (must match the ``{tid}_masks`` array naming).
    frame_start : int
        First frame index to process (inclusive).
    frame_end : int
        Last frame index to process (exclusive).
    downsample : int
        Spatial downscale factor applied before computing CoM.  ``4`` (default)
        gives 16× fewer pixels with negligible centroid error at full resolution.

    Returns
    -------
    centroids : dict
        ``{track_id: {frame_idx: (cx, cy)}}`` — float pixel coordinates in the
        original full-resolution frame.  Only frames where at least one mask
        pixel is present are included.
    """
    import numpy as np

    _BATCH = 500
    track_ids = list(track_ids)
    centroids = {tid: {} for tid in track_ids}
    n_frames = frame_end - frame_start
    n_batches_per_track = max(1, (n_frames + _BATCH - 1) // _BATCH)
    total_batches = len(track_ids) * n_batches_per_track
    _bar_width = 30
    batch_idx = 0

    for tid in track_ids:
        arr_key = f"{tid}_masks"
        if arr_key not in zarr_root:
            batch_idx += n_batches_per_track
            continue
        zarr_arr = zarr_root[arr_key]
        n_mask_frames = zarr_arr.shape[0]

        for batch_start in range(frame_start, min(frame_end, n_mask_frames), _BATCH):
            batch_end = min(batch_start + _BATCH, frame_end, n_mask_frames)
            raw = np.asarray(zarr_arr[batch_start:batch_end])  # (n, H, W)

            mask = (raw[:, ::downsample, ::downsample] == 1)  # (n, dH, dW) bool
            dH, dW = mask.shape[1], mask.shape[2]

            count = mask.sum(axis=(1, 2))
            ys = np.arange(dH, dtype=np.float32)[None, :, None]
            xs = np.arange(dW, dtype=np.float32)[None, None, :]
            cy_ds = (mask * ys).sum(axis=(1, 2)) / np.maximum(count, 1)
            cx_ds = (mask * xs).sum(axis=(1, 2)) / np.maximum(count, 1)

            cx_full = cx_ds * downsample
            cy_full = cy_ds * downsample

            for i, frame_idx in enumerate(range(batch_start, batch_end)):
                if count[i] > 0:
                    centroids[tid][frame_idx] = (float(cx_full[i]), float(cy_full[i]))

            batch_idx += 1
            pct = batch_idx / total_batches
            filled = int(_bar_width * pct)
            bar = "█" * filled + "░" * (_bar_width - filled)
            logger.debug(f"  [{bar}] track {tid} | {batch_start}-{batch_end} | {pct:.0%}")

    return centroids


def butterworth_smooth_tracklet_positions(pos_lookup, fps, cutoff_hz=2.0, order=4):
    """
    Apply a zero-phase Butterworth low-pass filter to the centroid trajectories
    in ``pos_lookup`` to remove high-frequency jitter while preserving real
    animal movement.

    Uses ``scipy.signal.filtfilt`` (forward + backward pass) for zero phase
    distortion and no lag — suitable for offline use where the full trajectory
    is available before rendering.

    Parameters
    ----------
    pos_lookup : dict
        ``{track_id: {frame_idx: pandas.Series}}`` as built by ``run_tracklets``.
        Modified in-place.
    fps : float
        Frame rate of the video, used to normalise ``cutoff_hz``.
    cutoff_hz : float
        Low-pass cutoff frequency in Hz.  Frequencies above this are attenuated.
        Default 2.0 Hz.
    order : int
        Butterworth filter order.  Higher = steeper rolloff.  Default 4.

    Returns
    -------
    pos_lookup : dict
        The same dict, modified in-place.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt

    if not cutoff_hz or cutoff_hz <= 0:
        return pos_lookup

    nyquist = fps / 2.0
    cutoff_norm = cutoff_hz / nyquist
    if cutoff_norm >= 1.0:
        logger.warning(f"Warning: cutoff_hz={cutoff_hz} exceeds Nyquist ({nyquist:.1f} Hz); skipping smoothing.")
        return pos_lookup

    b, a = butter(order, cutoff_norm, btype="low")
    min_len = 3 * max(len(a), len(b)) + 1

    for tid in list(pos_lookup.keys()):
        frames = sorted(pos_lookup[tid].keys())
        n = len(frames)
        if n < min_len:
            continue
        xs = np.array([pos_lookup[tid][f]["pos_x"] for f in frames], dtype=float)
        ys = np.array([pos_lookup[tid][f]["pos_y"] for f in frames], dtype=float)
        xs_smooth = filtfilt(b, a, xs)
        ys_smooth = filtfilt(b, a, ys)
        for i, f in enumerate(frames):
            row = pos_lookup[tid][f].copy()
            row["pos_x"] = xs_smooth[i]
            row["pos_y"] = ys_smooth[i]
            pos_lookup[tid][f] = row

    logger.info(f"Tracklet centroids smoothed (Butterworth order={order}, cutoff={cutoff_hz} Hz)")
    return pos_lookup


def interpolate_tracklet_gaps(pos_lookup, max_gap):
    """
    Fill gaps in tracklet centroid trajectories using cubic spline interpolation.

    Only gaps between two known positions are filled (leading and trailing gaps
    are left as black frames).  Gaps wider than ``max_gap`` frames are left
    untouched.  Interpolated rows are synthetic copies of the nearest real row
    with ``pos_x`` / ``pos_y`` replaced by the spline values.

    Applied after Butterworth smoothing so the spline fits the already-smooth
    trajectory.

    Parameters
    ----------
    pos_lookup : dict
        ``{track_id: {frame_idx: pandas.Series}}`` — modified in-place.
    max_gap : int
        Maximum gap width (in frames) to interpolate across.  0 = disabled.

    Returns
    -------
    pos_lookup : dict
        The same dict, modified in-place.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline

    if not max_gap or max_gap <= 0:
        return pos_lookup

    total_filled = 0
    for tid in list(pos_lookup.keys()):
        frames = sorted(pos_lookup[tid].keys())
        if len(frames) < 2:
            continue

        xs = np.array([pos_lookup[tid][f]["pos_x"] for f in frames], dtype=float)
        ys = np.array([pos_lookup[tid][f]["pos_y"] for f in frames], dtype=float)
        cs_x = CubicSpline(frames, xs)
        cs_y = CubicSpline(frames, ys)

        for i in range(len(frames) - 1):
            gap_start = frames[i]
            gap_end = frames[i + 1]
            gap = gap_end - gap_start - 1
            if gap <= 0 or gap > max_gap:
                continue
            ref_row = pos_lookup[tid][gap_start]
            for f in range(gap_start + 1, gap_end):
                row = ref_row.copy()
                row["pos_x"] = float(cs_x(f))
                row["pos_y"] = float(cs_y(f))
                pos_lookup[tid][f] = row
                total_filled += 1

    if total_filled:
        logger.info(f"Tracklet gaps interpolated (cubic spline, max_gap={max_gap}): {total_filled} frame(s) filled")
    return pos_lookup


def _load_results(predictions_path, video_path):
    """Load YOLO_results and optionally override the video path."""
    from octron.yolo_octron.helpers.yolo_results import YOLO_results

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_path}")

    results = YOLO_results(predictions_path, verbose=True)

    if video_path is not None:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        from octron.sam_octron.helpers.video_loader import probe_video
        from napari_pyav._reader import FastVideoReader

        results.video = FastVideoReader(video_path, read_format="rgb24")
        results.video_dict = probe_video(video_path, verbose=False)
        results.height = results.video_dict["height"]
        results.width = results.video_dict["width"]
        results.num_frames = results.video_dict["num_frames"]

    if results.video_dict is None:
        raise FileNotFoundError(
            "Could not locate the original video automatically. "
            "Use --video to specify its path."
        )

    return results


def report_bbox_sizes(predictions_path):
    """
    Scan all tracking CSVs in a predictions directory and report the maximum
    bounding-box dimensions per track, so users can choose an appropriate
    ``size`` before rendering tracklets.

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory (same as used for ``run_render``).

    Returns
    -------
    stats : dict
        ``{track_id: {'label': str, 'max_w': int, 'max_h': int}}``
    """
    import numpy as np

    results = _load_results(predictions_path, video_path=None)
    tracking_data = results.get_tracking_data(interpolate=False)

    stats = {}
    for tid, td in tracking_data.items():
        feats = td["features"]
        max_w = int(np.ceil((feats["bbox_x_max"] - feats["bbox_x_min"]).max()))
        max_h = int(np.ceil((feats["bbox_y_max"] - feats["bbox_y_min"]).max()))
        stats[tid] = {"label": td["label"], "max_w": max_w, "max_h": max_h}

    overall_max = max((max(s["max_w"], s["max_h"]) for s in stats.values()), default=0)

    logger.info("Bounding-box size report:")
    for tid, s in sorted(stats.items()):
        logger.info(f"  Track {tid} ({s['label']}): max {s['max_w']}×{s['max_h']} px (w×h)")
    logger.info(
        f"→ Recommended minimum tracklet size: {overall_max} px "
        f"(add padding as desired, default is 160 px)"
    )
    return stats


# ---------------------------------------------------------------------------
# Main render functions
# ---------------------------------------------------------------------------

def run_tracklets(
    predictions_path,
    video_path=None,
    output_path=None,
    preset="draft",
    also_overlay=False,
    size=160,
    mask_centroids=False,
    smooth_cutoff_hz=2.0,
    smooth_order=4,
    alpha=0.4,
    draw_masks=False,
    draw_boxes=False,
    start=None,
    end=None,
    min_track_frames=0,
    interpolate_max_gap=0,
):
    """
    Render one stabilised crop video per tracked animal.

    Crops are centred on the smoothed centroid trajectory and extracted at
    sub-pixel accuracy via bilinear interpolation (``cv2.getRectSubPix``).

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory produced by ``octron analyze``.
    video_path : str or Path, optional
        Path to the original video.  Auto-detected if alongside
        ``octron_predictions/``.  Pass an optic-flow video here to crop flow
        fields instead of raw frames.
    output_path : str or Path, optional
        Output directory.  Defaults to ``<predictions_path>/rendered/``.
    preset : {'preview', 'draft', 'final'}
        Sets the output resolution for ``--tracklet-overlay`` mode only;
        the raw tracklet crops are always at full resolution.
    also_overlay : bool
        If True, render masks and boxes onto the tracklet crops (useful for
        inspecting centroid placement).  Default False.
    size : int
        Side length in pixels of each square tracklet crop.  Default 160.
    mask_centroids : bool
        If True, replace bbox-derived centroid positions with the centre-of-mass
        computed from the segmentation masks.  Default False.
    smooth_cutoff_hz : float
        Butterworth low-pass cutoff (Hz) applied to the centroid trajectories.
        Set to 0 to disable.  Default 2.0 Hz.
    smooth_order : int
        Butterworth filter order.  Default 4.
    alpha : float
        Mask overlay opacity when ``also_overlay=True`` (0–1).  Default 0.4.
    draw_masks : bool
        Draw segmentation masks when ``also_overlay=True``.  Default False.
    draw_boxes : bool
        Draw bounding boxes when ``also_overlay=True``.  Default False.
        Labels are never drawn on tracklet crops.
    start : int, optional
        First frame index (inclusive).  Default: beginning of video.
    end : int, optional
        Last frame index (exclusive).  Default: end of video.
    """
    import cv2
    import numpy as np
    import time
    from collections import deque

    results = _load_results(predictions_path, video_path)

    src_path = results.video_dict["video_file_path"]
    fps = results.video_dict["fps"]
    height = results.height
    width = results.width
    num_frames = results.num_frames

    frame_start = start if start is not None else 0
    frame_end = end if end is not None else num_frames
    n_frames = frame_end - frame_start

    output_path = (
        Path(output_path) if output_path is not None else Path(predictions_path) / "rendered"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    scale = PRESETS[preset]["scale"]
    out_h = int(height * scale)
    out_w = int(width * scale)
    out_h += out_h % 2
    out_w += out_w % 2

    tracking_data = results.get_tracking_data(interpolate=False)

    pos_lookup = {}
    bbox_lookup = {}
    for tid, td in tracking_data.items():
        pos_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["data"].iterrows()}
        bbox_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["features"].iterrows()}
        n = len(pos_lookup[tid])
        skipped = " (skipped)" if min_track_frames > 0 and n < min_track_frames else ""
        logger.info(f"  Track {tid} ({td['label']}): {n} frames{skipped}")

    render_tids = [
        tid for tid in results.track_ids
        if min_track_frames <= 0 or len(pos_lookup.get(tid, {})) >= min_track_frames
    ]
    if min_track_frames > 0:
        logger.info(f"  Keeping {len(render_tids)}/{len(results.track_ids)} tracks with ≥{min_track_frames} frames")

    if mask_centroids and results.has_masks:
        logger.info("Computing mask centre-of-mass centroids...")
        mask_cent = compute_mask_centroids(
            results.zarr_root, results.track_ids, frame_start, frame_end,
        )
        replaced = 0
        for tid, frame_centroids in mask_cent.items():
            for frame_idx, (cx, cy) in frame_centroids.items():
                if frame_idx in pos_lookup.get(tid, {}):
                    row = pos_lookup[tid][frame_idx].copy()
                    row["pos_x"] = cx
                    row["pos_y"] = cy
                    pos_lookup[tid][frame_idx] = row
                    replaced += 1
        logger.info(f"  Replaced {replaced} centroid(s) with mask CoM")

    if smooth_cutoff_hz and smooth_cutoff_hz > 0:
        butterworth_smooth_tracklet_positions(
            pos_lookup, fps=fps, cutoff_hz=smooth_cutoff_hz, order=smooth_order,
        )

    if interpolate_max_gap and interpolate_max_gap > 0:
        interpolate_tracklet_gaps(pos_lookup, max_gap=interpolate_max_gap)

    track_colors = {}
    for tid in results.track_ids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)

    # Pre-flight: warn if any bbox exceeds the requested crop size
    for tid, td in tracking_data.items():
        feats = td["features"]
        max_w = int(np.ceil((feats["bbox_x_max"] - feats["bbox_x_min"]).max()))
        max_h = int(np.ceil((feats["bbox_y_max"] - feats["bbox_y_min"]).max()))
        if max(max_w, max_h) > size:
            logger.info(
                f"Warning: track {tid} ({td['label']}) has bounding boxes up to "
                f"{max_w}×{max_h} px, which exceeds size={size}. "
                f"Consider --tracklet-size {max(max_w, max_h) + 20}."
            )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    for tid in render_tids:
        label = results.track_id_label[tid]
        tp = output_path / f"tracklet_{label}_track{tid}_{preset}.mp4"
        writers[tid] = cv2.VideoWriter(str(tp), fourcc, fps, (size, size))

    # Pre-compute downscale indices for overlay masks
    _y_idx = _x_idx = None
    if also_overlay and draw_masks and scale != 1.0 and results.has_masks:
        for tid in results.track_ids:
            arr_key = f"{tid}_masks"
            if arr_key in results.zarr_root:
                _H, _W = results.zarr_root[arr_key].shape[1:3]
                _y_idx = np.round(np.linspace(0, _H - 1, out_h)).astype(int)
                _x_idx = np.round(np.linspace(0, _W - 1, out_w)).astype(int)
                break

    _BATCH = 500
    _batch_masks = {}
    _batch_start = None

    logger.info(f"Rendering {n_frames} tracklet frames | preset={preset} | size={size}px")
    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        ok, orig_frame = cap.read()
        if not ok:
            logger.warning(f"Warning: could not read frame {frame_idx}, stopping early.")
            break

        # Lazy-load overlay mask batches when needed
        if also_overlay and draw_masks and results.has_masks \
                and (_batch_start is None or frame_idx >= _batch_start + _BATCH):
            _batch_start = frame_idx
            _batch_end = min(_batch_start + _BATCH, frame_end)
            _batch_masks = {}
            for tid in results.track_ids:
                arr_key = f"{tid}_masks"
                if arr_key not in results.zarr_root:
                    continue
                zarr_arr = results.zarr_root[arr_key]
                end_idx = min(_batch_end, zarr_arr.shape[0])
                if _batch_start >= end_idx:
                    continue
                raw = np.asarray(zarr_arr[_batch_start:end_idx])
                if _y_idx is not None:
                    raw = raw[:, _y_idx[:, None], _x_idx[None, :]]
                _batch_masks[tid] = raw

        j = frame_idx - _batch_start if _batch_start is not None else 0

        # Build overlay frame if requested
        out_frame = None
        if also_overlay:
            if scale != 1.0:
                frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = orig_frame
            overlay = frame_small.astype(np.float32)

            for tid in results.track_ids:
                color_bgr = track_colors[tid]
                label = results.track_id_label[tid]

                if draw_masks and results.has_masks and tid in _batch_masks:
                    batch = _batch_masks[tid]
                    if j < batch.shape[0]:
                        obj = batch[j] == 1
                        if obj.any():
                            color_f = np.array(color_bgr, dtype=np.float32)
                            overlay[obj] = (1 - alpha) * overlay[obj] + alpha * color_f

                if draw_boxes:
                    row = bbox_lookup.get(tid, {}).get(frame_idx)
                    if row is not None:
                        x1 = int(row["bbox_x_min"] * scale)
                        y1 = int(row["bbox_y_min"] * scale)
                        x2 = int(row["bbox_x_max"] * scale)
                        y2 = int(row["bbox_y_max"] * scale)
                        lw = max(1, int(2 * scale + 0.5))
                        cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, lw)

            out_frame = np.clip(overlay, 0, 255).astype(np.uint8)

        # Crop each track
        for tid in render_tids:
            row = pos_lookup.get(tid, {}).get(frame_idx)
            if row is not None:
                if also_overlay and out_frame is not None:
                    cx = float(row["pos_x"]) * scale
                    cy = float(row["pos_y"]) * scale
                    crop = cv2.getRectSubPix(out_frame, (size, size), (cx, cy))
                else:
                    cx = float(row["pos_x"])
                    cy = float(row["pos_y"])
                    crop = cv2.getRectSubPix(orig_frame, (size, size), (cx, cy))
            else:
                crop = np.zeros((size, size, 3), dtype=np.uint8)
            writers[tid].write(crop)

        now = time.time()
        _frame_times.append(now - _t_frame)
        _t_frame = now
        avg_ft = sum(_frame_times) / len(_frame_times)
        fps_disp = 1.0 / avg_ft if avg_ft > 0 else 0.0
        remaining = n_frames - (i + 1)
        eta_s = int(remaining * avg_ft) if avg_ft > 0 else 0
        eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"
        pct = (i + 1) / n_frames
        filled = int(_bar_width * pct)
        bar = "█" * filled + "░" * (_bar_width - filled)
        logger.debug(
            f"  [{bar}] {i + 1}/{n_frames} | {pct:.0%} | {fps_disp:.1f} fps | ETA {eta_str}"
        )

    cap.release()
    for w in writers.values():
        w.release()
    logger.info(f"Tracklet(s) saved → {output_path}")


def run_render(
    predictions_path,
    video_path=None,
    output_path=None,
    preset="draft",
    tracklets=False,
    also_overlay=False,
    tracklet_size=160,
    tracklet_mask_centroids=False,
    tracklet_smooth_cutoff_hz=2.0,
    tracklet_smooth_order=4,
    tracklet_min_frames=0,
    tracklet_interpolate_max_gap=0,
    alpha=0.4,
    draw_masks=True,
    draw_boxes=True,
    draw_labels=True,
    start=None,
    end=None,
):
    """
    Render annotated video(s) from OCTRON prediction output.

    When ``tracklets=True``, delegates to ``run_tracklets`` and skips the
    overlay video.  When ``tracklets=False``, renders the overlay video only.

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory produced by ``octron analyze``.
    video_path : str or Path, optional
        Path to the original video.  Auto-detected if alongside
        ``octron_predictions/``.
    output_path : str or Path, optional
        Output directory.  Defaults to ``<predictions_path>/rendered/``.
    preset : {'preview', 'draft', 'final'}
        Quality / speed trade-off.
    tracklets : bool
        If True, render tracklet crops instead of the overlay video.
    also_overlay : bool
        When ``tracklets=True``, render the overlay onto the tracklet crops.
    tracklet_size : int
        Side length in pixels of each tracklet crop.  Default 160.
    tracklet_mask_centroids : bool
        Use mask CoM instead of bbox centre for tracklet positioning.
    tracklet_smooth_cutoff_hz : float
        Butterworth cutoff (Hz) for centroid smoothing.  0 = off.  Default 2.0.
    tracklet_smooth_order : int
        Butterworth filter order.  Default 4.
    alpha : float
        Mask overlay opacity (0–1).  Default 0.4.
    draw_masks : bool
        Render segmentation mask fills.  Default True (overlay); False (tracklets).
    draw_boxes : bool
        Render bounding boxes.  Default True (overlay); False (tracklets).
    draw_labels : bool
        Render label text above bounding boxes (overlay only; never on tracklets).
        Default True.
    start : int, optional
        First frame index (inclusive).
    end : int, optional
        Last frame index (exclusive).
    """
    if tracklets:
        run_tracklets(
            predictions_path=predictions_path,
            video_path=video_path,
            output_path=output_path,
            preset=preset,
            also_overlay=also_overlay,
            size=tracklet_size,
            mask_centroids=tracklet_mask_centroids,
            smooth_cutoff_hz=tracklet_smooth_cutoff_hz,
            smooth_order=tracklet_smooth_order,
            alpha=alpha,
            draw_masks=draw_masks,
            draw_boxes=draw_boxes,
            start=start,
            end=end,
            min_track_frames=tracklet_min_frames,
            interpolate_max_gap=tracklet_interpolate_max_gap,
        )
        return

    import cv2
    import numpy as np
    import time
    from collections import deque

    results = _load_results(predictions_path, video_path)

    src_path = results.video_dict["video_file_path"]
    fps = results.video_dict["fps"]
    height = results.height
    width = results.width
    num_frames = results.num_frames

    frame_start = start if start is not None else 0
    frame_end = end if end is not None else num_frames
    n_frames = frame_end - frame_start

    output_path = (
        Path(output_path) if output_path is not None else Path(predictions_path) / "rendered"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    scale = PRESETS[preset]["scale"]
    out_h = int(height * scale)
    out_w = int(width * scale)
    out_h += out_h % 2
    out_w += out_w % 2

    tracking_data = results.get_tracking_data(interpolate=False)

    bbox_lookup = {}
    for tid, td in tracking_data.items():
        bbox_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["features"].iterrows()}
        logger.debug(f"  Track {tid} ({td['label']}): {len(bbox_lookup[tid])} frames with bbox data")

    track_colors = {}
    for tid in results.track_ids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_out = output_path / f"overlay_{preset}.mp4"
    writer = cv2.VideoWriter(str(overlay_out), fourcc, fps, (out_w, out_h))

    # Pre-compute downscale indices for masks
    _y_idx = _x_idx = None
    if draw_masks and scale != 1.0 and results.has_masks:
        for tid in results.track_ids:
            arr_key = f"{tid}_masks"
            if arr_key in results.zarr_root:
                _H, _W = results.zarr_root[arr_key].shape[1:3]
                _y_idx = np.round(np.linspace(0, _H - 1, out_h)).astype(int)
                _x_idx = np.round(np.linspace(0, _W - 1, out_w)).astype(int)
                break

    _BATCH = 500
    _batch_masks = {}
    _batch_start = None

    logger.info(
        f"Rendering {n_frames} frames | preset={preset} | scale={scale:.0%} "
        f"| output {out_w}×{out_h} px"
    )
    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        ok, orig_frame = cap.read()
        if not ok:
            logger.warning(f"Warning: could not read frame {frame_idx}, stopping early.")
            break

        if draw_masks and results.has_masks \
                and (_batch_start is None or frame_idx >= _batch_start + _BATCH):
            _batch_start = frame_idx
            _batch_end = min(_batch_start + _BATCH, frame_end)
            _batch_masks = {}
            for tid in results.track_ids:
                arr_key = f"{tid}_masks"
                if arr_key not in results.zarr_root:
                    continue
                zarr_arr = results.zarr_root[arr_key]
                end_idx = min(_batch_end, zarr_arr.shape[0])
                if _batch_start >= end_idx:
                    continue
                raw = np.asarray(zarr_arr[_batch_start:end_idx])
                if _y_idx is not None:
                    raw = raw[:, _y_idx[:, None], _x_idx[None, :]]
                _batch_masks[tid] = raw

        j = frame_idx - _batch_start if _batch_start is not None else 0

        if scale != 1.0:
            frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = orig_frame
        overlay = frame_small.astype(np.float32)

        for tid in results.track_ids:
            color_bgr = track_colors[tid]
            label = results.track_id_label[tid]

            if draw_masks and results.has_masks and tid in _batch_masks:
                batch = _batch_masks[tid]
                if j < batch.shape[0]:
                    obj = batch[j] == 1
                    if obj.any():
                        color_f = np.array(color_bgr, dtype=np.float32)
                        overlay[obj] = (1 - alpha) * overlay[obj] + alpha * color_f

            if draw_boxes:
                row = bbox_lookup.get(tid, {}).get(frame_idx)
                if row is not None:
                    x1 = int(row["bbox_x_min"] * scale)
                    y1 = int(row["bbox_y_min"] * scale)
                    x2 = int(row["bbox_x_max"] * scale)
                    y2 = int(row["bbox_y_max"] * scale)
                    lw = max(1, int(2 * scale + 0.5))
                    cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, lw)
                    if draw_labels:
                        txt = f"{label} {tid}"
                        font_scale = max(0.3, 0.45 * scale / 0.5)
                        txt_lw = max(1, lw)
                        cv2.putText(overlay, txt, (x1 + 1, max(y1 - 5, 13)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), txt_lw + 1, cv2.LINE_AA)
                        cv2.putText(overlay, txt, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, txt_lw, cv2.LINE_AA)

        writer.write(np.clip(overlay, 0, 255).astype(np.uint8))

        now = time.time()
        _frame_times.append(now - _t_frame)
        _t_frame = now
        avg_ft = sum(_frame_times) / len(_frame_times)
        fps_disp = 1.0 / avg_ft if avg_ft > 0 else 0.0
        remaining = n_frames - (i + 1)
        eta_s = int(remaining * avg_ft) if avg_ft > 0 else 0
        eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"
        pct = (i + 1) / n_frames
        filled = int(_bar_width * pct)
        bar = "█" * filled + "░" * (_bar_width - filled)
        logger.debug(
            f"  [{bar}] {i + 1}/{n_frames} | {pct:.0%} | {fps_disp:.1f} fps | ETA {eta_str}"
        )

    cap.release()
    writer.release()
    logger.info(f"Overlay saved → {overlay_out}")
