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

import subprocess
import shutil
from pathlib import Path

PRESETS = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def _detect_encoder():
    """Return the best available video encoder: 'h264_nvenc', 'libx264', or 'mp4v'."""
    if shutil.which("ffmpeg") is None:
        return "mp4v"
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True, text=True, timeout=5,
        )
        text = result.stdout + result.stderr
        if "h264_nvenc" in text:
            return "h264_nvenc"
        if "libx264" in text:
            return "libx264"
    except Exception:
        pass
    return "mp4v"


def _open_ffmpeg_writer(output_path, fps, width, height, encoder):
    """Open an ffmpeg subprocess pipe for encoding. Returns a Popen object."""
    if encoder == "h264_nvenc":
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
    ] + codec_args + [str(output_path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)


def _start_mask_prefetch(zarr_root, track_ids, frame_start, frame_end, batch_size):
    """
    Start a background thread that pre-loads zarr mask batches into a queue.

    Masks are stored at their native inference resolution (e.g. 640×384).
    The remap to output resolution is applied per-frame in the blend loop so
    that the prefetch thread never allocates more than
    ``batch_size × n_tracks × inference_H × inference_W`` bytes at once
    instead of ``batch_size × n_tracks × output_H × output_W``.

    Returns a ``queue.Queue`` that yields ``(batch_start, batch_end, masks_dict)``
    tuples in order, followed by a ``None`` sentinel when all batches are done.

    Using ``maxsize=2`` means the thread runs at most one batch ahead of the
    consumer, bounding peak RAM usage while hiding I/O latency.
    """
    import threading
    import queue as _queue_module
    import numpy as np
    import time
    from loguru import logger

    q = _queue_module.Queue(maxsize=2)

    def _worker():
        for bs in range(frame_start, frame_end, batch_size):
            be = min(bs + batch_size, frame_end)
            t0 = time.perf_counter()
            masks = {}
            for tid in track_ids:
                arr_key = f"{tid}_masks"
                if arr_key not in zarr_root:
                    continue
                zarr_arr = zarr_root[arr_key]
                end_idx = min(be, zarr_arr.shape[0])
                if bs >= end_idx:
                    continue
                masks[tid] = np.asarray(zarr_arr[bs:end_idx])
            dt_ms = (time.perf_counter() - t0) * 1000
            logger.debug(
                f"[mask_load] batch {bs}–{be} ({len(masks)} tracks) in {dt_ms:.1f}ms"
            )
            q.put((bs, be, masks))
        q.put(None)  # sentinel — no more batches

    t = threading.Thread(target=_worker, daemon=True, name="mask-prefetch")
    t.start()
    return q


def _letterbox_bounds(mask_h, mask_w, vid_h, vid_w):
    """
    Compute the video-content region within a mask stored at inference resolution.

    YOLO (retina_masks=False) letterboxes frames to fit within imgsz and then
    pads to the nearest multiple of stride (32).  The padding is centred, so the
    actual video content starts at (pad_y, pad_x) within the stored mask.

    Returns
    -------
    content_h, content_w : int
        Height and width of the video content area within the mask.
    pad_y, pad_x : int
        Top and left padding in mask pixels.
    """
    content_h = round(vid_h * mask_w / vid_w)
    if content_h <= mask_h:
        # Width is the constraining dimension; y-axis may have padding
        content_w = mask_w
        pad_y = (mask_h - content_h) // 2
        pad_x = 0
    else:
        # Height is the constraining dimension; x-axis may have padding
        content_h = mask_h
        content_w = round(vid_w * mask_h / vid_h)
        pad_y = 0
        pad_x = (mask_w - content_w) // 2
    return content_h, content_w, pad_y, pad_x


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
        # Masks may be stored at inference resolution; read stored video dims so
        # centroids are returned in video-pixel coordinates, not mask-pixel coords.
        _mask_h, _mask_w = zarr_arr.shape[1], zarr_arr.shape[2]
        _vid_h = zarr_arr.attrs.get('video_height', _mask_h) or _mask_h
        _vid_w = zarr_arr.attrs.get('video_width',  _mask_w) or _mask_w
        # Account for letterbox padding: the mask includes stride-alignment padding
        # that does not correspond to video content; subtract it before scaling.
        _c_h, _c_w, _p_y, _p_x = _letterbox_bounds(_mask_h, _mask_w, _vid_h, _vid_w)

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

            # Convert from downsampled-mask coords → full-mask coords → content coords → video coords
            cx_full = (cx_ds * downsample - _p_x) * _vid_w / _c_w
            cy_full = (cy_ds * downsample - _p_y) * _vid_h / _c_h

            for i, frame_idx in enumerate(range(batch_start, batch_end)):
                if count[i] > 0:
                    centroids[tid][frame_idx] = (float(cx_full[i]), float(cy_full[i]))

            batch_idx += 1
            pct = batch_idx / total_batches
            filled = int(_bar_width * pct)
            bar = "█" * filled + "░" * (_bar_width - filled)
            print(f"  [{bar}] track {tid} | {batch_start}-{batch_end} | {pct:.0%}",
                  end="\r")

    print()
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
        print(f"Warning: cutoff_hz={cutoff_hz} exceeds Nyquist ({nyquist:.1f} Hz); skipping smoothing.")
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

    from loguru import logger
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
        from loguru import logger
        logger.info(f"Tracklet gaps interpolated (cubic spline, max_gap={max_gap}): {total_filled} frame(s) filled")
    return pos_lookup


def _load_results(predictions_path, video_path):
    """Load YOLO_results and optionally override the video path."""
    from octron.yolo_octron.helpers.yolo_results import YOLO_results

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_path}")

    results = YOLO_results(predictions_path, verbose=True,
                           **({"video_path": video_path} if video_path is not None else {}))

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

    print("Bounding-box size report:")
    for tid, s in sorted(stats.items()):
        print(f"  Track {tid} ({s['label']}): max {s['max_w']}×{s['max_h']} px (w×h)")
    print(
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
    size=None,
    mask_centroids=False,
    smooth_cutoff_hz=2.0,
    smooth_order=4,
    alpha=0.4,
    draw_masks=False,
    draw_boxes=False,
    segment_only=False,
    segment_keep_n=0,
    offset=(0, 0),
    start=None,
    end=None,
    min_track_frames=0,
    interpolate_max_gap=0,
    track_ids=None,
    min_observations=0,
    trim=False,
    min_confidence=0.5,
    debug=False,
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
    segment_only : bool
        If True, black out all pixels outside each animal's segmentation mask.
        Requires mask data; pixels with no mask are set to zero.  Default False.
    segment_keep_n : int
        When ``segment_only=True``, keep only the N largest connected components
        of the mask (by pixel area in video space).  0 = keep all components.
        Default 0.
    start : int, optional
        First frame index (inclusive).  Default: beginning of video.
    end : int, optional
        Last frame index (exclusive).  Default: end of video.
    """
    try:
        ox, oy = offset
        offset = (int(ox), int(oy))
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"offset must be a (dx, dy) pair of ints; got {offset!r}"
        ) from e

    import cv2
    import numpy as np
    import time
    from collections import deque
    from loguru import logger
    from octron._logging import setup_logging as _setup_logging
    _setup_logging(debug=debug)

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

    # Pre-filter: combine explicit track_ids with min_observations row-count check
    # so get_tracking_data only loads (and warns about) the CSVs we will render.
    _prefilter_ids = set(track_ids) if track_ids is not None else None
    if min_observations > 0:
        _obs_counts = results.get_observation_counts()
        _obs_ids = {tid for tid, n in _obs_counts.items() if n >= min_observations}
        _prefilter_ids = _obs_ids if _prefilter_ids is None else _prefilter_ids & _obs_ids

    tracking_data = results.get_tracking_data(
        interpolate=False,
        track_ids=list(_prefilter_ids) if _prefilter_ids is not None else None,
    )

    pos_lookup = {}
    bbox_lookup = {}
    for tid, td in tracking_data.items():
        pos_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["data"].iterrows()}
        bbox_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["features"].iterrows()}

    # Only include tracks that have at least one detection in [frame_start, frame_end).
    active_tids = [
        tid for tid in results.track_ids
        if any(frame_start <= f < frame_end for f in pos_lookup.get(tid, {}))
    ]
    skipped_range = len(results.track_ids) - len(active_tids)
    logger.info(
        f"Tracks in range: {len(active_tids)}/{len(results.track_ids)}"
        + (f" ({skipped_range} skipped — no detections in frames {frame_start}–{frame_end})" if skipped_range else "")
    )

    render_tids = [
        tid for tid in active_tids
        if min_track_frames <= 0 or len(pos_lookup.get(tid, {})) >= min_track_frames
    ]
    if min_track_frames > 0:
        skipped_min = len(active_tids) - len(render_tids)
        logger.info(f"Keeping {len(render_tids)}/{len(active_tids)} tracks with ≥{min_track_frames} frames ({skipped_min} skipped)")

    if track_ids is not None:
        missing = set(track_ids) - set(render_tids)
        if missing:
            logger.warning(f"Requested track ID(s) not found: {sorted(missing)}")
        render_tids = [t for t in render_tids if t in set(track_ids)]
    if min_observations > 0:
        render_tids = [t for t in render_tids if len(pos_lookup.get(t, {})) >= min_observations]
    if track_ids is not None or min_observations > 0:
        logger.info(f"Rendering {len(render_tids)}/{len(results.track_ids)} track(s) after filtering")

    if size is None:
        max_dim = 0
        for tid in render_tids:
            feats = tracking_data[tid]["features"]
            max_w = int(np.ceil((feats["bbox_x_max"] - feats["bbox_x_min"]).max()))
            max_h = int(np.ceil((feats["bbox_y_max"] - feats["bbox_y_min"]).max()))
            max_dim = max(max_dim, max_w, max_h)
        size = max_dim + 20 if max_dim > 0 else 160
        logger.info(f"Auto tracklet size: {size}px (largest bbox {max_dim}px + 20px padding)")

    if size is not None:  # always true now, but defensive
        for tid in render_tids:
            td = tracking_data[tid]
            feats = td["features"]
            max_w = int(np.ceil((feats["bbox_x_max"] - feats["bbox_x_min"]).max()))
            max_h = int(np.ceil((feats["bbox_y_max"] - feats["bbox_y_min"]).max()))
            if max(max_w, max_h) > size:
                logger.warning(
                    f"Track {tid} ({td['label']}) has bounding boxes up to "
                    f"{max_w}×{max_h} px, which exceeds size={size}. "
                    f"Consider --tracklet-size {max(max_w, max_h) + 20}."
                )

    if mask_centroids and results.has_masks:
        logger.info("Computing mask centre-of-mass centroids...")
        mask_cent = compute_mask_centroids(
            results.zarr_root, render_tids, frame_start, frame_end,
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
        logger.info(f"Replaced {replaced} centroid(s) with mask CoM")

    if smooth_cutoff_hz and smooth_cutoff_hz > 0:
        butterworth_smooth_tracklet_positions(
            pos_lookup, fps=fps, cutoff_hz=smooth_cutoff_hz, order=smooth_order,
        )

    if interpolate_max_gap and interpolate_max_gap > 0:
        interpolate_tracklet_gaps(pos_lookup, max_gap=interpolate_max_gap)

    # Trim: per-track active span (first→last observation within current range)
    trim_starts = {}
    trim_ends = {}
    if trim:
        for tid in render_tids:
            frames_in_range = [f for f in pos_lookup.get(tid, {}) if frame_start <= f < frame_end]
            if frames_in_range:
                trim_starts[tid] = min(frames_in_range)
                trim_ends[tid] = max(frames_in_range) + 1
            else:
                trim_starts[tid] = frame_start
                trim_ends[tid] = frame_start  # zero-length — writer will be empty
        # Shrink the global decode range to avoid reading frames no track needs
        if trim_starts:
            frame_start = min(trim_starts.values())
            frame_end = max(trim_ends.values())
            n_frames = frame_end - frame_start
            logger.info(f"Trim enabled: decoding frames {frame_start}–{frame_end} (union of track spans)")

    track_colors = {}
    for tid in render_tids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {}
    for tid in render_tids:
        label = results.track_id_label[tid]
        tp = output_path / f"tracklet_{label}_track{tid}_{preset}.mp4"
        writers[tid] = cv2.VideoWriter(str(tp), fourcc, fps, (size, size))

    # Detect inference resolution and letterbox bounds for overlay masks.
    _mask_inf_h = _mask_inf_w = None
    _lb_py = _lb_px = _lb_ch = _lb_cw = 0
    _need_masks = (also_overlay and draw_masks or segment_only) and results.has_masks
    if _need_masks:
        for tid in render_tids:
            arr_key = f"{tid}_masks"
            if arr_key in results.zarr_root:
                _mask_inf_h, _mask_inf_w = results.zarr_root[arr_key].shape[1:3]
                _lb_ch, _lb_cw, _lb_py, _lb_px = _letterbox_bounds(
                    _mask_inf_h, _mask_inf_w, height, width
                )
                break

    _BATCH = 100
    _batch_masks = {}
    _batch_start = -1
    _batch_end = -1

    # Start background zarr prefetch thread when masks are needed.
    _mask_q = None
    if _need_masks:
        logger.info("Starting mask prefetch thread...")
        _mask_q = _start_mask_prefetch(
            results.zarr_root, render_tids,
            frame_start, frame_end, _BATCH,
        )
        _first = _mask_q.get(timeout=120)
        if _first is not None:
            _batch_start, _batch_end, _batch_masks = _first

    logger.info(f"Rendering {n_frames} tracklet frames | preset={preset} | size={size}px | {len(render_tids)} track(s)")
    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    # Debug timing accumulators
    if debug:
        _dbg_n = 0
        _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_crop = 0.0
        _dbg_next_report = time.perf_counter() + 2.0

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        if debug:
            _t0 = time.perf_counter()

        ok, orig_frame = cap.read()
        if not ok:
            logger.warning(f"Could not read frame {frame_idx}, stopping early.")
            break

        if debug:
            _t1 = time.perf_counter()
            _dbg_t_decode += _t1 - _t0

        # Advance to next mask batch when current one is exhausted.
        if _mask_q is not None and frame_idx >= _batch_end:
            _item = _mask_q.get(timeout=120)
            if _item is not None:
                _batch_start, _batch_end, _batch_masks = _item
            else:
                _batch_masks = {}

        j = frame_idx - _batch_start if _batch_start >= 0 else 0

        if debug:
            _t2 = time.perf_counter()
            _dbg_t_queue += _t2 - _t1

        # Build overlay frame if requested
        out_frame = None
        if also_overlay:
            if scale != 1.0:
                frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = orig_frame

            # Build combined mask at inference resolution, crop letterbox,
            # then resize once to output — same approach as run_render.
            if draw_masks and results.has_masks and _batch_masks and _mask_inf_h:
                _comb_mask = np.zeros((_mask_inf_h, _mask_inf_w), dtype=np.uint8)
                _comb_color = np.zeros((_mask_inf_h, _mask_inf_w, 3), dtype=np.uint8)
                _drew_any = False
                for tid in render_tids:
                    if tid in _batch_masks:
                        _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
                        if _conf_row is None or _conf_row["confidence"] < min_confidence:
                            continue
                        batch = _batch_masks[tid]
                        if j < batch.shape[0]:
                            obj = batch[j] == 1
                            if obj.any():
                                _comb_mask[obj] = 255
                                _comb_color[obj] = track_colors[tid]
                                _drew_any = True
                if _drew_any:
                    _mc = _comb_mask[_lb_py:_lb_py + _lb_ch, _lb_px:_lb_px + _lb_cw]
                    _cc = _comb_color[_lb_py:_lb_py + _lb_ch, _lb_px:_lb_px + _lb_cw]
                    mask_full = cv2.resize(_mc, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
                    color_full = cv2.resize(_cc, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                    color_layer = frame_small.copy()
                    color_layer[mask_full] = color_full[mask_full]
                    out_frame = cv2.addWeighted(frame_small, 1.0 - alpha, color_layer, alpha, 0)
                else:
                    out_frame = frame_small.copy()
            else:
                out_frame = frame_small.copy()

            for tid in render_tids:
                color_bgr = track_colors[tid]
                if draw_boxes:
                    row = bbox_lookup.get(tid, {}).get(frame_idx)
                    if row is not None and row["confidence"] >= min_confidence:
                        x1 = int(row["bbox_x_min"] * scale)
                        y1 = int(row["bbox_y_min"] * scale)
                        x2 = int(row["bbox_x_max"] * scale)
                        y2 = int(row["bbox_y_max"] * scale)
                        lw = max(1, int(2 * scale + 0.5))
                        cv2.rectangle(out_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color_bgr, lw)

        if debug:
            _t3 = time.perf_counter()
            _dbg_t_blend += _t3 - _t2

        # Crop each track
        for tid in render_tids:
            if trim and not (trim_starts[tid] <= frame_idx < trim_ends[tid]):
                continue
            _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
            row = pos_lookup.get(tid, {}).get(frame_idx)
            if row is not None and (_conf_row is None or _conf_row["confidence"] < min_confidence):
                row = None  # below confidence threshold — render as black frame
            if row is not None:
                if also_overlay and out_frame is not None:
                    cx = float(row["pos_x"]) * scale + offset[0]
                    cy = float(row["pos_y"]) * scale + offset[1]
                    crop = cv2.getRectSubPix(out_frame, (size, size), (cx, cy))
                else:
                    cx = float(row["pos_x"]) + offset[0]
                    cy = float(row["pos_y"]) + offset[1]
                    crop = cv2.getRectSubPix(orig_frame, (size, size), (cx, cy))

                if segment_only:
                    masked = False
                    if _batch_masks and tid in _batch_masks:
                        batch = _batch_masks[tid]
                        if j < batch.shape[0]:
                            # zarr fill_value=-1 (int8); use == 1 to get clean binary uint8
                            raw_mask = (batch[j] == 1).astype(np.uint8)
                            mask_content = raw_mask[_lb_py:_lb_py + _lb_ch, _lb_px:_lb_px + _lb_cw]
                            # Resize mask to match the source frame resolution used for cropping
                            if also_overlay and out_frame is not None:
                                mask_src = cv2.resize(mask_content, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                            else:
                                mask_src = cv2.resize(mask_content, (width, height), interpolation=cv2.INTER_NEAREST)
                            # Optionally keep only the N largest connected components
                            if segment_keep_n > 0 and mask_src.any():
                                n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_src, connectivity=8)
                                if n_labels > 1:  # label 0 is background
                                    areas = [(stats[i, cv2.CC_STAT_AREA], i) for i in range(1, n_labels)]
                                    areas.sort(reverse=True)
                                    keep = {i for _, i in areas[:segment_keep_n]}
                                    mask_src = np.where(np.isin(labels, list(keep)), mask_src, 0).astype(np.uint8)
                            mask_crop = cv2.getRectSubPix(mask_src.astype(np.float32), (size, size), (cx, cy))
                            crop[mask_crop < 0.5] = 0
                            masked = True
                    if not masked:
                        crop[:] = 0
            else:
                crop = np.zeros((size, size, 3), dtype=np.uint8)
            writers[tid].write(crop)

        if debug:
            _t4 = time.perf_counter()
            _dbg_t_crop += _t4 - _t3
            _dbg_n += 1
            now_pc = _t4
            if now_pc >= _dbg_next_report and _dbg_n > 0:
                logger.debug(
                    f"[decode] {_dbg_t_decode/_dbg_n*1000:.1f}ms  "
                    f"[queue-wait] {_dbg_t_queue/_dbg_n*1000:.1f}ms  "
                    f"[blend] {_dbg_t_blend/_dbg_n*1000:.1f}ms  "
                    f"[crop+write] {_dbg_t_crop/_dbg_n*1000:.1f}ms  "
                    f"(avg over {_dbg_n} frames)"
                )
                _dbg_n = 0
                _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_crop = 0.0
                _dbg_next_report = now_pc + 2.0

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
        print(
            f"  [{bar}] {i + 1}/{n_frames} | {pct:.0%} | {fps_disp:.1f} fps | ETA {eta_str}",
            end="\r",
        )

    print()  # close the \r progress line
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
    tracklet_size=None,
    tracklet_mask_centroids=False,
    tracklet_smooth_cutoff_hz=2.0,
    tracklet_smooth_order=4,
    tracklet_min_frames=0,
    tracklet_interpolate_max_gap=0,
    tracklet_segment_only=False,
    tracklet_segment_keep_n=0,
    tracklet_offset=(0, 0),
    track_ids=None,
    min_observations=0,
    trim=False,
    min_confidence=0.5,
    alpha=0.4,
    draw_masks=True,
    draw_boxes=True,
    draw_labels=True,
    start=None,
    end=None,
    debug=False,
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
            segment_only=tracklet_segment_only,
            segment_keep_n=tracklet_segment_keep_n,
            offset=tracklet_offset,
            start=start,
            end=end,
            min_track_frames=tracklet_min_frames,
            interpolate_max_gap=tracklet_interpolate_max_gap,
            track_ids=track_ids,
            min_observations=min_observations,
            trim=trim,
            min_confidence=min_confidence,
            debug=debug,
        )
        return

    import cv2
    import numpy as np
    import time
    from collections import deque
    from loguru import logger
    from octron._logging import setup_logging as _setup_logging
    _setup_logging(debug=debug)

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

    # Pre-filter: combine explicit track_ids with min_observations row-count check
    # so get_tracking_data only loads (and warns about) the CSVs we will render.
    _prefilter_ids = set(track_ids) if track_ids is not None else None
    if min_observations > 0:
        _obs_counts = results.get_observation_counts()
        _obs_ids = {tid for tid, n in _obs_counts.items() if n >= min_observations}
        _prefilter_ids = _obs_ids if _prefilter_ids is None else _prefilter_ids & _obs_ids

    tracking_data = results.get_tracking_data(
        interpolate=False,
        track_ids=list(_prefilter_ids) if _prefilter_ids is not None else None,
    )

    bbox_lookup = {}
    for tid, td in tracking_data.items():
        bbox_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["features"].iterrows()}

    # Only render tracks that have at least one detection in [frame_start, frame_end).
    active_tids = [
        tid for tid in results.track_ids
        if any(frame_start <= f < frame_end for f in bbox_lookup.get(tid, {}))
    ]
    skipped = len(results.track_ids) - len(active_tids)
    logger.info(
        f"Tracks in range: {len(active_tids)}/{len(results.track_ids)}"
        + (f" ({skipped} skipped — no detections in frames {frame_start}–{frame_end})" if skipped else "")
    )

    render_tids = list(active_tids)
    if track_ids is not None:
        missing = set(track_ids) - set(render_tids)
        if missing:
            logger.warning(f"Requested track ID(s) not found: {sorted(missing)}")
        render_tids = [t for t in render_tids if t in set(track_ids)]
    if min_observations > 0:
        render_tids = [t for t in render_tids if len(bbox_lookup.get(t, {})) >= min_observations]
    if track_ids is not None or min_observations > 0:
        logger.info(f"Rendering {len(render_tids)}/{len(results.track_ids)} track(s) after filtering")

    track_colors = {}
    for tid in render_tids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)

    # Detect best encoder and open writer
    overlay_out = output_path / f"overlay_{preset}.mp4"
    _encoder = _detect_encoder()
    if _encoder != "mp4v":
        logger.info(f"Encoder:  {_encoder} (ffmpeg)")
        _ffmpeg_proc = _open_ffmpeg_writer(overlay_out, fps, out_w, out_h, _encoder)
        _cv2_writer = None
    else:
        logger.info("Encoder:  mp4v (cv2.VideoWriter)")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        _cv2_writer = cv2.VideoWriter(str(overlay_out), fourcc, fps, (out_w, out_h))
        _ffmpeg_proc = None

    # Detect inference resolution and letterbox bounds.
    # Masks are stored at inference resolution (e.g. 640×384) and may include
    # stride-alignment padding.  _letterbox_bounds() gives the content-only
    # slice so upscaling aligns mask pixels with video-coordinate bboxes.
    _mask_inf_h = _mask_inf_w = None
    _lb_py = _lb_px = _lb_ch = _lb_cw = 0
    if draw_masks and results.has_masks:
        for tid in render_tids:
            arr_key = f"{tid}_masks"
            if arr_key in results.zarr_root:
                _mask_inf_h, _mask_inf_w = results.zarr_root[arr_key].shape[1:3]
                _lb_ch, _lb_cw, _lb_py, _lb_px = _letterbox_bounds(
                    _mask_inf_h, _mask_inf_w, height, width
                )
                logger.debug(
                    f"Mask dims: {_mask_inf_w}×{_mask_inf_h}  "
                    f"content: {_lb_cw}×{_lb_ch}  padding: top={_lb_py}px left={_lb_px}px"
                )
                break

    # Batch size for zarr mask reads.  100 frames balances first-frame latency
    # (~1–7s on network) against memory usage; the prefetch thread ensures the
    # next batch is ready before the current one is exhausted.
    _BATCH = 100
    _batch_masks = {}
    _batch_start = -1
    _batch_end = -1

    logger.info(
        f"Rendering {n_frames} frames | preset={preset} | scale={scale:.0%} "
        f"| output {out_w}×{out_h} px"
    )

    # Start background zarr prefetch thread (only when masks are present).
    _mask_q = None
    if draw_masks and results.has_masks:
        logger.info("Starting mask prefetch thread...")
        _mask_q = _start_mask_prefetch(
            results.zarr_root, render_tids,
            frame_start, frame_end, _BATCH,
        )
        # Pull first batch — blocks for initial load (batch_size frames only)
        _first = _mask_q.get(timeout=120)
        if _first is not None:
            _batch_start, _batch_end, _batch_masks = _first
            logger.debug(f"[mask_load] first batch ready: frames {_batch_start}–{_batch_end}")

    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    # Debug timing accumulators (all using perf_counter for consistency)
    if debug:
        _dbg_n = 0
        _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_encode = 0.0
        _dbg_next_report = time.perf_counter() + 2.0

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        if debug:
            _t0 = time.perf_counter()

        ok, orig_frame = cap.read()
        if not ok:
            print(f"\nWarning: could not read frame {frame_idx}, stopping early.")
            break

        if debug:
            _t1 = time.perf_counter()
            _dbg_t_decode += _t1 - _t0

        # Advance to next mask batch when the current one is exhausted.
        # The prefetch thread already has the next batch loading in the background,
        # so queue.get() returns almost instantly in the common case.
        # Any non-trivial wait here means zarr I/O is slower than rendering.
        if _mask_q is not None and frame_idx >= _batch_end:
            _item = _mask_q.get(timeout=120)
            if _item is not None:
                _batch_start, _batch_end, _batch_masks = _item
            else:
                _batch_masks = {}  # no more masks

        j = frame_idx - _batch_start if _batch_start >= 0 else 0

        if debug:
            _t2 = time.perf_counter()
            _dbg_t_queue += _t2 - _t1

        if scale != 1.0:
            frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = orig_frame

        # Build combined mask at inference resolution, crop letterbox padding,
        # then resize to output once.  This replaces n_tracks per-frame fancy-
        # index remap ops (each allocating out_H×out_W bools) with a single
        # cv2.resize call, giving ~10× speedup on large outputs with many tracks.
        if draw_masks and results.has_masks and _batch_masks and _mask_inf_h:
            _comb_mask = np.zeros((_mask_inf_h, _mask_inf_w), dtype=np.uint8)
            _comb_color = np.zeros((_mask_inf_h, _mask_inf_w, 3), dtype=np.uint8)
            _drew_any = False
            for tid in render_tids:
                if tid in _batch_masks:
                    _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
                    if _conf_row is None or _conf_row["confidence"] < min_confidence:
                        continue
                    batch = _batch_masks[tid]
                    if j < batch.shape[0]:
                        obj = batch[j] == 1
                        if obj.any():
                            _comb_mask[obj] = 255
                            _comb_color[obj] = track_colors[tid]
                            _drew_any = True
            if _drew_any:
                # Crop letterbox padding so the content region maps 1:1 to output
                _mc = _comb_mask[_lb_py:_lb_py + _lb_ch, _lb_px:_lb_px + _lb_cw]
                _cc = _comb_color[_lb_py:_lb_py + _lb_ch, _lb_px:_lb_px + _lb_cw]
                mask_full = cv2.resize(_mc, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
                color_full = cv2.resize(_cc, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                color_layer = frame_small.copy()
                color_layer[mask_full] = color_full[mask_full]
                out_frame = cv2.addWeighted(frame_small, 1.0 - alpha, color_layer, alpha, 0)
            else:
                out_frame = frame_small.copy()
        else:
            out_frame = frame_small.copy()

        for tid in render_tids:
            color_bgr = track_colors[tid]
            label = results.track_id_label[tid]

            if draw_boxes:
                row = bbox_lookup.get(tid, {}).get(frame_idx)
                if row is not None and row["confidence"] >= min_confidence:
                    x1 = int(row["bbox_x_min"] * scale)
                    y1 = int(row["bbox_y_min"] * scale)
                    x2 = int(row["bbox_x_max"] * scale)
                    y2 = int(row["bbox_y_max"] * scale)
                    lw = max(1, int(2 * scale + 0.5))
                    cv2.rectangle(out_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color_bgr, lw)
                    if draw_labels:
                        txt = f"{label} {tid}"
                        font_scale = max(0.3, 0.45 * scale / 0.5)
                        txt_lw = max(1, lw)
                        cv2.putText(out_frame, txt, (x1 + 1, max(y1 - 5, 13)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), txt_lw + 1, cv2.LINE_AA)
                        cv2.putText(out_frame, txt, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, txt_lw, cv2.LINE_AA)

        if debug:
            _t3 = time.perf_counter()
            _dbg_t_blend += _t3 - _t2

        if _ffmpeg_proc is not None:
            _ffmpeg_proc.stdin.write(out_frame.tobytes())
        else:
            _cv2_writer.write(out_frame)

        if debug:
            _t4 = time.perf_counter()
            _dbg_t_encode += _t4 - _t3
            _dbg_n += 1
            now = _t4
            if now >= _dbg_next_report and _dbg_n > 0:
                logger.debug(
                    f"[decode] {_dbg_t_decode/_dbg_n*1000:.1f}ms  "
                    f"[queue-wait] {_dbg_t_queue/_dbg_n*1000:.1f}ms  "
                    f"[blend] {_dbg_t_blend/_dbg_n*1000:.1f}ms  "
                    f"[encode] {_dbg_t_encode/_dbg_n*1000:.1f}ms  "
                    f"(avg over {_dbg_n} frames)"
                )
                _dbg_n = 0
                _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_encode = 0.0
                _dbg_next_report = now + 2.0

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
        print(
            f"  [{bar}] {i + 1}/{n_frames} | {pct:.0%} | {fps_disp:.1f} fps | ETA {eta_str}",
            end="\r",
        )

    print()
    cap.release()
    if _ffmpeg_proc is not None:
        _ffmpeg_proc.stdin.close()
        rc = _ffmpeg_proc.wait()
        if rc != 0:
            logger.warning(f"ffmpeg exited with code {rc} — output may be incomplete.")
    else:
        _cv2_writer.release()
    logger.info(f"Overlay saved → {overlay_out}")
