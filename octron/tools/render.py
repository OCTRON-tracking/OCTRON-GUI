"""
OCTRON video rendering pipeline.

Reads prediction output (zarr masks + CSV tracks + metadata) produced by
``octron predict`` and renders annotated videos.

Public functions
----------------
run_render      : Render an overlay video (masks, boxes, labels) from predictions.
run_tracklets   : Render one stabilised crop video per tracked animal.
report_bbox_sizes : Report bounding-box sizes to help choose tracklet crop size.
"""

import subprocess
import tempfile
from pathlib import Path

from octron.tools._ffmpeg import h264_codec_args, EVEN_DIM_YUV420P, resolve_encoder, _FfmpegWriter

PRESETS = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}


def _validate_render_args(preset, min_confidence, alpha):
    """Defensive input checks shared by run_render and run_tracklets.

    This is the tool-layer guard for programmatic/notebook callers. The
    ``octron render`` CLI also validates these before calling in (friendly
    ``typer.BadParameter`` for preset, Typer ``min``/``max`` for min_confidence
    and alpha); the overlap is intentional layering, not redundancy.
    """
    if preset not in PRESETS:
        raise ValueError(
            f"preset must be one of {sorted(PRESETS)}; got {preset!r}"
        )
    if not 0.0 <= float(min_confidence) <= 1.0:
        raise ValueError(
            f"min_confidence must be in [0, 1]; got {min_confidence!r}"
        )
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(
            f"alpha must be in [0, 1]; got {alpha!r}"
        )


def _coerce_track_ids(track_ids):
    """Coerce track_ids to list[int] or None (tool layer, for programmatic callers).

    Accepts None, int, str ('1,3,5'), or an iterable of ints. The CLI already
    pre-parses its comma-separated ``--track-ids`` string; this defensively
    handles the other shapes so notebook/library callers can pass a string or
    ints directly.
    """
    if track_ids is None:
        return None
    if isinstance(track_ids, str):
        try:
            return [int(x.strip()) for x in track_ids.split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(
                f"track_ids string must be comma-separated ints (e.g. '1,3,5'); got {track_ids!r}"
            ) from e
    if isinstance(track_ids, int):
        return [track_ids]
    try:
        return [int(x) for x in track_ids]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"track_ids must be a list/tuple/str of ints or a single int; got {track_ids!r}"
        ) from e


def _select_render_frames(per_track_frames, frame_start, frame_end):
    """Sorted unique union of per-track frame indices within ``[frame_start, frame_end)``.

    ``per_track_frames`` maps ``track_id -> iterable of frame indices`` that will
    render content (used by ``--skip-empty``). Frames outside the range are
    dropped. Pure/stdlib-only so it is cheap to unit-test.
    """
    frames = set()
    for track_frames in per_track_frames.values():
        frames.update(f for f in track_frames if frame_start <= f < frame_end)
    return sorted(frames)


# ---------------------------------------------------------------------------
# Encoder helpers
# ---------------------------------------------------------------------------

def _open_ffmpeg_writer(output_path, fps, width, height, encoder):
    """Open an ffmpeg subprocess pipe for encoding. Returns a _FfmpegWriter.

    ffmpeg's stderr is captured to a temp file (via the writer) so that if
    ffmpeg exits early -- e.g. h264_nvenc can't initialise -- the real reason is
    surfaced instead of an opaque broken-pipe error on the first frame write.
    """
    codec_args = h264_codec_args(encoder, crf=20, preset="fast")
    # Apply the shared even-dimension + yuv420p filter (mirrors transcode.py).
    # Without it, encoding rgb24 input defaults to yuv444p (4:4:4), which macOS
    # players (QuickTime/AVFoundation/Preview) cannot decode; odd crop sizes
    # (e.g. the 47px auto tracklet size) compound the problem. The filter rounds
    # each dimension down to even and converts to the widely-playable 4:2:0.
    cmd = [
        "ffmpeg", "-y", "-nostats", "-loglevel", "warning",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vf", EVEN_DIM_YUV420P,
    ] + codec_args + [str(output_path)]
    # stderr -> temp file (not an unread PIPE, which could deadlock the encode).
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_file)
    return _FfmpegWriter(proc, stderr_file, encoder, output_path)


class _MaskPrefetchError:
    """Sentinel placed on the prefetch queue when the worker thread fails.

    Lets a zarr/network read error surface in the consumer immediately instead
    of blocking on an empty queue until the ``get()`` timeout fires.
    """
    __slots__ = ("exc",)

    def __init__(self, exc):
        self.exc = exc


def _start_mask_prefetch(zarr_root, track_ids, frame_start, frame_end, batch_size):
    """
    Start a background thread that pre-loads zarr mask batches into a queue.

    Masks are stored at full video resolution (retina_masks=True), and the
    remap to output resolution is applied per-frame in the blend loop.  The
    prefetch thread therefore holds at most
    ``batch_size × n_tracks × video_H × video_W`` bytes at once.

    Returns a ``queue.Queue`` that yields ``(batch_start, batch_end, masks_dict)``
    tuples in order, followed by a ``None`` sentinel when all batches are done.
    If the worker hits an error (e.g. a failed zarr read) it enqueues a
    ``_MaskPrefetchError`` instead; consume the queue via ``_next_mask_batch`` so
    that error is re-raised in the consumer rather than blocking until timeout.

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
        try:
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
        except Exception as exc:  # zarr/network read failure — surface to consumer
            logger.error(f"Mask prefetch thread failed: {exc}")
            q.put(_MaskPrefetchError(exc))
        else:
            q.put(None)  # sentinel — no more batches

    t = threading.Thread(target=_worker, daemon=True, name="mask-prefetch")
    t.start()
    return q


def _next_mask_batch(q, timeout=120):
    """Return the next prefetch item, re-raising worker errors.

    Yields the worker's ``(batch_start, batch_end, masks)`` tuple, or ``None``
    when the worker has finished.  If the worker thread raised, the original
    exception is re-raised here (in the consumer) instead of the consumer
    blocking on the queue until ``timeout`` and failing with an opaque Empty.
    """
    item = q.get(timeout=timeout)
    if isinstance(item, _MaskPrefetchError):
        raise RuntimeError("Mask prefetch thread failed") from item.exc
    return item


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
    smooth_sigma=2.0,
    alpha=0.4,
    draw_masks=False,
    draw_boxes=False,
    segment_only=False,
    segment_keep_n=0,
    offset=(0, 0),
    start=None,
    end=None,
    min_track_frames=0,
    interpolate_limit=0,
    track_ids=None,
    min_observations=0,
    trim=False,
    skip_empty=False,
    min_confidence=0.5,
    encoder="auto",
    debug=False,
):
    """
    Render one stabilised crop video per tracked animal.

    Crops are centred on the smoothed centroid trajectory and extracted at
    sub-pixel accuracy via bilinear interpolation (``cv2.getRectSubPix``).

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory produced by ``octron predict``.
    video_path : str or Path, optional
        Path to the original video.  Auto-detected if alongside
        ``octron_predictions/``.  Pass an optic-flow video here to crop flow
        fields instead of raw frames.
    output_path : str or Path, optional
        Output directory.  Defaults to ``<predictions_path>/rendered/``.
    preset : {'preview', 'draft', 'final'}
        Sets the output resolution for overlay mode (``also_overlay=True``) only;
        the raw tracklet crops are always at full resolution.
    also_overlay : bool
        If True, render masks and boxes onto the tracklet crops (useful for
        inspecting centroid placement).  Default False.
    size : int, optional
        Side length in pixels of each square tracklet crop.  If None (the CLI
        default ``auto``), the size is auto-computed from the largest bounding
        box (plus 20px padding), falling back to 160px when no boxes exist.
    smooth_sigma : float
        Gaussian smoothing strength (standard deviation in frames) applied to the
        centroid trajectories via core ``get_tracking_data(sigma=...)``.  0 = off.
        Default 2.0.
    interpolate_limit : int
        Fill interior gaps in the centroid trajectory by linear interpolation in
        core ``get_tracking_data``.  Caps how many consecutive missing frames to
        bridge (longer interior gaps are partially filled; leading/trailing gaps
        are never filled).  0 disables interpolation.  Default 0.
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
    offset : tuple of int, optional
        ``(dx, dy)`` shift of each crop centre, in source-video pixels (applied
        identically whether or not ``also_overlay`` is set).  Positive values
        move the crop right/down.  Default ``(0, 0)``.
    skip_empty : bool, optional
        If True, render only frames that show content for a track (a detection
        at/above ``min_confidence``). Each track's crop keeps only its own such
        frames, so per-track videos can differ in length. Supersedes ``trim``.
        Default False.
    start : int, optional
        First frame index (inclusive).  Default: beginning of video.
    end : int, optional
        Last frame index (exclusive).  Default: end of video.
    """
    _validate_render_args(preset, min_confidence, alpha)
    track_ids = _coerce_track_ids(track_ids)
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

    # Interpolation and smoothing are done in core get_tracking_data so the CLI
    # render path and the napari GUI share one implementation.  --tracklet-interpolate
    # maps to the pandas consecutive-NaN `limit` (0 = off); --tracklet-smooth-sigma
    # to `sigma`.  Interpolation runs first, then smoothing (core order).
    # min_observations is applied as a cheap CSV row-count pre-filter inside
    # get_tracking_data, so tracks below threshold are never parsed/warned about.
    _interp_limit = interpolate_limit if (interpolate_limit and interpolate_limit > 0) else None
    tracking_data = results.get_tracking_data(
        interpolate=_interp_limit is not None,
        interpolate_limit=_interp_limit,
        sigma=smooth_sigma if (smooth_sigma and smooth_sigma > 0) else 0,
        track_ids=track_ids,
        min_observations=min_observations,
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

    # --skip-empty supersedes --trim (it removes all gaps, not just leading/
    # trailing ones).
    if skip_empty and trim:
        logger.info("--skip-empty is set; ignoring --trim (skip-empty already removes gaps).")
        trim = False

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

    # --skip-empty: render only frames that show content for a track (a detection
    # at/above min_confidence). bbox_lookup keys are exactly the predicted frames
    # (one CSV row per prediction; consistent with the zarr 'annotated_frames'
    # attribute) and carry confidence. Each track keeps its own frames, so the
    # decode loop walks the union and writes a crop only on that track's frames.
    nonempty_frames = {}
    if skip_empty:
        nonempty_frames = {
            tid: {
                f for f, row in bbox_lookup.get(tid, {}).items()
                if frame_start <= f < frame_end and row.get("confidence", 1.0) >= min_confidence
            }
            for tid in render_tids
        }
        frames_to_render = _select_render_frames(nonempty_frames, frame_start, frame_end)
        n_frames = len(frames_to_render)
        if not frames_to_render:
            logger.warning(
                f"--skip-empty: no detections at/above min_confidence={min_confidence} "
                f"in frames {frame_start}–{frame_end}; tracklet video(s) will be empty."
            )
    else:
        frames_to_render = range(frame_start, frame_end)

    track_colors = {}
    for tid in render_tids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (r, g, b)  # RGB to match FastVideoReader frames

    # One ffmpeg pipe per tracklet (replaces cv2.VideoWriter).
    _encoder = resolve_encoder(encoder)
    logger.info(f"Encoder:  {_encoder} (ffmpeg)")
    writers = {}
    for tid in render_tids:
        label = results.track_id_label[tid]
        tp = output_path / f"tracklet_{label}_track{tid}_{preset}.mp4"
        writers[tid] = _open_ffmpeg_writer(tp, fps, size, size, _encoder)

    _need_masks = (also_overlay and draw_masks or segment_only) and results.has_masks

    def _read_mask_frame(tid, frame_idx):
        """Read one (H, W) mask for *tid* at an absolute frame index, or None.

        Used by --skip-empty (random access) in place of the contiguous prefetch.
        Mask zarr arrays are full-length and indexed by absolute frame index.
        """
        root = results.zarr_root
        key = f"{tid}_masks"
        if root is None or key not in root:
            return None
        arr = root[key]
        if frame_idx >= arr.shape[0]:
            return None
        return np.asarray(arr[frame_idx])

    _BATCH = 100
    _batch_masks = {}
    _batch_start = -1
    _batch_end = -1

    # Start background zarr prefetch thread when masks are needed. Skipped for
    # --skip-empty, which reads the few needed masks on demand instead.
    _mask_q = None
    if _need_masks and not skip_empty:
        logger.info("Starting mask prefetch thread...")
        _mask_q = _start_mask_prefetch(
            results.zarr_root, render_tids,
            frame_start, frame_end, _BATCH,
        )
        _first = _next_mask_batch(_mask_q)
        if _first is not None:
            _batch_start, _batch_end, _batch_masks = _first

    logger.info(f"Rendering {n_frames} tracklet frames | preset={preset} | size={size}px | {len(render_tids)} track(s)")

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    # Debug timing accumulators
    if debug:
        _dbg_n = 0
        _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_crop = 0.0
        _dbg_next_report = time.perf_counter() + 2.0

    for i, frame_idx in enumerate(frames_to_render):
        if debug:
            _t0 = time.perf_counter()

        try:
            orig_frame = results.video[frame_idx]
        except Exception:
            logger.warning(f"Could not read frame {frame_idx}, stopping early.")
            break
        if orig_frame is None:
            logger.warning(f"Could not read frame {frame_idx}, stopping early.")
            break
        orig_frame = np.ascontiguousarray(orig_frame)

        if debug:
            _t1 = time.perf_counter()
            _dbg_t_decode += _t1 - _t0

        # Advance to next mask batch when current one is exhausted.
        if _mask_q is not None and frame_idx >= _batch_end:
            _item = _next_mask_batch(_mask_q)
            if _item is not None:
                _batch_start, _batch_end, _batch_masks = _item
            else:
                _batch_masks = {}

        j = frame_idx - _batch_start if _batch_start >= 0 else 0

        if debug:
            _t2 = time.perf_counter()
            _dbg_t_queue += _t2 - _t1

        # Masks for this frame: prefetch batch (dense) or on-demand random reads
        # (--skip-empty). Maps track_id -> (H, W) array.
        cur_masks = {}
        if _need_masks:
            if skip_empty:
                for _tid in render_tids:
                    if frame_idx in nonempty_frames[_tid]:
                        _m = _read_mask_frame(_tid, frame_idx)
                        if _m is not None:
                            cur_masks[_tid] = _m
            elif _batch_masks:
                for _tid in render_tids:
                    _batch = _batch_masks.get(_tid)
                    if _batch is not None and j < _batch.shape[0]:
                        cur_masks[_tid] = _batch[j]

        # Build overlay frame if requested
        out_frame = None
        if also_overlay:
            if orig_frame.shape[1] != out_w or orig_frame.shape[0] != out_h:
                frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
            else:
                frame_small = orig_frame

            # Build combined mask at video resolution, then resize once to output.
            if draw_masks and results.has_masks and cur_masks:
                _comb_mask = np.zeros((height, width), dtype=np.uint8)
                _comb_color = np.zeros((height, width, 3), dtype=np.uint8)
                _drew_any = False
                for tid in render_tids:
                    if tid in cur_masks:
                        _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
                        if _conf_row is None or _conf_row.get("confidence", 1.0) < min_confidence:
                            continue
                        obj = cur_masks[tid] == 1
                        if obj.any():
                            _comb_mask[obj] = 255
                            _comb_color[obj] = track_colors[tid]
                            _drew_any = True
                if _drew_any:
                    mask_full = cv2.resize(_comb_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
                    color_full = cv2.resize(_comb_color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                    color_layer = frame_small.copy()
                    color_layer[mask_full] = color_full[mask_full]
                    out_frame = cv2.addWeighted(frame_small, 1.0 - alpha, color_layer, alpha, 0)
                else:
                    out_frame = frame_small.copy()
            else:
                out_frame = frame_small.copy()

            for tid in render_tids:
                color_rgb = track_colors[tid]
                if draw_boxes:
                    row = bbox_lookup.get(tid, {}).get(frame_idx)
                    if row is not None and row["confidence"] >= min_confidence:
                        x1 = int(row["bbox_x_min"] * scale)
                        y1 = int(row["bbox_y_min"] * scale)
                        x2 = int(row["bbox_x_max"] * scale)
                        y2 = int(row["bbox_y_max"] * scale)
                        lw = max(1, int(2 * scale + 0.5))
                        cv2.rectangle(out_frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                        cv2.rectangle(out_frame, (x1, y1), (x2, y2), color_rgb, lw)

        if debug:
            _t3 = time.perf_counter()
            _dbg_t_blend += _t3 - _t2

        # Crop each track
        for tid in render_tids:
            if skip_empty and frame_idx not in nonempty_frames[tid]:
                continue  # this track has no in-confidence detection here
            if trim and not (trim_starts[tid] <= frame_idx < trim_ends[tid]):
                continue
            _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
            row = pos_lookup.get(tid, {}).get(frame_idx)
            if row is not None and (_conf_row is None or _conf_row["confidence"] < min_confidence):
                row = None  # below confidence threshold — render as black frame
            if row is not None:
                if also_overlay and out_frame is not None:
                    # offset is in source-video pixels; scale it together with the
                    # centroid so --tracklet-offset means the same in both modes.
                    cx = (float(row["pos_x"]) + offset[0]) * scale
                    cy = (float(row["pos_y"]) + offset[1]) * scale
                    crop = cv2.getRectSubPix(out_frame, (size, size), (cx, cy))
                else:
                    # raw crop from the full-resolution frame; offset is already
                    # in source-video pixels.
                    cx = float(row["pos_x"]) + offset[0]
                    cy = float(row["pos_y"]) + offset[1]
                    crop = cv2.getRectSubPix(orig_frame, (size, size), (cx, cy))

                if segment_only:
                    masked = False
                    if tid in cur_masks:
                        # zarr fill_value=-1 (int8); use == 1 to get clean binary uint8
                        raw_mask = (cur_masks[tid] == 1).astype(np.uint8)
                        # Resize mask to match the source frame resolution used for cropping
                        if also_overlay and out_frame is not None:
                            mask_src = cv2.resize(raw_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                        else:
                            mask_src = cv2.resize(raw_mask, (width, height), interpolation=cv2.INTER_NEAREST)
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
            writers[tid].write(np.ascontiguousarray(crop).tobytes())

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
    for w in writers.values():
        w.close()
    logger.info(f"Tracklet(s) saved → {output_path}")


def run_render(
    predictions_path,
    video_path=None,
    output_path=None,
    preset="draft",
    tracklets=False,
    also_overlay=False,
    tracklet_size=None,
    tracklet_smooth_sigma=2.0,
    tracklet_min_frames=0,
    tracklet_interpolate_limit=0,
    tracklet_segment_only=False,
    tracklet_segment_keep_n=0,
    tracklet_offset=(0, 0),
    track_ids=None,
    min_observations=0,
    trim=False,
    skip_empty=False,
    min_confidence=0.5,
    alpha=0.4,
    draw_masks=True,
    draw_boxes=True,
    draw_labels=True,
    start=None,
    end=None,
    encoder="auto",
    debug=False,
):
    """
    Render annotated video(s) from OCTRON prediction output.

    When ``tracklets=True``, delegates to ``run_tracklets`` and skips the
    overlay video.  When ``tracklets=False``, renders the overlay video only.

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory produced by ``octron predict``.
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
    tracklet_size : int, optional
        Side length in pixels of each tracklet crop.  If None (CLI default
        ``auto``), auto-computed from the largest bounding box (+20px padding),
        falling back to 160px.
    tracklet_smooth_sigma : float
        Gaussian smoothing strength (standard deviation in frames) for centroid
        smoothing.  0 = off.  Default 2.0.
    alpha : float
        Mask overlay opacity (0–1).  Default 0.4.
    draw_masks : bool
        Render segmentation mask fills.  Default True (overlay); False (tracklets).
    draw_boxes : bool
        Render bounding boxes.  Default True (overlay); False (tracklets).
    draw_labels : bool
        Render label text above bounding boxes (overlay only; never on tracklets).
        Default True.
    skip_empty : bool, optional
        If True, render only frames that have at least one detection at/above
        ``min_confidence`` (drops the empty frames left by ``predict
        --skip-frames``). Default False.
    start : int, optional
        First frame index (inclusive).
    end : int, optional
        Last frame index (exclusive).
    """
    _validate_render_args(preset, min_confidence, alpha)
    track_ids = _coerce_track_ids(track_ids)

    if tracklets:
        run_tracklets(
            predictions_path=predictions_path,
            video_path=video_path,
            output_path=output_path,
            preset=preset,
            also_overlay=also_overlay,
            size=tracklet_size,
            smooth_sigma=tracklet_smooth_sigma,
            alpha=alpha,
            draw_masks=draw_masks,
            draw_boxes=draw_boxes,
            segment_only=tracklet_segment_only,
            segment_keep_n=tracklet_segment_keep_n,
            offset=tracklet_offset,
            start=start,
            end=end,
            min_track_frames=tracklet_min_frames,
            interpolate_limit=tracklet_interpolate_limit,
            track_ids=track_ids,
            min_observations=min_observations,
            trim=trim,
            skip_empty=skip_empty,
            min_confidence=min_confidence,
            encoder=encoder,
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

    # min_observations is applied as a cheap CSV row-count pre-filter inside
    # get_tracking_data, so tracks below threshold are never parsed/warned about.
    tracking_data = results.get_tracking_data(
        interpolate=False,
        track_ids=track_ids,
        min_observations=min_observations,
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
        track_colors[tid] = (r, g, b)  # RGB to match FastVideoReader frames

    # --skip-empty: render only frames where at least one track has a detection
    # at/above min_confidence (drops the empty frames left by predict
    # --skip-frames). bbox_lookup keys are exactly the predicted frames and carry
    # confidence (consistent with the zarr 'annotated_frames' attribute).
    nonempty_frames = {}
    if skip_empty:
        nonempty_frames = {
            tid: {
                f for f, row in bbox_lookup.get(tid, {}).items()
                if frame_start <= f < frame_end and row.get("confidence", 1.0) >= min_confidence
            }
            for tid in render_tids
        }
        frames_to_render = _select_render_frames(nonempty_frames, frame_start, frame_end)
        n_frames = len(frames_to_render)
        if not frames_to_render:
            logger.warning(
                f"--skip-empty: no detections at/above min_confidence={min_confidence} "
                f"in frames {frame_start}–{frame_end}; overlay video will be empty."
            )
    else:
        frames_to_render = range(frame_start, frame_end)

    # Resolve the encoder (honours --encoder/--no-nvenc) and open the ffmpeg writer
    overlay_out = output_path / f"overlay_{preset}.mp4"
    _encoder = resolve_encoder(encoder)
    logger.info(f"Encoder:  {_encoder} (ffmpeg)")
    _ffmpeg_writer = _open_ffmpeg_writer(overlay_out, fps, out_w, out_h, _encoder)

    # Batch size for zarr mask reads.  100 frames balances first-frame latency
    # (~1–7s on network) against memory usage; the prefetch thread ensures the
    # next batch is ready before the current one is exhausted.
    _need_masks = draw_masks and results.has_masks
    _BATCH = 100
    _batch_masks = {}
    _batch_start = -1
    _batch_end = -1

    def _read_mask_frame(tid, frame_idx):
        """Read one (H, W) mask for *tid* at an absolute frame index, or None.

        Used by --skip-empty (random access) in place of the contiguous prefetch.
        Mask zarr arrays are full-length and indexed by absolute frame index.
        """
        root = results.zarr_root
        key = f"{tid}_masks"
        if root is None or key not in root:
            return None
        arr = root[key]
        if frame_idx >= arr.shape[0]:
            return None
        return np.asarray(arr[frame_idx])

    logger.info(
        f"Rendering {n_frames} frames | preset={preset} | scale={scale:.0%} "
        f"| output {out_w}×{out_h} px"
    )

    # Start background zarr prefetch thread (only when masks are present). Skipped
    # for --skip-empty, which reads the few needed masks on demand instead.
    _mask_q = None
    if _need_masks and not skip_empty:
        logger.info("Starting mask prefetch thread...")
        _mask_q = _start_mask_prefetch(
            results.zarr_root, render_tids,
            frame_start, frame_end, _BATCH,
        )
        # Pull first batch — blocks for initial load (batch_size frames only)
        _first = _next_mask_batch(_mask_q)
        if _first is not None:
            _batch_start, _batch_end, _batch_masks = _first
            logger.debug(f"[mask_load] first batch ready: frames {_batch_start}–{_batch_end}")

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    # Debug timing accumulators (all using perf_counter for consistency)
    if debug:
        _dbg_n = 0
        _dbg_t_decode = _dbg_t_queue = _dbg_t_blend = _dbg_t_encode = 0.0
        _dbg_next_report = time.perf_counter() + 2.0

    for i, frame_idx in enumerate(frames_to_render):
        if debug:
            _t0 = time.perf_counter()

        try:
            orig_frame = results.video[frame_idx]
        except Exception:
            print(f"\nWarning: could not read frame {frame_idx}, stopping early.")
            break
        if orig_frame is None:
            print(f"\nWarning: could not read frame {frame_idx}, stopping early.")
            break
        orig_frame = np.ascontiguousarray(orig_frame)

        if debug:
            _t1 = time.perf_counter()
            _dbg_t_decode += _t1 - _t0

        # Advance to next mask batch when the current one is exhausted.
        # The prefetch thread already has the next batch loading in the background,
        # so queue.get() returns almost instantly in the common case.
        # Any non-trivial wait here means zarr I/O is slower than rendering.
        if _mask_q is not None and frame_idx >= _batch_end:
            _item = _next_mask_batch(_mask_q)
            if _item is not None:
                _batch_start, _batch_end, _batch_masks = _item
            else:
                _batch_masks = {}  # no more masks

        j = frame_idx - _batch_start if _batch_start >= 0 else 0

        if debug:
            _t2 = time.perf_counter()
            _dbg_t_queue += _t2 - _t1

        # Masks for this frame: prefetch batch (dense) or on-demand random reads
        # (--skip-empty). Maps track_id -> (H, W) array.
        cur_masks = {}
        if _need_masks:
            if skip_empty:
                for _tid in render_tids:
                    if frame_idx in nonempty_frames[_tid]:
                        _m = _read_mask_frame(_tid, frame_idx)
                        if _m is not None:
                            cur_masks[_tid] = _m
            elif _batch_masks:
                for _tid in render_tids:
                    _batch = _batch_masks.get(_tid)
                    if _batch is not None and j < _batch.shape[0]:
                        cur_masks[_tid] = _batch[j]

        if orig_frame.shape[1] != out_w or orig_frame.shape[0] != out_h:
            frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = orig_frame

        # Build combined mask at video resolution, then resize to output once.
        # This replaces n_tracks per-frame fancy-index remap ops (each allocating
        # out_H×out_W bools) with a single cv2.resize call, giving ~10× speedup on
        # large outputs with many tracks.
        if draw_masks and results.has_masks and cur_masks:
            _comb_mask = np.zeros((height, width), dtype=np.uint8)
            _comb_color = np.zeros((height, width, 3), dtype=np.uint8)
            _drew_any = False
            for tid in render_tids:
                if tid in cur_masks:
                    _conf_row = bbox_lookup.get(tid, {}).get(frame_idx)
                    if _conf_row is None or _conf_row.get("confidence", 1.0) < min_confidence:
                        continue
                    obj = cur_masks[tid] == 1
                    if obj.any():
                        _comb_mask[obj] = 255
                        _comb_color[obj] = track_colors[tid]
                        _drew_any = True
            if _drew_any:
                mask_full = cv2.resize(_comb_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST) > 0
                color_full = cv2.resize(_comb_color, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                color_layer = frame_small.copy()
                color_layer[mask_full] = color_full[mask_full]
                out_frame = cv2.addWeighted(frame_small, 1.0 - alpha, color_layer, alpha, 0)
            else:
                out_frame = frame_small.copy()
        else:
            out_frame = frame_small.copy()

        for tid in render_tids:
            color_rgb = track_colors[tid]
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
                    cv2.rectangle(out_frame, (x1, y1), (x2, y2), color_rgb, lw)
                    if draw_labels:
                        txt = f"{label} {tid}"
                        font_scale = max(0.3, 0.45 * scale / 0.5)
                        txt_lw = max(1, lw)
                        cv2.putText(out_frame, txt, (x1 + 1, max(y1 - 5, 13)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), txt_lw + 1, cv2.LINE_AA)
                        cv2.putText(out_frame, txt, (x1, max(y1 - 6, 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_rgb, txt_lw, cv2.LINE_AA)

        if debug:
            _t3 = time.perf_counter()
            _dbg_t_blend += _t3 - _t2

        _ffmpeg_writer.write(np.ascontiguousarray(out_frame).tobytes())

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
    _ffmpeg_writer.close()
    logger.info(f"Overlay saved → {overlay_out}")
