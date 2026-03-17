"""
OCTRON video rendering pipeline.

Reads prediction output (zarr masks + CSV tracks + metadata) produced by
``octron analyze`` and renders annotated videos.

Two output modes (combinable via --tracklets flag):
  - Overlay video   : original frames with masks/boxes and track IDs burned in.
  - Tracklet videos : one small crop video per tracked animal, centred on the
    animal's bounding box throughout the sequence.
"""

from pathlib import Path

PRESETS = {
    "preview": {"scale": 0.25},
    "draft": {"scale": 0.5},
    "final": {"scale": 1.0},
}


def report_bbox_sizes(predictions_path):
    """
    Scan all tracking CSVs in a predictions directory and report the maximum
    bounding-box dimensions per track, so users can choose an appropriate
    ``tracklet_size`` before rendering.

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
    from octron.yolo_octron.helpers.yolo_results import YOLO_results

    predictions_path = Path(predictions_path)
    results = YOLO_results(predictions_path, verbose=False)
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


def run_render(
    predictions_path,
    video_path=None,
    output_path=None,
    preset="draft",
    tracklets=False,
    tracklet_size=160,
    alpha=0.4,
    draw_masks=True,
    draw_boxes=True,
    start=None,
    end=None,
):
    """
    Render annotated video(s) from OCTRON prediction output.

    Parameters
    ----------
    predictions_path : str or Path
        Path to a prediction output directory produced by ``octron analyze``
        (e.g. ``octron_predictions/video_ByteTrack/``).  Must contain
        ``prediction_metadata.json`` and either ``predictions.zarr`` (for
        segmentation models) or CSV tracks (for detection models).
    video_path : str or Path, optional
        Path to the original video file.  If None, the path stored in
        ``prediction_metadata.json`` / inferred from the directory layout is
        used.
    output_path : str or Path, optional
        Directory where rendered files are written.  Defaults to
        ``<predictions_path>/rendered/``.
    preset : {'preview', 'draft', 'final'}
        Quality / speed trade-off.
        - ``preview`` : 0.25× resolution — for quick inspection.
        - ``draft``   : 0.5× resolution, balanced (default).
        - ``final``   : full resolution, high-quality encode.
    tracklets : bool
        If True, generate one crop video per tracked animal (``tracklet_size``
        × ``tracklet_size`` pixels, centred on the animal's bounding box).
    tracklet_size : int
        Side length in pixels of each tracklet crop (default 160).
    alpha : float
        Opacity of mask overlays blended onto the original frame (0–1).
    draw_masks : bool
        Render semi-transparent segmentation mask fills (default True).
        Requires a segmentation model prediction (zarr masks).
    draw_boxes : bool
        Render bounding boxes and label text (default True).
    start : int, optional
        First frame index to render (inclusive).  None = beginning of video.
    end : int, optional
        Last frame index to render (exclusive).  None = end of video.
    """
    import cv2
    import numpy as np
    from octron.yolo_octron.helpers.yolo_results import YOLO_results

    predictions_path = Path(predictions_path)
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions directory not found: {predictions_path}")

    # Load prediction results (auto-discovers video, zarr, CSVs)
    results = YOLO_results(predictions_path, verbose=True)

    # Override video path if explicitly provided
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

    src_path = results.video_dict["video_file_path"]
    fps = results.video_dict["fps"]
    height = results.height
    width = results.width
    num_frames = results.num_frames

    # Frame range
    frame_start = start if start is not None else 0
    frame_end = end if end is not None else num_frames
    n_frames = frame_end - frame_start

    # Output directory
    output_path = (
        Path(output_path) if output_path is not None else predictions_path / "rendered"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # Preset scaling — force even dimensions (required by most codecs)
    scale = PRESETS[preset]["scale"]
    out_h = int(height * scale)
    out_w = int(width * scale)
    out_h += out_h % 2
    out_w += out_w % 2

    # Load tracking data: {track_id: {'label', 'data' (positions), 'features' (bbox)}}
    tracking_data = results.get_tracking_data(interpolate=False)

    # Build per-frame index lookups for fast O(1) access during frame loop
    # Use plain Python dicts keyed by int(frame_idx) to avoid pandas type-matching issues
    bbox_lookup = {}  # {track_id: {frame_idx: Series}}
    pos_lookup = {}  # {track_id: {frame_idx: Series}}
    for tid, td in tracking_data.items():
        bbox_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["features"].iterrows()}
        pos_lookup[tid] = {int(r["frame_idx"]): r for _, r in td["data"].iterrows()}
        n_bbox = len(bbox_lookup[tid])
        print(f"  Track {tid} ({td['label']}): {n_bbox} frames with bbox data")

    # Color per track: RGBA [0-1] from OCTRON's color system → BGR uint8 for OpenCV
    track_colors = {}
    for tid in results.track_ids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)  # BGR

    # --- Video writers ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    overlay_out = output_path / f"overlay_{preset}.mp4"
    writer = cv2.VideoWriter(str(overlay_out), fourcc, fps, (out_w, out_h))

    tracklet_writers = {}
    if tracklets:
        # Pre-flight: warn if any bbox exceeds the requested tracklet_size
        for tid, td in tracking_data.items():
            feats = td["features"]
            max_w = int(np.ceil((feats["bbox_x_max"] - feats["bbox_x_min"]).max()))
            max_h = int(np.ceil((feats["bbox_y_max"] - feats["bbox_y_min"]).max()))
            if max(max_w, max_h) > tracklet_size:
                print(
                    f"Warning: track {tid} ({td['label']}) has bounding boxes up to "
                    f"{max_w}×{max_h} px (w×h), which exceeds tracklet_size={tracklet_size}. "
                    f"Animal will be cropped. Consider --tracklet-size {max(max_w, max_h) + 20}."
                )
        for tid in results.track_ids:
            label = results.track_id_label[tid]
            tp = output_path / f"tracklet_{label}_track{tid}_{preset}.mp4"
            tracklet_writers[tid] = cv2.VideoWriter(
                str(tp), fourcc, fps, (tracklet_size, tracklet_size)
            )

    print(
        f"Rendering {n_frames} frames | preset={preset} | scale={scale:.0%} "
        f"| output {out_w}×{out_h} px"
    )

    import time
    from collections import deque

    # Pre-compute nearest-neighbour downscale indices for masks (vectorised batch resize)
    _y_idx = _x_idx = None
    if draw_masks and scale != 1.0 and results.has_masks:
        for tid in results.track_ids:
            arr_key = f"{tid}_masks"
            if arr_key in results.zarr_root:
                _H, _W = results.zarr_root[arr_key].shape[1:3]
                _y_idx = np.round(np.linspace(0, _H - 1, out_h)).astype(int)
                _x_idx = np.round(np.linspace(0, _W - 1, out_w)).astype(int)
                break

    # Zarr batch size — match chunk_size so each chunk is decompressed exactly once
    _BATCH = 500
    _batch_masks = {}   # tid -> (n, out_h, out_w) uint8, current batch
    _batch_start = None  # first frame_idx of current batch

    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    _bar_width = 30
    _frame_times = deque(maxlen=30)
    _t_frame = time.time()

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        ok, orig_frame = cap.read()
        if not ok:
            print(f"\nWarning: could not read frame {frame_idx}, stopping early.")
            break

        # Lazy-load the next zarr batch when the current one is exhausted
        if draw_masks and results.has_masks and (_batch_start is None or frame_idx >= _batch_start + _BATCH):
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
                raw = np.asarray(zarr_arr[_batch_start:end_idx])  # (n, H, W)
                if _y_idx is not None:
                    raw = raw[:, _y_idx[:, None], _x_idx[None, :]]  # (n, out_h, out_w)
                _batch_masks[tid] = raw

        j = frame_idx - _batch_start if _batch_start is not None else 0

        # Downscale frame for overlay; keep orig_frame for tracklet crops
        if scale != 1.0:
            frame_small = cv2.resize(orig_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = orig_frame
        overlay = frame_small.astype(np.float32)

        for tid in results.track_ids:
            color_bgr = track_colors[tid]
            label = results.track_id_label[tid]

            # Mask overlay (segmentation models only)
            if draw_masks and results.has_masks and tid in _batch_masks:
                batch = _batch_masks[tid]
                if j < batch.shape[0]:
                    mask = batch[j]
                    obj = mask == 1
                    if obj.any():
                        color_f = np.array(color_bgr, dtype=np.float32)
                        overlay[obj] = (1 - alpha) * overlay[obj] + alpha * color_f

            # Bounding box + label text
            if draw_boxes:
                row = bbox_lookup.get(tid, {}).get(frame_idx)
                if row is not None:
                    x1 = int(row["bbox_x_min"] * scale)
                    y1 = int(row["bbox_y_min"] * scale)
                    x2 = int(row["bbox_x_max"] * scale)
                    y2 = int(row["bbox_y_max"] * scale)
                    lw = max(1, int(2 * scale + 0.5))
                    # Dark outline + coloured inner box for contrast against mask fill
                    cv2.rectangle(overlay, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 0, 0), lw)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, lw)
                    txt = f"{label} {tid}"
                    font_scale = max(0.3, 0.45 * scale / 0.5)
                    txt_lw = max(1, lw)
                    # Dark shadow then coloured text
                    cv2.putText(overlay, txt, (x1 + 1, max(y1 - 5, 13)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), txt_lw + 1, cv2.LINE_AA)
                    cv2.putText(overlay, txt, (x1, max(y1 - 6, 12)),
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_bgr, txt_lw, cv2.LINE_AA)

        out_frame = np.clip(overlay, 0, 255).astype(np.uint8)
        writer.write(out_frame)

        # Tracklet crops — use orig_frame (full resolution) for crops
        if tracklets:
            for tid in results.track_ids:
                crop = np.zeros((tracklet_size, tracklet_size, 3), dtype=np.uint8)
                row = pos_lookup.get(tid, {}).get(frame_idx)
                if row is not None:
                    cx = int(row["pos_x"])
                    cy = int(row["pos_y"])
                    half = tracklet_size // 2
                    # Source coords in original frame (clamped to frame bounds)
                    src_x1 = max(0, cx - half)
                    src_x2 = min(width, cx + half)
                    src_y1 = max(0, cy - half)
                    src_y2 = min(height, cy + half)
                    # Destination coords in crop (offset by how much was clamped,
                    # so the centroid stays at the centre of the crop)
                    dst_x1 = src_x1 - (cx - half)
                    dst_y1 = src_y1 - (cy - half)
                    dst_x2 = dst_x1 + (src_x2 - src_x1)
                    dst_y2 = dst_y1 + (src_y2 - src_y1)
                    crop[dst_y1:dst_y2, dst_x1:dst_x2] = orig_frame[
                        src_y1:src_y2, src_x1:src_x2
                    ]
                tracklet_writers[tid].write(crop)

        now = time.time()
        _frame_times.append(now - _t_frame)
        _t_frame = now
        avg_ft = sum(_frame_times) / len(_frame_times)
        fps = 1.0 / avg_ft if avg_ft > 0 else 0.0
        remaining = n_frames - (i + 1)
        eta_s = int(remaining * avg_ft) if avg_ft > 0 else 0
        eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"
        pct = (i + 1) / n_frames
        filled = int(_bar_width * pct)
        bar = "█" * filled + "░" * (_bar_width - filled)
        print(
            f"  [{bar}] {i + 1}/{n_frames} | {pct:.0%} | {fps:.1f} fps | ETA {eta_str}",
            end="\r",
        )

    print()
    cap.release()
    writer.release()
    print(f"Overlay saved → {overlay_out}")

    for tw in tracklet_writers.values():
        tw.release()
    if tracklets:
        print(f"Tracklets saved → {output_path}")
