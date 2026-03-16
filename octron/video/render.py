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
    'preview': {'scale': 0.25},
    'draft':   {'scale': 0.5},
    'final':   {'scale': 1.0},
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
        feats = td['features']
        max_w = int(np.ceil((feats['bbox_x_max'] - feats['bbox_x_min']).max()))
        max_h = int(np.ceil((feats['bbox_y_max'] - feats['bbox_y_min']).max()))
        stats[tid] = {'label': td['label'], 'max_w': max_w, 'max_h': max_h}

    overall_max = max((max(s['max_w'], s['max_h']) for s in stats.values()), default=0)

    print('Bounding-box size report:')
    for tid, s in sorted(stats.items()):
        print(f"  Track {tid} ({s['label']}): max {s['max_w']}×{s['max_h']} px (w×h)")
    print(f"→ Recommended minimum tracklet size: {overall_max} px "
          f"(add padding as desired, default is 160 px)")
    return stats


def run_render(
    predictions_path,
    video_path=None,
    output_path=None,
    preset='draft',
    tracklets=False,
    tracklet_size=160,
    alpha=0.4,
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
        raise FileNotFoundError(f'Predictions directory not found: {predictions_path}')

    # Load prediction results (auto-discovers video, zarr, CSVs)
    results = YOLO_results(predictions_path, verbose=True)

    # Override video path if explicitly provided
    if video_path is not None:
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f'Video not found: {video_path}')
        from octron.sam_octron.helpers.video_loader import probe_video
        from napari_pyav._reader import FastVideoReader
        results.video = FastVideoReader(video_path, read_format='rgb24')
        results.video_dict = probe_video(video_path, verbose=False)
        results.height = results.video_dict['height']
        results.width = results.video_dict['width']
        results.num_frames = results.video_dict['num_frames']

    if results.video_dict is None:
        raise FileNotFoundError(
            "Could not locate the original video automatically. "
            "Use --video to specify its path."
        )

    src_path = results.video_dict['video_file_path']
    fps = results.video_dict['fps']
    height = results.height
    width = results.width
    num_frames = results.num_frames

    # Frame range
    frame_start = start if start is not None else 0
    frame_end = end if end is not None else num_frames
    n_frames = frame_end - frame_start

    # Output directory
    output_path = (
        Path(output_path) if output_path is not None
        else predictions_path / 'rendered'
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # Preset scaling — force even dimensions (required by most codecs)
    scale = PRESETS[preset]['scale']
    out_h = int(height * scale)
    out_w = int(width * scale)
    out_h += out_h % 2
    out_w += out_w % 2

    # Load tracking data: {track_id: {'label', 'data' (positions), 'features' (bbox)}}
    tracking_data = results.get_tracking_data(interpolate=False)

    # Build per-frame index lookups for fast O(1) access during frame loop
    bbox_lookup = {}   # {track_id: DataFrame indexed by frame_idx}
    pos_lookup = {}    # {track_id: DataFrame indexed by frame_idx}
    for tid, td in tracking_data.items():
        bbox_lookup[tid] = td['features'].set_index('frame_idx')
        pos_lookup[tid] = td['data'].set_index('frame_idx')

    # Color per track: RGBA [0-1] from OCTRON's color system → BGR uint8 for OpenCV
    track_colors = {}
    for tid in results.track_ids:
        rgba, _ = results.get_color_for_track_id(tid)
        r, g, b = int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
        track_colors[tid] = (b, g, r)  # BGR

    # --- Video writers ---
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    overlay_out = output_path / f'overlay_{preset}.mp4'
    writer = cv2.VideoWriter(str(overlay_out), fourcc, fps, (out_w, out_h))

    tracklet_writers = {}
    if tracklets:
        # Pre-flight: warn if any bbox exceeds the requested tracklet_size
        for tid, td in tracking_data.items():
            feats = td['features']
            max_w = int(np.ceil((feats['bbox_x_max'] - feats['bbox_x_min']).max()))
            max_h = int(np.ceil((feats['bbox_y_max'] - feats['bbox_y_min']).max()))
            if max(max_w, max_h) > tracklet_size:
                print(
                    f'Warning: track {tid} ({td["label"]}) has bounding boxes up to '
                    f'{max_w}×{max_h} px (w×h), which exceeds tracklet_size={tracklet_size}. '
                    f'Animal will be cropped. Consider --tracklet-size {max(max_w, max_h) + 20}.'
                )
        for tid in results.track_ids:
            label = results.track_id_label[tid]
            tp = output_path / f'tracklet_{label}_track{tid}_{preset}.mp4'
            tracklet_writers[tid] = cv2.VideoWriter(
                str(tp), fourcc, fps, (tracklet_size, tracklet_size)
            )

    print(
        f'Rendering {n_frames} frames | preset={preset} | scale={scale:.0%} '
        f'| output {out_w}×{out_h} px'
    )

    cap = cv2.VideoCapture(str(src_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    for i, frame_idx in enumerate(range(frame_start, frame_end)):
        ok, frame_bgr = cap.read()
        if not ok:
            print(f'\nWarning: could not read frame {frame_idx}, stopping early.')
            break

        overlay = frame_bgr.astype(np.float32)

        for tid in results.track_ids:
            color_bgr = track_colors[tid]
            label = results.track_id_label[tid]

            # Mask overlay (segmentation models only)
            if results.has_masks:
                arr_key = f'{tid}_masks'
                if arr_key in results.zarr_root and frame_idx < results.zarr_root[arr_key].shape[0]:
                    mask = np.asarray(results.zarr_root[arr_key][frame_idx])
                    obj = mask == 1
                    if obj.any():
                        color_f = np.array(color_bgr, dtype=np.float32)
                        overlay[obj] = (1 - alpha) * overlay[obj] + alpha * color_f

            # Bounding box + label text
            feat = bbox_lookup.get(tid)
            if feat is not None and frame_idx in feat.index:
                row = feat.loc[frame_idx]
                # guard against duplicate index entries returning a DataFrame
                if hasattr(row, 'iloc'):
                    row = row.iloc[0]
                x1 = int(row['bbox_x_min'])
                y1 = int(row['bbox_y_min'])
                x2 = int(row['bbox_x_max'])
                y2 = int(row['bbox_y_max'])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, 1)
                cv2.putText(
                    overlay, f'{label} {tid}',
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA,
                )

        out_frame = np.clip(overlay, 0, 255).astype(np.uint8)
        if scale != 1.0:
            out_frame = cv2.resize(out_frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(out_frame)

        # Tracklet crops — black frame when animal is not detected
        if tracklets:
            for tid in results.track_ids:
                crop = np.zeros((tracklet_size, tracklet_size, 3), dtype=np.uint8)
                pos = pos_lookup.get(tid)
                if pos is not None and frame_idx in pos.index:
                    row = pos.loc[frame_idx]
                    if hasattr(row, 'iloc'):
                        row = row.iloc[0]
                    cx = int(row['pos_x'])
                    cy = int(row['pos_y'])
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
                    crop[dst_y1:dst_y2, dst_x1:dst_x2] = frame_bgr[src_y1:src_y2, src_x1:src_x2]
                tracklet_writers[tid].write(crop)

        if (i + 1) % 100 == 0 or (i + 1) == n_frames:
            pct = (i + 1) / n_frames * 100
            print(f'  {i + 1}/{n_frames} frames ({pct:.0f}%)', end='\r')

    print()
    cap.release()
    writer.release()
    print(f'Overlay saved → {overlay_out}')

    for tw in tracklet_writers.values():
        tw.release()
    if tracklets:
        print(f'Tracklets saved → {output_path}')
