"""
OCTRON tracking export pipeline.

Converts zarr mask archives and per-track CSVs produced by ``octron predict``
into clean, analysis-ready CSV files with resolved scalar columns.

All columns that contain tuple-strings (produced when multiple disconnected
segments are detected in a single frame) are resolved to scalars using the
chosen ``centroid_method``.  The same strategy applies to every column —
positions, areas, shape descriptors, and intensity properties — so the output
is always fully numeric.

Centroid / resolution methods
------------------------------
largest   : use the value from the single largest segment per frame (default).
            Fast, CSV-only, no zarr required.
weighted  : area-weighted mean across all segments per frame.
            CSV-only.  ``area`` becomes the *sum* of segment areas.
mask_com  : ``pos_x``/``pos_y`` replaced by full centre-of-mass from zarr masks
            (most robust against flickering outlier segments).  All other
            columns resolved via weighted.  ``area`` is the sum of segment areas.
            Requires the zarr archive.

Special cases
-------------
area      : ``largest`` → area of the largest segment.
            ``weighted`` / ``mask_com`` → sum of all segment areas.
orientation : always resolved via ``largest`` (circular quantity; weighted mean
              is undefined for angles).

Public functions
----------------
export_tracking        : Export CSVs from an existing predictions directory.
compute_mask_centroids : Compute per-frame centre-of-mass from zarr masks.
"""

import ast
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np


# ---------------------------------------------------------------------------
# Letterbox helper (shared with render.py)
# ---------------------------------------------------------------------------

def _letterbox_bounds(mask_h, mask_w, vid_h, vid_w):
    """
    Compute the video-content region within a mask stored at inference resolution.

    YOLO (retina_masks=False) letterboxes frames to fit within imgsz and then
    pads to the nearest multiple of stride (32).  The padding is centred, so the
    actual video content starts at (pad_y, pad_x) within the stored mask.

    Returns
    -------
    content_h, content_w : int
    pad_y, pad_x : int
    """
    content_h = round(vid_h * mask_w / vid_w)
    if content_h <= mask_h:
        content_w = mask_w
        pad_y = (mask_h - content_h) // 2
        pad_x = 0
    else:
        content_h = mask_h
        content_w = round(vid_w * mask_h / vid_h)
        pad_y = 0
        pad_x = (mask_w - content_w) // 2
    return content_h, content_w, pad_y, pad_x


# ---------------------------------------------------------------------------
# compute_mask_centroids (moved from render.py)
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
        Root zarr group from a predictions.zarr archive.
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
        original full-resolution frame.  Only frames with at least one mask
        pixel are included.
    """
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
        _mask_h, _mask_w = zarr_arr.shape[1], zarr_arr.shape[2]
        _vid_h = zarr_arr.attrs.get('video_height', _mask_h) or _mask_h
        _vid_w = zarr_arr.attrs.get('video_width',  _mask_w) or _mask_w
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


# ---------------------------------------------------------------------------
# Tuple-resolution helpers
# ---------------------------------------------------------------------------

def _parse_tuple_or_scalar(val):
    """Return a tuple of floats from a scalar value or a tuple-string '(1.0, 2.0)'."""
    if isinstance(val, (int, float, np.integer, np.floating)):
        return (float(val),)
    if isinstance(val, str) and val.startswith('('):
        try:
            parsed = ast.literal_eval(val)
            if isinstance(parsed, tuple):
                return tuple(float(v) for v in parsed)
        except (ValueError, SyntaxError):
            pass
    try:
        return (float(val),)
    except (TypeError, ValueError):
        return (float('nan'),)


def _largest_idx(area_vals):
    """Return the index of the largest segment given a tuple of area values."""
    if len(area_vals) == 1:
        return 0
    return int(np.argmax(area_vals))


def _resolve_xy_largest(xs, ys, areas):
    """Pick the position of the largest segment for each row."""
    cx = np.empty(len(xs), dtype=float)
    cy = np.empty(len(ys), dtype=float)
    for i, (x, y, a) in enumerate(zip(xs, ys, areas)):
        xv = _parse_tuple_or_scalar(x)
        yv = _parse_tuple_or_scalar(y)
        av = _parse_tuple_or_scalar(a)
        idx = _largest_idx(av)
        cx[i] = xv[idx] if idx < len(xv) else xv[0]
        cy[i] = yv[idx] if idx < len(yv) else yv[0]
    return cx, cy


def _resolve_xy_weighted(xs, ys, areas):
    """Compute the area-weighted mean position across all segments per row."""
    cx = np.empty(len(xs), dtype=float)
    cy = np.empty(len(ys), dtype=float)
    for i, (x, y, a) in enumerate(zip(xs, ys, areas)):
        xv = np.array(_parse_tuple_or_scalar(x))
        yv = np.array(_parse_tuple_or_scalar(y))
        av = np.array(_parse_tuple_or_scalar(a))
        total = av.sum()
        if total > 0:
            cx[i] = float((xv * av).sum() / total)
            cy[i] = float((yv * av).sum() / total)
        else:
            cx[i] = float(xv.mean())
            cy[i] = float(yv.mean())
    return cx, cy


def _resolve_prop_largest(values, areas):
    """Pick the regionprop value for the largest segment per row."""
    out = np.empty(len(values), dtype=float)
    for i, (v, a) in enumerate(zip(values, areas)):
        vv = _parse_tuple_or_scalar(v)
        av = _parse_tuple_or_scalar(a)
        idx = _largest_idx(av)
        out[i] = vv[idx] if idx < len(vv) else vv[0]
    return out


def _resolve_prop_weighted(values, areas):
    """Compute the area-weighted mean of a regionprop across all segments per row."""
    out = np.empty(len(values), dtype=float)
    for i, (v, a) in enumerate(zip(values, areas)):
        vv = np.array(_parse_tuple_or_scalar(v))
        av = np.array(_parse_tuple_or_scalar(a))
        total = av.sum()
        out[i] = float((vv * av).sum() / total) if total > 0 else float(vv.mean())
    return out


def _sum_segments_area(areas):
    """Sum all segment areas per row to give total mask area."""
    out = np.empty(len(areas), dtype=float)
    for i, a in enumerate(areas):
        out[i] = float(sum(_parse_tuple_or_scalar(a)))
    return out


# ---------------------------------------------------------------------------
# Header helpers
# ---------------------------------------------------------------------------

def _build_header(video_metadata):
    lines = [
        f"video_name: {video_metadata.get('video_name', 'unknown')}",
        f"frame_count: {video_metadata.get('frame_count', '')}",
        f"frame_count_analyzed: {video_metadata.get('frame_count_analyzed', '')}",
        f"video_height: {video_metadata.get('video_height', '')}",
        f"video_width: {video_metadata.get('video_width', '')}",
        f"created_at: {video_metadata.get('created_at', str(datetime.now()))}",
        "",  # empty separator line
    ]
    return '\n'.join(lines) + '\n'


def _read_csv_metadata(csv_path, n_header_lines=7):
    """Parse the key: value metadata from the top of a prediction CSV."""
    meta = {}
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i >= n_header_lines - 1:
                break
            line = line.strip()
            if ':' in line:
                key, _, val = line.partition(':')
                meta[key.strip()] = val.strip()
    return meta


# ---------------------------------------------------------------------------
# Private core function
# ---------------------------------------------------------------------------

def _export_tracking_from_data(
    output_dir,
    track_ids,
    labels,
    frame_counters,
    frame_indices,
    confidence,
    segments_x,
    segments_y,
    segments_area,
    bbox,
    region_props,
    video_metadata,
    centroid_method="largest",
    zarr_root=None,
    combined=False,
):
    """
    Build and write tracking CSVs from raw per-track arrays.

    This is the single place where prediction CSVs are written — called by
    both ``predict_batch`` (in-memory path) and ``export_tracking`` (disk path).

    All tuple-valued columns (produced when multiple disconnected segments are
    detected in a frame) are resolved to scalars using ``centroid_method``.
    The same strategy applies to positions, areas, and all regionprops columns,
    with two exceptions:

    - ``area`` uses the *largest segment's area* for ``"largest"``, and the
      *sum of all segment areas* for ``"weighted"`` and ``"mask_com"``.
    - ``orientation`` always uses the largest segment (circular quantity).

    Parameters
    ----------
    output_dir : Path or str
    track_ids : list[int]
    labels : dict[int, str]
    frame_counters : dict[int, array-like]
        Sequential counter within the analyzed frames (frame_no level).
    frame_indices : dict[int, array-like]
        Actual video frame indices (frame_idx level).
    confidence : dict[int, array-like]
    segments_x : dict[int, array-like]
        Raw pos_x values — scalars or tuple-strings when multi-segment.
    segments_y : dict[int, array-like]
    segments_area : dict[int, array-like]
        Raw area values — scalars or tuple-strings when multi-segment.
    bbox : dict[int, dict[str, array-like]]
        Bounding-box columns keyed by column name.
    region_props : dict[int, dict[str, array-like]]
        Extra regionprop columns (all columns except the above).
    video_metadata : dict
        Keys: video_name, frame_count, frame_count_analyzed,
              video_height, video_width, created_at.
    centroid_method : {"largest", "weighted", "mask_com"}
    zarr_root : zarr.Group or None
        Required only when centroid_method == "mask_com".
    combined : bool
        True → single all_tracks.csv; False → one file per track.
    """
    import pandas as pd
    from loguru import logger

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute mask centroids once across all tracks when needed
    mask_centroids_lookup = {}
    if centroid_method == "mask_com" and zarr_root is not None:
        all_frames = [int(f) for tid in track_ids for f in frame_indices.get(tid, [])]
        if all_frames:
            print("Computing mask centroids from zarr...")
            mask_centroids_lookup = compute_mask_centroids(
                zarr_root, track_ids,
                frame_start=min(all_frames),
                frame_end=max(all_frames) + 1,
            )

    # Determine per-prop resolver based on centroid_method.
    # mask_com falls back to weighted for all non-centroid metrics.
    _use_weighted = centroid_method in ("weighted", "mask_com")

    header = _build_header(video_metadata)
    dfs = {}

    for tid in track_ids:
        if tid not in frame_indices or len(frame_indices[tid]) == 0:
            continue

        label = labels.get(tid, "unknown")
        n = len(frame_indices[tid])

        sx = np.asarray(segments_x.get(tid, np.full(n, np.nan)), dtype=object)
        sy = np.asarray(segments_y.get(tid, np.full(n, np.nan)), dtype=object)
        sa = np.asarray(segments_area.get(tid, np.ones(n)), dtype=object)

        # --- pos_x / pos_y ---
        if centroid_method == "weighted":
            px, py = _resolve_xy_weighted(sx, sy, sa)
        elif centroid_method == "mask_com" and mask_centroids_lookup.get(tid):
            fi_arr = np.asarray(frame_indices[tid])
            lookup = mask_centroids_lookup[tid]
            px = np.array([lookup.get(int(f), (np.nan, np.nan))[0] for f in fi_arr])
            py = np.array([lookup.get(int(f), (np.nan, np.nan))[1] for f in fi_arr])
        else:
            # "largest" or mask_com fallback when zarr unavailable
            px, py = _resolve_xy_largest(sx, sy, sa)

        # --- area ---
        # "largest" → area of the largest segment.
        # "weighted" / "mask_com" → sum of all segment areas (total mask coverage).
        if _use_weighted:
            area = _sum_segments_area(sa)
        else:
            area = _resolve_prop_largest(sa, sa)

        # --- all other regionprops ---
        resolved_props = {}
        for prop_name, prop_values in (region_props.get(tid) or {}).items():
            pv = np.asarray(prop_values, dtype=object)
            # orientation is a circular quantity — always use largest segment.
            if prop_name == "orientation" or not _use_weighted:
                resolved_props[prop_name] = _resolve_prop_largest(pv, sa)
            else:
                resolved_props[prop_name] = _resolve_prop_weighted(pv, sa)

        # --- Build DataFrame ---
        data = {
            "label":      np.full(n, label),
            "confidence": np.asarray(confidence.get(tid, np.full(n, np.nan)), dtype=float),
            "pos_x":      px,
            "pos_y":      py,
        }

        for col, arr in (bbox.get(tid) or {}).items():
            data[col] = np.asarray(arr)

        data["area"] = area
        data.update(resolved_props)

        idx = pd.MultiIndex.from_arrays(
            [
                np.asarray(frame_counters[tid]),
                np.asarray(frame_indices[tid]),
                np.full(n, tid, dtype=int),
            ],
            names=["frame_counter", "frame_idx", "track_id"],
        )
        dfs[tid] = (label, pd.DataFrame(data, index=idx))

    if not dfs:
        logger.warning("No tracking data to export.")
        return

    if combined:
        all_df = pd.concat([df for _, df in dfs.values()]).sort_index()
        csv_path = output_dir / "all_tracks.csv"
        with open(csv_path, "w") as f:
            f.write(header)
            all_df.to_csv(f, na_rep="NaN", lineterminator="\n")
        logger.debug(f"Saved combined tracking data ({len(dfs)} tracks) to all_tracks.csv")
    else:
        for tid, (label, df) in dfs.items():
            csv_path = output_dir / f"{label}_track_{tid}.csv"
            with open(csv_path, "w") as f:
                f.write(header)
                df.to_csv(f, na_rep="NaN", lineterminator="\n")
            logger.debug(f"Saved tracking data for '{label}' (track ID: {tid}) to {csv_path.name}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------

_CSV_HEADER_LINES = 7
_BBOX_COLS = frozenset({"bbox_x_min", "bbox_x_max", "bbox_y_min", "bbox_y_max",
                        "bbox_area", "bbox_aspect_ratio"})
_BASE_COLS = frozenset({"frame_counter", "frame_idx", "track_id", "label",
                        "confidence", "pos_x", "pos_y", "area"}) | _BBOX_COLS


def export_tracking(
    predictions_path,
    output_dir=None,
    centroid_method: Literal["largest", "weighted", "mask_com"] = "largest",
    region_properties=None,
    combined=False,
):
    """
    Export tracking CSVs from an existing OCTRON predictions directory.

    Reads the per-track CSVs and (optionally) zarr masks produced by
    ``octron predict``, resolves all tuple-valued columns to scalars using
    the chosen strategy, and writes new CSVs.

    Parameters
    ----------
    predictions_path : str or Path
        Path to an ``octron_predictions/<video>/`` output directory.
    output_dir : str or Path or None
        Where to write the output CSVs.  Defaults to *predictions_path*.
    centroid_method : {"largest", "weighted", "mask_com"}
        How to resolve multi-segment rows.  Applied consistently to all
        columns (positions, areas, regionprops).  See module docstring.
    region_properties : None, "all", or list[str]
        Which regionprop columns to include in the output.
        ``None`` / ``"all"`` → keep every column found in the existing CSV.
        A list of names → include only those columns (if present in the CSV).
    combined : bool
        If True write a single ``all_tracks.csv``; otherwise one file per track.
    """
    import zarr
    from natsort import natsorted
    import pandas as pd
    from loguru import logger

    predictions_path = Path(predictions_path)

    # --- Discover CSV files ---
    csvs = natsorted(predictions_path.rglob("*track_*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No tracking CSV files found in {predictions_path}")
    logger.info(f"Found {len(csvs)} tracking CSV(s) in {predictions_path.name}")

    # --- Discover zarr (optional; required only for mask_com) ---
    zarr_root = None
    zarr_paths = list(predictions_path.rglob("predictions.zarr"))
    if zarr_paths:
        store = zarr.storage.LocalStore(zarr_paths[0], read_only=True)
        zarr_root = zarr.open_group(store=store, mode="r")
        logger.info("Found zarr archive")
    elif centroid_method == "mask_com":
        logger.warning(
            "centroid_method='mask_com' requested but no zarr archive found — "
            "falling back to 'largest'."
        )
        centroid_method = "largest"

    # --- Read each CSV ---
    track_ids      = []
    labels         = {}
    frame_counters = {}
    frame_indices  = {}
    confidence_d   = {}
    segments_x_d   = {}
    segments_y_d   = {}
    segments_area_d = {}
    bbox_d         = {}
    region_props_d = {}
    video_metadata = {}

    for csv_file in csvs:
        meta = _read_csv_metadata(csv_file, _CSV_HEADER_LINES)
        if not video_metadata:
            video_metadata = meta

        df = pd.read_csv(csv_file, skiprows=_CSV_HEADER_LINES)
        if df.empty:
            continue

        tid = int(df["track_id"].iloc[0])
        track_ids.append(tid)
        labels[tid] = str(df["label"].iloc[0])

        # frame_counter is the first column (MultiIndex level written by to_csv)
        frame_counters[tid] = df.iloc[:, 0].to_numpy()
        frame_indices[tid]  = df["frame_idx"].to_numpy()
        confidence_d[tid]   = df["confidence"].to_numpy()
        segments_x_d[tid]   = df["pos_x"].to_numpy()
        segments_y_d[tid]   = df["pos_y"].to_numpy()

        area_col = "area" if "area" in df.columns else None
        segments_area_d[tid] = (
            df[area_col].to_numpy() if area_col else np.ones(len(df))
        )

        bbox_d[tid] = {c: df[c].to_numpy() for c in _BBOX_COLS if c in df.columns}

        # Extra regionprop columns
        extra_cols = [c for c in df.columns if c not in _BASE_COLS]
        if isinstance(region_properties, list):
            extra_cols = [c for c in extra_cols if c in region_properties]
        # region_properties=None or "all" → keep all extra columns
        region_props_d[tid] = {c: df[c].to_numpy() for c in extra_cols}

    output_dir = Path(output_dir) if output_dir is not None else predictions_path

    _export_tracking_from_data(
        output_dir=output_dir,
        track_ids=track_ids,
        labels=labels,
        frame_counters=frame_counters,
        frame_indices=frame_indices,
        confidence=confidence_d,
        segments_x=segments_x_d,
        segments_y=segments_y_d,
        segments_area=segments_area_d,
        bbox=bbox_d,
        region_props=region_props_d,
        video_metadata=video_metadata,
        centroid_method=centroid_method,
        zarr_root=zarr_root,
        combined=combined,
    )
