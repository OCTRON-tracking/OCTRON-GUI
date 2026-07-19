"""OCTRON prediction pipeline.

Wraps ``YOLO_octron.predict_batch()`` into a single callable usable from
the CLI or programmatically.

Local prediction caching (staging output on a fast local disk, then moving each
finished video to its final destination) now lives in core ``predict_batch`` so
the GUI and programmatic callers share it. It is opt-in: set
``prediction_cache_dir`` in ``config.yaml`` (or pass ``local_cache_dir`` here /
``--local-cache-dir`` on the CLI). ``run_predict`` only prints progress.
"""

import time
from collections import deque
from pathlib import Path


def run_predict(
    videos,
    model_path,
    tracker_name=None,
    tracker_cfg_path=None,
    tracker_params=None,
    device="auto",
    conf_thresh=0.5,
    iou_thresh=0.7,
    skip_frames=0,
    one_object_per_label=False,
    opening_radius=0,
    overwrite=False,
    buffer_size=500,
    region_properties=None,
    output_dir=None,
    debug=False,
    local_cache_dir=None,
):
    """Run YOLO prediction and tracking on one or more videos.

    Parameters
    ----------
    videos : str, Path, or list
        Single video path, a directory of ``.mp4`` files, or a list of paths.
    model_path : str or Path
        Path to a trained YOLO ``.pt`` file, or a directory containing one.
    tracker_name : str, optional
        Tracker to use (e.g. 'ByteTrack', 'BotSort'). Either this or
        ``tracker_cfg_path`` must be provided.
    tracker_cfg_path : str or Path, optional
        Path to a custom boxmot tracker config YAML (takes priority over
        ``tracker_name``).
    tracker_params : dict, optional
        Parameter overrides applied on top of the resolved tracker config.
    device : str
        Device to run inference on ('auto', 'cpu', 'cuda', 'mps'). 'auto'
        selects CUDA if available, then MPS, then CPU.
    conf_thresh : float
        Confidence threshold for detection.
    iou_thresh : float
        IOU threshold for detection.
    skip_frames : int
        Number of frames to skip between predictions.
    one_object_per_label : bool
        Track only the highest-confidence detection per label per frame.
    opening_radius : int
        Morphological opening radius applied to masks to remove noise.
    overwrite : bool
        Overwrite existing prediction results.
    buffer_size : int
        Number of frames buffered before writing to zarr.
    region_properties : tuple or None
        Region property names to extract via skimage.measure.regionprops_table.
        Pass DEFAULT_REGION_PROPERTIES for the standard set, or None to skip.
    output_dir : str or Path, optional
        Root directory for octron_predictions output. Defaults to alongside
        each video file.
    debug : bool
        Enable DEBUG-level progress logging.
    local_cache_dir : str or Path, optional
        Override the configured prediction cache directory. When a cache dir is
        set (here or via ``config.yaml`` ``prediction_cache_dir``), core
        ``predict_batch`` stages each video's output under
        ``<cache>/octron_cache_<pid>`` and moves the finished folder to
        ``output_dir``. Caching is OFF unless a cache dir is configured.

    """
    if isinstance(videos, (str, Path)):
        videos = [videos]
    expanded = []
    for v in videos:
        v = Path(v)
        if v.is_dir():
            found = sorted(
                f for f in v.iterdir() if f.suffix.lower() == ".mp4"
            )
            if not found:
                raise ValueError(f"No .mp4 files found in directory: {v}")
            print(f"Found {len(found)} video(s) in {v}")
            expanded.extend(found)
        else:
            expanded.append(v)
    videos = expanded

    model_path = Path(model_path)
    if model_path.is_dir():
        candidates = [
            model_path / "weights" / "best.pt",
            model_path / "training" / "weights" / "best.pt",
            model_path / "model" / "training" / "weights" / "best.pt",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found is None:
            raise FileNotFoundError(
                f"model_path is a directory but no best.pt found "
                f"inside: {model_path}"
            )
        model_path = found

    from loguru import logger

    from octron.test_gpu import auto_device
    from octron.yolo_octron.yolo_octron import YOLO_octron

    # Silence boxmot's verbose INFO chatter (tracker init parameter dumps).
    logger.disable("boxmot")

    # Unwrap a Device enum to its plain string value (mirrors run_training).
    # boxmot's select_device() calls str(device).lower(); a (str, Enum)
    # member stringifies to e.g. "Device.mps" -> "device.mps", which it
    # then misreads as a CUDA device and rejects. Passing the bare value
    # ("mps") avoids that.
    device = device.value if hasattr(device, "value") else str(device)

    if device == "auto":
        device = auto_device()

    yolo = YOLO_octron()

    _wall_start = time.time()
    # Consumer-side fps: timestamps of the last _FPS_WINDOW seconds of
    # frames. Time-based eviction means stall outliers fall out of the
    # window immediately after the stall ends, avoiding both
    # mean-poisoning and median bimodal issues.
    _FPS_WINDOW = 5.0  # seconds
    _fps_ts: deque[float] = deque()
    _current_video_frames = 0
    _last_print_t = 0.0
    _last_video_index = -1
    # In debug mode slow progress updates to 2 Hz so they don't fight with the
    # timing diagnostic lines; otherwise cap at 30 Hz to avoid CONPTY overhead.
    _PRINT_INTERVAL = 0.5 if debug else 1 / 30
    # Track whether a \r progress line is open so we can close it (with a plain
    # newline) before emitting any logger output.
    _progress_line_open = False

    def _close_progress():
        nonlocal _progress_line_open
        if _progress_line_open:
            print()
            _progress_line_open = False

    try:
        for progress in yolo.predict_batch(
            videos=videos,
            model_path=model_path,
            device=device,
            tracker_name=tracker_name,
            tracker_cfg_path=tracker_cfg_path,
            tracker_params=tracker_params,
            skip_frames=skip_frames,
            one_object_per_label=one_object_per_label,
            iou_thresh=iou_thresh,
            conf_thresh=conf_thresh,
            opening_radius=opening_radius,
            overwrite=overwrite,
            buffer_size=buffer_size,
            region_properties=region_properties,
            output_dir=output_dir,
            local_cache_dir=local_cache_dir,
        ):
            stage = progress.get("stage", "")

            if stage == "skipped_video":
                _close_progress()
                logger.warning(
                    f"Skipped {progress.get('video_name', '')} — "
                    f"predictions already exist at "
                    f"{progress.get('save_dir', '')}. "
                    f"Use --overwrite to replace."
                )
                continue

            if stage == "video_complete":
                _close_progress()
                logger.info(f"Saved: {progress.get('save_dir', '')}")
                continue

            if stage == "complete":
                _close_progress()
                continue

            if stage != "processing":
                continue

            video_index = progress.get("video_index", 0)
            total_f = progress.get("total_frames", 0)

            # Per-video header on the first processing frame of each new video.
            if video_index != _last_video_index:
                _last_video_index = video_index
                _current_video_frames = total_f
                _fps_ts.clear()  # reset fps window per video
                _close_progress()
                total_videos = progress.get("total_videos", "?")
                logger.info(
                    f"Video {video_index + 1}/{total_videos}: "
                    f"{progress.get('video_name', '')}  ({total_f:,} frames)"
                )

            frame = progress.get("frame", 0)
            total_f = _current_video_frames or total_f

            now_t = time.time()
            _fps_ts.append(now_t)
            while _fps_ts[0] < now_t - _FPS_WINDOW:
                _fps_ts.popleft()
            fps = (
                (len(_fps_ts) - 1) / (_fps_ts[-1] - _fps_ts[0])
                if len(_fps_ts) >= 2
                else 0.0
            )

            pct = 100.0 * frame / total_f if total_f > 0 else 0.0
            eta_s = int((total_f - frame) / fps) if fps > 0 else 0
            eta_h, eta_rem = divmod(eta_s, 3600)
            eta_m, eta_sec = divmod(eta_rem, 60)
            eta_str = f"{eta_h:02d}:{eta_m:02d}:{eta_sec:02d}"

            if now_t - _last_print_t >= _PRINT_INTERVAL:
                progress_msg = (
                    f"{frame:,}/{total_f:,} | {pct:.1f}% | "
                    f"{fps:.1f} fps | ETA: {eta_str}"
                )
                if debug:
                    logger.info(f"[progress] {progress_msg}")
                else:
                    print(
                        f"\r\033[K  {progress_msg}",
                        end="",
                        flush=True,
                    )
                    _progress_line_open = True
                _last_print_t = now_t

    except KeyboardInterrupt:
        _close_progress()
        logger.warning(
            "Interrupted — stopping prediction. Completed videos are saved; "
            "the in-progress video is discarded."
        )
        return

    elapsed = time.time() - _wall_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    logger.info(f"Done. Total time: {h:02d}:{m:02d}:{s:02d}")
