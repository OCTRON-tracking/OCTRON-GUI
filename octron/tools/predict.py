"""
OCTRON prediction pipeline.

Wraps YOLO_octron.predict_batch() into a single callable function
that can be used from the CLI or called programmatically.
"""

from collections import deque
from pathlib import Path
import time


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
    buffer_size=200,
    region_properties=None,
    infer_batch_size=8,
    output_dir=None,
    debug=False,
):
    """
    Run YOLO prediction and tracking on one or more videos.

    Parameters
    ----------
    videos : str, Path, or list
        Single video path, or a list of video paths.
    model_path : str or Path
        Path to the trained YOLO model (.pt file).
    device : str
        Device to run inference on ('auto', 'cpu', 'cuda', 'mps'). 'auto'
        selects CUDA if available, then MPS, then CPU.
    tracker_name : str, optional
        Tracker to use (e.g. 'ByteTrack', 'BotSort'). Either this or
        tracker_cfg_path must be provided.
    tracker_cfg_path : str or Path, optional
        Path to a custom boxmot tracker config YAML. Takes priority over
        tracker_name when provided.
    tracker_params : dict, optional
        Parameter overrides applied on top of the resolved tracker config
        (e.g. {'det_thresh': 0.5, 'max_age': 100}).
    skip_frames : int
        Number of frames to skip between predictions.
    one_object_per_label : bool
        Track only the highest-confidence detection per label per frame.
    iou_thresh : float
        IOU threshold for detection.
    conf_thresh : float
        Confidence threshold for detection.
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
        Enable DEBUG-level logging (per-stage timing diagnostics).
    """
    from loguru import logger
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device

    # Silence boxmot's verbose INFO chatter (tracker init parameter dumps).
    logger.disable("boxmot")

    if device == "auto":
        device = auto_device()

    yolo = YOLO_octron()

    _wall_start = time.time()
    frame_time = 0.0
    _frame_times = deque(maxlen=300)
    _current_video_frames = 0
    _last_print_t = 0.0
    # In debug mode slow progress updates to 2 Hz so they don't fight with
    # timing diagnostic lines. In normal mode cap at 30 Hz to avoid
    # Windows CONPTY overhead.
    _PRINT_INTERVAL = 0.5 if debug else 1 / 30

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
            infer_batch_size=infer_batch_size,
            output_dir=output_dir,
        ):
            stage = progress.get("stage", "")

            if stage == "predict_init":
                task     = progress.get("model_task", "?")
                imgsz    = progress.get("imgsz", "?")
                tracker  = progress.get("tracker", "?")
                n_videos = progress.get("total_videos", "?")
                skip     = progress.get("skip_frames", 0)
                model_p  = Path(progress.get("model_path", str(model_path)))
                flags = []
                if skip > 0:
                    flags.append(f"skip={skip}")
                if progress.get("one_object_per_label"):
                    flags.append("one-per-label")
                flag_str = f"  [{', '.join(flags)}]" if flags else ""
                print(f"\n  Model:    {model_p.name}  [{task} · imgsz={imgsz}]")
                print(f"  Tracker:  {tracker}")
                print(f"  Device:   {device}")
                print(f"  Videos:   {n_videos}{flag_str}")
                continue

            if stage == "video_init":
                video_name = progress.get("video_name", "")
                vidx       = progress.get("video_index", 0) + 1
                total_v    = progress.get("total_videos", "?")
                num_frames = progress.get("num_frames", 0)
                save_dir   = progress.get("save_dir", "")
                _current_video_frames = num_frames
                _frame_times.clear()  # reset rolling average per video
                print(f"\nVideo {vidx}/{total_v}: {video_name}")
                print(f"  Frames:   {num_frames:,}")
                print(f"  Output:   {save_dir}")
                continue

            if stage == "skipped_video":
                video_name = progress.get("video_name", "")
                save_dir   = progress.get("save_dir", "")
                print(f"\nVideo: {video_name}")
                print(f"  [skipped] predictions already exist at {save_dir}")
                print(f"  Use --overwrite to replace.")
                continue

            if stage != "processing":
                continue

            frame      = progress.get("frame", 0)
            total_f    = _current_video_frames or progress.get("total_frames", 0)
            frame_time = progress.get("frame_time", frame_time)
            if frame_time > 0:
                _frame_times.append(frame_time)
            avg_frame_time = sum(_frame_times) / len(_frame_times) if _frame_times else 0.0

            pct = 100.0 * frame / total_f if total_f > 0 else 0.0
            remaining = total_f - frame
            eta = remaining * avg_frame_time if avg_frame_time > 0 else 0.0
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            eta_s = int(eta)
            eta_str = f"{eta_s // 3600:02d}:{(eta_s % 3600) // 60:02d}:{eta_s % 60:02d}"

            now_t = time.time()
            if now_t - _last_print_t >= _PRINT_INTERVAL:
                if debug:
                    logger.info(
                        f"[progress] {frame:,}/{total_f:,} | {pct:.1f}% | {fps:.1f} fps | ETA: {eta_str}"
                    )
                else:
                    print(
                        f"\r\033[K  {frame:,}/{total_f:,} | {pct:.1f}% | {fps:.1f} fps | ETA: {eta_str}",
                        end="",
                        flush=True,
                    )
                _last_print_t = now_t

    except KeyboardInterrupt:
        if not debug:
            print()  # close the \r progress line
        logger.warning("Interrupted — stopping prediction.")
        return

    if not debug:
        print()  # close the \r progress line
    elapsed = time.time() - _wall_start
    h, rem = divmod(int(elapsed), 3600)
    m, s = divmod(rem, 60)
    elapsed_str = f"{h:02d}:{m:02d}:{s:02d}"
    if debug:
        logger.info(f"Done. Total time: {elapsed_str}")
    else:
        print(f"\nDone. Total time: {elapsed_str}")
