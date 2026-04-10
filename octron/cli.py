"""
OCTRON command-line interface.

Entry point: `octron`

Subcommands
-----------
  gui         Launch the OCTRON napari GUI
  gpu-test    Check GPU availability
  split       Prepare and export train/val/test data from an OCTRON project
  train       Prepare training data and run YOLO model training
  predict     Run YOLO prediction and tracking on one or more videos
  render      Render annotated video(s) from prediction output
              (use --bbox-sizes to report bbox sizes instead of rendering)
  transcode   Transcode video files to MP4 (H.264/libx264) using ffmpeg
"""


from typing import List, Optional
from pathlib import Path
from enum import Enum
import typer
from loguru import logger


class YOLOModel(str, Enum):
    yolo11m = "yolo11m"
    yolo11l = "yolo11l"
    yolo26m = "yolo26m"
    yolo26l = "yolo26l"


class TrainMode(str, Enum):
    segment = "segment"
    detect = "detect"


class TrackerName(str, Enum):
    bytetrack = "bytetrack"
    ocsort = "ocsort"
    botsort = "botsort"
    docsort = "d-ocsort"
    hybridsort = "hybridsort"
    boosttrack = "boosttrack"


class Device(str, Enum):
    auto = "auto"
    cpu = "cpu"
    cuda = "cuda"
    mps = "mps"

app = typer.Typer(
    name="octron",
    help="OCTRON – segmentation and tracking for animal behavior quantification.",
    invoke_without_command=True,
)


@app.callback()
def default(ctx: typer.Context):
    """Launch the OCTRON napari GUI (default), or run a subcommand."""
    # ctx.resilient_parsing is True during shell-completion parsing.
    # ctx.args contains args that will be forwarded to the subcommand (e.g. ['--help']).
    # Using ctx.args works with both real sys.argv and typer.testing.CliRunner.
    _remaining = list(ctx.protected_args or []) + list(ctx.args or [])
    if not ctx.resilient_parsing and "--help" not in _remaining and "-h" not in _remaining:
        from octron._logging import setup_logging, print_welcome
        setup_logging()
        print_welcome()
    if ctx.invoked_subcommand is None:
        logger.info("Loading libraries (this may take a moment)...")
        from octron.main import octron_gui

        octron_gui()


@app.command()
def gui():
    """Launch the OCTRON napari GUI."""
    logger.info("Loading libraries (this may take a moment)...")
    from octron.main import octron_gui

    octron_gui()


@app.command("gpu-test")
def gpu_test():
    """Check GPU availability (CUDA / MPS)."""
    from octron.test_gpu import check_gpu_access

    check_gpu_access()


@app.command()
def split(
    project_path: Path = typer.Argument(..., help="Path to the OCTRON project directory."),
    train_mode: TrainMode = typer.Option(TrainMode.segment, "--mode", help="Training mode."),
    train_fraction: float = typer.Option(0.7, "--train", help="Fraction of frames for training."),
    val_fraction: float = typer.Option(0.15, "--val", help="Fraction of frames for validation. The remainder becomes the test split."),
    seed: int = typer.Option(88, "--seed", help="Random seed for reproducibility."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print split sizes without writing files."),
):
    """Prepare and export train/val/test data from an OCTRON project."""
    from octron.tools.split import run_split

    run_split(
        project_path=project_path,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
        train_mode=train_mode,
        dry_run=dry_run,
    )


@app.command()
def train(
    project_path: Path = typer.Argument(..., help="Path to the OCTRON project directory."),
    model: YOLOModel = typer.Option(YOLOModel.yolo26m, help="YOLO model to train."),
    train_mode: TrainMode = typer.Option(TrainMode.segment, "--mode", help="Training mode."),
    device: Device = typer.Option(Device.auto, help="Device to train on."),
    epochs: int = typer.Option(250, help="Number of training epochs."),
    imagesz: int = typer.Option(640, help="Input image size."),
    save_period: int = typer.Option(50, help="Save a checkpoint every N epochs."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite an existing trained model. Default: skip if best.pt already exists."),
    resume: bool = typer.Option(False, help="Resume from an existing last.pt checkpoint."),
    no_split: bool = typer.Option(False, "--no-split", help="Skip data preparation. Use when 'octron split' has already been run."),
    train_fraction: float = typer.Option(0.7, "--train", help="Fraction of frames for training (ignored with --no-split)."),
    val_fraction: float = typer.Option(0.15, "--val", help="Fraction of frames for validation (ignored with --no-split)."),
    seed: int = typer.Option(88, "--seed", help="Random seed for the split (ignored with --no-split)."),
):
    """Prepare training data and run YOLO model training on an OCTRON project."""
    from octron.tools.train import run_training

    run_training(
        project_path=project_path,
        model=model,
        train_mode=train_mode,
        device=device,
        epochs=epochs,
        imagesz=imagesz,
        save_period=save_period,
        overwrite=overwrite,
        resume=resume,
        skip_split=no_split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )


@app.command()
def predict(
    videos: List[Path] = typer.Argument(..., help="One or more video file paths."),
    model_path: Path = typer.Option(..., "--model", help="Path to a trained YOLO .pt file."),
    tracker: TrackerName = typer.Option(TrackerName.bytetrack, "--tracker", help="Tracker algorithm."),
    tracker_config: Optional[Path] = typer.Option(None, "--tracker-config", help="Path to a custom tracker config YAML (overrides --tracker)."),
    device: Device = typer.Option(Device.auto, help="Device to run inference on."),
    conf_thresh: float = typer.Option(0.5, help="Confidence threshold for detection."),
    iou_thresh: float = typer.Option(0.7, help="IOU threshold for NMS."),
    skip_frames: int = typer.Option(0, help="Number of frames to skip between predictions."),
    one_object_per_label: bool = typer.Option(False, help="Track only the top-confidence detection per label."),
    opening_radius: int = typer.Option(0, help="Morphological opening radius applied to masks."),
    overwrite: bool = typer.Option(False, "--overwrite", help="Overwrite existing prediction results. Default: skip videos that already have predictions."),
    detailed: bool = typer.Option(False, "--detailed", help="Extract detailed region properties (area, eccentricity, solidity, …) from segmentation masks via scikit-image. Ignored for detection models."),
    buffer_size: int = typer.Option(500, help="Frame buffer size before writing to zarr."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory where octron_predictions/ folders are written. Defaults to alongside each video file."),
    local_cache_dir: Optional[Path] = typer.Option(None, "--local-cache-dir", help="Write zarr output here first, then move to --output-dir when each video finishes. Enabled automatically for network/UNC paths. Useful for NVMe scratch space."),
):
    """Run YOLO prediction and tracking on one or more videos."""
    from octron.tools.predict import run_predict
    from octron.test_gpu import auto_device

    expanded = []
    for p in videos:
        if p.is_dir():
            found = sorted(f for f in p.iterdir() if f.suffix.lower() == ".mp4")
            if not found:
                raise typer.BadParameter(f"No .mp4 files found in directory: {p}", param_hint="videos")
            print(f"Found {len(found)} video(s) in {p}")
            expanded.extend(found)
        else:
            expanded.append(p)
    videos = expanded

    if model_path.is_dir():
        candidates = [
            model_path / "weights" / "best.pt",
            model_path / "training" / "weights" / "best.pt",
            model_path / "model" / "training" / "weights" / "best.pt",
        ]
        found = next((c for c in candidates if c.exists()), None)
        if found:
            model_path = found
        else:
            raise typer.BadParameter(
                f"Directory given but no best.pt found inside: {model_path}",
                param_hint="'--model'",
            )

    if device == "auto":
        device = auto_device()
    if tracker_config is not None:
        tracker_name = None
        tracker_cfg_path = tracker_config
    else:
        tracker_name = tracker.value
        tracker_cfg_path = None
    from octron.yolo_octron.constants import DEFAULT_REGION_PROPERTIES
    run_predict(
        videos=videos,
        model_path=model_path,
        device=device,
        tracker_name=tracker_name,
        tracker_cfg_path=tracker_cfg_path,
        skip_frames=skip_frames,
        one_object_per_label=one_object_per_label,
        iou_thresh=iou_thresh,
        conf_thresh=conf_thresh,
        opening_radius=opening_radius,
        overwrite=overwrite,
        buffer_size=buffer_size,
        region_properties=DEFAULT_REGION_PROPERTIES if detailed else None,
        output_dir=output_dir,
        local_cache_dir=local_cache_dir,
    )


@app.command()
def render(
    predictions_path: Path = typer.Argument(
        ..., help="Path to a prediction output directory (octron_predictions/video_Tracker/).",
    ),
    # --- Input / output ---
    video_path: Path = typer.Option(
        None, "--video",
        help="Path to the original video. Auto-detected if alongside octron_predictions/.",
    ),
    output_path: Path = typer.Option(
        None, "--output", "-o",
        help="Output directory. Defaults to <predictions_path>/rendered/.",
    ),
    preset: str = typer.Option(
        "draft", "--preset",
        help="Quality preset: 'preview' (0.25×), 'draft' (0.5×), 'final' (full resolution).",
    ),
    start: int = typer.Option(None, "--start", help="First frame to render (inclusive)."),
    end: int = typer.Option(None, "--end", help="Last frame to render (exclusive)."),
    # --- Overlay options ---
    alpha: float = typer.Option(0.4, "--alpha", help="Mask overlay opacity (0–1)."),
    draw_masks: Optional[bool] = typer.Option(
        None, "--masks/--no-masks",
        help="Render segmentation masks. Default: on for overlay, off for tracklets.",
    ),
    draw_boxes: Optional[bool] = typer.Option(
        None, "--boxes/--no-boxes",
        help="Render bounding boxes. Default: on for overlay, off for tracklets.",
    ),
    draw_labels: Optional[bool] = typer.Option(
        None, "--labels/--no-labels",
        help="Render label text above bounding boxes (overlay mode only; never drawn on tracklets).",
    ),
    # --- Tracklet options ---
    tracklets: bool = typer.Option(False, "--tracklets", help="Generate one crop video per tracked animal."),
    tracklet_size: Optional[str] = typer.Option("auto", "--tracklet-size", help="Side length in pixels of each tracklet crop, or 'auto' to use the largest bounding box + 20px padding."),
    tracklet_mask_centroids: bool = typer.Option(
        False, "--tracklet-mask-centroids",
        help="Use mask centre-of-mass instead of bbox centre for tracklet positioning.",
    ),
    tracklet_smooth_cutoff_hz: float = typer.Option(
        2.0, "--tracklet-smooth-cutoff", help="Butterworth low-pass cutoff (Hz) for centroid smoothing. 0=off.",
    ),
    tracklet_smooth_order: int = typer.Option(
        4, "--tracklet-smooth-order", help="Butterworth filter order (higher = steeper rolloff).",
    ),
    tracklet_interpolate: int = typer.Option(
        0, "--tracklet-interpolate", help="Fill gaps between track segments with cubic spline interpolation. Value is the maximum gap in frames to bridge. 0 = off.",
    ),
    tracklet_segment_only: bool = typer.Option(
        False, "--tracklet-segment-only",
        help="Black out all pixels outside each animal's segmentation mask (requires mask data).",
    ),
    tracklet_segment_keep: int = typer.Option(
        0, "--tracklet-segment-keep",
        help="When --tracklet-segment-only is set, keep only the N largest connected components of the mask. 0 = keep all components (default).",
    ),
    # --- Filtering ---
    track_ids: Optional[str] = typer.Option(
        None, "--track-ids",
        help="Comma-separated list of track IDs to render (e.g. 1,3,5). Renders all tracks if omitted.",
    ),
    min_observations: int = typer.Option(
        0, "--min-observations", help="Skip tracks with fewer than this many observations. 0 = keep all tracks.",
    ),
    trim: bool = typer.Option(
        False, "--trim", help="Trim each tracklet video to the track's first and last observation (within --start/--end if given).",
    ),
    bbox_sizes: bool = typer.Option(
        False, "--bbox-sizes",
        help="Report per-track bounding-box sizes to help choose --tracklet-size, then exit.",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Enable DEBUG-level logging with per-stage timing (decode / blend / encode).",
    ),
):
    """Render annotated video(s) from OCTRON prediction output."""
    from octron.tools.render import run_render, PRESETS, report_bbox_sizes

    if bbox_sizes:
        report_bbox_sizes(predictions_path)
        return

    if preset not in PRESETS:
        raise typer.BadParameter(
            f"preset must be one of {list(PRESETS)}.", param_hint="'--preset'"
        )

    # Resolve mode-specific defaults: overlay → everything on; tracklets → nothing on
    if tracklets:
        resolved_masks = draw_masks if draw_masks is not None else False
        resolved_boxes = draw_boxes if draw_boxes is not None else False
        resolved_labels = False  # never draw labels on tracklet crops
    else:
        resolved_masks = draw_masks if draw_masks is not None else True
        resolved_boxes = draw_boxes if draw_boxes is not None else True
        resolved_labels = draw_labels if draw_labels is not None else True

    parsed_track_ids = (
        [int(x.strip()) for x in track_ids.split(",") if x.strip()]
        if track_ids is not None else None
    )

    parsed_tracklet_size = None if tracklet_size is None or tracklet_size.lower() == "auto" else int(tracklet_size)

    run_render(
        predictions_path=predictions_path,
        video_path=video_path,
        output_path=output_path,
        preset=preset,
        start=start,
        end=end,
        alpha=alpha,
        draw_masks=resolved_masks,
        draw_boxes=resolved_boxes,
        draw_labels=resolved_labels,
        tracklets=tracklets,
        also_overlay=resolved_masks or resolved_boxes,
        tracklet_size=parsed_tracklet_size,
        tracklet_mask_centroids=tracklet_mask_centroids,
        tracklet_smooth_cutoff_hz=tracklet_smooth_cutoff_hz,
        tracklet_smooth_order=tracklet_smooth_order,
        tracklet_interpolate_max_gap=tracklet_interpolate,
        tracklet_segment_only=tracklet_segment_only,
        tracklet_segment_keep_n=tracklet_segment_keep,
        track_ids=parsed_track_ids,
        min_observations=min_observations,
        trim=trim,
        debug=debug,
    )


@app.command()
def transcode(
    videos: List[Path] = typer.Argument(
        ...,
        help="One or more video file paths, or a directory containing video files.",
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to mp4_transcoded/ next to the input(s).",
    ),
    crf: int = typer.Option(
        23,
        "--crf",
        help="Constant Rate Factor (0–51). Lower = better quality. Default 23.",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output files."
    ),
):
    """Transcode video files to MP4 (H.264/libx264) using ffmpeg."""
    from octron.tools.transcode import run_transcode

    run_transcode(
        videos=videos,
        output_path=output_path,
        crf=crf,
        overwrite=overwrite,
    )


def main():
    app()
