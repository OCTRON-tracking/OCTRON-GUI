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
  dump-tracker-config  Print a tracker's default config YAML (edit, then pass via --tracker-config)
  render      Render annotated video(s) from prediction output
              (use --bbox-sizes to report bbox sizes instead of rendering)
  transcode   Transcode video files to MP4 (H.264/libx264) using ffmpeg
  gif         Convert MP4/MOV/AVI videos to GIF (GUI)
  download-yolo   Download/refresh YOLO base weights into the model cache
  download-sam2   Download/refresh SAM2 checkpoints into the model cache
  download-sam3   Download/refresh the SAM3 checkpoint (needs HuggingFace access)
"""


from typing import List, Optional, TYPE_CHECKING
from pathlib import Path
from enum import Enum
import re
import typer
import yaml
from loguru import logger

from octron.yolo_octron.constants import ALL_REGION_PROPERTIES

_PKG_DIR = Path(__file__).parent

# Flat tuple of all valid scikit-image region-property names (for predict --detailed).
_REGION_PROPERTY_NAMES = tuple(
    name for group in ALL_REGION_PROPERTIES.values() for name in group
)
_DETAILED_HELP = (
    "Extract scikit-image region properties from each segmentation mask "
    "(ignored for detection models). Pass a comma-separated list of property "
    "names, or 'all'. Custom measurement functions are supported only via the "
    "Python API (predict_batch extra_properties). Available: "
    + ", ".join(_REGION_PROPERTY_NAMES) + "."
)


def _parse_region_properties(detailed):
    """Parse the predict ``--detailed`` value into region-property names.

    Returns ``None`` (no extraction) when not given, the full catalog for
    ``all``, or the validated comma-separated names (de-duplicated, order
    preserved). Raises ``typer.BadParameter`` on unknown names.
    """
    if detailed is None:
        return None
    text = detailed.strip()
    if not text:
        return None
    if text.lower() == "all":
        return _REGION_PROPERTY_NAMES
    names = [n.strip() for n in text.split(",") if n.strip()]
    unknown = [n for n in names if n not in _REGION_PROPERTY_NAMES]
    if unknown:
        raise typer.BadParameter(
            f"Unknown region property name(s): {', '.join(unknown)}. "
            f"Valid names: {', '.join(_REGION_PROPERTY_NAMES)}.",
            param_hint="'--detailed'",
        )
    return tuple(dict.fromkeys(names))


def _sanitize_identifier(value: str) -> str:
    """Turn an arbitrary string into a valid Python identifier for an Enum member.

    Non-alphanumeric characters become ``_`` and a leading digit is prefixed
    with ``_`` so catalog keys like ``3d-sort`` or ``bot.sort`` do not make the
    functional ``Enum(...)`` call raise at import time.
    """
    member = re.sub(r"\W", "_", value)
    if not member or member[0].isdigit():
        member = f"_{member}"
    return member


def _enum_from_yaml(
    name: str, yaml_path: Path, *, available_only: bool = False, fallback: Optional[str] = None
) -> type:
    """Build a (str, Enum) from a YAML mapping. Keys are lower-cased for CLI
    values and sanitised into valid identifiers for the member names.

    The build is defensive so a single malformed/missing catalog does not break
    the import of every CLI subcommand: on a read/parse failure, or when no
    usable entries are produced, a minimal enum containing only ``fallback`` is
    returned (with a warning). If no fallback is given in that case, a clear
    error naming the catalog is raised.
    """
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(
            f"Could not read {name} catalog '{yaml_path}': {e}. "
            f"Falling back to a minimal set of choices."
        )
        data = {}

    members = {}
    for key, info in data.items():
        if available_only and isinstance(info, dict) and not info.get("available", False):
            continue
        value = str(key).lower()
        members.setdefault(_sanitize_identifier(value), value)  # first spelling wins on collision

    if not members:
        if fallback is None:
            raise RuntimeError(f"No usable entries found in {name} catalog '{yaml_path}'.")
        members = {_sanitize_identifier(fallback): fallback}

    try:
        return Enum(name, members, type=str)
    except Exception as e:
        logger.warning(f"Could not build {name} from '{yaml_path}': {e}. Falling back.")
        fb = fallback if fallback is not None else next(iter(members.values()))
        return Enum(name, {_sanitize_identifier(fb): fb}, type=str)


# The CLI choice enums are built dynamically from the YAML catalogs so the
# choices always match the shipped models/trackers. Static type checkers can't
# see the members of a runtime-built Enum (hence "Variable not allowed in type
# expression"), so under TYPE_CHECKING we give them lightweight stand-ins that
# carry only the members used as defaults below; the real enums are built at
# runtime in the else branch.
if TYPE_CHECKING:
    class YOLOModel(str, Enum):
        yolo26m = "yolo26m"

    class TrackerName(str, Enum):
        bytetrack = "bytetrack"
else:
    YOLOModel = _enum_from_yaml(
        "YOLOModel", _PKG_DIR / "yolo_octron" / "yolo_models.yaml",
        fallback="yolo26m",
    )
    TrackerName = _enum_from_yaml(
        "TrackerName", _PKG_DIR / "tracking" / "boxmot_trackers.yaml",
        available_only=True, fallback="bytetrack",
    )


class TrainMode(str, Enum):
    segment = "segment"
    detect = "detect"


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
    """Show the available commands (default), or run a subcommand.

    Running ``octron`` with no subcommand prints this help. Use ``octron gui``
    to launch the napari GUI.
    """
    # ctx.resilient_parsing is True during shell-completion parsing.
    if ctx.invoked_subcommand is None:
        # Bare `octron` behaves like `octron --help`.
        if not ctx.resilient_parsing:
            typer.echo(ctx.get_help())
        raise typer.Exit()
    # A subcommand was given: print the welcome banner, unless the user is just
    # asking for help. We inspect the raw argv because, across Click versions,
    # the parent callback cannot reliably see the subcommand's --help token via
    # ctx.args/ctx.protected_args (and protected_args is removed in Click 9).
    import sys
    _argv = sys.argv[1:]
    _wants_help = "--help" in _argv or "-h" in _argv
    if not ctx.resilient_parsing and not _wants_help:
        from octron._logging import setup_logging, print_welcome
        setup_logging()
        print_welcome()


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
    detailed: Optional[str] = typer.Option(None, "--detailed", help=_DETAILED_HELP),
    buffer_size: int = typer.Option(500, help="Frame buffer size before writing to zarr."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Directory where octron_predictions/ folders are written. Defaults to alongside each video file."),
    local_cache_dir: Optional[Path] = typer.Option(None, "--local-cache-dir", help="Stage prediction output in this local dir, then move each finished video to --output-dir. Overrides the config.yaml 'prediction_cache_dir'; if neither is set, caching is off."),
):
    """Run YOLO prediction and tracking on one or more videos."""
    # Validate --detailed up front (before any heavy import) so a typo fails fast.
    region_properties = _parse_region_properties(detailed)

    from octron.tools.predict import run_predict

    if tracker_config is not None:
        tracker_name = None
        tracker_cfg_path = tracker_config
    else:
        tracker_name = tracker.value
        tracker_cfg_path = None
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
        region_properties=region_properties,
        output_dir=output_dir,
        local_cache_dir=local_cache_dir,
    )


@app.command("dump-tracker-config")
def dump_tracker_config(
    tracker: TrackerName = typer.Argument(
        TrackerName.bytetrack, help="Tracker whose default config to dump.",
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Write to this file instead of printing to stdout.",
    ),
):
    """Print (or write with -o) a tracker's default config YAML.

    A starting point for customization: edit the values, then pass the file to
    `octron predict --tracker-config <file>`.
    """
    from octron.tracking.helpers.tracker_checks import load_boxmot_trackers, resolve_tracker

    trackers_dict = load_boxmot_trackers(_PKG_DIR / "tracking" / "boxmot_trackers.yaml")
    tracker_id, info = resolve_tracker(tracker.value, trackers_dict)
    src = _PKG_DIR / info["config_path"]
    if not src.exists():
        raise typer.BadParameter(f"Tracker config not found: {src}", param_hint="TRACKER")
    content = src.read_text()
    if output is None:
        typer.echo(content)
    else:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        logger.info(
            f"Wrote '{tracker_id}' tracker config to {output.as_posix()}. "
            f"Edit it and pass it to `octron predict --tracker-config`."
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
    tracklet_smooth_sigma: float = typer.Option(
        2.0, "--tracklet-smooth-sigma",
        help="Gaussian smoothing strength (standard deviation in frames) for centroid smoothing. 0=off.",
    ),
    tracklet_interpolate: int = typer.Option(
        0, "--tracklet-interpolate", help="Fill gaps between track segments by linear interpolation. Value caps how many consecutive missing frames to bridge (longer gaps are partially filled); 0 = off.",
    ),
    tracklet_segment_only: bool = typer.Option(
        False, "--tracklet-segment-only",
        help="Black out all pixels outside each animal's segmentation mask (requires mask data).",
    ),
    tracklet_segment_keep: int = typer.Option(
        0, "--tracklet-segment-keep",
        help="When --tracklet-segment-only is set, keep only the N largest connected components of the mask. 0 = keep all components (default).",
    ),
    tracklet_offset: Optional[str] = typer.Option(
        None, "--tracklet-offset",
        help="Offset of the tracklet crop centre as 'DX,DY' in source-video pixels (e.g. '20,-30'). Positive = right/down. Applied consistently in overlay and raw modes. Default: 0,0.",
    ),
    # --- Filtering ---
    track_ids: Optional[str] = typer.Option(
        None, "--track-ids",
        help="Comma-separated list of track IDs to render (e.g. 1,3,5). Renders all tracks if omitted.",
    ),
    min_observations: int = typer.Option(
        0, "--min-observations", help="Skip tracks with fewer than this many observations. 0 = keep all tracks.",
    ),
    min_confidence: float = typer.Option(
        0.5, "--min-confidence", min=0.0, max=1.0,
        help="Skip individual detections whose confidence is below this threshold (0–1). Default 0.5.",
    ),
    trim: bool = typer.Option(
        False, "--trim", help="Trim each tracklet video to the track's first and last observation (within --start/--end if given).",
    ),
    skip_empty: bool = typer.Option(
        False, "--skip-empty",
        help="Skip frames with no detection at/above --min-confidence (e.g. from predict --skip-frames). "
             "Tracklets: each track's video keeps only its own detected frames. Overlay: keeps frames with any detection.",
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

    # Validation/parsing layering (convention):
    #   CLI layer (here) owns user-facing parsing and friendly errors:
    #     typer.BadParameter for --preset/--tracklet-offset, Typer min/max for
    #     --min-confidence/--alpha, and splitting --track-ids / --tracklet-size.
    #   Tool layer (render.py: _validate_render_args, _coerce_track_ids, and the
    #     offset unpack in run_tracklets) re-validates defensively for
    #     programmatic/notebook callers that bypass the CLI.
    #   The overlap on preset/min_confidence is intentional (two audiences),
    #   not redundancy: keep the friendly CLI message AND the library-safe guard.
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

    if tracklet_offset is None or not tracklet_offset.strip():
        parsed_tracklet_offset = (0, 0)
    else:
        try:
            parts = [int(x.strip()) for x in tracklet_offset.split(",")]
        except ValueError:
            raise typer.BadParameter(
                f"--tracklet-offset must be 'DX,DY' integers (e.g. '20,-30'); got {tracklet_offset!r}.",
                param_hint="'--tracklet-offset'",
            )
        if len(parts) != 2:
            raise typer.BadParameter(
                f"--tracklet-offset must be exactly two integers separated by a comma; got {tracklet_offset!r}.",
                param_hint="'--tracklet-offset'",
            )
        parsed_tracklet_offset = (parts[0], parts[1])

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
        tracklet_smooth_sigma=tracklet_smooth_sigma,
        tracklet_interpolate_limit=tracklet_interpolate,
        tracklet_segment_only=tracklet_segment_only,
        tracklet_segment_keep_n=tracklet_segment_keep,
        tracklet_offset=parsed_tracklet_offset,
        track_ids=parsed_track_ids,
        min_observations=min_observations,
        trim=trim,
        skip_empty=skip_empty,
        min_confidence=min_confidence,
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
    fps: float = typer.Option(
        None,
        "--fps",
        help="Output framerate. Videos: reinterprets source frames at this fps "
             "(changes playback speed). TIFF stacks: sets playback fps (default 20). "
             "Omit to keep source fps for videos.",
    ),
    no_audio: bool = typer.Option(
        False, "--no-audio", help="Drop the audio track instead of re-encoding it to AAC."
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing output files."
    ),
):
    """Transcode video files or multi-frame TIFF stacks to MP4 (H.264) using ffmpeg.

    Accepts video files, multi-frame TIFF stacks, or a directory containing them.
    Odd-dimension inputs are handled automatically (scaled to even dimensions);
    audio is kept (re-encoded to AAC) by default for inputs that have it.
    """
    from octron.tools.transcode import run_transcode

    run_transcode(
        videos=videos,
        output_path=output_path,
        crf=crf,
        overwrite=overwrite,
        fps=fps,
        keep_audio=not no_audio,
    )


@app.command()
def gif():
    """Launch the OCTRON video→GIF converter (GUI) for MP4/MOV/AVI files."""
    logger.info("Loading libraries (this may take a moment)...")
    from octron.tools.mp4_to_gif import main as gif_main

    gif_main()


@app.command("download-yolo")
def download_yolo(
    force: bool = typer.Option(
        False, "--force", help="Re-download even if the weight files already exist."
    ),
):
    """Download YOLO base model weights into the model cache.

    Files are stored under config.get_yolo_models_dir() (the per-user model
    cache, or model_cache_dir from config.yaml). Missing files are always
    fetched; pass --force to re-download ones that already exist.
    """
    from octron import config
    from octron.yolo_octron.helpers.yolo_checks import check_yolo_models

    yaml_path = _PKG_DIR / "yolo_octron" / "yolo_models.yaml"
    check_yolo_models(YOLO_BASE_URL=None, models_yaml_path=yaml_path, force_download=force)
    logger.info(f"YOLO weights are in: {config.get_yolo_models_dir().as_posix()}")


@app.command("download-sam2")
def download_sam2(
    force: bool = typer.Option(
        False, "--force", help="Re-download even if the checkpoint files already exist."
    ),
):
    """Download SAM2 checkpoints into the model cache.

    Files are stored under config.get_sam_checkpoints_dir() (the per-user model
    cache, or model_cache_dir from config.yaml). Missing files are always
    fetched; pass --force to re-download ones that already exist.
    """
    from octron import config
    from octron.sam_octron.helpers.sam2_checks import check_sam2_models

    yaml_path = _PKG_DIR / "sam_octron" / "sam2_models.yaml"
    check_sam2_models(SAM2p1_BASE_URL="", models_yaml_path=yaml_path, force_download=force)
    logger.info(f"SAM checkpoints are in: {config.get_sam_checkpoints_dir().as_posix()}")


@app.command("download-sam3")
def download_sam3(
    force: bool = typer.Option(
        False, "--force", help="Re-download even if the checkpoint files already exist."
    ),
):
    """Download the SAM3 checkpoint into the model cache (requires HuggingFace access).

    Needs licence acceptance + `huggingface-cli login` (see the SAM3 model card).
    Files are stored under config.get_sam_checkpoints_dir(). Missing files are
    always fetched; pass --force to re-download ones that already exist.
    """
    from octron import config
    from octron.sam_octron.helpers.sam3_checks import check_sam3_models

    yaml_path = _PKG_DIR / "sam_octron" / "sam3_models.yaml"
    result = check_sam3_models(models_yaml_path=yaml_path, force_download=force)
    if not result:
        raise typer.Exit(code=1)
    logger.info(f"SAM checkpoints are in: {config.get_sam_checkpoints_dir().as_posix()}")


def main():
    app()
