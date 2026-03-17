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
  bbox-sizes  Report bounding-box sizes to inform --tracklet-size choice
  transcode   Transcode video files to MP4 (H.264/libx264) using ffmpeg
"""


from typing import List
from pathlib import Path
import typer

app = typer.Typer(
    name="octron",
    help="OCTRON – segmentation and tracking for animal behavior quantification.",
    invoke_without_command=True,
)


@app.callback()
def default(ctx: typer.Context):
    """Launch the OCTRON napari GUI (default), or run a subcommand."""
    if ctx.invoked_subcommand is None:
        from octron.main import octron_gui

        octron_gui()


@app.command()
def gui():
    """Launch the OCTRON napari GUI."""
    from octron.main import octron_gui

    octron_gui()


@app.command("gpu-test")
def gpu_test():
    """Check GPU availability (CUDA / MPS)."""
    from octron.test_gpu import check_gpu_access

    check_gpu_access()


@app.command()
def split(
    project_path: Path = typer.Argument(
        ..., help="Path to the OCTRON project directory."
    ),
    train_fraction: float = typer.Option(0.7, "--train", help="Fraction of frames for training."),
    val_fraction: float = typer.Option(0.15, "--val", help="Fraction of frames for validation. The remainder becomes the test split."),
    seed: int = typer.Option(88, "--seed", help="Random seed for reproducibility."),
    train_mode: str = typer.Option("segment", "--mode", help="'segment' or 'detect'."),
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
    project_path: Path = typer.Argument(
        ..., help="Path to the OCTRON project directory."
    ),
    model: str = typer.Option("YOLO26m", help="YOLO model name or path to a .pt file."),
    device: str = typer.Option(
        "auto", help="Device to train on ('auto', 'cpu', 'cuda', 'mps')."
    ),
    epochs: int = typer.Option(250, help="Number of training epochs."),
    imagesz: int = typer.Option(640, help="Input image size."),
    save_period: int = typer.Option(50, help="Save a checkpoint every N epochs."),
    train_mode: str = typer.Option("segment", "--mode", help="'segment' or 'detect'."),
    resume: bool = typer.Option(
        False, help="Resume from an existing last.pt checkpoint."
    ),
    no_split: bool = typer.Option(
        False, "--no-split", help="Skip data preparation. Use when 'octron split' has already been run."
    ),
    train_fraction: float = typer.Option(0.7, "--train", help="Fraction of frames for training (ignored with --no-split)."),
    val_fraction: float = typer.Option(0.15, "--val", help="Fraction of frames for validation (ignored with --no-split)."),
    seed: int = typer.Option(88, "--seed", help="Random seed for the split (ignored with --no-split)."),
):
    """Prepare training data and run YOLO model training on an OCTRON project."""
    from octron.tools.train import run_training

    run_training(
        project_path=project_path,
        model=model,
        device=device,
        epochs=epochs,
        imagesz=imagesz,
        save_period=save_period,
        train_mode=train_mode,
        resume=resume,
        skip_split=no_split,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )


@app.command()
def predict(
    videos: List[Path] = typer.Argument(..., help="One or more video file paths."),
    model_path: Path = typer.Option(
        ..., "--model", help="Path to a trained YOLO .pt file."
    ),
    device: str = typer.Option(
        "auto", help="Device to run inference on ('auto', 'cpu', 'cuda', 'mps')."
    ),
    tracker: str = typer.Option(
        "ByteTrack",
        "--tracker",
        help="Tracker name (e.g. 'ByteTrack', 'BotSort') or path to a tracker config YAML.",
    ),
    skip_frames: int = typer.Option(
        0, help="Number of frames to skip between predictions."
    ),
    one_object_per_label: bool = typer.Option(
        False, help="Track only the top-confidence detection per label."
    ),
    iou_thresh: float = typer.Option(0.7, help="IOU threshold for detection."),
    conf_thresh: float = typer.Option(0.5, help="Confidence threshold for detection."),
    opening_radius: int = typer.Option(
        0, help="Morphological opening radius applied to masks."
    ),
    overwrite: bool = typer.Option(True, help="Overwrite existing prediction results."),
    buffer_size: int = typer.Option(
        500, help="Frame buffer size before writing to zarr."
    ),
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
        candidate = model_path / "weights" / "best.pt"
        if candidate.exists():
            model_path = candidate
        else:
            raise typer.BadParameter(
                f"Directory given but no weights/best.pt found inside: {model_path}",
                param_hint="'--model'",
            )

    if device == "auto":
        device = auto_device()
    tracker_name = None
    tracker_cfg_path = None
    if tracker.endswith((".yaml", ".yml")):
        tracker_cfg_path = Path(tracker)
    else:
        tracker_name = tracker
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
    )


@app.command()
def render(
    predictions_path: Path = typer.Argument(
        ...,
        help="Path to a prediction output directory (octron_predictions/video_Tracker/).",
    ),
    video_path: Path = typer.Option(
        None,
        "--video",
        help="Path to the original .mp4 video. Auto-detected if it sits alongside octron_predictions/. Required if the video has been moved or lives elsewhere.",
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory. Defaults to <predictions_path>/rendered/.",
    ),
    preset: str = typer.Option(
        "draft",
        "--preset",
        help="Quality preset: 'preview' (0.25×), 'draft' (0.5×), or 'final' (full resolution).",
    ),
    tracklets: bool = typer.Option(
        False, "--tracklets", help="Also generate one crop video per tracked animal."
    ),
    tracklet_size: int = typer.Option(
        160, "--tracklet-size", help="Side length in pixels of each tracklet crop."
    ),
    alpha: float = typer.Option(0.4, "--alpha", help="Mask overlay opacity (0–1)."),
    draw_masks: bool = typer.Option(True, "--masks/--no-masks", help="Render segmentation mask fills."),
    draw_boxes: bool = typer.Option(True, "--boxes/--no-boxes", help="Render bounding boxes and label text."),
    start: int = typer.Option(
        None, "--start", help="First frame to render (inclusive)."
    ),
    end: int = typer.Option(None, "--end", help="Last frame to render (exclusive)."),
):
    """Render annotated video(s) from OCTRON prediction output."""
    from octron.tools.render import run_render, PRESETS

    if preset not in PRESETS:
        raise typer.BadParameter(
            f"preset must be one of {list(PRESETS)}.", param_hint="'--preset'"
        )
    run_render(
        predictions_path=predictions_path,
        video_path=video_path,
        output_path=output_path,
        preset=preset,
        tracklets=tracklets,
        tracklet_size=tracklet_size,
        alpha=alpha,
        draw_masks=draw_masks,
        draw_boxes=draw_boxes,
        start=start,
        end=end,
    )


@app.command("bbox-sizes")
def bbox_sizes(
    predictions_path: Path = typer.Argument(
        ..., help="Path to a prediction output directory."
    ),
):
    """Report per-track bounding-box sizes to help choose --tracklet-size."""
    from octron.tools.render import report_bbox_sizes

    report_bbox_sizes(predictions_path)


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
