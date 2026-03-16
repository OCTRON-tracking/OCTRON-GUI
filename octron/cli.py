"""
OCTRON command-line interface.

Entry point: `octron`

Subcommands
-----------
  gui         Launch the OCTRON napari GUI
  gpu-test    Check GPU availability
  train       Run the full YOLO training pipeline on an OCTRON project
  predict     Run YOLO prediction and tracking on one or more videos
  render      Render annotated video(s) from prediction output
  bbox-sizes  Report bounding-box sizes to inform --tracklet-size choice
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
def train(
    project_path: Path = typer.Argument(
        ..., help="Path to the OCTRON project directory."
    ),
    model: str = typer.Option("YOLO11m", help="YOLO model name or path to a .pt file."),
    device: str = typer.Option(
        "auto", help="Device to train on ('auto', 'cpu', 'cuda', 'mps')."
    ),
    epochs: int = typer.Option(30, help="Number of training epochs."),
    imagesz: int = typer.Option(640, help="Input image size."),
    save_period: int = typer.Option(15, help="Save a checkpoint every N epochs."),
    train_mode: str = typer.Option("segment", help="'segment' or 'detect'."),
    resume: bool = typer.Option(
        False, help="Resume from an existing last.pt checkpoint."
    ),
):
    """Run the full YOLO training pipeline on an OCTRON project."""
    from octron.train import run_training
    from octron.test_gpu import auto_device

    if device == "auto":
        device = auto_device()
    run_training(
        project_path=project_path,
        model=model,
        device=device,
        epochs=epochs,
        imagesz=imagesz,
        save_period=save_period,
        train_mode=train_mode,
        resume=resume,
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
    from octron.predict import run_predict
    from octron.test_gpu import auto_device

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
    predictions_path: Path = typer.Argument(..., help='Path to a prediction output directory (octron_predictions/video_Tracker/).'),
    video_path: Path = typer.Option(None, '--video', help='Path to the original video. Auto-detected if it sits alongside octron_predictions/ and is .mp4/.avi/.mov/.mkv. Required if the video has been moved or lives elsewhere.'),
    output_path: Path = typer.Option(None, '--output', '-o', help='Output directory. Defaults to <predictions_path>/rendered/.'),
    preset: str = typer.Option('draft', '--preset', help="Quality preset: 'preview' (0.25×), 'draft' (0.5×), or 'final' (full resolution)."),
    tracklets: bool = typer.Option(False, '--tracklets', help='Also generate one crop video per tracked animal.'),
    tracklet_size: int = typer.Option(160, '--tracklet-size', help='Side length in pixels of each tracklet crop.'),
    alpha: float = typer.Option(0.4, '--alpha', help='Mask overlay opacity (0–1).'),
    start: int = typer.Option(None, '--start', help='First frame to render (inclusive).'),
    end: int = typer.Option(None, '--end', help='Last frame to render (exclusive).'),
):
    """Render annotated video(s) from OCTRON prediction output."""
    from octron.video.render import run_render, PRESETS
    if preset not in PRESETS:
        raise typer.BadParameter(f"preset must be one of {list(PRESETS)}.", param_hint="'--preset'")
    run_render(
        predictions_path=predictions_path,
        video_path=video_path,
        output_path=output_path,
        preset=preset,
        tracklets=tracklets,
        tracklet_size=tracklet_size,
        alpha=alpha,
        start=start,
        end=end,
    )


@app.command('bbox-sizes')
def bbox_sizes(
    predictions_path: Path = typer.Argument(..., help='Path to a prediction output directory.'),
):
    """Report per-track bounding-box sizes to help choose --tracklet-size."""
    from octron.video.render import report_bbox_sizes
    report_bbox_sizes(predictions_path)


def main():
    app()
