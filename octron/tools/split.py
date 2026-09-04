"""OCTRON training-data split pipeline.

Prepares and exports train/val/test data from an OCTRON project without
running model training.  The `octron train` command calls this internally;
users can also run it standalone via `octron split`.
"""

from collections import Counter
from pathlib import Path

_MODELS_YAML = (
    Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"
)

# Terminal timeline styling. Click strips these ANSI colors automatically
# when stdout is not a TTY (e.g. piped to a file), so the bar degrades to a
# plain-text block/shade fallback without any extra dependency.
_SPLIT_COLORS = {"train": "green", "val": "cyan", "test": "yellow"}
_UNANNOTATED_COLOR = "bright_black"
_BLOCK = "\u2588"  # full block: annotated frames
_EMPTY = "\u2591"  # light shade: unannotated frames


def run_split(
    project_path,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=88,
    train_mode="segment",
    dry_run=False,
):
    """Prepare and export train/val/test data for an OCTRON project.

    Steps
    -----
    1. Collect labels from project annotation files.
    2. Generate polygons (segment) or bounding boxes (detect).
    3. Split frames into train / val / test.
    4. Export images + label files to ``<project>/model/training_data/``.

    Parameters
    ----------
    project_path : str or Path
        Path to the OCTRON project directory.
    train_fraction : float
        Fraction of frames assigned to the training split (default 0.7).
    val_fraction : float
        Fraction of frames assigned to the validation split (default 0.15).
        The remainder becomes the test split.
    seed : int
        Random seed for reproducibility (default 88).
    train_mode : str
        ``'segment'`` for instance segmentation, ``'detect'`` for bounding-box
        detection only.
    dry_run : bool
        If ``True``, print split sizes without writing anything to disk.

    """
    from octron.yolo_octron.yolo_octron import YOLO_octron

    train_mode = (
        train_mode.value if hasattr(train_mode, "value") else str(train_mode)
    )

    # Validate fractions up front using the core guard (also enforced
    # inside prepare_split) so the CLI fails before any model, label, or
    # geometry work.
    YOLO_octron._validate_split_fractions(train_fraction, val_fraction)

    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
    )
    yolo.train_mode = train_mode

    # --- Step 1: collect labels ---
    print("Preparing labels...")
    yolo.prepare_labels()

    # --- Step 2: generate geometry (polygons for segment, bboxes for
    # detect) ---
    print(
        "Generating polygons..."
        if train_mode == "segment"
        else "Generating bounding boxes..."
    )
    for (
        no_entry,
        total,
        label,
        frame_no,
        total_frames,
    ) in yolo.prepare_geometry():
        print(
            f"  [{no_entry}/{total}] {label}: frame {frame_no}/{total_frames}",
            end="\r",
        )
    print()

    # --- Step 3: split ---
    print("Splitting data into train/val/test sets...")
    yolo.prepare_split(
        training_fraction=train_fraction,
        validation_fraction=val_fraction,
        random_seed=seed,
    )

    # Print summary table + colored whole-video timelines
    _print_split_summary(yolo.label_dict, seed)
    _print_split_timelines(yolo.label_dict)

    if dry_run:
        print("Dry run — no files written.")
        return

    # --- Step 4: export to disk ---
    print("Exporting training data...")
    for (
        no_entry,
        total,
        label,
        split,
        frame_no,
        total_frames,
    ) in yolo.create_training_data():
        print(
            f"  [{no_entry}/{total}] {label} ({split}): "
            f"frame {frame_no}/{total_frames}",
            end="\r",
        )
    print()

    yolo.write_yolo_config(train_mode=train_mode)
    print("Training data export complete.")


def _print_split_summary(label_dict, seed):
    """Print a per-label, per-split frame count table."""
    rows = []
    for subfolder, labels in label_dict.items():
        for entry, info in labels.items():
            if entry in ("video", "video_file_path"):
                continue
            split = info.get("frames_split", {})
            rows.append(
                (
                    Path(subfolder).name,
                    info["label"],
                    len(split.get("train", [])),
                    len(split.get("val", [])),
                    len(split.get("test", [])),
                    len(info["frames"]),
                )
            )

    if not rows:
        return

    col_w = [max(len(str(r[i])) for r in rows) for i in range(6)]
    col_w = [
        max(w, h) for w, h in zip(col_w, [8, 5, 5, 3, 4, 5], strict=False)
    ]
    header = (
        f"{'Subfolder':{col_w[0]}}  "
        f"{'Label':{col_w[1]}}  "
        f"{'Train':>{col_w[2]}}  "
        f"{'Val':>{col_w[3]}}  "
        f"{'Test':>{col_w[4]}}  "
        f"{'Total':>{col_w[5]}}"
    )
    sep = "-" * len(header)
    print(f"\nSplit summary  (seed={seed})")
    print(sep)
    print(header)
    print(sep)
    for subfolder, label, tr, va, te, tot in rows:  # codespell:ignore te
        print(
            f"{subfolder:{col_w[0]}}  "
            f"{label:{col_w[1]}}  "
            f"{tr:>{col_w[2]}}  "
            f"{va:>{col_w[3]}}  "
            f"{te:>{col_w[4]}}  "  # codespell:ignore te
            f"{tot:>{col_w[5]}}"
        )
    print(sep)
    print()


def _build_frame_to_split(labels):
    """Map each annotated frame index to its split for one subfolder.

    Aggregates every label in the subfolder. With empty-label pruning
    (the default) all labels share the same frames, so the mapping is
    unambiguous; without it, the last label wins on the rare conflict.
    """
    frame_to_split = {}
    for entry, info in labels.items():
        if entry in ("video", "video_file_path"):
            continue
        split = info.get("frames_split", {})
        for name in ("train", "val", "test"):
            for frame in split.get(name, []):
                frame_to_split[int(frame)] = name
    return frame_to_split


def _num_frames_for(labels, frame_to_split):
    """Best-effort whole-video length for a subfolder.

    Prefers the mask zarr length (the true video length); falls back to
    the largest annotated index when masks are unavailable.
    """
    for entry, info in labels.items():
        if entry in ("video", "video_file_path"):
            continue
        masks = info.get("masks")
        if masks:
            try:
                return int(masks[0].shape[0])
            except (AttributeError, IndexError, TypeError):
                pass
    return (max(frame_to_split) + 1) if frame_to_split else 0


def _timeline_bins(frame_to_split, num_frames, width):
    """Reduce a frame->split mapping into ``width`` columns.

    Each column reports the dominant split among the frames that fall in
    it, or ``None`` when no annotated frame lands there (unannotated).
    """
    buckets = [Counter() for _ in range(width)]
    for frame, name in frame_to_split.items():
        if 0 <= frame < num_frames:
            col = min(width - 1, int(frame * width / num_frames))
            buckets[col][name] += 1
    return [c.most_common(1)[0][0] if c else None for c in buckets]


def _render_split_timeline(labels, subfolder_name, width=60):
    """Print a colored whole-video timeline of the train/val/test split.

    The bar spans the entire video (frame 0..num_frames); contiguous
    train/val/test blocks show as colored runs and unannotated regions
    as a dim shade, so the episode structure of the split is visible.
    """
    import click

    frame_to_split = _build_frame_to_split(labels)
    if not frame_to_split:
        return
    num_frames = _num_frames_for(labels, frame_to_split)
    if num_frames <= 0:
        return

    width = min(width, num_frames)
    bins = _timeline_bins(frame_to_split, num_frames, width)
    bar = "".join(
        click.style(_BLOCK, fg=_SPLIT_COLORS[b])
        if b is not None
        else click.style(_EMPTY, fg=_UNANNOTATED_COLOR)
        for b in bins
    )
    legend = "  ".join(
        (
            click.style(_BLOCK, fg=_SPLIT_COLORS["train"]) + " train",
            click.style(_BLOCK, fg=_SPLIT_COLORS["val"]) + " val",
            click.style(_BLOCK, fg=_SPLIT_COLORS["test"]) + " test",
            click.style(_EMPTY, fg=_UNANNOTATED_COLOR) + " unannotated",
        )
    )
    click.echo(
        f"\nTimeline: {subfolder_name}  "
        f"({num_frames} frames, {len(frame_to_split)} annotated)"
    )
    click.echo(f"0 {bar} {num_frames}")
    click.echo(f"Legend: {legend}")


def _print_split_timelines(label_dict, width=60):
    """Print a colored split timeline per subfolder (whole-video view)."""
    for subfolder, labels in label_dict.items():
        _render_split_timeline(labels, Path(subfolder).name, width=width)
