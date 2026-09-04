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
_EMPTY = "\u2591"  # light shade: unannotated frames (within an episode)
_ELLIPSIS = "\u2026"  # … : elided unannotated gap between episodes

# The colored bar shares this many columns across all annotated episodes
# (each episode gets a share proportional to its frame count). Long gaps
# between episodes are collapsed to "…" instead of drawn to scale, so
# sparsely annotated but very long videos stay compact and legible.
_TIMELINE_BUDGET = 60
_MIN_EPISODE_WIDTH = 1  # smallest visible episode bar
_MAX_TIMELINE_EPISODES = 40  # above this, fall back to a plain linear bar


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
    # tqdm (inside prepare_geometry) renders the per-label progress bar on
    # stderr. We only drive the generator here; printing our own
    # carriage-return line to stdout in lockstep with tqdm makes the bar
    # "staircase" onto new lines (most visibly on Windows).
    for _ in yolo.prepare_geometry():
        pass

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
    # As above: tqdm owns the export progress bar; we just consume the
    # generator so a competing stdout writer can't break the bar.
    for _ in yolo.create_training_data():
        pass

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


def _segment_episodes(frames, gap):
    """Group sorted frame indices into episodes, splitting at gaps > ``gap``.

    An episode is a run of annotated frames whose internal gaps never
    exceed ``gap`` frames; larger gaps are elided (``…``) in the bar.
    """
    episodes = [[frames[0]]]
    for frame in frames[1:]:
        if frame - episodes[-1][-1] > gap:
            episodes.append([frame])
        else:
            episodes[-1].append(frame)
    return episodes


def _episode_bar(frames, frame_to_split, width, click):
    """Render one episode's frame span as ``width`` colored blocks."""
    start, end = frames[0], frames[-1]
    span = max(1, end - start + 1)
    buckets = [Counter() for _ in range(width)]
    for frame in frames:
        col = min(width - 1, int((frame - start) * width / span))
        buckets[col][frame_to_split[frame]] += 1
    cells = []
    for counter in buckets:
        if counter:
            split = counter.most_common(1)[0][0]
            cells.append(click.style(_BLOCK, fg=_SPLIT_COLORS[split]))
        else:
            cells.append(click.style(_EMPTY, fg=_UNANNOTATED_COLOR))
    return "".join(cells)


def _linear_bar(frame_to_split, num_frames, width, click):
    """Render a plain, to-scale whole-video bar (no gap elision)."""
    bins = _timeline_bins(frame_to_split, num_frames, min(width, num_frames))
    return "".join(
        click.style(_BLOCK, fg=_SPLIT_COLORS[b])
        if b is not None
        else click.style(_EMPTY, fg=_UNANNOTATED_COLOR)
        for b in bins
    )


def _render_split_timeline(labels, subfolder_name, width=_TIMELINE_BUDGET):
    """Print a colored timeline of the train/val/test split.

    Annotated episodes are drawn as train/val/test runs sized in
    proportion to their frame counts, and long unannotated gaps between
    episodes are collapsed to ``…``. This keeps the annotated structure
    legible even for sparsely annotated, very long videos (where a
    to-scale bar would be almost all empty). If the annotations are too
    fragmented to compress usefully, fall back to a plain linear bar.
    """
    import click

    frame_to_split = _build_frame_to_split(labels)
    if not frame_to_split:
        return
    num_frames = _num_frames_for(labels, frame_to_split)
    if num_frames <= 0:
        return

    frames = sorted(frame_to_split)
    # A gap wider than ~3% of the video is elided rather than drawn.
    gap = max(10, num_frames // 33)
    episodes = _segment_episodes(frames, gap)
    dim_ellipsis = click.style(f" {_ELLIPSIS} ", fg=_UNANNOTATED_COLOR)

    if len(episodes) > _MAX_TIMELINE_EPISODES:
        bar = _linear_bar(frame_to_split, num_frames, width, click)
    else:
        total = len(frames)
        parts = []
        if frames[0] > gap:
            parts.append(dim_ellipsis)  # gap before the first episode
        for i, episode in enumerate(episodes):
            ep_w = max(_MIN_EPISODE_WIDTH, round(width * len(episode) / total))
            parts.append(_episode_bar(episode, frame_to_split, ep_w, click))
            if i < len(episodes) - 1:
                parts.append(dim_ellipsis)
        if num_frames - 1 - frames[-1] > gap:
            parts.append(dim_ellipsis)  # gap after the last episode
        bar = "".join(parts)

    click.echo(
        f"\nTimeline: {subfolder_name}  ({num_frames} frames, "
        f"{len(frames)} annotated, {len(episodes)} episode(s))"
    )
    click.echo(f"0 {bar} {num_frames}")
    legend = "  ".join(
        (
            click.style(_BLOCK, fg=_SPLIT_COLORS["train"]) + " train",
            click.style(_BLOCK, fg=_SPLIT_COLORS["val"]) + " val",
            click.style(_BLOCK, fg=_SPLIT_COLORS["test"]) + " test",
            click.style(_EMPTY, fg=_UNANNOTATED_COLOR) + " unannotated",
            f"{_ELLIPSIS} gap",
        )
    )
    click.echo(f"Legend: {legend}")


def _print_split_timelines(label_dict, width=_TIMELINE_BUDGET):
    """Print a colored split timeline per subfolder (whole-video view)."""
    for subfolder, labels in label_dict.items():
        _render_split_timeline(labels, Path(subfolder).name, width=width)
