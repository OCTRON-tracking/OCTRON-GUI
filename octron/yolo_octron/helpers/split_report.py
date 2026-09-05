"""Shared reporting for the train/val/test split.

This module is the single source of truth for the ``octron split`` summary
table and the colored, episode-aware timeline, so the CLI (``run_split``)
and the GUI both render an identical report. It is deliberately split into
two layers:

- ``build_split_report(label_dict)`` computes pure, click-free structured
  data (per-subfolder counts and timeline column structure). This is what
  :meth:`YOLO_octron.summarize_split` returns, so callers can inspect the
  split programmatically or feed a widget.
- ``render_split_report(report, seed)`` prints that data to the console:
  the count table via ``print`` and the colored timeline via ``click``
  (which strips ANSI automatically when stdout is not a TTY).
"""

from collections import Counter
from pathlib import Path

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


# ---------------------------------------------------------------------------
# Pure data helpers (no click / no printing)
# ---------------------------------------------------------------------------


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


def _annotated_frame_count(labels):
    """Count all annotated frames in a subfolder (union across labels).

    This is the pre-split total (matching the summary's ``Total``). It can
    exceed the number of frames assigned to train/val/test because
    ``train_test_val`` drops ``buffer`` frames at block boundaries; those
    dropped frames show as gaps in the timeline.
    """
    frames = set()
    for entry, info in labels.items():
        if entry in ("video", "video_file_path"):
            continue
        for frame in info.get("frames", []):
            frames.add(int(frame))
    return len(frames)


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


def _episode_cols(frames, frame_to_split, width):
    """Reduce one episode's frame span into ``width`` dominant-split columns.

    Returns a list of split names (or ``None`` for a column with no frames);
    the renderer maps these to colored blocks.
    """
    start, end = frames[0], frames[-1]
    span = max(1, end - start + 1)
    buckets = [Counter() for _ in range(width)]
    for frame in frames:
        col = min(width - 1, int((frame - start) * width / span))
        buckets[col][frame_to_split[frame]] += 1
    return [c.most_common(1)[0][0] if c else None for c in buckets]


def _build_timeline(labels):
    """Return timeline data for one subfolder, or ``None`` if unsplit.

    ``segments`` is an ordered list describing the bar: ``("gap",)`` for an
    elided unannotated stretch and ``("blocks", cols)`` for a run of
    dominant-split columns (``cols`` entries are ``"train"``/``"val"``/
    ``"test"``/``None``). Annotated episodes are sized proportional to
    their frame count over a shared column budget; long gaps between
    episodes collapse to ``("gap",)``. Falls back to a single to-scale
    ``("blocks", ...)`` segment when the annotations are too fragmented.
    """
    frame_to_split = _build_frame_to_split(labels)
    if not frame_to_split:
        return None
    num_frames = _num_frames_for(labels, frame_to_split)
    if num_frames <= 0:
        return None

    frames = sorted(frame_to_split)
    gap = max(10, num_frames // 33)
    episodes = _segment_episodes(frames, gap)
    segments = []
    if len(episodes) > _MAX_TIMELINE_EPISODES:
        width = min(_TIMELINE_BUDGET, num_frames)
        segments.append(
            ("blocks", _timeline_bins(frame_to_split, num_frames, width))
        )
    else:
        total = len(frames)
        if frames[0] > gap:
            segments.append(("gap",))
        for i, episode in enumerate(episodes):
            ep_w = max(
                _MIN_EPISODE_WIDTH,
                round(_TIMELINE_BUDGET * len(episode) / total),
            )
            segments.append(
                ("blocks", _episode_cols(episode, frame_to_split, ep_w))
            )
            if i < len(episodes) - 1:
                segments.append(("gap",))
        if num_frames - 1 - frames[-1] > gap:
            segments.append(("gap",))

    return {
        "num_frames": num_frames,
        "assigned": len(frames),
        "buffered": max(0, _annotated_frame_count(labels) - len(frames)),
        "n_episodes": len(episodes),
        "segments": segments,
    }


def build_split_report(label_dict):
    """Compute the structured split report (counts + timeline) per subfolder.

    Pure and click-free; this is what :meth:`YOLO_octron.summarize_split`
    returns and what :func:`render_split_report` consumes.
    """
    report = []
    for subfolder, labels in label_dict.items():
        rows = []
        for entry, info in labels.items():
            if entry in ("video", "video_file_path"):
                continue
            split = info.get("frames_split", {})
            rows.append(
                (
                    info["label"],
                    len(split.get("train", [])),
                    len(split.get("val", [])),
                    len(split.get("test", [])),
                    len(info.get("frames", [])),
                )
            )
        report.append(
            {
                "name": Path(subfolder).name,
                "rows": rows,
                "timeline": _build_timeline(labels),
            }
        )
    return report


# ---------------------------------------------------------------------------
# Rendering (console: print for the table, click for the colored timeline)
# ---------------------------------------------------------------------------


def _render_summary_table(report, seed):
    """Print a per-label, per-split frame count table."""
    rows = []
    for sub in report:
        for label, tr, va, test_n, tot in sub["rows"]:
            rows.append((sub["name"], label, tr, va, test_n, tot))
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
    for subfolder, label, tr, va, test_n, tot in rows:
        print(
            f"{subfolder:{col_w[0]}}  "
            f"{label:{col_w[1]}}  "
            f"{tr:>{col_w[2]}}  "
            f"{va:>{col_w[3]}}  "
            f"{test_n:>{col_w[4]}}  "
            f"{tot:>{col_w[5]}}"
        )
    print(sep)
    print()


def _render_timeline(sub, click):
    """Print one subfolder's colored whole-video timeline."""
    tl = sub["timeline"]
    if tl is None:
        return
    dim_ellipsis = click.style(f" {_ELLIPSIS} ", fg=_UNANNOTATED_COLOR)
    parts = []
    for segment in tl["segments"]:
        if segment[0] == "gap":
            parts.append(dim_ellipsis)
            continue
        for split in segment[1]:
            if split is not None:
                parts.append(click.style(_BLOCK, fg=_SPLIT_COLORS[split]))
            else:
                parts.append(click.style(_EMPTY, fg=_UNANNOTATED_COLOR))
    bar = "".join(parts)
    click.echo(
        f"\nTimeline: {sub['name']}  ({tl['num_frames']} frames, "
        f"{tl['assigned']} assigned, {tl['buffered']} buffered, "
        f"{tl['n_episodes']} episode(s))"
    )
    click.echo(f"0 {bar} {tl['num_frames']}")
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


def render_split_report(report, seed):
    """Print the split summary table and colored timelines to the console.

    Shared by the CLI (``run_split``) and the GUI so both surfaces show an
    identical report. ``report`` is the output of :func:`build_split_report`
    (or :meth:`YOLO_octron.summarize_split`); ``seed`` is shown in the table
    header only.
    """
    import click

    _render_summary_table(report, seed)
    for sub in report:
        _render_timeline(sub, click)
