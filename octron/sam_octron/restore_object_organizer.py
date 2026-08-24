"""Rebuild a missing ``object_organizer.json`` from a video subfolder.

OCTRON stores, per annotated video, a subfolder named after the video's
abbreviated hash (e.g. ``<project>/<hash>/``) containing:

* one ``<label>[ <suffix>] masks.zarr`` store per annotated object,
* a ``video data.zarr`` store (preprocessed frames), and
* an ``object_organizer.json`` index tying the mask stores to labels and
  the source video.

If the JSON is lost, the annotations themselves are still intact inside the
mask ``.zarr`` stores. This module reconstructs a schema-correct
``object_organizer.json`` from those stores (array shape + attributes) plus
the ``video_info.txt`` written alongside them, so the project loads and
trains again.

Recovered exactly: labels, per-object mask store paths, data shapes,
annotated-frame counts and the video hash.
Regenerated (cosmetic / not persisted anywhere): object IDs, label IDs and
colors. Not recoverable (never persisted): the raw point/box prompts -- but
the propagated masks they produced live in the zarr and are fully preserved.

Usage (dry run prints the JSON without writing)::

    python -m octron.sam_octron.restore_object_organizer <video_subfolder>
    python -m octron.sam_octron.restore_object_organizer <folder> --write
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import zarr

# Constants mirroring napari/OCTRON mask-layer creation.
_MASK_OPACITY = 0.4
_ZARR_SUFFIX = ".zarr"
_MASK_MARKER = " masks"  # prediction layer name = f"{label} {suffix} masks"
_VIDEO_DATA_ZARR = "video data.zarr"
_N_LABELS_MAX = 10
_N_SUBCOLORS = 50


def _load_color_helpers():
    """Return OCTRON's (create_label_colors, sample_maximally_different).

    Falls back to ``(None, None)`` when OCTRON's color module cannot be
    imported, so recovery still works standalone. Colors are cosmetic here.
    """
    try:
        from octron.sam_octron.helpers.octron_colors import (
            create_label_colors,
            sample_maximally_different,
        )

        return create_label_colors, sample_maximally_different
    except Exception:  # pragma: no cover - defensive fallback
        return None, None


def _fallback_color(label_id: int, sub_index: int) -> list[float]:
    """Return a deterministic RGBA color without OCTRON's colormap."""
    import colorsys

    hue = ((label_id * 0.6180339887) + (sub_index * 0.11)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.65, 0.95)
    return [float(r), float(g), float(b), 1.0]


def _compute_colors(
    pairs: list[tuple[str, str]],
) -> tuple[list[list[float]], dict[str, int]]:
    """Assign an RGBA color per (label, suffix), mirroring ObjectOrganizer.

    ``pairs`` are the (label, suffix) tuples in entry-creation order.
    Returns the per-pair colors and a ``label -> label_id`` mapping.
    """
    create_label_colors, sample_maximally_different = _load_color_helpers()
    if create_label_colors is not None:
        submaps = create_label_colors(
            cmap="cmr.tropical",
            n_labels=_N_LABELS_MAX,
            n_colors_submap=_N_SUBCOLORS,
        )
        idx_labels = sample_maximally_different(list(range(_N_LABELS_MAX)))
        idx_sub = sample_maximally_different(list(range(_N_SUBCOLORS)))
    else:
        submaps = idx_labels = idx_sub = None

    label_id_map: dict[str, int] = {}
    next_label_id = 0
    per_label_count: dict[str, int] = {}
    colors: list[list[float]] = []

    for label, _suffix in pairs:
        if label not in label_id_map:
            label_id_map[label] = next_label_id
            next_label_id += 1
        label_id = label_id_map[label]
        k = per_label_count.get(label, 0)
        per_label_count[label] = k + 1

        if submaps is not None:
            ci = idx_labels[label_id % _N_LABELS_MAX]
            si = idx_sub[k % _N_SUBCOLORS]
            colors.append([float(c) for c in submaps[ci][si]])
        else:
            colors.append(_fallback_color(label_id, k))

    return colors, label_id_map


def _label_suffix_from_zarr(zarr_folder: Path) -> tuple[str, str, list[str]]:
    """Derive (label, suffix, warnings) from a mask zarr folder name.

    Folder names follow ``"{label} {suffix} masks.zarr"``; an empty suffix
    yields ``"{label} masks.zarr"``.
    """
    warnings: list[str] = []
    stem = zarr_folder.name[: -len(_ZARR_SUFFIX)]  # strip ".zarr"
    if stem.endswith(_MASK_MARKER):
        base = stem[: -len(_MASK_MARKER)]
    else:
        base = stem
        warnings.append(
            f"'{zarr_folder.name}' does not end in "
            f"'{_MASK_MARKER}{_ZARR_SUFFIX}'; using the full name as label."
        )

    tokens = base.split()
    if len(tokens) <= 1:
        label, suffix = (tokens[0] if tokens else base), ""
    elif len(tokens) == 2:
        label, suffix = tokens[0], tokens[1]
    else:
        label, suffix = tokens[0], " ".join(tokens[1:])
        warnings.append(
            f"Ambiguous name '{base}': assuming label='{label}', "
            f"suffix='{suffix}'. Edit the JSON if this is wrong."
        )
    return label, suffix, warnings


def _read_mask_store(zarr_folder: Path) -> dict:
    """Read shape + attributes from a mask ``.zarr`` (its ``masks`` array)."""
    store = zarr.storage.LocalStore(zarr_folder, read_only=True)
    root = zarr.open_group(store=store, mode="r")
    if "masks" not in root:
        raise ValueError(f"No 'masks' array in {zarr_folder.name}")
    arr = root["masks"]
    shape = tuple(int(s) for s in arr.shape)
    if len(shape) != 3:
        raise ValueError(
            f"Expected a 3-D (frames, H, W) mask array in "
            f"{zarr_folder.name}, got shape {shape}."
        )
    num_frames, height, width = shape
    attrs = dict(arr.attrs)
    annotated = attrs.get("annotated_frames")
    if annotated is None:
        # Old store without the attribute: scan the (frames, 0, 0) column.
        annotated = np.where(np.asarray(arr[:, 0, 0]) >= 0)[0].tolist()
    return {
        "num_frames": num_frames,
        "height": height,
        "width": width,
        "video_hash": attrs.get("video_hash"),
        "annotated_frames": [int(f) for f in annotated],
    }


def _read_video_info(subfolder: Path) -> dict:
    """Parse ``video_info.txt`` for the source video path and hash."""
    info: dict[str, str | None] = {
        "video_file_path": None,
        "video_hash": None,
    }
    info_path = subfolder / "video_info.txt"
    if not info_path.exists():
        return info
    for line in info_path.read_text().splitlines():
        if line.startswith("Video path:"):
            info["video_file_path"] = line.split(":", 1)[1].strip()
        elif line.startswith("Video abbreviated hash:"):
            info["video_hash"] = line.split(":", 1)[1].strip()
    return info


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as posix, else absolute posix."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except (ValueError, OSError):
        return path.as_posix()


def reconstruct(
    subfolder: str | Path,
    project_root: str | Path | None = None,
    video_path: str | None = None,
) -> tuple[dict, list[str]]:
    """Rebuild the object-organizer mapping for one video subfolder.

    Parameters
    ----------
    subfolder : str or Path
        Per-video folder holding the mask ``.zarr`` stores and
        ``video_info.txt`` (e.g. ``<project>/<hash>``).
    project_root : str or Path, optional
        Root that ``zarr_path`` is stored relative to. Defaults to
        ``subfolder.parent`` (the standard ``<project>/<hash>`` layout).
    video_path : str, optional
        Source video path to use when ``video_info.txt`` is absent.

    Returns
    -------
    (data, warnings) : tuple[dict, list[str]]
        ``data`` is the reconstructed mapping; ``warnings`` lists notes to
        review before trusting the result.
    """
    subfolder = Path(subfolder)
    if not subfolder.is_dir():
        raise NotADirectoryError(f"Not a folder: {subfolder}")
    root = Path(project_root) if project_root else subfolder.parent

    warnings: list[str] = []

    zarr_folders = sorted(
        p
        for p in subfolder.glob(f"*{_ZARR_SUFFIX}")
        if p.is_dir()
        and p.name != _VIDEO_DATA_ZARR
        and p.name.lower().endswith(f"masks{_ZARR_SUFFIX}")
    )
    if not zarr_folders:
        raise FileNotFoundError(
            f"No '*{_MASK_MARKER}{_ZARR_SUFFIX}' stores found in {subfolder}"
        )

    info = _read_video_info(subfolder)
    video_file = video_path or info["video_file_path"]
    if not video_file:
        warnings.append(
            "Could not determine the source video path (no video_info.txt "
            "and no --video-path); set 'video_file_path' in the JSON by hand."
        )

    parsed: list[tuple[Path, str, str]] = []
    for zf in zarr_folders:
        label, suffix, warns = _label_suffix_from_zarr(zf)
        warnings.extend(warns)
        parsed.append((zf, label, suffix))

    colors, label_id_map = _compute_colors(
        [(label, suffix) for _zf, label, suffix in parsed]
    )

    entries: dict[str, Any] = {}
    hashes: set[str] = set()
    for obj_id, ((zf, label, suffix), color) in enumerate(
        zip(parsed, colors)
    ):
        store = _read_mask_store(zf)
        if store["video_hash"]:
            hashes.add(store["video_hash"])
        layer_name = f"{label} {suffix}".strip()
        entries[str(obj_id)] = {
            "label": label,
            "suffix": suffix,
            "label_id": label_id_map[label],
            "color": color,
            "prediction_layer_metadata": {
                "name": f"{layer_name} masks",
                "type": "Labels",
                "num_predicted_indices": len(store["annotated_frames"]),
                "data_shape": [
                    store["num_frames"],
                    store["height"],
                    store["width"],
                ],
                "ndim": 3,
                "visible": True,
                "opacity": _MASK_OPACITY,
                "zarr_path": _relative_posix(zf, root),
                # Stored verbatim (may be absolute / from another OS); the
                # loader joins it onto the project root and absolute wins.
                "video_file_path": video_file,
                "video_hash": store["video_hash"],
            },
        }

    # Cross-checks that mirror collect_labels' expectations.
    if len(hashes) > 1:
        warnings.append(
            f"Mask stores reference multiple video hashes ({sorted(hashes)}); "
            "one subfolder should map to a single video."
        )
    info_hash = info["video_hash"]
    if info_hash and hashes and info_hash not in hashes:
        warnings.append(
            f"video_info.txt hash ({info_hash}) does not match the mask "
            f"store hash(es) {sorted(hashes)}."
        )

    data = {
        "entries": entries,
        "settings": {},
        "time_last_changed": datetime.now().isoformat(),
    }
    return data, warnings


def _write_json(data: dict, target: Path, force: bool) -> None:
    """Write ``data`` to ``target``; refuse to clobber unless ``force``."""
    if target.exists() and not force:
        raise FileExistsError(
            f"{target} already exists; pass --force to overwrite."
        )
    with open(target, "w") as f:
        json.dump(data, f, indent=2)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Dry-run by default; ``--write`` to persist."""
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a missing object_organizer.json from the mask .zarr "
            "stores in an OCTRON per-video subfolder."
        )
    )
    parser.add_argument(
        "subfolder",
        help="Per-video folder (.../<project>/<video_hash>).",
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Root paths are relative to (default: the subfolder's parent).",
    )
    parser.add_argument(
        "--video-path",
        default=None,
        help="Source video path if video_info.txt is missing/incorrect.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write object_organizer.json (otherwise just print it).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing object_organizer.json.",
    )
    args = parser.parse_args(argv)

    subfolder = Path(args.subfolder)
    data, warnings = reconstruct(
        subfolder,
        project_root=args.project_root,
        video_path=args.video_path,
    )

    print(json.dumps(data, indent=2))
    print("\n--- summary ---")
    for obj_id, entry in data["entries"].items():
        meta = entry["prediction_layer_metadata"]
        print(
            f"  id {obj_id}: label='{entry['label']}' "
            f"suffix='{entry['suffix']}' "
            f"frames={meta['num_predicted_indices']} "
            f"shape={tuple(meta['data_shape'])} "
            f"zarr='{meta['zarr_path']}'"
        )
    hashes = sorted(
        {
            e["prediction_layer_metadata"]["video_hash"]
            for e in data["entries"].values()
        }
    )
    print(f"  video_hash(es): {hashes}")
    for w in warnings:
        print(f"  WARNING: {w}")

    target = subfolder / "object_organizer.json"
    if args.write:
        _write_json(data, target, force=args.force)
        print(f"\nWrote {target}")
    else:
        print(
            "\n(dry run - nothing written) "
            f"Re-run with --write to create {target}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
