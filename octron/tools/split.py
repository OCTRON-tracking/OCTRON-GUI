"""
OCTRON training-data split pipeline.

Prepares and exports train/val/test data from an OCTRON project without
running model training.  The `octron train` command calls this internally;
users can also run it standalone via `octron split`.
"""

from pathlib import Path

_MODELS_YAML = Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"


def run_split(
    project_path,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=88,
    train_mode="segment",
    dry_run=False,
):
    """
    Prepare and export train/val/test data for an OCTRON project.

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
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1); got {train_fraction!r}"
        )
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(
            f"val_fraction must be in [0, 1); got {val_fraction!r}"
        )
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            f"train_fraction + val_fraction must be < 1 (test split = remainder); "
            f"got {train_fraction} + {val_fraction} = {train_fraction + val_fraction}"
        )

    from octron.yolo_octron.yolo_octron import YOLO_octron

    train_mode = train_mode.value if hasattr(train_mode, 'value') else str(train_mode)

    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
    )
    yolo.train_mode = train_mode

    # --- Step 1: collect labels ---
    print("Preparing labels...")
    yolo.prepare_labels()

    # --- Step 2: generate polygons / bboxes ---
    if train_mode == "segment":
        print("Generating polygons...")
        for no_entry, total, label, frame_no, total_frames in yolo.prepare_polygons():
            print(
                f"  [{no_entry}/{total}] {label}: frame {frame_no}/{total_frames}",
                end="\r",
            )
        print()
    else:
        print("Generating bounding boxes...")
        for no_entry, total, label, frame_no, total_frames in yolo.prepare_bboxes():
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
    )

    # Print summary table
    _print_split_summary(yolo.label_dict, seed)

    if dry_run:
        print("Dry run — no files written.")
        return

    # --- Step 4: export to disk ---
    print("Exporting training data...")
    if train_mode == "segment":
        gen = yolo.create_training_data_segment()
    else:
        gen = yolo.create_training_data_detect()
    for no_entry, total, label, split, frame_no, total_frames in gen:
        print(
            f"  [{no_entry}/{total}] {label} ({split}): frame {frame_no}/{total_frames}",
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
    col_w = [max(w, h) for w, h in zip(col_w, [8, 5, 5, 3, 4, 5])]
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
    for subfolder, label, tr, va, te, tot in rows:
        print(
            f"{subfolder:{col_w[0]}}  "
            f"{label:{col_w[1]}}  "
            f"{tr:>{col_w[2]}}  "
            f"{va:>{col_w[3]}}  "
            f"{te:>{col_w[4]}}  "
            f"{tot:>{col_w[5]}}"
        )
    print(sep)
    print()
