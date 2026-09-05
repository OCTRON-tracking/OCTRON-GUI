"""OCTRON training-data split pipeline.

Prepares and exports train/val/test data from an OCTRON project without
running model training.  The `octron train` command calls this internally;
users can also run it standalone via `octron split`.
"""

from pathlib import Path

_MODELS_YAML = (
    Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"
)


def run_split(
    project_path,
    train_fraction=None,
    val_fraction=None,
    seed=None,
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
    train_fraction : float or None
        Fraction of frames for the training split. ``None`` (the CLI
        default) reads ``split_train_fraction`` from ``config.yaml``.
    val_fraction : float or None
        Fraction of frames for the validation split; the remainder
        becomes the test split. ``None`` reads ``split_val_fraction``
        from config.
    seed : int or None
        Random seed for reproducibility. ``None`` reads ``split_seed``
        from config.
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

    # Resolve unset split parameters from config.yaml. Precedence: a CLI
    # flag (non-None) overrides config, which overrides the built-in
    # default. The GUI reads the same config for its defaults.
    if train_fraction is None or val_fraction is None or seed is None:
        from octron import config

        if train_fraction is None or val_fraction is None:
            cfg_train, cfg_val = config.get_split_fractions()
            if train_fraction is None:
                train_fraction = cfg_train
            if val_fraction is None:
                val_fraction = cfg_val
        if seed is None:
            seed = config.get_split_seed()

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

    # Print summary table + colored whole-video timelines (shared w/ GUI)
    from octron.yolo_octron.helpers.split_report import render_split_report

    render_split_report(yolo.summarize_split(), seed)

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
