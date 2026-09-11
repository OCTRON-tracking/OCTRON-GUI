"""OCTRON training-data split pipeline.

Prepares and exports train/val/test data from an OCTRON project without
running model training.  The `octron train` command calls this internally;
users can also run it standalone via `octron split`.
"""

from pathlib import Path

_MODELS_YAML = (
    Path(__file__).parent.parent / "analysis_octron" / "analysis_models.yaml"
)


def run_split(
    project_path,
    train_fraction=None,
    val_fraction=None,
    seed=None,
    buffer=None,
    prune=False,
    watershed=False,
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
        Fraction of frames for the training split. None (the CLI
        default) reads ``split_train_fraction`` from ``config.yaml``.
    val_fraction : float or None
        Fraction of frames for the validation split; the remainder
        becomes the test split. None reads ``split_val_fraction``
        from config.
    seed : int or None
        Random seed for reproducibility. None reads ``split_seed``
        from config.
    buffer : int or None
        Frames dropped at each train/val/test block boundary to add a
        temporal gap between splits. None (the CLI default) reads
        ``split_buffer`` from ``config.yaml``.
    prune : bool
        Drop frames where not all labels are annotated (threaded to
        ``prepare_labels(prune_empty_labels=...)``). Default ``False``,
        matching the GUI's Prune checkbox.
    watershed : bool
        Watershed touching same-label masks into separate instances
        before geometry generation. Default ``False``.
    train_mode : str
        ``'segment'`` for instance segmentation, ``'detect'`` for bounding-box
        detection only.
    dry_run : bool
        If ``True``, print split sizes without writing anything to disk.

    """
    from octron.analysis_octron.analysis_octron import AnalysisOctron

    train_mode = (
        train_mode.value if hasattr(train_mode, "value") else str(train_mode)
    )

    # Resolve unset split parameters from config.yaml. Precedence: a CLI
    # flag (non-None) overrides config, which overrides the built-in
    # default. The GUI reads the same config for its defaults.
    if (
        train_fraction is None
        or val_fraction is None
        or seed is None
        or buffer is None
    ):
        from octron import config

        if train_fraction is None or val_fraction is None:
            cfg_train, cfg_val = config.get_split_fractions()
            if train_fraction is None:
                train_fraction = cfg_train
            if val_fraction is None:
                val_fraction = cfg_val
        if seed is None:
            seed = config.get_split_seed()
        if buffer is None:
            buffer = config.get_split_buffer()

    # Validate fractions up front using the core guard (also enforced
    # inside prepare_split) so the CLI fails before any model, label, or
    # geometry work.
    AnalysisOctron._validate_split_fractions(train_fraction, val_fraction)

    analysis = AnalysisOctron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
    )
    analysis.train_mode = train_mode
    analysis.enable_watershed = watershed

    # --- Step 1: collect labels ---
    print("Preparing labels...")
    analysis.prepare_labels(prune_empty_labels=prune)

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
    for _ in analysis.prepare_geometry():
        pass

    # --- Step 3: split ---
    print("Splitting data into train/val/test sets...")
    analysis.prepare_split(
        training_fraction=train_fraction,
        validation_fraction=val_fraction,
        random_seed=seed,
        buffer=buffer,
    )

    # Print summary table + colored whole-video timelines (shared w/ GUI)
    from octron.analysis_octron.helpers.split_report import render_split_report

    render_split_report(analysis.summarize_split(), seed)

    if dry_run:
        print("Dry run — no files written.")
        return

    # --- Step 4: export to disk ---
    print("Exporting training data...")
    # As above: tqdm owns the export progress bar; we just consume the
    # generator so a competing stdout writer can't break the bar.
    for _ in analysis.create_training_data():
        pass

    analysis.write_analysis_config(train_mode=train_mode)
    print("Training data export complete.")
