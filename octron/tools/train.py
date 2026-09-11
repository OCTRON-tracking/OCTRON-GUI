"""OCTRON training pipeline.

Wraps the AnalysisOctron model-loading and training steps into a single
callable. By default, training data is prepared automatically via
``run_split()``. Pass ``skip_split=True`` if ``octron split`` has already
been run.
"""

from pathlib import Path

_MODELS_YAML = (
    Path(__file__).parent.parent / "analysis_octron" / "analysis_models.yaml"
)


def run_training(
    project_path,
    model="YOLO26m",
    train_mode="segment",
    device=None,
    epochs=250,
    imagesz=640,
    save_period=50,
    overwrite=False,
    resume=False,
    skip_split=False,
    train_fraction=None,
    val_fraction=None,
    seed=None,
    buffer=None,
    prune=False,
    watershed=False,
):
    """Run the OCTRON training pipeline.

    By default this prepares and exports training data before training.
    Pass ``skip_split=True`` to skip that step when data is already up to date.

    Parameters
    ----------
    project_path : str or Path
        Path to the OCTRON project directory.
    model : str or Path
        Model name (e.g. 'YOLO11m') or path to an existing model file.
    device : str or None
        Device to train on ('auto', 'cpu', 'cuda', 'mps'). None reads
        ``device`` from ``config.yaml`` (default 'auto' selects CUDA if
        available, then MPS, then CPU).
    epochs : int
        Number of training epochs.
    imagesz : int
        Input image size for training.
    save_period : int
        Save a checkpoint every N epochs.
    train_mode : str
        'segment' for instance segmentation, 'detect' for bounding-box
        detection.
    overwrite : bool
        Train from scratch, discarding any existing checkpoint. Overwrite
        always wins over ``resume``.
    resume : bool
        Resume training from an existing last.pt checkpoint.
    skip_split : bool
        Skip data preparation. Use when ``octron split`` has already been run
        and the training data is up to date.
    train_fraction : float or None
        Fraction of frames for training. None reads ``config.yaml``
        (ignored when ``skip_split=True``).
    val_fraction : float or None
        Fraction of frames for validation. None reads ``config.yaml``
        (ignored when ``skip_split=True``).
    seed : int or None
        Random seed for the split. None reads ``config.yaml``
        (ignored when ``skip_split=True``).
    buffer : int or None
        Frames dropped at each train/val/test block boundary. None
        reads ``config.yaml`` (ignored when ``skip_split=True``).
    prune : bool
        Drop frames where not all labels are annotated (ignored when
        ``skip_split=True``). Default ``False``.
    watershed : bool
        Watershed touching same-label masks into separate instances
        (ignored when ``skip_split=True``). Default ``False``.

    """
    from octron import config
    from octron.analysis_octron.analysis_octron import AnalysisOctron
    from octron.test_gpu import auto_device
    from octron.tools.split import run_split

    # Unwrap enums to plain strings so they are never serialised as Python
    # object tags when written into YAML config files downstream.
    train_mode = (
        train_mode.value if hasattr(train_mode, "value") else str(train_mode)
    )
    # Resolve device from config.yaml when not set explicitly (the CLI
    # passes None). Precedence: explicit arg > config.yaml > 'auto'.
    if device is None:
        device = config.get_device()
    device = device.value if hasattr(device, "value") else str(device)

    if device == "auto":
        device = auto_device()

    # Create the model wrapper first so the resume/overwrite decision and the
    # config-path resolution both live in core (shared with the GUI).
    analysis = AnalysisOctron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
        clean_training_dir=False,
    )
    analysis.train_mode = train_mode

    # Decide fresh vs. strict-resume vs. continue-from-completed-checkpoint.
    state = analysis.resolve_resume_state(resume=resume, overwrite=overwrite)
    action = state["action"]
    if action in ("completed", "error"):
        print(state["message"])
        return
    print(state["message"])

    # Fail fast if the chosen base model does not support the requested
    # task (e.g. an RT-DETR model with --mode segment). This mirrors the
    # GUI, which hides unsupported models from the menu. Only relevant
    # for a fresh run; resume/continue reload the existing checkpoint.
    if action not in ("resume", "init_from_checkpoint") and (
        not analysis.supports_task(model, train_mode)
    ):
        resolved = analysis.resolve_model_name(model)
        task_label = "detection" if train_mode == "detect" else "segmentation"
        other_label = "segmentation" if train_mode == "detect" else "detection"
        display = analysis.models_dict[resolved].get("name", resolved)
        print(
            f"Model '{display}' does not support {task_label}. "
            f"Use --mode {other_label} or choose a "
            f"{task_label}-capable model."
        )
        return

    # --- Steps 1–4: prepare and export training data ---
    if not skip_split:
        run_split(
            project_path=project_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed,
            buffer=buffer,
            prune=prune,
            watershed=watershed,
            train_mode=train_mode,
            dry_run=False,
        )

    # --- Step 5: load the base model, or last.pt when resuming/continuing ---
    if action in ("resume", "init_from_checkpoint"):
        # Image size is recovered from the checkpoint, overriding --imagesz.
        imagesz = state["imgsz"]
        analysis.load_model(state["checkpoint"], train_mode=train_mode)
    else:
        model_name = model.value if hasattr(model, "value") else model
        print(f"Loading model: {model_name}...")
        analysis.load_model(model, train_mode=train_mode)

    # --- Step 6: train (core resolves a cached AutoBatch size for CUDA) ---
    print(f"Training for {epochs} epochs on {device}...")
    for progress in analysis.train(
        device=device,
        imagesz=imagesz,
        epochs=epochs,
        save_period=save_period,
        train_mode=train_mode,
        resume=(action == "resume"),
    ):
        epoch = progress.get("epoch", "?")
        total_epochs = progress.get("total_epochs", "?")
        remaining = progress.get("remaining_time", 0)
        print(
            f"  Epoch {epoch}/{total_epochs} | ETA: {remaining:.0f}s", end="\r"
        )
    print()
    print("Training complete.")
