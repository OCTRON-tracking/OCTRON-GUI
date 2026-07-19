"""OCTRON training pipeline.

Wraps the YOLO_octron model-loading and training steps into a single callable.
By default, training data is prepared automatically via ``run_split()``.
Pass ``skip_split=True`` if ``octron split`` has already been run.
"""

from pathlib import Path

_MODELS_YAML = (
    Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"
)


def run_training(
    project_path,
    model="YOLO26m",
    train_mode="segment",
    device="auto",
    epochs=250,
    imagesz=640,
    save_period=50,
    overwrite=False,
    resume=False,
    skip_split=False,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=88,
):
    """Run the OCTRON/YOLO training pipeline.

    By default this prepares and exports training data before training.
    Pass ``skip_split=True`` to skip that step when data is already up to date.

    Parameters
    ----------
    project_path : str or Path
        Path to the OCTRON project directory.
    model : str or Path
        YOLO model name (e.g. 'YOLO11m') or path to an existing model file.
    device : str
        Device to train on ('auto', 'cpu', 'cuda', 'mps'). 'auto' selects
        CUDA if available, then MPS, then CPU.
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
    train_fraction : float
        Fraction of frames for training (ignored when ``skip_split=True``).
    val_fraction : float
        Fraction of frames for validation (ignored when ``skip_split=True``).
    seed : int
        Random seed for the split (ignored when ``skip_split=True``).

    """
    from octron.test_gpu import auto_device
    from octron.tools.split import run_split
    from octron.yolo_octron.yolo_octron import YOLO_octron

    # Unwrap enums to plain strings so they are never serialised as Python
    # object tags when written into YAML config files downstream.
    train_mode = (
        train_mode.value if hasattr(train_mode, "value") else str(train_mode)
    )
    device = device.value if hasattr(device, "value") else str(device)

    if device == "auto":
        device = auto_device()

    # Create the model wrapper first so the resume/overwrite decision and the
    # config-path resolution both live in core (shared with the GUI).
    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
        clean_training_dir=False,
    )
    yolo.train_mode = train_mode

    # Decide fresh vs. strict-resume vs. continue-from-completed-checkpoint.
    state = yolo.resolve_resume_state(resume=resume, overwrite=overwrite)
    action = state["action"]
    if action in ("completed", "error"):
        print(state["message"])
        return
    print(state["message"])

    # --- Steps 1–4: prepare and export training data ---
    if not skip_split:
        run_split(
            project_path=project_path,
            train_fraction=train_fraction,
            val_fraction=val_fraction,
            seed=seed,
            train_mode=train_mode,
            dry_run=False,
        )

    # --- Step 5: load the base model, or last.pt when resuming/continuing ---
    if action in ("resume", "init_from_checkpoint"):
        # Image size is recovered from the checkpoint, overriding --imagesz.
        imagesz = state["imgsz"]
        yolo.load_model(state["checkpoint"], train_mode=train_mode)
    else:
        model_name = model.value if hasattr(model, "value") else model
        print(f"Loading model: {model_name}...")
        yolo.load_model(model, train_mode=train_mode)

    # --- Step 6: train (core resolves a cached AutoBatch size for CUDA) ---
    print(f"Training for {epochs} epochs on {device}...")
    for progress in yolo.train(
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
