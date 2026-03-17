"""
OCTRON training pipeline.

Wraps the YOLO_octron model-loading and training steps into a single callable.
By default, training data is prepared automatically via ``run_split()``.
Pass ``skip_split=True`` if ``octron split`` has already been run.
"""

import json
from pathlib import Path

_MODELS_YAML = Path(__file__).parent.parent / "yolo_octron" / "yolo_models.yaml"


def _get_batch_size(model, imgsz, device, cache_path):
    """
    Return the optimal batch size for training, using a cache to avoid
    running AutoBatch on every training run.

    The cache is stored at ``cache_path`` and keyed by model architecture,
    image size, and GPU name so it stays valid across runs with the same
    hardware and model.
    """
    import torch

    gpu_name = (
        torch.cuda.get_device_name(device) if torch.cuda.is_available() else "cpu"
    )
    cache_key = f"{model.model.model.__class__.__name__}_{imgsz}_{gpu_name}"

    cache_path = Path(cache_path)
    cache = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    if cache_key in cache:
        batch = cache[cache_key]
        print(f"AutoBatch: using cached batch size {batch} for {gpu_name} (imgsz={imgsz})")
        return batch

    print("AutoBatch: running batch size search (result will be cached) ...")
    from ultralytics.utils.autobatch import check_train_batch_size
    batch = check_train_batch_size(model.model.model, imgsz=imgsz, amp=True)

    cache[cache_key] = batch
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f)
    print(f"AutoBatch: batch size {batch} cached to {cache_path}")
    return batch


def run_training(
    project_path,
    model="YOLO26m",
    device="auto",
    epochs=250,
    imagesz=640,
    save_period=50,
    train_mode="segment",
    resume=False,
    skip_split=False,
    train_fraction=0.7,
    val_fraction=0.15,
    seed=88,
):
    """
    Run the OCTRON/YOLO training pipeline.

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
        'segment' for instance segmentation, 'detect' for bounding-box detection.
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
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device
    from octron.tools.split import run_split

    if device == "auto":
        device = auto_device()

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

    # --- Step 5: load the base model ---
    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
        clean_training_dir=False,
    )
    yolo.train_mode = train_mode
    yolo.config_path = yolo.data_path / "yolo_config.yaml"

    print(f"Loading model: {model}...")
    yolo.load_model(model, train_mode=train_mode)

    # --- Step 6: train ---
    batch_cache = Path(project_path) / "model" / "autobatch_cache.json"
    batch = _get_batch_size(yolo, imagesz, device, batch_cache)

    print(f"Training for {epochs} epochs on {device}...")
    for progress in yolo.train(
        device=device,
        imagesz=imagesz,
        epochs=epochs,
        save_period=save_period,
        train_mode=train_mode,
        resume=resume,
        batch=batch,
    ):
        epoch = progress.get("epoch", "?")
        total_epochs = progress.get("total_epochs", "?")
        remaining = progress.get("remaining_time", 0)
        print(f"  Epoch {epoch}/{total_epochs} | ETA: {remaining:.0f}s", end="\r")
    print()
    print("Training complete.")
