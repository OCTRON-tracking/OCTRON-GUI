"""
OCTRON training pipeline.

Wraps the YOLO_octron training steps into a single callable function
that can be used from the CLI or called programmatically.
"""

from pathlib import Path

_MODELS_YAML = Path(__file__).parent / 'yolo_octron' / 'yolo_models.yaml'


def run_training(
    project_path,
    model='YOLO11m',
    device='auto',
    epochs=30,
    imagesz=640,
    save_period=15,
    train_mode='segment',
    resume=False,
):
    """
    Run the full OCTRON/YOLO training pipeline.

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
    """
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device
    if device == 'auto':
        device = auto_device()

    yolo = YOLO_octron(
        models_yaml_path=_MODELS_YAML,
        project_path=project_path,
    )
    yolo.train_mode = train_mode

    # --- Step 1: collect labels from project annotation files ---
    print('Preparing labels...')
    yolo.prepare_labels()

    # --- Step 2: generate polygons or bboxes (generator, consume for progress) ---
    if train_mode == 'segment':
        print('Generating polygons...')
        for no_entry, total, label, frame_no, total_frames in yolo.prepare_polygons():
            print(f'  [{no_entry}/{total}] {label}: frame {frame_no}/{total_frames}', end='\r')
        print()
    else:
        print('Generating bounding boxes...')
        for no_entry, total, label, frame_no, total_frames in yolo.prepare_bboxes():
            print(f'  [{no_entry}/{total}] {label}: frame {frame_no}/{total_frames}', end='\r')
        print()

    # --- Step 3: split into train / val / test ---
    print('Splitting data into train/val/test sets...')
    yolo.prepare_split()

    # --- Step 4: export training data to disk ---
    print('Exporting training data...')
    if train_mode == 'segment':
        gen = yolo.create_training_data_segment()
    else:
        gen = yolo.create_training_data_detect()
    for no_entry, total, label, split, frame_no, total_frames in gen:
        print(f'  [{no_entry}/{total}] {label} ({split}): frame {frame_no}/{total_frames}', end='\r')
    print()

    # --- Step 5: load the base model ---
    print(f'Loading model: {model}...')
    yolo.load_model(model, train_mode=train_mode)

    # --- Step 6: train ---
    print(f'Training for {epochs} epochs on {device}...')
    for progress in yolo.train(
        device=device,
        imagesz=imagesz,
        epochs=epochs,
        save_period=save_period,
        train_mode=train_mode,
        resume=resume,
    ):
        epoch = progress.get('epoch', '?')
        total_epochs = progress.get('total_epochs', '?')
        remaining = progress.get('remaining_time', 0)
        print(f'  Epoch {epoch}/{total_epochs} | ETA: {remaining:.0f}s', end='\r')
    print()
    print('Training complete.')
