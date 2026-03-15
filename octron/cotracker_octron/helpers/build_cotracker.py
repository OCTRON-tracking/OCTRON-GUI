import torch
from cotracker.predictor import CoTrackerOnlinePredictor


def build_cotracker(ckpt_path):
    """Build the CoTracker model from a checkpoint path.

    It assumes the checkpoint is already downloaded and available at ckpt_path.
    Returns the model and the torch device it's on.
    """
    # Select device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # Instantiate CoTracker model from checkpoint and move to device
    model = CoTrackerOnlinePredictor(checkpoint=ckpt_path)
    model = model.to(device)

    return model, device
