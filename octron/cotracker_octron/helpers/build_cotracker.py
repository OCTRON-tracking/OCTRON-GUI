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

    # Get online CoTracker predictor and move to device
    model = CoTrackerOnlinePredictor(checkpoint=ckpt_path)
    model = model.to(device)

    # Cotracker does not require the input image to be a certain
    # size, but internally its working resolution is 384×512. It then
    # scales the predicted tracks back to the original video resolution. 
    # For the zarr cache we set it to the actual video resolution to match
    # the napari videos and the cotracker results. 
    model.image_size = None

    return model, device
