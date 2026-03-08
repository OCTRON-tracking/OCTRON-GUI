"""

Follow the pattern in build_sam3_octron.py:

Load CoTracker3 checkpoint (from torch hub or local file)
Move to device, set eval mode
Return (model, device)


cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online").to(device)

# Run Online CoTracker, the same model with a different API:
# Initialize online processing
cotracker(video_chunk=video, is_first_step=True, grid_size=grid_size)

# Process the video
for ind in range(0, video.shape[1] - cotracker.step, cotracker.step):
    pred_tracks, pred_visibility = cotracker(
        video_chunk=video[:, ind : ind + cotracker.step * 2]
    )  # B T N 2,  B T N 1
"""

import torch
from cotracker.predictor import CoTrackerOnlinePredictor


def build_cotracker(ckpt_path):
    """Build the CoTracker model from a checkpoint path.

    It assumes the checkpoint is already downloaded and available at ckpt_path.
    Returns the model and the torch device it's on.
    """
    # Find out which device to use
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
