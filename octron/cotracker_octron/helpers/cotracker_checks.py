# Code for checking the availability of the Cotracker model checkpoint
from pathlib import Path

from huggingface_hub import hf_hub_download

COTRACKER_HF_REPO = "facebook/cotracker3"
COTRACKER_FILES = ["scaled_online.pth"]


def download_cotracker_file(filename, local_dir, overwrite=False):
    """
    Download a single file from the Cotracker HuggingFace repository.

    Parameters
    ----------
    filename : str
        Name of the file to download from the HF repository at
        https://huggingface.co/facebook/cotracker3/tree/main
        (e.g. "scaled_online.pth").
    local_dir : str or Path
        Local directory to download into.
    overwrite : bool
        If True, re-download even if the file already exists locally.

    Returns
    -------
    local_path : Path
        Path to the downloaded file.
    """
    local_dir = Path(local_dir)
    local_path = local_dir / filename

    if local_path.exists() and not overwrite:
        print(f"File '{local_path}' exists. Skipping download.")
        return local_path

    print(f"Downloading {filename} from {COTRACKER_HF_REPO} ...")
    cached_path = hf_hub_download(
        repo_id=COTRACKER_HF_REPO,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"💾 Saved {filename} to {cached_path}")
    return Path(cached_path)


def check_cotracker_models(checkpoints_dir, force_download=False):
    """
    Ensure the Cotracker files are available locally.
    Downloads them from HuggingFace if missing.

    Returns a dictionary for the model-selection dropdown.

    Parameters
    ----------
    checkpoints_dir : str or Path
        Directory where the files should reside
        (typically ``<base_path>/cotracker_octron/checkpoints``).
    force_download : bool
        Re-download even if files exist.

    Returns
    -------
    cotracker_models_dict : dict
        Keys are model IDs, values contain 'name', 'checkpoint_path'.
        Returns an empty dict if download fails.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

    try:
        for fname in COTRACKER_FILES:
            download_cotracker_file(
                filename=fname,
                local_dir=checkpoints_dir,
                overwrite=force_download,
            )
    except Exception as e:
        print(f"⚠️  Could not download Cotracker file: {e}")
        return {}

    # Verify the checkpoint actually landed
    ckpt_path = checkpoints_dir / "scaled_online.pth"
    if not ckpt_path.exists():
        print("⚠️  cotracker checkpoint not found after download attempt.")
        return {}

    # Build the models dict (relative to cotracker_octron/)
    cotracker_models_dict = {
        "cotracker": {
            "name": "Cotracker3",
            "checkpoint_path": "checkpoints/scaled_online.pth",
        }
    }
    return cotracker_models_dict
