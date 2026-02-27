# Code for checking the availability of the SAM3 model checkpoint and config
import os
from pathlib import Path


SAM3_HF_REPO = "facebook/sam3"
SAM3_FILES = ["sam3.pt", "config.json"]


def download_sam3_file(filename, local_dir, overwrite=False):
    """
    Download a single file from the SAM3 HuggingFace repository.
    
    Uses huggingface_hub to handle authentication (for gated models)
    and caching. The user must have run `huggingface-cli login` or set
    HF_TOKEN in their environment.
    
    Parameters
    ----------
    filename : str
        Name of the file to download (e.g. "sam3.pt", "config.json").
    local_dir : str or Path
        Local directory to download into.
    overwrite : bool
        If True, re-download even if the file already exists locally.
    
    Returns
    -------
    local_path : Path
        Path to the downloaded file.
    """
    from huggingface_hub import hf_hub_download

    local_dir = Path(local_dir)
    local_path = local_dir / filename

    if local_path.exists() and not overwrite:
        print(f"File '{local_path}' exists. Skipping download.")
        return local_path

    print(f"Downloading {filename} from {SAM3_HF_REPO} ...")
    cached_path = hf_hub_download(
        repo_id=SAM3_HF_REPO,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"💾 Saved {filename} to {cached_path}")
    return Path(cached_path)


def check_sam3_models(checkpoints_dir, force_download=False):
    """
    Ensure the SAM3 checkpoint and config are available locally.
    Downloads them from HuggingFace if missing.
    
    Returns a dictionary describing the two SAM3 modes that can be
    added to the model-selection dropdown.
    
    Parameters
    ----------
    checkpoints_dir : str or Path
        Directory where sam3.pt and config.json should reside
        (typically ``<base_path>/sam2_octron/checkpoints``).
    force_download : bool
        Re-download even if files exist.
    
    Returns
    -------
    sam3_models_dict : dict
        Keys are model IDs, values contain 'name', 'checkpoint_path',
        and 'semantic' (bool).  Returns an empty dict if download fails.
    """
    checkpoints_dir = Path(checkpoints_dir)
    if not checkpoints_dir.exists():
        os.makedirs(checkpoints_dir, exist_ok=True)

    try:
        for fname in SAM3_FILES:
            download_sam3_file(
                filename=fname,
                local_dir=checkpoints_dir,
                overwrite=force_download,
            )
    except Exception as e:
        print(f"⚠️  Could not download SAM3 files: {e}")
        print("   Make sure you have accepted the licence at "
              "https://huggingface.co/facebook/sam3 and run "
              "`huggingface-cli login`.")
        return {}

    # Verify the checkpoint actually landed
    ckpt_path = checkpoints_dir / "sam3.pt"
    if not ckpt_path.exists():
        print("⚠️  sam3.pt not found after download attempt.")
        return {}

    # Build the models dict (relative to sam2_octron/)
    rel_ckpt = f"checkpoints/sam3.pt"

    sam3_models_dict = {
        "sam3_mode_a": {
            "name": "SAM3 (tracking)",
            "checkpoint_path": rel_ckpt,
            "semantic": False,
        },
        "sam3_mode_b": {
            "name": "SAM3 Semantic (detect+track)",
            "checkpoint_path": rel_ckpt,
            "semantic": True,
        },
    }
    return sam3_models_dict
