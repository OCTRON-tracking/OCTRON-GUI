# Code for checking the availability of the SAM3 model checkpoint and config
import os
from pathlib import Path
import yaml
from loguru import logger


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
        logger.info(f"File '{local_path}' exists. Skipping download.")
        return local_path

    logger.info(f"Downloading {filename} from {SAM3_HF_REPO} ...")
    cached_path = hf_hub_download(
        repo_id=SAM3_HF_REPO,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    logger.info(f"💾 Saved {filename} to {cached_path}")
    return Path(cached_path)


def check_sam3_models(models_yaml_path, force_download=False):
    """
    Load SAM3 model definitions from a YAML file and ensure the
    checkpoint files are available locally (downloading from
    HuggingFace if missing).
    
    Parameters
    ----------
    models_yaml_path : str or Path
        Path to the sam3_models.yaml file.
    force_download : bool
        Re-download even if files exist.
    
    Returns
    -------
    models_dict : dict
        Keys are model IDs, values contain 'name', 'checkpoint_path',
        'semantic' (bool), and optionally 'tooltip'.
        Returns an empty dict if download fails.
    """
    models_yaml_path = Path(models_yaml_path)
    sam3_path = models_yaml_path.parent
    assert sam3_path.exists(), f"Path {sam3_path} does not exist"
    assert models_yaml_path.exists(), f"Path {models_yaml_path} does not exist"

    # Load the model YAML file
    with open(models_yaml_path, 'r') as file:
        models_dict = yaml.safe_load(file)

    for model_id in models_dict:
        assert 'name' in models_dict[model_id], f"Name not found for model {model_id} in yaml file"
        assert 'checkpoint_path' in models_dict[model_id], f"Checkpoint path not found for model {model_id} in yaml file"

    # Collect unique checkpoint files that need downloading
    checkpoints_dir = sam3_path / 'checkpoints'
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
        logger.warning(f"⚠️  Could not download SAM3 files: {e}")
        logger.warning("   Make sure you have accepted the licence at "
              "https://huggingface.co/facebook/sam3 and run "
              "`huggingface-cli login`.")
        return {}

    # Verify the checkpoint actually landed
    ckpt_path = checkpoints_dir / "sam3.pt"
    if not ckpt_path.exists():
        logger.warning("⚠️  sam3.pt not found after download attempt.")
        return {}

    # Verify checkpoint paths from YAML actually exist
    for model_id, model in models_dict.items():
        ckpt_path = sam3_path / model['checkpoint_path']
        if not ckpt_path.exists():
            print(f"⚠️ {ckpt_path} not found after download attempt.")
            return {}

    return models_dict
