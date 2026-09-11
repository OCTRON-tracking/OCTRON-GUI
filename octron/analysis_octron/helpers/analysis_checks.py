"""Code for checking the availability of the base analysis models."""

from pathlib import Path

import requests
import yaml
from loguru import logger

from octron.url_check import check_url_availability


def download_analysis_model(url, fpath, overwrite=False):
    """Download the model file from a URL.

    Parameters
    ----------
    url : str
        URL to download the model from.
        For example "https://github.com/ultralytics/assets/releases/
        download/v8.3.0/yolo11l-seg.pt"
    fpath : str or Path
        Destination path to save the model to. For example
        "analysis_octron/models/yolo11l-seg.pt"
    overwrite : bool
        If True, overwrite the file if it already exists.
        If False, skip the download if the file already exists.
        Default is False.

    """
    fpath = Path(fpath)
    output_folder = fpath.parent
    assert output_folder.is_dir(), (
        f"Destination folder '{output_folder}' does not exist"
    )

    if fpath.exists() and not overwrite:
        logger.info(f"File '{fpath}' exists. Skipping download.")
        return
    else:
        logger.info(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(fpath, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"Saved model to {fpath}")
        else:
            pass


def check_analysis_models(
    YOLO_BASE_URL,
    models_yaml_path,
    force_download=False,
):
    """Check the availability of the base analysis model(s).

    Optionally download the model file if they are not available
    or if force_download is set to True.

    Parameters
    ----------
    YOLO_BASE_URL : str
        Base URL to download the models from.
        For example "https://github.com/ultralytics/assets/releases/download/v8.3.0"
        If not provided, the default URL is used.
    models_yaml_path : str or Path
        Path to the YAML file containing the model information.
        For example "analysis_octron/models.yaml"
    force_download : bool
        If True, download the model even if it already exists.
        Default is False.

    Returns
    -------
    models_dict : dict
        Dictionary of the models and their paths.
        For example:
        {
            'YOLO11m': {
                'name': 'YOLO11m',
                'model_path_seg': 'yolo11m-seg.pt',
                'model_path_detect': 'yolo11m.pt',
            },

        ...

    """
    if not YOLO_BASE_URL:
        # Archiving the YOLO github releases URL here for now ...
        YOLO_BASE_URL = (
            "https://github.com/ultralytics/assets/releases/download/v8.4.0"
        )

    models_yaml_path = Path(models_yaml_path)
    assert models_yaml_path.exists(), f"Path {models_yaml_path} does not exist"
    # Downloaded weights live in the per-user cache
    # (config.get_analysis_models_dir()), NOT inside the installed package
    # (which may be read-only and is wiped on reinstall). The manifest
    # YAML stays bundled with the package.
    from octron import config

    analysis_model_path = config.get_analysis_models_dir()

    # Load the model YAML file and convert it to a dictionary
    with open(models_yaml_path) as file:
        models_dict = yaml.safe_load(file)

    for model in models_dict:
        # Perform some sanity checks on the dictionary
        assert "name" in models_dict[model], (
            f"Name not found for model {model} in yaml file"
        )

        # A model must support at least one task. Capability is derived
        # from which variants are non-empty: a model may ship only a
        # segmentation variant, only a detection variant (e.g. RT-DETR),
        # or both. Empty/missing variants mark an unsupported task and
        # are skipped rather than downloaded.
        variants = {
            path_key: models_dict[model].get(path_key)
            for path_key in ("model_path_seg", "model_path_detect")
        }
        if not any(variants.values()):
            raise AssertionError(
                f"Model {model} has neither a segmentation nor a "
                f"detection variant in the yaml file"
            )

        # Download each available (non-empty) model variant.
        for path_key, variant in variants.items():
            if not variant:
                # Unsupported task for this model — nothing to download.
                continue
            model_path = analysis_model_path / variant
            # Check if the model file exists. If not, download it.
            if model_path.exists() and not force_download:
                logger.info(
                    f"Model file {model_path} exists. Skipping download."
                )
            else:
                logger.info(
                    f"Trying to download {path_key} for {model} "
                    f"(force_download={force_download})"
                )
                model_name = model_path.name
                model_url = f"{YOLO_BASE_URL}/{model_name}"
                assert check_url_availability(model_url), (
                    f"URL {model_url} is not available."
                )
                download_analysis_model(
                    url=model_url, fpath=model_path, overwrite=True
                )

    return models_dict
