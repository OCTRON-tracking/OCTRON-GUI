# Code for checking the availability of the SAM2 model configurations and checkpoints
import os 
from pathlib import Path
import requests
import yaml
from octron.url_check import check_url_availability


def download_sam2_checkpoint(url, 
                            fpath, 
                            overwrite=False
                            ):
    """
    Parameters
    ----------
    url : str
        URL to download the model from. 
        For example "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    fpath : str or Path
        Destination path to save the model to. For example "sam_octron/checkpoints/sam2.1_hiera_large.pt"
    overwrite : bool
        If True, overwrite the file if it already exists. 
        If False, skip the download if the file already exists. Default is False.
        
    
    """
    fpath = Path(fpath)
    output_folder = fpath.parent
    assert output_folder.is_dir(), f"Destination folder '{output_folder}' does not exist"

    if fpath.exists() and not overwrite:
        print(f"File '{fpath}' exists. Skipping download.")
        return
    else:
        print(f"Downloading model from {url}")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(fpath, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            print(f"💾 Saved SAM2 model to {fpath}")  
        else:
            pass
            
            
            
            
def check_sam2_models(SAM2p1_BASE_URL, 
                      models_yaml_path,
                      force_download = False,
                      ):
    """
    Check the availability of the SAM2 model configurations and checkpoints.
    Optionally download the files if they are not available or if force_download is set to True.
    
    
    Parameters
    ----------
    SAM2p1_BASE_URL : str
        Base URL to download the models from. 
        For example "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
        If not provided, the default URL will be used.
    models_yaml_path : str or Path
        Path to the models yaml file. 
        For example "sam_octron/sam2_models.yaml"
    force_download : bool
        If True, download the model even if it already exists. 
        Default is False.

    Returns
    -------
    models_dict : dict
        Dictionary of the models and their configurations. 
        For example:
        {
            'sam2_base_plus': {
                'name: 'SAM2 Base Plus',
                'config_path': 'configs/sam2.1_hiera_large.yaml',
                'checkpoint_path': 'checkpoints/sam2.1_hiera_large.pt'
            },
            
        ...
          
    """
    if not SAM2p1_BASE_URL:
        # Archiving the SAM2 URL here for now ...
        SAM2p1_BASE_URLs =["https://dl.fbaipublicfiles.com/segment_anything_2/092824",
                           "https://huggingface.co/lkeab/hq-sam/resolve/main"]
    else:
        SAM2p1_BASE_URLs = [SAM2p1_BASE_URL]
    
    models_yaml_path = Path(models_yaml_path)
    sam2_path = models_yaml_path.parent
    assert sam2_path.exists(), f"Path {sam2_path} does not exist"
    assert models_yaml_path.exists(), f"Path {models_yaml_path} does not exist"
    
    # Load the model YAML file and convert it to a dictionary
    with open(models_yaml_path, 'r') as file:
        models_dict = yaml.safe_load(file)
    
    for model in models_dict:
        # Perform some sanity checks on the dictionary 
        assert 'name' in models_dict[model], f"Name not found for model {model} in yaml file"
        assert 'config_path' in models_dict[model], f"Config path not found for model {model} in yaml file"
        assert 'checkpoint_path' in models_dict[model], f"Checkpoint path not found for model {model} in yaml file"

        # Some sanity checks on the actual paths
        model_config_path = (sam2_path  / models_dict[model]['config_path'])
        assert model_config_path.exists(), f"Config file {model_config_path} does not exist" 
        model_checkpt_path = (sam2_path  / models_dict[model]['checkpoint_path'])
        if not model_checkpt_path.parent.exists():
            os.mkdir(model_checkpt_path.parent)
        assert model_checkpt_path.parent.exists(), f"Checkpoint folder {model_checkpt_path.parent} does not exist"
        
        # Check if the checkpoint file exists. If not, download it.
        if model_checkpt_path.exists() and not force_download:
            print(f"Checkpoint file {model_checkpt_path} exists. Skipping download.")
        else:
            print(f'Trying to download the checkpoint file (force_download={force_download})')
            checkpoint_path = model_checkpt_path
            checkpoint_name = checkpoint_path.name
            for download_url in SAM2p1_BASE_URLs:
                try:
                    model_url = f"{download_url}/{checkpoint_name}"
                    assert check_url_availability(model_url), f"URL {model_url} is not available."
                    download_sam2_checkpoint(url=model_url, 
                                            fpath=checkpoint_path, 
                                            overwrite=True
                                            )
                except AssertionError as e:
                    #print(f"Error: {e}")
                    #print(f"Trying next URL...")
                    continue
                
    return models_dict