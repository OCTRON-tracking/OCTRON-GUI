# Code for checking the availability of the YOLO models
import os 
from pathlib import Path
import requests
import yaml
from octron.url_check import check_url_availability


def download_yolo_model(url, 
                        fpath, 
                        overwrite=False
                        ):
    """
    Parameters
    ----------
    url : str
        URL to download the model from. 
        For example "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt"
    fpath : str or Path
        Destination path to save the model to. For example "yolo_octron/models/yolo11l-seg.pt"
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
            print(f"💾 Saved YOLO model to {fpath}")  
        else:
            pass
            
               
            
def check_yolo_models(YOLO_BASE_URL, 
                      models_yaml_path,
                      force_download = False,
                      ):
    """
    Check the availability of the YOLO model.
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
        For example "yolo_octron/models.yaml"    
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
        YOLO_BASE_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0" 
    
    assert models_yaml_path.exists(), f"Path {models_yaml_path} does not exist"
    yolo_model_path = models_yaml_path.parent / 'models' # OCTRON convention. Currently not changeable.
    if yolo_model_path.exists():
        print(f"Models folder {yolo_model_path} exists.")
    else:
        os.mkdir(yolo_model_path)
        print(f"Created YOLO models folder {yolo_model_path}")
    
    # Load the model YAML file and convert it to a dictionary
    with open(models_yaml_path, 'r') as file:
        models_dict = yaml.safe_load(file)
    
    for model in models_dict:
        # Perform some sanity checks on the dictionary 
        assert 'name' in models_dict[model], f"Name not found for model {model} in yaml file"
        assert 'model_path_seg' in models_dict[model], f"Segmentation model path not found for model {model} in yaml file"
        assert 'model_path_detect' in models_dict[model], f"Detection model path not found for model {model} in yaml file"

        # Download both segmentation and detection model variants
        for path_key in ('model_path_seg', 'model_path_detect'):
            model_path = (yolo_model_path / models_dict[model][path_key])
            # Check if the model file exists. If not, download it.
            if model_path.exists() and not force_download:
                print(f"Model file {model_path} exists. Skipping download.")
            else:
                print(f'Trying to download {path_key} for {model} (force_download={force_download})')
                model_name = model_path.name
                model_url = f"{YOLO_BASE_URL}/{model_name}"
                assert check_url_availability(model_url), f"URL {model_url} is not available."
                download_yolo_model(url=model_url, 
                                   fpath=model_path, 
                                   overwrite=True
                                 )
     
    return models_dict
