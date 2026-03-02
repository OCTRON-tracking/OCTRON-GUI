from pathlib import Path
import yaml

def load_boxmot_trackers(trackers_yaml_path):
    """
    Load boxmot tracker overview .yaml.
    This yaml contains info about the tracker names, and where 
    their configuration files are saved. 
    
    Parameters
    ----------
    trackers_yaml_path : str or Path
        Path to the trackers yaml file. 
        For example "tracking/boxmot_trackers.yaml"

    Returns
    -------
    trackers_dict : dict
        Dictionary of the models and their configurations. 
        For example:
        {
            'BaseTracker': {
                'name: 'base tracker',
                'config_path': 'configs/basetracker.yaml',
            },
            
        ...
          
    """
    trackers_yaml_path = Path(trackers_yaml_path)
    assert trackers_yaml_path.exists(), f"Path {trackers_yaml_path} does not exist"
    
    # Load the model YAML file and convert it to a dictionary
    with open(trackers_yaml_path, 'r') as file:
        trackers_dict = yaml.safe_load(file)
    return trackers_dict


def load_boxmot_tracker_config(config_yaml_path):
    
    """
    Load OCTRON boxmot tracker configuration file.
    
    Parameters
    ----------
    config_yaml_path : str or Path
        Path to the tracker config yaml file. 
        For example "configs/bytetrack.yaml"
        (This is in octron/tracking/)

    Returns
    -------
    config_dict : dict
        Config dictionary

    """
    config_yaml_path = Path(config_yaml_path)
    assert config_yaml_path.exists(), f"Path {config_yaml_path} does not exist"
    
    # Load the model YAML file and convert it to a dictionary
    with open(config_yaml_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return config_dict


def list_available_trackers(trackers_dict):
    """
    List all available trackers from a loaded boxmot_trackers dictionary.
    
    Parameters
    ----------
    trackers_dict : dict
        Dictionary loaded from boxmot_trackers.yaml via load_boxmot_trackers()

    Returns
    -------
    available : dict
        Dictionary of {tracker_id: display_name} for all trackers with available: true
        For example: {'ByteTrack': 'ByteTrack', 'OcSort': 'OcSort', ...}
    """
    available = {}
    for tracker_id, info in trackers_dict.items():
        if info.get('available', False):
            available[tracker_id] = info.get('name', tracker_id)
    return available


def resolve_tracker(tracker_name, trackers_dict):
    """
    Resolve a user-provided tracker name to its tracker ID and info dictionary.
    
    This performs a flexible lookup in the following order:
    1. Exact key match (e.g. "ByteTrack")
    2. Case-insensitive key match (e.g. "bytetrack" -> "ByteTrack")
    3. Match on the 'name' field (e.g. "D-OcSort")
    
    Only trackers with ``available: true`` are considered.
    
    Parameters
    ----------
    tracker_name : str
        Name of the tracker to resolve. Can be the YAML key, or the display 
        name, case-insensitive.
    trackers_dict : dict
        Dictionary loaded from boxmot_trackers.yaml via load_boxmot_trackers()

    Returns
    -------
    tracker_id : str
        The canonical key in boxmot_trackers.yaml (e.g. "ByteTrack")
    tracker_info : dict
        The info dictionary for that tracker (contains 'name', 'config_path', etc.)

    Raises
    ------
    ValueError
        If the tracker cannot be resolved. The error message lists all 
        available trackers.
    """
    name = tracker_name.strip()
    
    # Build lookup of available trackers only
    available = {tid: info for tid, info in trackers_dict.items()
                 if info.get('available', False)}
    
    if not available:
        raise ValueError("No trackers are marked as available in the tracker configuration.")
    
    # 1. Exact key match
    if name in available:
        return name, available[name]
    
    # 2. Case-insensitive key match
    name_lower = name.lower()
    for tid, info in available.items():
        if tid.lower() == name_lower:
            return tid, info
    
    # 3. Match on the 'name' field (case-insensitive)
    for tid, info in available.items():
        if info.get('name', '').lower() == name_lower:
            return tid, info
    
    # Not found — build a helpful error message
    available_list = "\n".join(
        f"  - {tid}  ({info.get('name', tid)})" for tid, info in available.items()
    )
    raise ValueError(
        f"Tracker '{tracker_name}' not found. Available trackers:\n{available_list}"
    )
