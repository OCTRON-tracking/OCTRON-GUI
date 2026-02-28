# YOLO training related helpers
from pathlib import Path
import json
import numpy as np                                         
                
def find_files_with_depth_limit(base_path, pattern, max_depth=1):
    """
    Find files matching a pattern with a maximum depth limit.
    
    Parameters
    ----------
    base_path : Path
        Base directory to start the search
    pattern : str
        File pattern to match (e.g., '*.json')
    max_depth : int
        Maximum directory depth to search (0 = base_path only)
        
    Returns
    -------
    list
        List of Path objects matching the pattern within the depth limit
    """
    results = []
    
    # Process base directory (depth 0)
    for path in base_path.glob(pattern):
        if path.is_file():
            results.append(path)
    
    # Process subdirectories up to max_depth
    if max_depth > 0:
        for path in base_path.iterdir():
            if path.is_dir():
                # Recursively search subdirectories with reduced depth
                results.extend(find_files_with_depth_limit(path, pattern, max_depth - 1))
    
    return results
                                 
def load_object_organizer(file_path):
    """
    Load object organizer .json from disk and return
    its content as dictionary.
    The .json file itself has been created via the save_to_disk method in 
    the object_organizer class (octron.sam_octron.object_organizer.py).
    
    Parameters
    ----------
    file_path : str or Path
        Path to the .json file.
    
    Returns
    -------
    dict
        Dictionary containing all object organizer data.
    
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"No organizer file found at {file_path}")
        return
    if not file_path.suffix == '.json': 
        print(f"❌ File is not a json file: {file_path}")
        return
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"📖 Octron object organizer loaded from {file_path.as_posix()}")
        return data
    except Exception as e:
        print(f"❌ Error loading json: {e}")
        return 


def find_common_frames(frame_arrays):
    """
    Find frame indices that are present in all provided arrays.
    
    Parameters:
    -----------
    frame_arrays : list of numpy.ndarrays
        Numpy arrays containing frame indices
        
    Returns:
    --------
    numpy.ndarray
        Array containing only the frame indices present in all input arrays
    """
    if not frame_arrays:
        return np.array([], dtype=int)
    
    if len(frame_arrays) == 1:
        return frame_arrays[0]
    
    # Start with the first array
    common = frame_arrays[0]
    
    # Sequentially intersect with each additional array
    for frames in frame_arrays[1:]:
        common = np.intersect1d(common, frames)
        # Early exit if no common frames are found
        if len(common) == 0:
            break    
    return common

def pick_random_frames(frames, n=20):
    """
    Pick n random frames from a frames array without replacement
    
    Parameters
    ----------
    frames : numpy.ndarray
        Array of frame indices
    n : int
        Number of frames to pick
    
    Returns
    -------
    numpy.ndarray
        Array of randomly selected frame indices
    """
    # Determine the number of frames to pick (min of n and array length)
    num_to_pick = min(n, len(frames))
    
    # Pick random frames without replacement
    if num_to_pick > 0:
        random_frames = np.random.choice(frames, size=num_to_pick, replace=False)
        # Sort the frames to maintain chronological order if needed
        random_frames.sort()
        return random_frames
    else:
        return np.array([], dtype=frames.dtype)
    
def collect_labels(project_path, 
                   subfolder=None,
                   prune_empty_labels=True, 
                   min_num_frames=5,
                   verbose=False,
                   ):
    """
    Extract info from project path.
    Find all the object organizer json files and load them.
    The object organizer json files contain the information about the annotations 
    (the zarr arrays) as well as the associated video files.
    Both object organizer as well as zarr arrays (as attribute) contain the video hash.
    This hash is used to check if the correct video file is associated with the zarr array.

    Sanity checks:
    1. Data shape info in the zarr array and 
       the object organizer json file match the actual video file shape 
    2. The video hash in the zarr array and 
       the object organizer json file match the actual video file hash
    3. The label id to label name association is congruent across all entries


    Parameters
    ----------
    project_path : str or Path : Path to the project root directory.
        The jsons are saved in subfolders.
    prune_empty_labels : bool : Whether to prune frames that 
                                do not have annotation across all labels.
    min_num_frames : int : Minimum number of frames required 
                           for training data generation.
    verbose : bool : Whether to print debug info.
    
    Returns
    -------
    label_dict : dict : Dictionary containing project subfolders,
                        and their corresponding labels, 
                        annotated frames, masks and video data.
            keys: project_subfolder
            values: dict
                keys: label_id, video
                values: dict, video
                    dict:
                        keys: label, frames, masks, color
                        values: label (str), # Name of the label corresponding to current ID
                                frames (np.array), # Annotated frame indices for the label
                                masks (list of zarr arrays), # Mask zarr arrays
                                color (list) # Color of the label (RGBA, [0,1])
                    video: FastVideoReader object   

    """
    # Hiding some imports here to reduce initial loading time
    from napari_pyav._reader import FastVideoReader
    from octron.sam_octron.helpers.video_loader import get_vfile_hash
    from octron.sam_octron.helpers.sam2_zarr import load_image_zarr   

    project_path = Path(project_path)
    assert project_path.exists(), f'Project path not found at {project_path.as_posix()}'
    assert project_path.is_dir(), f'Project path should be a directory, not file'

    # Check whether .json files should be found in only a subfolder
    if subfolder is not None:
        json_parent_path = project_path / subfolder
        assert project_path.exists(), f'Subfolder not found at {project_path.as_posix()}'
        assert project_path.is_dir(), f'Subfolder should be a directory, not file'
    else:
        # If no subfolder is provided, search in the project root directory
        json_parent_path = project_path

    label_dict = {}
    # Create a (new) global mapping of label names to IDs for consistency across directories
    # This is important since the user might decide to add multiple labels with the same name
    # but in different orders across video projects. I.e. a label "worm" might have ID 0 in one
    # project and ID 1 in another. We want to make sure that the label ID to label name association
    # is consistent across all projects.
    label_id_map = {}
    current_label_id = 0
    
    for object_organizer in find_files_with_depth_limit(json_parent_path, 'object_organizer.json', 1):
        if verbose: print(object_organizer.parent)
        organizer_dict = load_object_organizer(object_organizer)  
        labels = {}
        video_hash_dict = {}   
        
        for entry in organizer_dict['entries'].values():
            original_label_id = int(entry['label_id'] )
            label    = entry['label'] 
            color    = entry['color']
            
            # Check if label already exists in labels
            if label in label_id_map:
                # Use existing ID for consistency
                label_id = label_id_map[label]
                if verbose: print(f'Using existing ID {label_id} for label {label}')
            else:
                # Assign a new ID and update mapping
                label_id = current_label_id
                label_id_map[label] = label_id
                current_label_id += 1
                if verbose: print(f'Created new ID {label_id} for label {label}')
            
            if verbose: print(f'Found label {label} with id {label_id}')   
            if label_id in labels:
                assert labels[label_id]['label'] == label, 'Label name vs. id do not match'
            else:
                # First time we see this label
                labels[label_id] = {'label':  label, 
                                    'original_id': original_label_id,
                                    'frames': [],
                                    'masks':  [], 
                                    'color': color,
                                    } 
            
            # Find out which frames were annotated
            zarr_path_relative = Path(entry['prediction_layer_metadata']['zarr_path'])
            zarr_path = project_path / zarr_path_relative
            assert zarr_path.exists(), f'Zarr file not found at {zarr_path}'
            num_frames, image_height, image_width = entry['prediction_layer_metadata']['data_shape']
            # Feed the expected shape to the loader.
            loaded_masks, status = load_image_zarr(zarr_path, 
                                        num_frames,
                                        image_height, 
                                        image_width, 
                                        num_ch=None,
                                        verbose=False,
                                        ) # Not doing hash comparison here! 
            assert status == True
            assert loaded_masks is not None
            # Do some sanity checks
            assert num_frames   == loaded_masks.shape[0]
            assert image_height == loaded_masks.shape[1]
            assert image_width  == loaded_masks.shape[2]
            labels[label_id]['masks'].append(loaded_masks) # This is the zarr array
            # Extract annotated frame indices
            # The fill value of the zarr array is -1, so we can use this to find annotated frames
            annotated_indices = np.where(loaded_masks[:,0,0] >= 0)[0]
            if verbose:
                print(f'Found {len(annotated_indices)} annotated frames for label {label} in {object_organizer.parent.name}')
            # if prune_empty_labels:
                
            #     # Also get rid of frames where the mask is all zeros
            #     # Why? 
            #     # Because frames that are not annotated and accidentally skipped contribute to 
            #     # "background" masks in YOLO. This will just spoil the actual training success.
            #     summed = np.sum(loaded_masks, axis=(1,2)) # TODO: This is a heavy computation!!
            #     annotated_indices = np.intersect1d(annotated_indices, np.where(summed > 0)[0])
            #     if verbose: 
            #         print(f'PRUNING: {len(annotated_indices)} remain after removing empty frames')
            labels[label_id]['frames'].extend(annotated_indices) 
            
            expected_video_hash_zarr = loaded_masks.attrs.get('video_hash', None)
            expected_video_hash_organizer = entry['prediction_layer_metadata']['video_hash']    
            
            video_file_path = project_path / Path(entry['prediction_layer_metadata']['video_file_path'])
            if not video_file_path in video_hash_dict:
                assert video_file_path.exists(), f'Video file not found at "{video_file_path.as_posix()}"' 
                actual_video_hash = get_vfile_hash(video_file_path)[-8:] # By default this is shortened to 8 characters
                video_hash_dict[video_file_path] = actual_video_hash 
            assert len(video_hash_dict) == 1, 'Different video files found for one object organizer json.'
            assert expected_video_hash_zarr == expected_video_hash_organizer == video_hash_dict[video_file_path], 'Video hash mismatch'
            
        # Maintain only unique entries in 'frames' lists
        for label_id in labels:
            _, i = np.unique(labels[label_id]['frames'], return_index=True)
            labels[label_id]['frames'] = np.array(labels[label_id]['frames'])[np.sort(i)]
            if verbose: print(f'Label {labels[label_id]["label"]} has {len(labels[label_id]["frames"])} annotated frames')
        
        # Prune frames that do not have annotation across all labels
        if prune_empty_labels:
            common_frames = find_common_frames([f['frames'] for f in labels.values()])
            for label_id in labels:
                labels[label_id]['frames'] = common_frames
                if verbose: 
                    print(f'PRUNING: Label {labels[label_id]["label"]} has {len(labels[label_id]["frames"])} common frames') 
        
        # Assert that there is a minimum number of frames available for training data generation
        if min_num_frames > 0: 
            for label_id in labels:
                no_frames_label = len(labels[label_id]['frames'])    
                label = labels[label_id]["label"]
                path = object_organizer.parent.as_posix()
                assert no_frames_label >= min_num_frames,\
                    f'Not enough frames for label "{label}" in "{path}": {no_frames_label} < {min_num_frames}'
        
        # Add the video file path and data to the dictionary      
        # video_file_path is generated anew for every object, however, 
        # we are making sure above that all videos are the same.
        video = FastVideoReader(video_file_path)
        labels['video_file_path'] = video_file_path
        labels['video'] = video
        label_dict[object_organizer.parent.as_posix()] = labels
    
    # Assert that label_id to label associations are congruent across label_dict
    # i.e. the numerical label_id is always associated with the same label name across all entries
    label_ids = []
    label_idnames = []    
    for labels in label_dict.values():
        for label_id in labels:
            if label_id == 'video' or label_id == 'video_file_path':
                continue
            label_ids.append(label_id)
            label_idnames.append(f'{label_id}-{labels[label_id]["label"]}')    
    assert len(set(label_ids)) == len(set(label_idnames)), \
        'A label id to label name association is not congruent across label_dict'
    
    return label_dict   



def draw_polygons(labels, 
                  video_data,
                  max_to_plot=2,
                  randomize=False,
                  ):   
    """
    Helper.
    Draw the polygons on the video frames, after having created the labels
    dictionary via the collect_labels() function.

    Parameters
    ----------
    labels : dict : Dictionary containing labels and their corresponding frames
            keys: label_id
            values: dict
                keys: label, frames, masks, polygons, color
                values: label (str), # Name of the label
                         frames (np.array), # Annotated frame indices for the label
                         masks (list of zarr arrays), # Masks for each frame
                         polygons (dict) # Polygons for each frame index
                         color (list) # Color of the label (RGBA, [0,1])   
    video_data : np.array : Video data array
    max_to_plot : int : Maximum number of frames to plot per label
    randomize : bool : Whether to plot random frames
    
    
    """
    # Check if cv2 is installed correctly
    try:
        import cv2 
    except ModuleNotFoundError:
        print('Please install cv2 first, via pip install opencv-python')
        return
    # ... and matplotlib
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        print('Please install matplotlib first, via pip install matplotlib')
        return

    if max_to_plot < 1:
        max_to_plot = 1
    print(f'Drawing polygons for {len(labels)} labels.')
    print(f'Max {max_to_plot} frame(s) per label will be plotted.')
    # Draw the polygons on the video frames
    for entry in labels:
        if entry == 'video' or entry == 'video_file_path':
            continue
        
        label = labels[entry]['label']
        frames = labels[entry]['frames']
        if randomize:
            frames = pick_random_frames(frames, n=max_to_plot)
        #color = np.array(labels[entry]['color'])[:-1]*255
        counter = 1
        for frame in frames:
            
            current_frame = video_data[frame].copy()
            polys = labels[entry]['polygons'][frame]  
            
            # # cv2.polylines() is fine but introduces some nasty artefacts in 
            # # cases where the polygons are not closed.
            # frame_and_polys = cv2.polylines(img=current_frame, 
            #                                 pts=polys, 
            #                                 isClosed=True, 
            #                                 color=color.tolist(), 
            #                                 thickness=5,
            #                                 )
            
            # Draw
            _, ax = plt.subplots(1,1)    
            ax.imshow(current_frame)
            
            # Draw polys as dots
            for no_p, p in enumerate(polys):
                ax.scatter(p[:,0], p[:,1], c='w', s=2, alpha=.5, marker='s')
                ax.scatter(p[:,0], p[:,1], c='k', s=.5, alpha=.5, marker='.')
                ax.plot(p[:,0], p[:,1], c='w')
                center_coord = p.mean(axis=0)
                ax.text(center_coord[0], center_coord[1], str(no_p), 
                        color='w',  # Text color
                        fontsize=10,
                        bbox=dict(
                            facecolor='black',  # Background color
                            alpha=0.7,          # Transparency
                            edgecolor='none',   # No edge color
                            boxstyle='round,pad=0.3'  # Rounded corners with padding
                        ),
                        ha='center',  # Horizontal alignment
                        va='center'   # Vertical alignment
                    )
    
            ax.set_xticks([])   
            ax.set_yticks([])
            ax.set_title(f'Label: "{label}" - frame {frame} {len(polys)}')
            plt.show()   
            if counter >= max_to_plot:
                break   
            counter += 1
            
            
def train_test_val(frame_indices,
                   training_fraction=0.8,
                   validation_fraction=0.1,
                   random_seed=88,
                   verbose=False,
                   ):
    """
    Perform a train-test-validation split on frame indices.
    
    Parameters
    ----------
    frame_indices : np.array : Array of frame indices
    training_fraction : float : Fraction of training data
    validation_fraction : float : Fraction of validation data
    random_seed : int : Random seed for reproducibility
    verbose : bool : Whether to print sizes of the splits
    
    Returns
    -------
    split_dict : dict : Dictionary containing the splits
        keys: 'train', 'val', 'test'
        values: np.array : Frame indices for each split
    
    """
    assert training_fraction + validation_fraction < 1, 'Fractions should sum to less than 1'
    assert training_fraction > validation_fraction, 'Training fraction should be greater than validation fraction'
    assert len(frame_indices) > 0, 'No frame indices provided'
    
    # Shuffle the indices
    np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(frame_indices))

    # Calculate split points
    train_size = int(training_fraction * len(frame_indices))
    val_size = int(validation_fraction * len(frame_indices))
    # test_size = len(frames) - train_size - val_size (remaining frames)

    # Split the shuffled indices
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    # Get the actual frame numbers for each split
    train_frames = frame_indices[train_indices]
    val_frames = frame_indices[val_indices]
    test_frames = frame_indices[test_indices]

    if verbose:
        # Print sizes
        print(f"Total frames: {len(frame_indices)}")
        print(f"Training set: {len(train_frames)} frames")
        print(f"Validation set: {len(val_frames)} frames")
        print(f"Test set: {len(test_frames)} frames")
        
    split_dict = {
        'train': train_frames,
        'val': val_frames,
        'test': test_frames
    }

    # Sanity check 
    collected_frames = []
    for frames in split_dict.values():
        assert len(frames) > 0, 'Empty split found'
        collected_frames.extend(frames)
    assert len(collected_frames) == len(frame_indices), 'Not all frames were collected in the splits'
    assert set(collected_frames) == set(frame_indices), 'Some frames are missing in the splits'

    return split_dict

