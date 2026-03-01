import json
from pathlib import Path
from natsort import natsorted
import zarr
import numpy as np
import pandas as pd
import warnings
# Plugins
from napari_pyav._reader import FastVideoReader
from napari.utils import DirectLabelColormap
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import remove_small_holes, binary_closing, disk
from tqdm import tqdm

class YOLO_results:
    def __init__(self, results_dir, verbose=True, **kwargs):
        """
        
        
        Parameters
        ----------
        results_dir : str or Path
            Path to the results directory. This should be the directory
            where the predictions are stored.
        verbose : bool, optional
            If True, print additional information. The default is True.
        **kwargs : dict, optional
            Additional keyword arguments. The default is None.
            - csv_header_lines : int, optional
                Number of header lines in the CSV files. The default is 7.
        
        
        """
        # Ignore specific Zarr warning about .DS_Store
        # That happens on Mac ... might exception handling here.
        warnings.filterwarnings(
            "ignore",
            message="Object at .DS_Store is not recognized as a component of a Zarr hierarchy.",
            category=UserWarning,
            module="zarr.core.group"
        )
        # Suppress vispy DeprecationWarnings (third-party, not actionable)
        warnings.filterwarnings(
            "ignore",
            category=DeprecationWarning,
            module="vispy"
        )
        
        # Process kwargs
        self.csv_header_lines = kwargs.get('csv_header_lines', 7)
        
        # Initialize some variables 
        self.verbose = verbose
        self.video, self.video_dict = None, None
        self.width, self.height, self.num_frames = None, None, None 
        self.csvs = None
        self.zarr, self.zarr_root = None, None
        self.has_masks = False
        self.classes = None  # Model class definitions {int_id: str_label}
        self.metadata = None  # Prediction metadata from JSON
        self.frame_indices = {} 
        results_dir = Path(results_dir)
        assert results_dir.exists(), f"Path {results_dir.as_posix()} does not exist"
        self.results_dir = results_dir
        
        # Load prediction metadata (JSON) if available
        self._load_metadata()
        # Find video, csv and zarr files associated with prediction output
        self.find_video()
        self.find_csv()
        self.find_zarr_root()
        # Ensure track IDs and labels are loaded
        # This creates: self.track_ids, self.labels, self.track_id_label 
        self.get_track_ids_labels(csv_header_lines=self.csv_header_lines)
        

    def _load_metadata(self):
        """
        Load prediction_metadata.json if it exists in the results directory.
        Extracts model_classes and stores the full metadata dict.
        """
        meta_path = self.results_dir / 'prediction_metadata.json'
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
                # Extract model classes {str_id: str_label} -> {int_id: str_label}
                raw_classes = self.metadata.get('model_classes', None)
                if raw_classes:
                    self.classes = {int(k): v for k, v in raw_classes.items()}
                # Extract video dimensions as fallback (overridden by video/zarr if available)
                video_info = self.metadata.get('video_info', {})
                if video_info:
                    self.num_frames = video_info.get('num_frames_original', None) # These are the original (video) num_frames
                    self.height = video_info.get('height', None)
                    self.width = video_info.get('width', None)
                if self.verbose:
                    print(f"Loaded prediction metadata from '{meta_path.name}'")
            except Exception as e:
                if self.verbose:
                    print(f"Could not load prediction metadata: {e}")
        elif self.verbose:
            print(f"No prediction_metadata.json found in '{self.results_dir.name}'")

    def find_video(self):
        """
        Check if video is present in the second parent directory, then probe it for properties.
        OCTRON saves results of analyzed mp4 files into a subdirectory /octron_predictions/VIDEONAME/
        """
        from octron.sam_octron.helpers.video_loader import probe_video
        results_dir = self.results_dir
        video = None
        video_dict = None
        for video_path in results_dir.parent.parent.glob('*.mp4'):
            if video_path.stem == '_'.join(results_dir.name.split('_')[:-1]):
                video = FastVideoReader(video_path, read_format='rgb24')
                video_dict = probe_video(video_path, verbose=self.verbose)
                self.height = video_dict['height']
                self.width = video_dict['width']
                self.num_frames = video_dict['num_frames']
                # If height, width, num_frames are not set, there is still a chance
                # to recover that info from the zarr array ...
                break
        if video is None and self.verbose:
            print(f"No video found for '{results_dir.name}'")
        self.video, self.video_dict = video, video_dict
            
    def find_csv(self):
        results_dir = self.results_dir
        csvs = natsorted(results_dir.rglob('*track_*.csv'))
        if not csvs and self.verbose:
            print(f"No tracking CSV files found in '{results_dir.name}'")
            self.csvs = None
        else:
            self.csvs = csvs
            if self.verbose:
                print(f"Found {len(csvs)} tracking CSV files in '{results_dir.name}'")
                      
    def find_zarr_root(self):
        """
        Find the zarr archive (if it exists) and open the root group.
        Detection predictions have no zarr — this is expected and not an error.
        """
        results_dir = self.results_dir
        zarrs = list(results_dir.rglob('predictions.zarr'))
        if len(zarrs) == 0:
            if self.verbose:
                print(f"No zarr archive in '{results_dir.name}' (detection predictions)")
            return
        if len(zarrs) > 1:
            raise ValueError(f"Expected at most one predictions.zarr, got {len(zarrs)}.")
        
        self.zarr = zarrs[0]
        if self.verbose:
            print(f"Found tracking zarr in '{results_dir.name}'")
        
        store = zarr.storage.LocalStore(self.zarr, read_only=False)
        root = zarr.open_group(store=store, mode='a')
        if self.verbose:
            print("Existing keys in zarr archive:", natsorted(root.array_keys()))
        self.zarr_root = root
        self.has_masks = any(k.endswith('_masks') for k in root.array_keys())
        
        # Try to get classes from zarr root attrs (legacy/segmentation)
        if self.classes is None:
            zarr_classes = root.attrs.get('classes', None)
            if zarr_classes:
                self.classes = {int(k): v for k, v in dict(zarr_classes).items()}
        
        # Extract video dimensions from zarr if not already set from video
        if (self.num_frames is None) or (self.height is None) or (self.width is None):
            example_array = next(iter(self.zarr_root.array_values()), None)
            if example_array is not None:
                self.num_frames = example_array.shape[0] if len(example_array.shape) > 0 else None
                self.height = example_array.shape[1] if len(example_array.shape) > 1 else None
                self.width = example_array.shape[2] if len(example_array.shape) > 2 else None   
                if self.verbose:
                    print(f"Extracted video dimensions from zarr: {self.num_frames} frames, {self.width}x{self.height}")
        
    def get_track_ids_labels(self, csv_header_lines=None): 
        """
        Compare track IDs that are present across csv files and the zarr archive.
        Track IDs are stored in the csvs (column "track_id"),
        and the zarr root as ".array_keys()".
        Corresponding label names are stored in the csvs (column "label").
        
        Creates
        -------
        self.track_ids : list
            Set of list of track IDs that are present in both the csvs and the zarr archive.
        self.labels : list
            Set of list of labels that are present in the csvs.
        self.track_id_label : dict
            Dictionary of track IDs and their corresponding labels.
            The keys are the track IDs and the values are the labels.
                
        """
        if csv_header_lines is not None:
            self.csv_header_lines = csv_header_lines
            
        def _track_ids_zarr():
            if self.zarr_root is not None:
                track_ids_zarr_keys = self.zarr_root.array_keys()
                # Keys are like "0_label", "1_label", etc.
                track_ids = [int(key.split('_')[0]) for key in track_ids_zarr_keys if key.split('_')[0].isdigit()]
                return sorted(list(set(track_ids)))
            else:
                if self.verbose:
                    print("Zarr root not found.")
                return []
        
        def _track_ids_labels_csv(header_lines):
            if self.csvs is not None:
                track_ids_from_csv = []
                labels_from_csv = []
                track_id_label_dict = {}
                for csv_file in self.csvs:
                    try:
                        df = pd.read_csv(csv_file, skiprows=header_lines, usecols=['track_id','label'])
                        if 'track_id' in df.columns:
                            assert set(df['track_id'].unique()) == set(df['track_id']), f"Duplicate track IDs found in {csv_file.name}"
                            track_ids_from_csv.extend(df['track_id'].unique().tolist())
                        elif self.verbose:
                            print(f"Column 'track_id' not found in {csv_file.name}")
                        if 'label' in df.columns:
                            assert set(df['label'].unique()) == set(df['label']), f"Duplicate labels found in {csv_file.name}"
                            labels_from_csv.extend(df['label'].unique().tolist())
                        elif self.verbose:
                            print(f"Column 'label' not found in {csv_file.name}")
                    except Exception as e:
                        if self.verbose:
                            print(f"Could not read track_ids and labels from {csv_file.name}: {e}")
                            
                # Write to a dictionary, sort the track_id key 
                for track_id, label in zip(track_ids_from_csv, labels_from_csv):
                    if track_id not in track_id_label_dict:
                        track_id_label_dict[track_id] = label
                    else:
                        raise ValueError(f"Duplicate track ID {track_id} found.")
                # Sort the dictionary by track_id
                track_id_label_dict = dict(sorted(track_id_label_dict.items()))
                return list(set(track_ids_from_csv)), list(set(labels_from_csv)), track_id_label_dict
            else:
                if self.verbose:
                    print("No CSV files found, cannot extract track IDs from CSVs.")
                return [], [], {}

        zarr_ids = _track_ids_zarr()
        csv_ids, csv_labels, track_id_label  = _track_ids_labels_csv(self.csv_header_lines)

        if not csv_ids:
            raise ValueError("No track IDs found in CSV files.")
        if not csv_labels:
            raise ValueError("No labels found in CSV files.")
        
        if zarr_ids:
            # Segmentation results: zarr and CSV track IDs must match
            assert set(zarr_ids) == set(csv_ids), f"Track IDs zarr and CSVs do not match: {set(zarr_ids) - set(csv_ids)}"
        elif self.verbose:
            # Detection results: no mask arrays in zarr, rely on CSVs only
            print("No mask track IDs in zarr (detection model). Using CSV track IDs only.")
        
        self.track_ids = sorted(csv_ids)
        self.labels = sorted(csv_labels)
        self.track_id_label = track_id_label
        if self.verbose:
            print(f"Found {len(self.track_id_label)} unique track IDs in zarr and CSVs: {self.track_id_label}")

    
    
    def get_track_id_for_label(self, label):
        """
        Get the track ID for a given label.
        
        Parameters
        ----------
        label : str
            The label to search for.
            
        Returns
        -------
        track_ids : list of int
            A list of track IDs associated with the given label.
            
        """
        track_ids = self._find_keys_for_value(self.track_id_label, label)
        if not track_ids:
            raise ValueError(f"Label '{label}' not found in track IDs.")
        
        return track_ids
    
    
    def get_label_for_track_id(self, track_id): 
        """
        Get the label for a given track ID.
        
        Parameters
        ----------
        track_id : int
            The track ID to search for.
            
        Returns
        -------
        label : str
            The label for the given track ID.
            
        """
        label = self.track_id_label.get(track_id, None)
        if label is None:
            raise ValueError(f"Track ID '{track_id}' not found in labels.")
        
        return label
    
    def define_colors(self, label_n=10, n_colors_submap=50):
        """
        Color handling: Recreate the colors here for the masks 
        This is a bit of a hack, but it works if the n_labels and n_colors_submap 
        match the original parameters used to create the colormap. 
        So, if we don't change these parameters, the colors will be the same, because 
        we are looking up object IDs from the original model classes
        (See self.get_color_for_track_id())
        """
        from octron.sam_octron.helpers.sam2_colors import (create_label_colors, 
                                                            sample_maximally_different
                                                           )
        all_labels_submaps = create_label_colors(n_labels=label_n,
                                                 n_colors_submap=n_colors_submap,
                                                )
        indices_max_diff_labels    = sample_maximally_different(list(range(label_n)))
        indices_max_diff_subcolors = sample_maximally_different(list(range(n_colors_submap)))
        return all_labels_submaps, indices_max_diff_labels, indices_max_diff_subcolors
    
    
    
    def get_color_for_track_id(self, track_id):
        """
        Get the color for a given track ID.
        I am using the same method here that I use to define the colors
        during annotation in OCTRON.
        This is a bit of a hack, but it works if the n_labels and n_colors_submap
        match the original parameters used to create the colormap.
        So, if we don't change these parameters, the colors will be the same, because
        we are looking up object IDs from the original model classes.
        
        Parameters
        ----------
        track_id : int
            The track ID to search for.


        Returns
        -------
        obj_color : list
            The RGBA color for the given track ID [0-1 range].
        napari_color : DirectLabelColormap
            This can be used during plotting in napari.
            
        """

        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        # Ensure track_id is an int, as it's used as a dict key
        track_id = int(track_id)
        label = self.track_id_label[track_id] # Get the label for this specific track_id

        # Get model class definitions {int_id: str_label}
        # Priority: self.classes (from metadata JSON or zarr root), then mask array attrs
        classes = self.classes
        if classes is None and self.has_masks:
            all_mask_data = self.get_mask_data()
            if track_id in all_mask_data:
                raw = all_mask_data[track_id]['data'].attrs.get('classes', None)
                if raw:
                    classes = {int(k): v for k, v in dict(raw).items()}
        if not classes:
            raise ValueError(f"Model class definitions not found for track_id {track_id}.")

        # Find the original integer class ID from the model's definition for this track's label
        original_class_id_keys = self._find_keys_for_value(classes, label)
        if not original_class_id_keys:
            raise ValueError(f"Label '{label}' not found in model class definitions: {classes}")
        if len(original_class_id_keys) > 1 and self.verbose:
            print(f"Warning: Label '{label}' maps to multiple original class IDs in model definition: {original_class_id_keys}. Using first one: {original_class_id_keys[0]}.")
        original_class_id = int(original_class_id_keys[0])
        
        # Determine the occurrence index of this track_id among all tracks sharing the same label string
        # This is used to pick a distinct sub-color.
        all_track_ids_for_this_label_str = sorted(self._find_keys_for_value(self.track_id_label, label))
        track_occurrence_index = all_track_ids_for_this_label_str.index(track_id)
        # Get the color for the label
        all_labels_submaps, indices_max_diff_labels, indices_max_diff_subcolors = self.define_colors()
        # Ensure indices are within bounds
        label_color_index = indices_max_diff_labels[original_class_id % len(indices_max_diff_labels)]
        subcolor_index = indices_max_diff_subcolors[track_occurrence_index % len(indices_max_diff_subcolors)]
        obj_color =  all_labels_submaps[label_color_index][subcolor_index]
                                          
        napari_color = DirectLabelColormap(color_dict={None: [0.,0.,0.,0.], 1: obj_color}, 
                                           use_selection=True, 
                                           selection=1,
                                          )
        return obj_color, napari_color
    
    #### CORE DATA EXTRACTION ##############################################################
    
    def get_tracking_data(self,
                          interpolate=True,
                          interpolate_method: str = 'linear',
                          interpolate_limit=None,
                          sigma=0,
                          ):
        """
        
        Get the tracking data for all csvs in a dictionary
        of track_id -> tracking columns
        
        Returns
        -------
        tracking_data : dict
            Dictionary of track_id -> tracking data
            tracking data is a dictionary with the keys:
            - 'label': The label for the track ID
            - 'data': The tracking data (pandas.DataFrame)
            - 'features': The features (pandas.DataFrame), like area, eccentricity, etc.
            
        """
        EXPECTED_CSV_COLUMNS = ["frame_idx",
                                "track_id",
                                "label",
                                "confidence", # Including this as minimum
                                "pos_x",
                                "pos_y",
        ]
        POSITION_COLUMNS = ["frame_idx",
                            "track_id",
                            "pos_x",
                            "pos_y",
                           ]
        FEATURE_COLUMNS = ["frame_idx",
                           "confidence",
                           "bbox_x_min",
                           "bbox_x_max",
                           "bbox_y_min",
                           "bbox_y_max",
                           "bbox_area",
                           "bbox_aspect_ratio",
                           "area",
                           "eccentricity",
                           "solidity",
                           "orientation",
        ]
        INTEGER_COLUMNS = ["frame_idx",
                           "track_id",
        ]
        if interpolate_limit is not None: 
            if interpolate_limit <= 0:
                interpolate_limit = None
        if sigma < 0:
            sigma = 0 
        if self.csvs is None:
            raise ValueError("No CSV files found, cannot extract tracking data.")
        
        tracking_data = {}
        for csv_file in self.csvs:
            try:
                df = pd.read_csv(csv_file, skiprows=self.csv_header_lines)
                assert set(EXPECTED_CSV_COLUMNS).issubset(df.columns), \
                    f"CSV file {csv_file.name} does not contain all expected columns: {EXPECTED_CSV_COLUMNS}"
                
                # Prune FEATURE_COLUMNS based on what is actually present in the CSV
                # This is to guard against older formats or changes in the CSV structure
                FEATURE_COLUMNS = [col for col in FEATURE_COLUMNS if col in df.columns]
                
                track_id = int(df.iloc[0].track_id) # Get the scalar track_id for this file
                label = df.iloc[0].label

                is_continuous = np.all(np.diff(df['frame_idx']) == 1)   
                df_position = df[POSITION_COLUMNS].copy() 
                df_features = df[FEATURE_COLUMNS].copy() 
                
                # Initial type conversion
                for col in POSITION_COLUMNS:
                    if col in INTEGER_COLUMNS:
                        df_position[col] = df_position[col].astype(int)
                    else:
                        df_position[col] = df_position[col].astype(float)
                for col in FEATURE_COLUMNS:
                    if col in INTEGER_COLUMNS:
                        df_features[col] = df_features[col].astype(int)
                    else:
                        df_features[col] = df_features[col].astype(float)
                
                # Interpolate?
                if interpolate and not is_continuous:
                    assert self.num_frames is not None, \
                        "Number of frames (self.num_frames) not set. Cannot interpolate."
                    if self.verbose:
                        print(f"Frames are not continuous for track_id {track_id} in {csv_file.name}... interpolating")
                    df_position_interp = pd.DataFrame({'frame_idx': range(0, self.num_frames)})
                    df_features_interp = pd.DataFrame({'frame_idx': range(0, self.num_frames)})
                    
                    # Merge with existing data to identify missing frames
                    df_position_interp =  df_position_interp.merge(df_position, on='frame_idx', how='left')
                    df_features_interp =  df_features_interp.merge(df_features, on='frame_idx', how='left') 
                    
                    # Interpolate the position columns
                    cols_to_interpolate = [col for col in POSITION_COLUMNS if col not in INTEGER_COLUMNS]
                    df_position_interp[cols_to_interpolate] = df_position_interp[cols_to_interpolate].interpolate(method=interpolate_method, 
                                                                                                                  limit=interpolate_limit, 
                                                                                                                  limit_area=None, 
                                                                                                                  )
                    # Interpolate the feature columns
                    cols_to_interpolate = [col for col in FEATURE_COLUMNS if col not in INTEGER_COLUMNS]
                    df_features_interp[cols_to_interpolate] = df_features_interp[cols_to_interpolate].interpolate(method=interpolate_method,
                                                                                                                  limit=interpolate_limit, 
                                                                                                                  limit_area=None, 
                                                                                                                  )
                    # After interpolation, get the valid frame indices from the tracking DataFrame
                    valid_frames = df_position_interp.dropna(subset=['pos_x', 'pos_y'])['frame_idx'].values
                    
                    # Use these valid frames to filter both DataFrames
                    df_position_interp = df_position_interp[df_position_interp['frame_idx'].isin(valid_frames)].reset_index(drop=True)
                    df_features_interp = df_features_interp[df_features_interp['frame_idx'].isin(valid_frames)].reset_index(drop=True)
                    assert np.array_equal(df_position_interp['frame_idx'].to_numpy(), 
                                          df_features_interp['frame_idx'].to_numpy()),\
                                          "Frame mismatch between tracking and feature dataframes after interpolation."
                    
                    df_position = df_position_interp.copy()
                    df_features = df_features_interp.copy()     
                    df_position.loc[:, 'track_id'] = track_id
                      
                df_position = df_position[['track_id', 'frame_idx', 'pos_y', 'pos_x' ]] # Just change column order here
                    
                # Smooth? 
                if sigma > 0:
                    if self.verbose:
                        print(f"Smoothing position data for track_id {track_id} with sigma={sigma}")
                    cols_to_smooth = [col for col in ['pos_x', 'pos_y'] if col in df_position.columns]
                    for col in cols_to_smooth:
                        df_position.loc[:, col] = gaussian_filter1d(df_position[col].astype(float), sigma=sigma)
                
                for col in INTEGER_COLUMNS:
                    if col in df_position.columns:
                        df_position[col] = df_position[col].astype(int)
                
                if track_id not in tracking_data:
                    tracking_data[track_id] = {
                        'label':    label,
                        'data':     df_position,
                        'features': df_features,
                    }
                else:
                    raise ValueError(f"Duplicate track ID {track_id} found.")
            except Exception as e:
                if self.verbose:
                    print(f"Could not read tracking data from {csv_file.name}: {e}")
        return tracking_data
    
    
    def get_tracking_for_label(self, 
                               label,
                               interpolate=True,
                               interpolate_method: str = 'linear',
                               interpolate_limit=None,
                               sigma=0,
                               ):
        """
        Get the tracking data (positions) for a given label.
        If the label maps to multiple track IDs, a warning is issued (if verbose=True)
        and data for the first track ID is returned.
        
        Parameters
        ----------
        label : str
            The label to search for.
        interpolate : bool, optional
            Whether to interpolate missing frames. Default is True.
        interpolate_method : str, optional
            Method for interpolation if `interpolate` is True. Default is 'linear'.
        interpolate_limit : int, optional
            Maximum number of consecutive NaNs to fill. Default is None (no limit).
        sigma : float, optional
            Sigma for Gaussian smoothing of position data. If 0, no smoothing. Default is 0.
            
        Returns
        -------
        pandas.DataFrame
            The tracking data (positions) for the (first) track ID associated with the given label.
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        list_of_track_ids = self.get_track_id_for_label(label) # Returns a list
        
        if not list_of_track_ids: # Should be caught by get_track_id_for_label, but defensive
            raise ValueError(f"Label '{label}' did not resolve to any track IDs.")

        track_id_to_use = list_of_track_ids[0]
        if len(list_of_track_ids) > 1 and self.verbose:
            print(f"Warning: Label '{label}' maps to multiple track IDs: {list_of_track_ids}. "
                  f"Using data for the first track ID: {track_id_to_use}.")
            
        all_tracking_data = self.get_tracking_data(interpolate=interpolate,
                                                   interpolate_method=interpolate_method,
                                                   interpolate_limit=interpolate_limit,
                                                   sigma=sigma,
                                                  )
        if track_id_to_use not in all_tracking_data:
            raise ValueError(f"Track ID '{track_id_to_use}' (for label '{label}') not found in processed tracking data.")

        return all_tracking_data[track_id_to_use]['data']
    
    def get_features_for_label(self, label):
        """
        Get the features for a given label.
        If the label maps to multiple track IDs, a warning is issued (if verbose=True)
        and features for the first track ID are returned.
        
        Parameters
        ----------
        label : str
            The label to search for.
            
        Returns
        -------
        pandas.DataFrame
            The features for the (first) track ID associated with the given label.
        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        list_of_track_ids = self.get_track_id_for_label(label)
        
        if not list_of_track_ids:
            raise ValueError(f"Label '{label}' did not resolve to any track IDs.")

        track_id_to_use = list_of_track_ids[0]
        if len(list_of_track_ids) > 1 and self.verbose:
            print(f"Warning: Label '{label}' maps to multiple track IDs: {list_of_track_ids}. "
                  f"Using features for the first track ID: {track_id_to_use}.")

        # Note: get_tracking_data() computes both 'data' and 'features'.
        # We call it here without interpolation/smoothing args as features are typically not interpolated/smoothed.
        all_tracking_data = self.get_tracking_data(interpolate=False, sigma=0) 
        
        if track_id_to_use not in all_tracking_data:
            raise ValueError(f"Track ID '{track_id_to_use}' (for label '{label}') not found in processed tracking data.")
        
        return all_tracking_data[track_id_to_use]['features']

    
    def get_mask_data(self, close_holes=False):
        """
        Get the mask data for all track IDs in a dictionary
        of track_id -> mask data
        
        Parameters
        ----------
        close_holes : bool, optional
            If True, close small holes in the masks using skimage.morphology.remove_small_holes.
            Default is False.
            
        Returns
        -------
        mask_data : dict
            Dictionary of track_id -> mask data
            mask_data is a dictionary with the keys:
            - 'label': The label for the track ID
            - 'data': The mask data (zarr.Array)
            - 'frame_indices': The frame indices for which data exist in this mask data array.
            
        """
        if self.zarr_root is None:
            raise ValueError("No zarr root found, cannot extract mask data.")
        mask_data = {}
        for track_id in self.track_ids:
            label = self.get_label_for_track_id(track_id)
            try:
                mask_key = f"{track_id}_masks"
                if mask_key in self.zarr_root.array_keys():
                    masks = self.zarr_root[mask_key]
                    
                    num_frames = masks.shape[0]
                    height = masks.shape[1]
                    width = masks.shape[2]
                    if self.num_frames is not None: 
                        assert num_frames == self.num_frames, \
                            f"Number of frames in mask data ({num_frames}) does not match number of frames in video ({self.num_frames})."
                    if self.height is not None:
                        assert height == self.height, \
                            f"Height in mask data ({height}) does not match height in video ({self.height})."
                    if self.width is not None:
                        assert width == self.width, \
                            f"Width in mask data ({width}) does not match width in video ({self.width})."
                    # Find out which indices have data
                    frame_indices = np.where(
                        (masks[:,0,0] != -1) # -1 indicates no data for the frame
                    )[0]
                    if len(frame_indices) == 0 and self.verbose: 
                        print(f"Warning: No valid frames found for track ID '{track_id}' (label '{label}') in mask data.")
                    if len(frame_indices) and close_holes: 
                        for f in tqdm(frame_indices, desc=f'Closing holes for id {track_id}', total=len(frame_indices)): 
                            m = masks[f,:,:]
                            d = m.sum()
                            if d > 0:
                                m = binary_closing(m.astype(bool), disk(5)) # THIS IS EXPENSIVE! WHY!
                                m = remove_small_holes(m, area_threshold=d, connectivity=1)
                                masks[f,:,:] = m
                    # Store the mask data
                    mask_data[track_id] = {
                        'label': label,
                        'data' : masks,
                        'frame_indices': frame_indices,
                    }
                    self.frame_indices[track_id] = frame_indices
                else:
                    if self.verbose:
                        print(f"Mask key '{mask_key}' not found in zarr archive.")  
            except Exception as e:
                if self.verbose:
                    print(f"Could not read mask data for track ID {track_id}: {e}")
        return mask_data
    
    def get_masks_for_label(self, label, close_holes=False):
        """
        Get the mask data for a given label.
        If the label maps to multiple track IDs, a warning is issued (if verbose=True)
        and masks for the first track ID are returned.
        
        Parameters
        ----------
        label : str
            The label to search for.
        close_holes : bool, optional
            If True, close small holes in the masks using skimage.morphology.remove_small_holes.
            Default is False.
            
        Returns
        -------
        mask_data_array : zarr.Array
            The mask data for the (first) track ID associated with the given label.
        frame_indices : numpy.ndarray
            The frame indices for which data exist in this mask data array.

        """
        if self.track_id_label is None:
            raise ValueError("No track IDs found. Please run get_track_ids_labels() first.")
        
        list_of_track_ids = self.get_track_id_for_label(label)
        if not list_of_track_ids:
            raise ValueError(f"Label '{label}' did not resolve to any track IDs.")

        track_id_to_use = list_of_track_ids[0]
        if len(list_of_track_ids) > 1 and self.verbose:
            print(f"Warning: Label '{label}' maps to multiple track IDs: {list_of_track_ids}. "
                  f"Using masks for the first track ID: {track_id_to_use}.")
            
        all_mask_data = self.get_mask_data(close_holes=close_holes)
        if track_id_to_use not in all_mask_data:
            raise ValueError(f"Track ID '{track_id_to_use}' (for label '{label}') not found in mask data.")
        
        specific_mask_data = all_mask_data[track_id_to_use]['data']
        frame_indices = all_mask_data[track_id_to_use]['frame_indices']
        return specific_mask_data, frame_indices

    
    def get_frame_indices(self):
        """
        Helper to just return the frame indices for all track IDs.
        This is coupled to .get_mask_data() and is called from there, 
        since frame indices can be different across track IDs.
        
        Returns
        -------
        frame_indices : dict
            Dictionary of track_id -> frame indices

        """
        if not self.frame_indices:
            _ = self.get_mask_data()
        return self.frame_indices
    
    
    def get_masked_video_frames(self, label, frame_indices): 
        """
        Get the masked video frames for a given label and frame indices.
        If the label maps to multiple track IDs, a warning is issued (if verbose=True)
        and masks for the first track ID are used.
        
        Parameters
        ----------
        label : str
            The label to search for.
        frame_indices : numpy.ndarray
            The frame indices to search for.
            
        Returns
        -------
        masked_frames : list or numpy.ndarray
            The masked video frames for the given label and frame indices.
            If only one frame is requested, a numpy array is returned.
            If multiple frames are requested, a list of numpy arrays is returned, one for each frame.
            
        """
        if self.video is None:
            raise ValueError("No video found, cannot extract masked video frames.")
        if isinstance(frame_indices, int):
            frame_indices = [frame_indices]
        mask_data, _ = self.get_masks_for_label(label)
        masked_frames = []
        for frame_idx in frame_indices:
            frame = self.video[frame_idx]
            mask = mask_data[frame_idx]
            if frame.ndim == 3:
                mask = mask[:,:, np.newaxis]
            masked_frame = frame.copy()
            masked_frame = np.multiply(masked_frame, mask)
            masked_frames.append(masked_frame)
        
        if len(masked_frames) == 1:
            masked_frames = masked_frames[0]
        return masked_frames
    
    
    def __repr__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"YOLO_results\n{self.results_dir}\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"YOLO_results\n{self.results_dir}"
    def __str__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"YOLO_results\n{self.results_dir}\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"YOLO_results\n{self.results_dir}"
    
    #### OTHER HELPERS #######################################################################
    
    def _find_keys_for_value(self, data_dict, value_to_find):
        """Helper to find all keys in a dictionary that map to a specific value."""
        if data_dict is None:
            return []
        return [key for key, value in data_dict.items() if value == value_to_find]
