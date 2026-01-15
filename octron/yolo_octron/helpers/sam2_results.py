from pathlib import Path
from natsort import natsorted
import zarr
import numpy as np
import pandas as pd
import warnings
# Plugins
from scipy.ndimage import gaussian_filter1d
from skimage.morphology import remove_small_holes, binary_closing, disk
from tqdm import tqdm
from skimage.measure import regionprops

class ANNOT_results:
    def __init__(self, annotation_dir, verbose=True, **kwargs):
        """
        
        
        Parameters
        ----------
        annotation_dir : str or Path
            Path to the annotation directory.
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
        warnings.filterwarnings(
            "ignore",
            message=r"Object at (\.DS_Store|desktop\.ini) is not recognized as a component of a Zarr hierarchy.",
            category=UserWarning,
            module="zarr.core.group"
        )
        
        self.annotation_dir = annotation_dir
        self.zarr_archives = []
        self.zarr_dict = {}
        self.tracking_dict = {}
        self.frame_indices_dict = {}
        # Initialize some variables 
        self.verbose = verbose
        self.width, self.height, self.num_frames = None, None, None 
       
        self.get_annotation_folder_info(self.annotation_dir)
        self.create_zarr_dict()

   
    def get_annotation_folder_info(self, annotation_folder):
        '''
        
        '''
        annotation_folder = Path(annotation_folder)
        assert annotation_folder.exists(), 'Annotation folder path does not exist'
        if (annotation_folder / 'video_info.txt').exists():
            info_file_path = annotation_folder / 'video_info.txt'
            creation_date = None
            video_path = None
            with open(info_file_path, 'r') as f:
                for line in f:
                    if line.startswith('Info file created on:'):
                        creation_date = line.split(':', 1)[1].strip()
                    elif line.startswith('Video path:'):
                        video_path = Path(line.split(':', 1)[1].strip())
            if creation_date:
                print(f"Annotations first started on: {creation_date}")
            if video_path:
                print(f"Video file name: {video_path.name}")
   
   
    def find_zarrs(self):
        self.zarr_archives = natsorted(Path(self.annotation_dir).rglob('[!video]*.zarr'))
        if self.verbose:
            print(f'Found {len(self.zarr_archives)} zarr archives\n{self.zarr_archives}')
               
    def create_zarr_dict(self):
        """
        Finds Zarr archives, populates a dictionary mapping labels to mask arrays,
        and extracts video dimensions and non-empty frame indices.
        
        This populates 
        - self.zarr_dict
        - self.frame_indices_dict (non empty frame indices)
        """
        self.find_zarrs()
        
        for z in self.zarr_archives:
            label_name = '_'.join(z.stem.split(' ')[:-1])
            store = zarr.storage.LocalStore(z, read_only=True)
            root = zarr.open_group(store=store, mode='r')
            zarr_root = root
            self.zarr_dict[label_name] = zarr_root['masks']
            # Check num_frames, height, width by loading a 
            # example array from zarr to extract these dimensions.
            if self.num_frames or self.height or self.width is None:
                example_array = next(iter(zarr_root.array_values()), None)
                assert example_array is not None, "No arrays found in zarr root."
                self.num_frames = example_array.shape[0] if len(example_array.shape) > 0 else None
                self.height = example_array.shape[1] if len(example_array.shape) > 1 else None
                self.width = example_array.shape[2] if len(example_array.shape) > 2 else None   
                if self.verbose:
                    print(f"Extracted video dimensions from zarr: {self.num_frames} frames, {self.width}x{self.height}")
            # Check which indices are empty 
            self.frame_indices_dict[label_name] = np.where(self.zarr_dict[label_name][:,0,0] != -1)[0]
    
    
    def create_tracking_dict(self):
        """
        Loop over the zarr dictionary and all masks inside,
        and extract the centroid of each mask. 
        This throws an error if more than one region is detected per frame 


        """
        
        for label, masks in self.zarr_dict.items():
            self.tracking_dict[label] = {}
            if self.verbose:
                print(f"Calculating centroids for {label}...")
                
            df_list = []
            for frame_idx, m in tqdm(enumerate(masks), disable=not self.verbose, total=masks.shape[0]): 
                if np.any(m):
                    props = regionprops(m.astype(int))
                    if len(props) > 1:
                        # For now, taking the largest
                        print(f"Warning: Multiple regions found for {label} at frame idx {frame_idx}. Using the largest one.")
                        largest_region = max(props, key=lambda x: x.area)
                        centroid = largest_region.centroid
                    elif len(props) == 1:
                        centroid = props[0].centroid
                    else: # Should not happen if np.any(m) is true, but for safety
                        centroid = (np.nan, np.nan)
                    df_list.append({'frame': frame_idx, 'y': centroid[0], 'x': centroid[1]})
                else:
                    df_list.append({'frame': frame_idx, 'y': np.nan, 'x': np.nan})
            
            df = pd.DataFrame(df_list)
            df['x'] = df['x'].astype(float)
            df['y'] = df['y'].astype(float)
            self.tracking_dict[label] = df
        
    
    def __repr__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"Annotations\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"Annotations\n"
    def __str__(self) -> str:
        if self.num_frames is not None and self.width is not None and self.height is not None:
            return f"Annotations\n{self.num_frames} frames, {self.width}x{self.height}"
        else:
            return f"Annotations\n"
