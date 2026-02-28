import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
import torch
from torchvision.transforms import Resize

import warnings 
warnings.simplefilter("ignore")
from ultralytics.data.loaders import SourceTypes


MIN_ZARR_CHUNK_SIZE = 50 # Setting minimum chunk size for zarr arrays
                         # to avoid excessive chunking for small arrays


def create_image_zarr(zarr_path, 
                      num_frames, 
                      image_height,
                      image_width=None,
                      chunk_size=20,
                      fill_value=np.nan,
                      dtype='float16',
                      num_ch=None,
                      video_hash_abbrev=None,
                      verbose=False,
                      ):
    """
    Creates a zarr archive for storing and retrieving image data.
    Depending on the number of channels, the zarr array will have shape
    (num_frames, num_ch, image_height, image_width) or
    (num_frames, image_height, image_width).
    
    Parameters
    ---------
    zarr_path : pathlib.Path
        Path to the zarr archive. Must end in .zarr
    num_frames : int
        Number of frames in the video
    image_height : int
        Height of the resized image (as in SAM2 input)
    image_width : int
        Width of the resized image (as in SAM2 input)
        If None, then image_width = image_height
    chunk_size : int, optional
        Size of the chunk to store in the zarr archive. 
    fill_value : int, optional
        Value to fill the zarr array with.
    dtype : str, optional
        Data type of the zarr array.
    num_ch : int, optional
        Number of channels in the image. Default is None.
        If None, then the channel dimension will not be included.
    video_hash_abbrev : str, optional
        Abbreviated hash of the video file. This is used as 
        a unique identifier for the corresponding video file throughout.
    verbose : bool, optional
        If True, print the zarr store info.
        
        
    Returns
    -------
    image_zarr : zarr.core.Array
        The zarr array to store the image data in 
        
    """
    assert image_height > 0, f'image_height must be > 0, not {image_height}'
    if image_width is None:
        image_width = image_height
    else:
        assert image_width > 0, f'image_width must be > 0, not {image_width}'
    assert isinstance(zarr_path, Path), f'zarr_path must be a pathlib.Path object, not {type(zarr_path)}'
    assert zarr_path.suffix == '.zarr', f'zarr_path must end in .zarr, not {zarr_path.suffix}'  
    
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    # Assuming local store on fast SSD, so no compression employed for now 
    store = zarr.storage.LocalStore(zarr_path, read_only=False)  
  
    chunk_size = max(chunk_size, MIN_ZARR_CHUNK_SIZE)
    if num_ch is not None: 
        image_zarr = zarr.create_array(store=store,
                                    name='masks',
                                    shape=(num_frames, num_ch, image_height, image_width),  
                                    chunks=(chunk_size, num_ch, image_height, image_width), 
                                    fill_value=fill_value,
                                    dtype=dtype,
                                    overwrite=True,
                                    )
    else:
        image_zarr = zarr.create_array(store=store,
                                    name='masks',
                                    shape=(num_frames, image_height, image_width),  
                                    chunks=(chunk_size, image_height, image_width), 
                                    fill_value=fill_value,
                                    dtype=dtype,
                                    overwrite=True,
                                    )
    image_zarr.attrs['created_at'] = str(datetime.now())
    image_zarr.attrs['video_hash'] = video_hash_abbrev
    if verbose:
        print('Zarr store info:')
        print(image_zarr.info)

    return image_zarr

def load_image_zarr(zarr_path, 
                    num_frames, 
                    image_height,
                    image_width=None,
                    chunk_size=20,
                    num_ch=None,
                    video_hash_abrrev=None,
                    verbose=True,
                    ):
    """
    Loads an existing zarr archive for storing and retrieving image data,
    and checks if the stored array has the expected parameters.
    
    Parameters
    ---------
    zarr_path : pathlib.Path
        Path to the zarr archive. Must end in .zarr.
    num_frames : int
        Expected number of frames in the video.
    image_height : int
        Expected height of the resized image (as in SAM2 input).
    image_width : int, optional
        Expected width of the resized image (as in SAM2 input). 
        If None, then image_width = image_height.
    chunk_size : int, optional
        Size of a chunk stored in the zarr archive.
    num_ch : int, optional
        Number of channels in the image. Default is None.
    video_hash_abrrev : str, optional
        Abbreviated hash of the video file. This is used as
        a unique identifier for the corresponding video file throughout.
    verbose : bool, optional
        If True, print the zarr store info.
        
    Returns
    -------
    image_zarr : zarr.core.Array
        The loaded zarr array storing the image data.
    status : bool
        True if the array was loaded successfully, False otherwise.   

    """
    assert zarr_path.exists(), f'Zarr folder {zarr_path.as_posix()} does not exist.'
    assert image_height > 0, f'image_height must be > 0, not {image_height}'
    if image_width is None:
        image_width = image_height
    else:
        assert image_width > 0, f'image_width must be > 0, not {image_width}'

    # Open the LocalStore and check if the group 'masks' exists
    store = zarr.storage.LocalStore(zarr_path, read_only=False)  
    root = zarr.open_group(store=store, mode='a')
    if verbose: 
        print("Existing keys in zarr archive:", list(root.array_keys()))
    # Attempt to load the array named 'masks'
    if 'masks' not in root:
        print(f"Array 'masks' not found in {zarr_path.as_posix()}")
        return None, False
    else:
        image_zarr = root['masks']
    
    if num_ch is not None:
        expected_shape = (num_frames, num_ch, image_height, image_width)
    else:
        expected_shape = (num_frames, image_height, image_width)
    if image_zarr.shape != expected_shape:
        print(f"Shape mismatch: expected {expected_shape}, got {image_zarr.shape}")
        return None, False
    
    # if num_ch is not None:
    #     expected_chunks = (chunk_size, num_ch, image_height, image_width)
    # else:
    #     expected_chunks = (chunk_size, image_height, image_width)
    # if image_zarr.chunks != expected_chunks:
    #     print(f"Chunk size mismatch: expected {expected_chunks}, got {image_zarr.chunks}")
    #     return None, False
    
    # Check video hash if provided
    if video_hash_abrrev is not None:
        stored_hash = image_zarr.attrs.get('video_hash', None)
        if stored_hash != video_hash_abrrev:
            print(f"❌ Video hash mismatch: expected {video_hash_abrrev}, got {stored_hash}")
            return None, False
        elif verbose:
            print(f"🔒 Video hash verified: {video_hash_abrrev}")
    
    if verbose:
        print('Zarr store info:')
        print(image_zarr.info) 
    return image_zarr, True



class OctoZarr:
    """
    Flexible subclass of zarr array that allows for image data retrieval
    
    The idea here was to just replace the possibly very large 
    image dictionary directly with a zarr array. 
    I.e. instead of pre-loading all images into the dictionary, 
    just lazy load them when needed, and save them
    into zarr, so the second time they are accessed, the access is faster. 
    
    This should be optimized...  This is a lot (!) of back and forth (torch->numpy and back)
    
    
    """
    def __init__(self, 
                 zarr_array, 
                 video_data,
                 running_buffer_size=50,
                 normalize_scale=True,
                 normalize_mean=True,
                 normalize_std=True,
                 ):
        self.zarr_array = zarr_array
        self.saved_indices = []
        
        # Collect some basic info 
        self.num_frames, self.num_chs, self.image_height, self.image_width = zarr_array.shape
        
        # Store normalization settings
        self.normalize_scale = normalize_scale
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        
        # Choose normalization constants based on whether we scale to (0,1) or keep (0,255)
        if self.normalize_scale:
            # ImageNet normalization (for 0-1 range)
            img_mean = (0.485, 0.456, 0.406)
            img_std = (0.229, 0.224, 0.225)
        else:
            # Raw pixel normalization (for 0-255 range)
            img_mean = (123.675, 116.28, 103.53)
            img_std = (58.395, 57.12, 57.375)
        
        self.img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
        
        
        # Initialize resizing function
        self._resize_img = Resize(size=(self.image_height, self.image_width))

        # Store the napari data layer   
        self.video_data = video_data
        self.cached_indices = np.full((running_buffer_size), np.nan)
        self.cached_images  = torch.empty(running_buffer_size, 
                                          self.num_chs, 
                                          self.image_height, 
                                          self.image_width,
                                          dtype=torch.float16
                                          )
        self.cur_cache_idx = 0 # Keep track of where you are in the cache currently

        # Add loader interface attributes to make it compatible with ultralytics loaders
        self.mode = "video"  # Set mode as video
        self.bs = 1  # Batch size (process one frame at a time)
        self.frame = 0  # Current frame index
        self.frames = self.num_frames  # Total number of frames
        self.fps = 30  # Default FPS
        self.count = 0  # Iterator count
        self.paths = [f"frame_{i}.jpg" for i in range(self.num_frames)]  # Generate dummy paths
        # Set source type to indicate this is a video stream
        self.source_type = SourceTypes(stream=True, screenshot=False, from_img=False, tensor=False)

    @property
    def indices_in_store(self):
        return self.saved_indices        

    def _save_to_zarr(self, batch, indices):
        """ 
        Save a batch of images to  zarr array at index position indices
        """
        assert len(indices), 'No indices provided'
        assert len(batch) == len(indices), 'Batch and indices should have the same length'
        
        if len(batch) == 1:
            batch = batch[0]
        if len(indices) == 1:
            indices_ = indices[0]
        else:
            indices_ = indices
        self.zarr_array[indices_,:,:,:] = batch.float().numpy().astype(np.float16)
        # ... see https://github.com/pytorch/pytorch/issues/90574#issuecomment-1983794341 
        self.saved_indices.extend(indices)   
         
    @torch.inference_mode()
    def _fetch_one(self, idx):
        img = self.video_data[idx]
        img = self._resize_img(torch.from_numpy(img).permute(2,0,1)).to(torch.float16)
        if self.normalize_scale:
            img /= 255.  
        if self.normalize_mean:
            img -= self.img_mean
        if self.normalize_std:
            img /= self.img_std     
        # Cache 
        self.cached_indices[self.cur_cache_idx] = idx
        self.cached_images[self.cur_cache_idx] = img
        self.cur_cache_idx += 1
        if self.cur_cache_idx == len(self.cached_indices):
            self.cur_cache_idx = 0  
        return img   
    
    @torch.inference_mode()
    def _fetch_many(self, indices):
        imgs = self.video_data[indices]
        imgs = self._resize_img(torch.from_numpy(imgs).permute(0,3,1,2)).to(torch.float16)
        if self.normalize_scale:
            imgs /= 255.  
        if self.normalize_mean:
            imgs -= self.img_mean
        if self.normalize_std:
            imgs /= self.img_std
        # Cache
        for i, idx in enumerate(indices):
            self.cached_indices[self.cur_cache_idx] = idx
            self.cached_images[self.cur_cache_idx] = imgs[i]
            self.cur_cache_idx += 1
            if self.cur_cache_idx == len(self.cached_indices):
                self.cur_cache_idx = 0
        return imgs
    
    def fetch(self, indices):   
        
        """
        Check if the images are already in the zarr store.
        
        The logic is the following: 
        - Enable "quick" loading of single indices without saving them into zarr array. This would 
          just slow things down. 
        - Enable slightly slower loading from and saving to zarr for batches of images
        
        Generally for multiple images (batches):
        For those images that are not in the store, prefetch them from the napari data layer, then
        - Resize the images
        - Normalize the images
        - Save the images to the zarr array
        Combine those with images loaded from zarr store 
        - Return the combined images as torch tensor
        
        """
        # Initialize empty torch arrach of length indices
        imgs_torch = torch.empty(len(indices), 
                                 self.num_chs, 
                                 self.image_height, 
                                 self.image_width,
                                 dtype=torch.float16
                                 )
        
        # First check whether the indices are in the cache
        # If they are, return them immediately
        cached_idx = np.where(np.isin(self.cached_indices, indices))[0]
        if len(cached_idx):
            #print(f'Cached at indices {self.cached_indices[cached_idx]}')
            imgs_cached = self.cached_images[cached_idx]
            imgs_torch[np.where(np.isin(indices, self.cached_indices))[0]] = imgs_cached
            
        # Cover cases for which there are indices left (images that are not in the rolling cache)
        # Subtract the cached indices from the indices
        indices = np.setdiff1d(indices, self.cached_indices)
        if len(indices) == 1:
            # Single image
            idx = indices[0]
            if idx in self.saved_indices:
                img = torch.from_numpy(self.zarr_array[idx]).to(torch.float16)
            else:
                img = self._fetch_one(idx=idx)
                self._save_to_zarr([img], [idx])
            imgs_torch[np.where(indices == idx)[0][0]] = img
            
        elif len(indices) > 1:
            # Create indices
            not_in_store = np.array([idx for idx in indices if idx not in self.saved_indices]).astype(int)
            in_store = np.array([idx for idx in indices if idx in self.saved_indices]).astype(int)

            if len(not_in_store):
                imgs = self._fetch_many(indices=not_in_store)
                imgs_torch[np.where(np.isin(indices, not_in_store))[0]] = imgs
                # Save this batch to zarr 
                self._save_to_zarr(imgs, not_in_store)
            if len(in_store):
                #print(f'Found in store (multiple): {in_store}')
                imgs_in_store = torch.from_numpy(self.zarr_array[in_store]).squeeze().to(torch.float16)
                imgs_torch[np.where(np.isin(indices, in_store))[0]] = imgs_in_store    

        return imgs_torch.squeeze()
    
    def __getitem__(self, frame_idx):
        """
        Normal "get" function 

        """
        if isinstance(frame_idx, slice):
            indices = np.arange(frame_idx.start, frame_idx.stop, frame_idx.step)
        elif isinstance(frame_idx, list):
            indices = np.array(frame_idx)
        elif isinstance(frame_idx, int):
            indices = [frame_idx]   
        else:
            raise ValueError(f'frame_idx should be int, list or slice, got {type(frame_idx)}')

        images = self.fetch(indices)
        return images
    
    def __iter__(self):
        """Return an iterator for the video frames."""
        self.count = 0
        self.frame = 0
        return self
    
    def __next__(self):
        """
        Return the next frame in the video sequence.
        Uses the fetch infrastructure to leverage caching and zarr storage.
        
        Returns:
            tuple: (paths, images, info) where:
                - paths: list of frame paths/identifiers
                - images: list of numpy arrays containing the frame data
                - info: list of string metadata about the frame
        """
        if self.count >= self.num_frames:
            raise StopIteration
        
        # Use fetch() directly to leverage caching and zarr storage
        img = self.fetch([self.count])        
        path = self.paths[self.count]
        info = f"video 1/1 (frame {self.count + 1}/{self.num_frames}): "
        
        self.count += 1
        self.frame = self.count
        
        return [path], [img], [info]
        
    def __repr__(self):
            return repr(self.zarr_array)


