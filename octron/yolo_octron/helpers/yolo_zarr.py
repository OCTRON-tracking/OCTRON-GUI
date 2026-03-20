# Very similar to the SAM2 zarr function - maybe unite in the future
import shutil
from datetime import datetime
from pathlib import Path
import zarr
from loguru import logger

MIN_ZARR_CHUNK_SIZE = 50 # Setting minimum chunk size for zarr arrays
                         # to avoid excessive chunking for small arrays

def create_prediction_store(zarr_path, 
                            verbose=False,
                            ):
    """
    Creates a zarr store (LocalStore) for storing and retrieving prediction data.
    This can then be supplemented with data during prediction. 
    
    Parameters
    ---------
    zarr_path : pathlib.Path
        Path to the zarr archive. Must end in .zarr
    verbose : bool, optional
        If True, print the zarr store info.
        
        
    Returns
    -------
    store : zar.storage.LocalStore
        Zarr store object.
        
    """
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    # Assuming local store on fast SSD, so no compression employed for now 
    store = zarr.storage.LocalStore(zarr_path, read_only=False)  

    return store


def create_prediction_zarr(store, 
                           array_name,
                           shape,
                           chunk_size=20,
                           fill_value=-1,
                           dtype='int8',
                           video_hash=None,
                           verbose=False,
                           ):
    """
    Creates a zarr archive for storing and retrieving prediction data.
    
    Parameters
    ---------
    store : zarr.storage.LocalStore
        Zarr store object. Created using create_prediction_store()
    array_name : str
        Name of the zarr array.
    shape : tuple
        Shape of the zarr array.
    chunk_size : int, optional
        Size of the chunk to store in the zarr archive. 
    fill_value : int, optional
        Value to fill the zarr array with.
    dtype : str, optional
        Data type of the zarr array.
    video_hash : str, optional
        Hash of the video file. This is used as 
        a unique identifier for the corresponding video file throughout.
    verbose : bool, optional
        If True, print the zarr store info.
        
        
    Returns
    -------
    image_zarr : zarr.core.Array
        Zarr array object.
        
    """
    assert chunk_size > 0, f'chunk_size must be > 0, not {chunk_size}'
    assert isinstance(store, zarr.storage.LocalStore), 'store must be a zarr.storage.LocalStore object'
    chunk_size = max(chunk_size, MIN_ZARR_CHUNK_SIZE)
    # Create chunks tuple based on shape dimensions
    # First dimension uses chunk_size, remaining dimensions use their full size
    chunks = (chunk_size,) + shape[1:] if len(shape) > 1 else (chunk_size,)
    
    image_zarr = zarr.create_array(store=store,
                                   name=array_name,
                                   shape=shape,  
                                   chunks=chunks, 
                                   fill_value=fill_value,
                                   dtype=dtype,
                                   overwrite=True,
                                   )
    image_zarr.attrs['created_at'] = str(datetime.now())
    image_zarr.attrs['video_hash'] = video_hash
    image_zarr.attrs['annotated_frames'] = []
    if verbose:
        logger.debug("Zarr array info: %s", image_zarr.info)
        
    return image_zarr