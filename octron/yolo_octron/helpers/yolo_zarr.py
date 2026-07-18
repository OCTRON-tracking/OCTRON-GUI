"""Zarr store helpers for YOLO prediction data.

Very similar to the SAM2 zarr function - maybe unite in the future.
"""

import shutil
from datetime import datetime

import zarr
from loguru import logger

MIN_ZARR_CHUNK_SIZE = 1  # One frame per chunk for prediction zarrs.
# Spatial dims (H×W) are already large; chunking at 1 frame
# means napari reads exactly one frame per disk - this makes scrubbing faster


def create_prediction_store(
    zarr_path,
    verbose=False,
):
    """Create a zarr store (LocalStore) for prediction data.

    This can then be supplemented with data during prediction.

    Parameters
    ----------
    zarr_path : pathlib.Path
        Path to the zarr archive. Must end in .zarr
    verbose : bool, optional
        If True, print the zarr store info.


    Returns
    -------
    store : zarr.storage.LocalStore
        Zarr store object.

    """
    if zarr_path.exists():
        shutil.rmtree(zarr_path)
    # Assuming local store on fast SSD, so no compression employed for now
    store = zarr.storage.LocalStore(zarr_path, read_only=False)

    return store


def create_prediction_zarr(
    store,
    array_name,
    shape,
    chunk_size=1,
    shard_size=None,
    fill_value=-1,
    dtype="int8",
    video_hash=None,
    verbose=False,
):
    """Create a zarr archive for storing and retrieving prediction data.

    Parameters
    ----------
    store : zarr.storage.LocalStore
        Zarr store object. Created using create_prediction_store()
    array_name : str
        Name of the zarr array.
    shape : tuple
        Shape of the zarr array.
    chunk_size : int, optional
        Inner chunk size (frames). This is the unit of random access — one
        chunk is what napari reads per frame render. Default is 1 (one frame).
    shard_size : int or None, optional
        Outer shard size (frames). Multiple chunks are packed into a single
        shard file on disk. Must be a multiple of chunk_size and should also
        be a divisor of the write buffer_size to avoid read-modify-write
        overhead. None disables sharding (one file per chunk).
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
    assert chunk_size > 0, f"chunk_size must be > 0, not {chunk_size}"
    assert isinstance(store, zarr.storage.LocalStore), (
        "store must be a zarr.storage.LocalStore object"
    )
    chunk_size = max(chunk_size, MIN_ZARR_CHUNK_SIZE)
    # Inner chunks: chunk_size frames, full spatial extent
    chunks = (chunk_size,) + shape[1:] if len(shape) > 1 else (chunk_size,)

    create_kwargs = dict(
        store=store,
        name=array_name,
        shape=shape,
        chunks=chunks,
        fill_value=fill_value,
        dtype=dtype,
        overwrite=True,
    )
    # Sharding: pack multiple chunks into fewer shard files to reduce
    # filesystem pressure while keeping single-frame random access.
    # shard_size must be a multiple of chunk_size and ideally a divisor
    # of the write buffer_size so every flush writes complete shards.
    if shard_size is not None and shard_size > chunk_size:
        shards = (shard_size,) + shape[1:] if len(shape) > 1 else (shard_size,)
        create_kwargs["shards"] = shards
        if verbose:
            logger.debug(
                f"Sharding enabled: {shard_size} frames per shard, "
                f"{chunk_size} frames per chunk"
            )

    image_zarr = zarr.create_array(**create_kwargs)
    image_zarr.attrs["created_at"] = str(datetime.now())
    image_zarr.attrs["video_hash"] = video_hash
    image_zarr.attrs["annotated_frames"] = []
    if verbose:
        logger.debug("Zarr array info: %s", image_zarr.info)

    return image_zarr
