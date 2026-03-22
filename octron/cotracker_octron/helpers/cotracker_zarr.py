"""Zarr utilities for storing CoTracker trajectory results.

Each tracked object gets its own zarr array with shape (T, N_keypoints, 3),
where the 3 columns are (x, y, visibility). Untracked frames are filled
with (0, 0, 0).
"""

import shutil
from pathlib import Path

import zarr


def create_trajectory_zarr(
    zarr_path,
    num_frames,
    n_keypoints,
    fill_value=0,
    dtype="float32",
    video_hash_abbrev=None,
):
    """Create a zarr array for storing per-object trajectory data.

    We create one zarr per object ID, with shape (T, N_keypoints, 3)
    where the 3 last channels are (x, y, visibility).

    Parameters
    ----------
    zarr_path : Path
        Path ending in .zarr for the trajectory archive.
    num_frames : int
        Total number of frames in the video (T dimension).
    n_keypoints : int
        Number of tracked keypoints for this object.
    fill_value : float
        Value used to fill the array (0 means untracked).
    dtype : str
        Data type of the array.
    video_hash_abbrev : str, optional
        Abbreviated video hash stored as metadata.

    Returns
    -------
    root : zarr.Group
        The root group containing a "tracks" array (T, N_keypoints, 3)
        for the predicted trajectories. Annotated frame indices are stored
        as a JSON attribute on the tracks array (same pattern as SAM2/3).
    """
    # Validate zarr path
    zarr_path = Path(zarr_path)
    assert (
        zarr_path.suffix == ".zarr"
    ), f"zarr_path must end in .zarr, got {zarr_path.suffix}"

    # Remove existing zarr if it exists to avoid conflicts
    if zarr_path.exists():
        shutil.rmtree(zarr_path)

    # Initialise zarr store
    store = zarr.storage.LocalStore(zarr_path)
    root = zarr.open_group(store=store, mode="w")

    # Add tracks array
    root.create_array(
        name="tracks",
        shape=(num_frames, n_keypoints, 3),
        chunks=(num_frames, n_keypoints, 3),
        dtype=dtype,
        fill_value=fill_value,
    )

    # Add metadata
    root.attrs["num_frames"] = num_frames
    root.attrs["n_keypoints"] = n_keypoints
    root.attrs["video_hash_abbrev"] = video_hash_abbrev

    return root


def load_trajectory_zarr(zarr_path, num_frames=None, n_keypoints=None):
    """Load an existing tracks zarr array and optionally validate its shape.

    Parameters
    ----------
    zarr_path : Path
        Path to the .zarr store containing the 'tracks' array.
    num_frames : int, optional
        Expected number of frames. Checked if provided.
    n_keypoints : int, optional
        Expected number of keypoints. Checked if provided.

    Returns
    -------
    root : zarr.Group or None
        The root group containing the "tracks" array,
        or None if validation fails.
    status : bool
        True if loaded successfully.
    """
    # Validate zarr path
    zarr_path = Path(zarr_path)
    if not zarr_path.exists():
        return None, False

    # Read zarr store in append mode
    store = zarr.storage.LocalStore(zarr_path, read_only=False)
    root = zarr.open_group(store=store, mode="a")

    # Get tracks array
    if "tracks" not in root:
        print(f"Array 'tracks' not found in {zarr_path}")
        return None, False
    trajectory_zarr = root["tracks"]

    # Validate shape if expected values are provided
    if num_frames is not None and trajectory_zarr.shape[0] != num_frames:
        print(
            f"Frame count mismatch: expected {num_frames}, got {trajectory_zarr.shape[0]}"
        )
        return None, False
    if n_keypoints is not None and trajectory_zarr.shape[1] != n_keypoints:
        print(
            f"Keypoint count mismatch: expected {n_keypoints}, got {trajectory_zarr.shape[1]}"
        )
        return None, False

    return root, True


def write_frame_tracks(trajectory_zarr, frame_idx, tracks_xyv):
    """Write tracking results for a single frame into the zarr array.

    Parameters
    ----------
    trajectory_zarr : zarr.core.Array
        Shape (T, N_keypoints, 3).
    frame_idx : int
        The frame index to write to.
    tracks_xyv : np.ndarray or torch.Tensor
        Shape (N_keypoints, 3) with columns (x, y, visibility).
    """
    if hasattr(tracks_xyv, "cpu"):
        tracks_xyv = tracks_xyv.cpu().numpy()

    trajectory_zarr[frame_idx, :, :] = tracks_xyv
