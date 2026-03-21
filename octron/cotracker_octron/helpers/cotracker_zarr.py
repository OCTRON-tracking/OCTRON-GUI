"""Zarr utilities for storing CoTracker trajectory results.

Each tracked object gets its own zarr array with shape (T, N_keypoints, 3),
where the 3 columns are (x, y, visibility). Untracked frames are filled
with (0, 0, 0).
"""

import shutil
from pathlib import Path

import numpy as np
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
        The root group containing two arrays: 
        - "tracks" (T, N_keypoints, 3), for the predicted trajectories 
        - "annotated" (T, N_keypoints), to log the manually labelled frames 
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

    # Add array for annotated frames
    root.create_array(
        name="annotated",
        shape=(num_frames, n_keypoints),
        chunks=(num_frames, n_keypoints),
        dtype="bool",
        fill_value=False,
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
        The root group containing "tracks" and "annotated" arrays,
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


def mark_keypoints_annotated(zarr_root, frame_idx, keypoint_indices):
    """Mark keypoints as manually annotated for a given frame.

    Parameters
    ----------
    zarr_root : zarr.Group
        The root group containing the "annotated" array.
    frame_idx : int
        The frame index.
    keypoint_indices : int or iterable of int
        Keypoint index or indices to mark as annotated.
    """
    if not isinstance(keypoint_indices, list):
        keypoint_indices = list(keypoint_indices)
    for kp_idx in keypoint_indices:
        zarr_root["annotated"][int(frame_idx), int(kp_idx)] = True 


def unmark_keypoints_annotated(zarr_root, frame_idx, keypoint_indices):
    """Unmark keypoints as manually annotated for a given frame.

    Applied if keypoints are deleted.

    Parameters
    ----------
    zarr_root : zarr.Group
        The root group containing the "annotated" array.
    frame_idx : int
        The frame index.
    keypoint_indices : int or iterable of int
        Keypoint index or indices to unmark.
    """
    if isinstance(keypoint_indices, (int, np.integer)):
        zarr_root["annotated"][int(frame_idx), int(keypoint_indices)] = False
    else:
        for kp_idx in keypoint_indices:
            zarr_root["annotated"][int(frame_idx), int(kp_idx)] = False


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
