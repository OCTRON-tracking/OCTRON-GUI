"""Napari layer creation helpers for CoTracker annotation."""

import numpy as np
import pandas as pd


def add_cotracker_points_layer(viewer, name, skeleton_definition, obj_color):
    """Create a napari Points layer for CoTracker keypoint annotation.

    The layer is set up with:
    - Per-keypoint colors from the skeleton definition
    - A features DataFrame with a ``skeleton_idx`` column
    - Text labels showing keypoint names

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    name : str
        Name of the new points layer.
    skeleton_definition : SkeletonDefinition
        Skeleton with keypoint names and colors.
    obj_color : list
        RGBA color for this object (used as border color).

    Returns
    -------
    points_layer : napari.layers.Points
        The created points layer.
    """
    # Initialise empty features with skeleton_idx column
    features = pd.DataFrame({"skeleton_idx": pd.Series(dtype=int)})

    points_layer = viewer.add_points(
        None,
        ndim=3,
        name=name,
        features=features,
        border_color=obj_color,
        border_width=0.2,
        opacity=0.6,
    )

    # Select layer and activate add mode
    viewer.layers.selection.active = points_layer
    viewer.layers.selection.active.mode = "add"

    return points_layer


def add_cotracker_tracks_layer(
    viewer, name, zarr_roots, n_keypoints, skeleton_definition=None
):
    """Build and add a napari Tracks layer from trajectory zarr stores.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    name : str
        Name for the tracks layer.
    zarr_roots : dict[int, zarr.Group]
        Mapping ``{obj_id: zarr_root}`` for each tracked object.
    n_keypoints : int
        Number of keypoints in the skeleton.
    skeleton_definition : SkeletonDefinition, optional
        If provided, used for per-keypoint colors.

    Returns
    -------
    tracks_layer : napari.layers.Tracks or None
        The created tracks layer, or None if no visible tracks.
    """
    all_tracks = []
    all_colors = []

    for obj_id, zarr_root in zarr_roots.items():
        tracks_data, track_colors = _build_tracks_array_from_zarr(
            zarr_root, obj_id, n_keypoints, skeleton_definition
        )
        if len(tracks_data):
            all_tracks.append(tracks_data)
            if track_colors is not None:
                all_colors.append(track_colors)

    if not all_tracks:
        return None

    combined_tracks = np.concatenate(all_tracks, axis=0)

    kwargs = {"name": name, "tail_length": 10}
    if all_colors:
        kwargs["color_by"] = "track_id"

    tracks_layer = viewer.add_tracks(combined_tracks, **kwargs)
    return tracks_layer


def _build_tracks_array_from_zarr(
    zarr_root, obj_id, n_keypoints, skeleton_definition=None
):
    """Build a napari Tracks array from a trajectory zarr.

    Reads the zarr ``"tracks"`` array of shape ``(T, N_keypoints, 3)``
    and converts it to napari Tracks format ``(ID, T, Y, X)``.

    Only frames where visibility > 0 for a keypoint are included.
    Track IDs are ``obj_id * n_keypoints + keypoint_idx``.

    Parameters
    ----------
    zarr_root : zarr.Group
        Zarr root group containing a ``"tracks"`` array.
    obj_id : int
        Object ID (used to build unique track IDs).
    n_keypoints : int
        Number of keypoints in the skeleton.
    skeleton_definition : SkeletonDefinition, optional
        If provided, used for per-keypoint colors.

    Returns
    -------
    tracks_data : np.ndarray
        Shape ``(N_visible, 4)`` with columns ``(track_id, frame, y, x)``.
    track_colors : np.ndarray or None
        Per-track RGBA colors if skeleton_definition is provided.
    """
    tracks_arr = np.array(zarr_root["tracks"])  # (T, N_keypoints, 3)
    n_frames = tracks_arr.shape[0]

    rows = []
    color_per_row = []
    kpt_colors = (
        skeleton_definition.get_keypoint_colors() if skeleton_definition else None
    )

    for kpt_idx in range(n_keypoints):
        track_id = obj_id * n_keypoints + kpt_idx
        for t in range(n_frames):
            x, y, vis = tracks_arr[t, kpt_idx]
            if vis > 0:
                rows.append([track_id, t, y, x])  # napari: (ID, T, Y, X)
                if kpt_colors is not None:
                    color_per_row.append(kpt_colors[kpt_idx])

    if not rows:
        return np.empty((0, 4)), None

    tracks_data = np.array(rows, dtype=np.float32)
    track_colors = np.array(color_per_row) if color_per_row else None

    return tracks_data, track_colors
