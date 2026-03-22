"""Napari layer creation helpers for CoTracker annotation."""

import numpy as np
import pandas as pd


def add_cotracker_prompts_layer(viewer, name, skeleton_definition):
    """Create a napari Points layer for CoTracker prompts.

    The layer is set up with:
    - Per-keypoint face colors from the skeleton definition
    - A features DataFrame with a ``skeleton_idx`` column

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    name : str
        Name of the new points layer.
    skeleton_definition : SkeletonDefinition
        Skeleton with keypoint names and colors.

    Returns
    -------
    points_layer : napari.layers.Points
        The created points layer.
    """
    # Initialise empty features with skeleton_idx column
    features = pd.DataFrame({"skeleton_idx": pd.Series(dtype=int)})

    # Pre-compute per-keypoint colors for dynamic face_color updates
    keypoint_colors = skeleton_definition.get_keypoint_colors()

    points_layer = viewer.add_points(
        None,
        ndim=3,
        name=name,
        features=features,
        border_width=0.0,
        opacity=0.6,
    )

    # Store keypoint colors on metadata for use in on_points_changed
    points_layer.metadata["_keypoint_colors"] = keypoint_colors

    # Select layer and activate add mode
    viewer.layers.selection.active = points_layer
    viewer.layers.selection.active.mode = "add"

    return points_layer


def add_cotracker_prediction_layer(
    viewer,
    name,
    zarr_root,
    n_keypoints,
    skeleton_definition=None,
    existing_layer=None,
):
    """Build and add (or update) a napari Points layer for CoTracker predictions.

    If ``existing_layer`` is provided, its data is updated in place.
    Otherwise a new layer is created.  When the zarr contains no visible
    predictions yet, the layer is created with ``None`` data and hidden.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer instance.
    name : str
        Name for the prediction points layer.
    zarr_root : zarr.Group
        Zarr root group containing a ``"tracks"`` array.
    n_keypoints : int
        Number of keypoints in the skeleton.
    skeleton_definition : SkeletonDefinition, optional
        If provided, used for per-keypoint colors.
    existing_layer : napari.layers.Points, optional
        If provided, update this layer's data instead of creating a new one.

    Returns
    -------
    points_layer : napari.layers.Points
        The created or updated prediction layer.
    """
    points_data, face_colors, skel_indices = _build_points_array_from_zarr(
        zarr_root, n_keypoints, skeleton_definition
    )
    has_data = len(points_data) > 0

    if existing_layer is not None:
        existing_layer.data = points_data if has_data else None
        if has_data:
            existing_layer.features = pd.DataFrame({"skeleton_idx": skel_indices})
            if face_colors is not None:
                existing_layer.face_color = face_colors
        existing_layer.visible = has_data
        existing_layer.refresh()
        return existing_layer

    features = (
        pd.DataFrame({"skeleton_idx": skel_indices})
        if has_data
        else pd.DataFrame({"skeleton_idx": pd.Series(dtype=int)})
    )
    points_layer = viewer.add_points(
        points_data if has_data else None,
        ndim=3,
        name=name,
        features=features,
        symbol="diamond",
        size=8,
        border_width=0.0,
        opacity=0.5,
        visible=has_data,
    )
    if face_colors is not None and has_data:
        points_layer.face_color = face_colors
    return points_layer


def _build_points_array_from_zarr(zarr_root, n_keypoints, skeleton_definition=None):
    """Build a napari Points array from a trajectory zarr.

    Reads the zarr ``"tracks"`` array of shape ``(T, N_keypoints, 3)``
    and converts it to napari Points format ``(T, Y, X)`` with ``ndim=3``.

    Only frames where visibility > 0 for a keypoint are included.

    Parameters
    ----------
    zarr_root : zarr.Group
        Zarr root group containing a ``"tracks"`` array.
    n_keypoints : int
        Number of keypoints in the skeleton.
    skeleton_definition : SkeletonDefinition, optional
        If provided, used for per-keypoint colors.

    Returns
    -------
    points_data : np.ndarray
        Shape ``(N_visible, 3)`` with columns ``(frame, y, x)``.
    face_colors : np.ndarray or None
        Per-point RGBA colors if skeleton_definition is provided.
    skeleton_indices : np.ndarray
        Per-point keypoint index, shape ``(N_visible,)``.
    """
    tracks_arr = np.array(zarr_root["tracks"])  # (T, N_keypoints, 3)
    n_frames = tracks_arr.shape[0]

    rows = []
    color_per_row = []
    skel_per_row = []
    kpt_colors = (
        skeleton_definition.get_keypoint_colors() if skeleton_definition else None
    )

    for kpt_idx in range(n_keypoints):
        for t in range(n_frames):
            x, y, vis = tracks_arr[t, kpt_idx]
            if vis > 0:
                rows.append([t, y, x])  # napari Points: (T, Y, X)
                skel_per_row.append(kpt_idx)
                if kpt_colors is not None:
                    color_per_row.append(kpt_colors[kpt_idx])

    if not rows:
        return np.empty((0, 3)), None, np.empty((0,), dtype=int)

    points_data = np.array(rows, dtype=np.float32)
    face_colors = np.array(color_per_row) if color_per_row else None
    skeleton_indices = np.array(skel_per_row, dtype=int)

    return points_data, face_colors, skeleton_indices
