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
