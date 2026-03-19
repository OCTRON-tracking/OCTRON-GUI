"""Skeleton definition for structured keypoint tracking.

A SkeletonDefinition holds the ordered list of keypoint names and
optional connectivity (which keypoints connect with lines).  It is
used by CoTracker pose-tracking to assign each user click to a named
body part and to allocate correctly-sized trajectory zarr arrays.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, field_validator

from octron.sam_octron.helpers.sam2_colors import (
    create_label_colors,
    sample_maximally_different,
)


class SkeletonDefinition(BaseModel):
    """Ordered keypoint names with optional connectivity.

    Parameters
    ----------
    keypoint_names : list[str]
        Ordered keypoint names, e.g. ["nose", "left_ear", "right_ear"].
        The index in this list becomes the keypoint_index stored in
        napari Point features and in the trajectory zarr column.
    connectivity : list[tuple[int, int]], optional
        Pairs of keypoint indices that should be connected with lines,
        e.g. [(0, 1), (0, 2)] to draw nose→left_ear and nose→right_ear.
    """

    keypoint_names: List[str]
    connectivity: Optional[List[Tuple[int, int]]] = None

    @field_validator("keypoint_names")
    @classmethod
    def names_must_be_non_empty(cls, v):
        """Check keypoint names are non-empty and unique."""
        if not v:
            raise ValueError("keypoint_names must contain at least one entry")
        if len(v) != len(set(v)):
            raise ValueError("keypoint_names must be unique")
        return v

    @field_validator("connectivity")
    @classmethod
    def connectivity_indices_in_range(cls, v, info):
        """Check connectivity indices are valid for the number of keypoints."""
        if v is None:
            return v
        names = info.data.get("keypoint_names", [])
        n = len(names)
        for i, j in v:
            if not (0 <= i < n and 0 <= j < n):
                raise ValueError(
                    f"Connectivity edge ({i}, {j}) out of range for {n} keypoints"
                )
        return v

    @property
    def n_keypoints(self) -> int:
        return len(self.keypoint_names)

    def get_keypoint_colors(self) -> List[list]:
        """Return a list of RGBA colors (one per keypoint).

        Uses the same maximally-different sampling strategy as the
        ObjectOrganizer label colors, but applied across keypoints.
        """
        n = self.n_keypoints
        if n == 0:
            return []

        # Use a single label slice and sample n maximally different
        # sub-colors from it.
        label_colors = create_label_colors(
            cmap="cmr.tropical",
            n_labels=1,
            n_colors_submap=max(n, 10),
        )
        subcolors = label_colors[0]  # list of RGBA arrays
        indices = sample_maximally_different(list(range(len(subcolors))))
        return [subcolors[indices[i % len(indices)]] for i in range(n)]

    def get_color_for_keypoint(self, keypoint_index: int) -> list:
        """Return the RGBA color for a single keypoint index."""
        return self.get_keypoint_colors()[keypoint_index]

    def save_yaml(self, path) -> None:
        """Save skeleton definition to a YAML file."""

        path = Path(path)
        with open(path, "w") as f:
            # .model_dump() returns a dict with keys "keypoint_names" 
            # and "connectivity"
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def load_yaml(cls, path) -> "SkeletonDefinition":
        """Load skeleton definition from a YAML file.

        Expected format::
            keypoint_names:
              - nose
              - left_ear
            connectivity:
              - [0, 1]
        """
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)
