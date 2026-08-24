"""Helpers for loading and analyzing SAM2 annotation results."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

# Plugins
from loguru import logger
from natsort import natsorted
from skimage.measure import regionprops
from tqdm import tqdm

from octron.sam_octron.helpers.sam_zarr import get_annotated_frames


class ANNOT_results:
    """Load and analyze SAM2 annotation results stored as Zarr archives."""

    def __init__(self, annotation_dir, verbose=True, **kwargs):
        """Initialize by locating annotation folders and Zarr archives.

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
            message="Object at .DS_Store is not recognized as a "
            "component of a Zarr hierarchy.",
            category=UserWarning,
            module="zarr.core.group",
        )
        warnings.filterwarnings(
            "ignore",
            message=r"Object at (\.DS_Store|desktop\.ini) is not "
            r"recognized as a component of a Zarr hierarchy.",
            category=UserWarning,
            module="zarr.core.group",
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
        """Log creation date and video file name from video_info.txt."""
        annotation_folder = Path(annotation_folder)
        assert annotation_folder.exists(), (
            "Annotation folder path does not exist"
        )
        if (annotation_folder / "video_info.txt").exists():
            info_file_path = annotation_folder / "video_info.txt"
            creation_date = None
            video_path = None
            with open(info_file_path) as f:
                for line in f:
                    if line.startswith("Info file created on:"):
                        creation_date = line.split(":", 1)[1].strip()
                    elif line.startswith("Video path:"):
                        video_path = Path(line.split(":", 1)[1].strip())
            if creation_date:
                logger.info(f"Annotations first started on: {creation_date}")
            if video_path:
                logger.info(f"Video file name: {video_path.name}")

    def find_zarrs(self):
        """Find and sort all Zarr archives in the annotation directory."""
        self.zarr_archives = natsorted(
            Path(self.annotation_dir).rglob("[!video]*.zarr")
        )
        if self.verbose:
            logger.info(
                f"Found {len(self.zarr_archives)} zarr archives\n"
                f"{self.zarr_archives}"
            )

    def create_zarr_dict(self):
        """Find Zarr archives and populate label-to-mask-array mapping.

        Also extracts video dimensions and non-empty frame indices.
        This populates
        - self.zarr_dict
        - self.frame_indices_dict (non empty frame indices)
        """
        self.find_zarrs()

        for z in self.zarr_archives:
            label_name = "_".join(z.stem.split(" ")[:-1])
            store = zarr.storage.LocalStore(z, read_only=True)
            root = zarr.open_group(store=store, mode="r")
            zarr_root = root
            self.zarr_dict[label_name] = zarr_root["masks"]
            # Check num_frames, height, width by loading a
            # example array from zarr to extract these dimensions.
            if self.num_frames or self.height or self.width is None:
                example_array = next(iter(zarr_root.array_values()), None)
                assert example_array is not None, (
                    "No arrays found in zarr root."
                )
                self.num_frames = (
                    example_array.shape[0]
                    if len(example_array.shape) > 0
                    else None
                )
                self.height = (
                    example_array.shape[1]
                    if len(example_array.shape) > 1
                    else None
                )
                self.width = (
                    example_array.shape[2]
                    if len(example_array.shape) > 2
                    else None
                )
                if self.verbose:
                    logger.debug(
                        f"Extracted video dimensions from zarr: "
                        f"{self.num_frames} frames, "
                        f"{self.width}x{self.height}"
                    )
            # Check which indices are empty
            self.frame_indices_dict[label_name] = get_annotated_frames(
                self.zarr_dict[label_name]
            )

    def create_tracking_dict(self):
        """Loop over the zarr dictionary and all masks to get centroids.

        Extracts the centroid of each mask. This throws an error if more
        than one region is detected per frame.

        """
        for label, masks in self.zarr_dict.items():
            self.tracking_dict[label] = {}
            if self.verbose:
                logger.info(f"Calculating centroids for {label}...")

            df_list = []
            for frame_idx, m in tqdm(
                enumerate(masks),
                disable=not self.verbose,
                total=masks.shape[0],
            ):
                if np.any(m):
                    props = regionprops(m.astype(int))
                    if len(props) > 1:
                        # For now, taking the largest
                        logger.warning(
                            f"Multiple regions found for {label} at "
                            f"frame idx {frame_idx}. Using the "
                            f"largest one."
                        )
                        largest_region = max(props, key=lambda x: x.area)
                        centroid = largest_region.centroid
                    elif len(props) == 1:
                        centroid = props[0].centroid
                    else:  # Should not happen if np.any(m) is true,
                        # but for safety
                        centroid = (np.nan, np.nan)
                    df_list.append(
                        {
                            "frame": frame_idx,
                            "y": centroid[0],
                            "x": centroid[1],
                        }
                    )
                else:
                    df_list.append(
                        {"frame": frame_idx, "y": np.nan, "x": np.nan}
                    )

            df = pd.DataFrame(df_list)
            df["x"] = df["x"].astype(float)
            df["y"] = df["y"].astype(float)
            self.tracking_dict[label] = df

    def __repr__(self) -> str:
        """Return a summary string with frame count and dimensions."""
        if (
            self.num_frames is not None
            and self.width is not None
            and self.height is not None
        ):
            return (
                f"Annotations\n{self.num_frames} frames, "
                f"{self.width}x{self.height}"
            )
        else:
            return "Annotations\n"

    def __str__(self) -> str:
        """Return a summary string with frame count and dimensions."""
        if (
            self.num_frames is not None
            and self.width is not None
            and self.height is not None
        ):
            return (
                f"Annotations\n{self.num_frames} frames, "
                f"{self.width}x{self.height}"
            )
        else:
            return "Annotations\n"
