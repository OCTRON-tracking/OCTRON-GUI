"""Helpers for staging prediction output in a local cache directory and moving
completed results to their final destination.

Used by YOLO_octron.predict_batch when a prediction cache directory is
configured (config.yaml prediction_cache_dir or an explicit override).
Writing zarr to a local SSD/NVMe cache and moving completed video folders to the
final destination avoids zarr atomic-write failures on SMB/network shares and
keeps remote I/O to a single sequential move per video.
"""

import os
import shutil
from pathlib import Path


def is_network_path(path) -> bool:
    r"""Return True for a UNC / network-share path (\\server\share or //server/share).

    Advisory only — used to warn the user, never to change behavior.
    """
    s = str(os.fspath(path)).replace("\\", "/")
    return s.startswith("//")


def move_prediction_folder(src_dir, dst_dir) -> None:
    """Move a completed prediction folder from src_dir to dst_dir.

    Any existing dst_dir is replaced. The parent of dst_dir is created
    if needed. Used to move one video's prediction folder out of the local
    cache to its final destination once that video is complete.

    Parameters
    ----------
    src_dir : str or Path
        The video's prediction folder inside the local cache.
    dst_dir : str or Path
        The final destination folder
        (e.g. <output>/octron_predictions/<video>_<tracker>).

    """
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.parent.mkdir(parents=True, exist_ok=True)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.move(str(src_dir), str(dst_dir))
