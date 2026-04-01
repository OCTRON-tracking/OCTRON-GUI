"""
Tests for pure utility functions in octron/tools/predict.py.
No heavy dependencies (torch, cv2, YOLO) required.

Covered
-------
_is_network_path()
    UNC paths with forward slashes (//server/share) → True
    UNC paths with backslashes (\\\\server\\share) → True
    Mixed-slash UNC paths → True
    Local absolute Unix paths (/home/user/...) → False
    Windows drive paths (C:\\...) → False
    Relative paths → False
    Single-slash paths (/tmp/...) → False
    Accepts pathlib.Path objects for both network and local paths
"""

from pathlib import Path
from octron.tools.predict import _is_network_path


# ---------------------------------------------------------------------------
# _is_network_path
# ---------------------------------------------------------------------------

def test_is_network_path_unc_forward_slashes():
    assert _is_network_path("//server/share") is True


def test_is_network_path_unc_backslashes():
    assert _is_network_path("\\\\server\\share") is True


def test_is_network_path_unc_mixed_slashes():
    assert _is_network_path("\\\\server/share/subdir") is True


def test_is_network_path_local_unix():
    assert _is_network_path("/home/user/data") is False


def test_is_network_path_windows_drive():
    assert _is_network_path("C:\\Users\\data") is False


def test_is_network_path_relative():
    assert _is_network_path("data/videos") is False


def test_is_network_path_single_slash():
    assert _is_network_path("/tmp/videos") is False


def test_is_network_path_accepts_path_object():
    assert _is_network_path(Path("//server/share/data")) is True


def test_is_network_path_local_path_object():
    assert _is_network_path(Path("/home/user/data")) is False
