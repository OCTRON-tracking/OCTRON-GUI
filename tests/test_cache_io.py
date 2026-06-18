"""
Tests for octron.yolo_octron.helpers.cache_io — the prediction cache staging
helpers used by core predict_batch.
"""

from octron.yolo_octron.helpers.cache_io import is_network_path, move_prediction_folder


# ---------------------------------------------------------------------------
# is_network_path (advisory only)
# ---------------------------------------------------------------------------

def test_is_network_path_detects_unc_and_posix_shares():
    assert is_network_path("//server/share/x")
    assert is_network_path("\\\\server\\share\\x")  # backslash UNC


def test_is_network_path_rejects_local_paths():
    assert not is_network_path("/local/path")
    assert not is_network_path("C:/Users/me")
    assert not is_network_path("relative/dir")


# ---------------------------------------------------------------------------
# move_prediction_folder
# ---------------------------------------------------------------------------

def test_move_prediction_folder_moves_contents(tmp_path):
    src = tmp_path / "cache" / "octron_predictions" / "vid_bytetrack"
    src.mkdir(parents=True)
    (src / "a_track_1.csv").write_text("data")
    (src / "predictions.zarr").mkdir()
    (src / "predictions.zarr" / "zarr.json").write_text("{}")

    dst = tmp_path / "final" / "octron_predictions" / "vid_bytetrack"
    move_prediction_folder(src, dst)

    assert not src.exists()
    assert (dst / "a_track_1.csv").read_text() == "data"
    assert (dst / "predictions.zarr" / "zarr.json").read_text() == "{}"


def test_move_prediction_folder_creates_parent(tmp_path):
    src = tmp_path / "cache" / "vid"
    src.mkdir(parents=True)
    (src / "f.txt").write_text("x")
    # Destination parent does not exist yet.
    dst = tmp_path / "new" / "deep" / "vid"

    move_prediction_folder(src, dst)

    assert (dst / "f.txt").read_text() == "x"


def test_move_prediction_folder_replaces_existing_dst(tmp_path):
    src = tmp_path / "cache" / "vid"
    src.mkdir(parents=True)
    (src / "new.txt").write_text("new")

    dst = tmp_path / "final" / "vid"
    dst.mkdir(parents=True)
    (dst / "old.txt").write_text("old")

    move_prediction_folder(src, dst)

    assert not src.exists()
    assert (dst / "new.txt").read_text() == "new"
    assert not (dst / "old.txt").exists()  # stale destination replaced
