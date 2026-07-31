"""Tests for the cheap CSV row-count helper on YOLO_results.

``_csv_observation_count`` is the single place that encodes the tracking-CSV
header offset (fixed metadata header lines + one column-header row);
it backs the
``min_observations`` pre-filter in ``get_tracking_data``. The heavy
``YOLO_results.__init__`` is bypassed via ``__new__`` so no prediction data is
needed.
"""

from octron.yolo_octron.helpers.yolo_results import YOLO_results


def _make_results(header_lines=7):
    obj = YOLO_results.__new__(YOLO_results)
    obj.csv_header_lines = header_lines
    return obj


def _write_tracking_csv(path, data_rows, header_lines=7):
    """Write a valid tracking CSV: metadata header + column row + data rows.

    Mirrors the on-disk layout ``_csv_observation_count`` expects:
    ``header_lines`` fixed metadata lines, one column-header row, then the
    given data rows.
    """
    lines = (
        [f"# meta {i}" for i in range(header_lines)]
        + ["frame_idx,track_id,label"]
        + list(data_rows)
    )
    path.write_text("\n".join(lines) + "\n")


def test_csv_observation_count_counts_data_rows(tmp_path):
    obj = _make_results(header_lines=7)
    csv = tmp_path / "clip_track_1.csv"
    _write_tracking_csv(csv, ["0,1,a", "1,1,a", "2,1,a"], header_lines=7)
    assert obj._csv_observation_count(csv) == 3


def test_csv_observation_count_floors_at_zero(tmp_path):
    obj = _make_results(header_lines=7)
    csv = tmp_path / "clip_track_2.csv"
    csv.write_text("only one line\n")  # fewer lines than the header offset
    assert obj._csv_observation_count(csv) == 0


def test_csv_observation_count_respects_header_lines(tmp_path):
    obj = _make_results(header_lines=3)
    csv = tmp_path / "clip_track_3.csv"
    _write_tracking_csv(csv, ["0,3,a", "1,3,a"], header_lines=3)
    assert obj._csv_observation_count(csv) == 2


# ---------------------------------------------------------------------------
# Video auto-detection (_candidate_video_path)
#
# The prediction folder is named '<video_stem>_<tracker>'. The video may sit
# directly beside the folder (sibling layout) or one level up, next to the
# octron_predictions/ directory (nested layout). Both must be found. Pure
# filesystem lookup, so no real video is opened.
# ---------------------------------------------------------------------------


def test_candidate_video_path_finds_sibling(tmp_path, touch_file):
    (tmp_path / "clipA_ByteTrack").mkdir()
    video = touch_file("clipA.mp4")
    obj = YOLO_results.__new__(YOLO_results)
    obj.results_dir = tmp_path / "clipA_ByteTrack"
    assert obj._candidate_video_path() == video


def test_candidate_video_path_finds_nested(tmp_path, touch_file):
    folder = tmp_path / "octron_predictions" / "clipB_ByteTrack"
    folder.mkdir(parents=True)
    video = touch_file("clipB.mp4")
    obj = YOLO_results.__new__(YOLO_results)
    obj.results_dir = folder
    assert obj._candidate_video_path() == video


def test_candidate_video_path_none_when_missing(tmp_path):
    (tmp_path / "clipC_ByteTrack").mkdir()
    obj = YOLO_results.__new__(YOLO_results)
    obj.results_dir = tmp_path / "clipC_ByteTrack"
    assert obj._candidate_video_path() is None
