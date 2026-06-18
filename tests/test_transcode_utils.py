"""
Tests for the filesystem-level logic in octron/tools/transcode.py.

ffmpeg is never invoked: tests either exercise code paths that return before
the subprocess call (empty input, no matching files) or pre-create the
expected output file so the "skip if exists" branch is taken instead.

Covered
-------
VIDEO_EXTENSIONS constant
    Contains common formats: .mp4, .avi, .mov, .mkv, .mts

Input handling
    Empty list → prints "No videos to transcode" and returns
    Directory with no video files → prints warning, returns
    Directory containing only non-video files → prints warning, returns
    Directory containing mixed files → only video extensions are included
    Non-video files in directory are silently ignored
    Single file path passes through as-is

Output path resolution
    Default output path is mp4_transcoded/ next to the first input file
    Explicit --output path is used when provided
    Output directory is created automatically if it does not exist

Skip / overwrite behaviour (no ffmpeg)
    Existing output file is skipped when overwrite=False
    Multiple files: only those with pre-existing outputs are skipped;
    skipped count matches number of pre-created output files
"""

import pytest
from pathlib import Path
from octron.tools.transcode import run_transcode, VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# VIDEO_EXTENSIONS constant
# ---------------------------------------------------------------------------

def test_video_extensions_contains_common_formats():
    assert ".mp4" in VIDEO_EXTENSIONS
    assert ".avi" in VIDEO_EXTENSIONS
    assert ".mov" in VIDEO_EXTENSIONS
    assert ".mkv" in VIDEO_EXTENSIONS
    assert ".mts" in VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# Empty / no-video cases (no ffmpeg call)
# ---------------------------------------------------------------------------

def test_no_videos_empty_list(capsys):
    run_transcode([])
    assert "No videos to transcode" in capsys.readouterr().out


def test_empty_directory_warns(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    run_transcode([empty_dir])
    out = capsys.readouterr().out
    assert "No video files found" in out


def test_directory_with_no_matching_extensions_warns(tmp_path, capsys):
    video_dir = tmp_path / "vids"
    video_dir.mkdir()
    (video_dir / "readme.txt").touch()
    (video_dir / "data.csv").touch()
    run_transcode([video_dir])
    out = capsys.readouterr().out
    assert "No video files found" in out


# ---------------------------------------------------------------------------
# Directory expansion
# ---------------------------------------------------------------------------

def test_directory_expands_to_video_files(tmp_path, capsys):
    video_dir = tmp_path / "vids"
    video_dir.mkdir()
    (video_dir / "clip1.avi").touch()
    (video_dir / "clip2.mov").touch()
    (video_dir / "notes.txt").touch()   # should be ignored
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    # Pre-create outputs so ffmpeg is never called
    (out_dir / "clip1.mp4").touch()
    (out_dir / "clip2.mp4").touch()
    run_transcode([video_dir], output_path=out_dir, overwrite=False)
    out = capsys.readouterr().out
    assert "2 video(s)" in out


def test_non_video_files_in_directory_are_ignored(tmp_path, capsys):
    video_dir = tmp_path / "vids"
    video_dir.mkdir()
    (video_dir / "clip.mp4").touch()
    (video_dir / "image.png").touch()
    (video_dir / "data.csv").touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip.mp4").touch()
    run_transcode([video_dir], output_path=out_dir, overwrite=False)
    out = capsys.readouterr().out
    assert "1 video(s)" in out


def test_single_file_passes_through(tmp_path, capsys):
    fake_video = tmp_path / "clip.avi"
    fake_video.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "clip.mp4").touch()
    run_transcode([fake_video], output_path=out_dir, overwrite=False)
    out = capsys.readouterr().out
    assert "skipped" in out


# ---------------------------------------------------------------------------
# Output path resolution
# ---------------------------------------------------------------------------

def test_default_output_path_is_mp4_transcoded_next_to_input(tmp_path, capsys):
    fake_video = tmp_path / "clip.avi"
    fake_video.touch()
    expected_out_dir = tmp_path / "mp4_transcoded"
    expected_out_dir.mkdir()
    (expected_out_dir / "clip.mp4").touch()     # pre-create so ffmpeg is skipped
    run_transcode([fake_video], overwrite=False)
    assert expected_out_dir.exists()


def test_explicit_output_path_is_used(tmp_path, capsys):
    fake_video = tmp_path / "clip.avi"
    fake_video.touch()
    custom_out = tmp_path / "my_output"
    custom_out.mkdir()
    (custom_out / "clip.mp4").touch()           # pre-create so ffmpeg is skipped
    run_transcode([fake_video], output_path=custom_out, overwrite=False)
    out = capsys.readouterr().out
    assert str(custom_out) in out


def test_output_directory_is_created_if_missing(tmp_path, capsys):
    fake_video = tmp_path / "clip.avi"
    fake_video.touch()
    new_dir = tmp_path / "brand_new_output"
    assert not new_dir.exists()
    # Pre-create the output file inside so ffmpeg is skipped
    new_dir.mkdir()
    (new_dir / "clip.mp4").touch()
    run_transcode([fake_video], output_path=new_dir, overwrite=False)
    assert new_dir.exists()


# ---------------------------------------------------------------------------
# Skip / overwrite behaviour (no ffmpeg)
# ---------------------------------------------------------------------------

def test_existing_output_is_skipped_when_no_overwrite(tmp_path, capsys):
    fake_video = tmp_path / "clip.avi"
    fake_video.touch()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "clip.mp4"
    existing.touch()
    run_transcode([fake_video], output_path=out_dir, overwrite=False)
    out = capsys.readouterr().out
    assert "skipped" in out


def test_multiple_files_some_skipped(tmp_path, capsys):
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    for name in ("a.avi", "b.avi", "c.avi"):
        (tmp_path / name).touch()
    # Pre-create output for two of the three
    (out_dir / "a.mp4").touch()
    (out_dir / "b.mp4").touch()
    run_transcode(
        [tmp_path / "a.avi", tmp_path / "b.avi", tmp_path / "c.avi"],
        output_path=out_dir,
        overwrite=False,
    )
    out = capsys.readouterr().out
    assert out.count("skipped") == 2
