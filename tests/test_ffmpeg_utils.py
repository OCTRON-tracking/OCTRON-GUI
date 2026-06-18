"""
Tests for the shared ffmpeg helpers in octron/tools/_ffmpeg.py.

ffmpeg itself is never invoked: codec-arg construction is pure, and the
encoder-detection failure path is exercised by monkeypatching shutil.which.

Covered
-------
EVEN_DIM_YUV420P
    Exact even-dimension + yuv420p filter string.
h264_codec_args
    libx264: -preset/-crf passthrough (and defaults).
    h264_nvenc: ignores the x264 preset (fixed p4) and maps crf -> -cq.
detect_h264_encoder
    Raises RuntimeError when ffmpeg is not on PATH.
"""

import pytest

import octron.tools._ffmpeg as ff
from octron.tools._ffmpeg import EVEN_DIM_YUV420P, h264_codec_args


def test_even_dim_filter_value():
    assert EVEN_DIM_YUV420P == "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"


def test_h264_codec_args_libx264_explicit():
    assert h264_codec_args("libx264", crf=18, preset="fast") == [
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
    ]


def test_h264_codec_args_libx264_defaults():
    # Default preset is superfast, default crf 23.
    assert h264_codec_args("libx264") == [
        "-c:v", "libx264", "-preset", "superfast", "-crf", "23",
    ]


def test_h264_codec_args_nvenc_uses_cq_and_fixed_preset():
    args = h264_codec_args("h264_nvenc", crf=20, preset="fast")
    assert args == ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "20"]
    # nvenc has no CRF mode and rejects x264 preset names.
    assert "-crf" not in args
    assert "fast" not in args


def test_detect_h264_encoder_raises_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        ff.detect_h264_encoder()


class _FakeRun:
    """Stand-in for subprocess.run with a fixed `ffmpeg -encoders` listing."""
    def __init__(self, text):
        self.stdout = text
        self.stderr = ""


def test_detect_h264_encoder_preference(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ff.subprocess, "run",
        lambda *a, **k: _FakeRun("V..... h264_nvenc\nV..... libx264\n"),
    )
    # Render (default) prefers the hardware encoder.
    assert ff.detect_h264_encoder() == "h264_nvenc"
    assert ff.detect_h264_encoder(prefer_hardware=True) == "h264_nvenc"
    # Transcode prefers libx264 even when nvenc is available.
    assert ff.detect_h264_encoder(prefer_hardware=False) == "libx264"


def test_detect_h264_encoder_falls_back_to_nvenc_without_libx264(monkeypatch):
    monkeypatch.setattr(ff.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        ff.subprocess, "run",
        lambda *a, **k: _FakeRun("V..... h264_nvenc\n"),
    )
    # libx264 unavailable: prefer_hardware=False still yields a usable encoder.
    assert ff.detect_h264_encoder(prefer_hardware=False) == "h264_nvenc"
