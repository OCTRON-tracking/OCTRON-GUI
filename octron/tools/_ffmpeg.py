"""
Shared ffmpeg helpers for OCTRON's video tools.

Centralises the H.264 encoder selection, codec-argument construction, and the
even-dimension scaling filter so the transcode path (``octron/tools/transcode.py``
and the napari reader dialog) and the render pipeline (``octron/tools/render.py``)
all agree on how to drive ffmpeg.

Imports are deliberately light (stdlib only) so this is cheap to import from the
CLI, the GUI reader, and core.
"""

import shutil
import subprocess


# Even-dimension + yuv420p filter.
#
# H.264 (libx264/h264_nvenc) with yuv420p requires even width and height; the
# ``trunc(.../2)*2`` rounds each dimension down to the nearest even number so
# odd-dimension inputs do not make ffmpeg fail.
EVEN_DIM_YUV420P = "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p"


def detect_h264_encoder(prefer_hardware=True):
    """Return an available H.264 encoder: ``'h264_nvenc'`` or ``'libx264'``.

    ffmpeg is a hard requirement for every OCTRON encode path, so this raises
    rather than returning a sentinel that each caller has to re-check.

    Parameters
    ----------
    prefer_hardware : bool
        When True (the default, used by the render pipeline) prefer the hardware
        encoder ``h264_nvenc`` for speed, falling back to ``libx264``. When
        False (used by the transcode path) prefer ``libx264`` for reproducible,
        broadly-compatible output, falling back to ``h264_nvenc`` only if
        libx264 is unavailable.

    Returns
    -------
    str
        ``'h264_nvenc'`` or ``'libx264'``, according to availability and
        ``prefer_hardware``.

    Raises
    ------
    RuntimeError
        If ffmpeg is not on PATH, or is present but exposes no usable H.264
        encoder (neither h264_nvenc nor libx264).
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is required but was not found on PATH. Please install ffmpeg."
        )
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-v", "quiet"],
            capture_output=True, text=True, timeout=5,
        )
        text = result.stdout + result.stderr
        candidates = (
            ["h264_nvenc", "libx264"] if prefer_hardware else ["libx264", "h264_nvenc"]
        )
        for name in candidates:
            if name in text:
                return name
    except Exception:
        pass
    raise RuntimeError(
        "ffmpeg is installed but exposes no usable H.264 encoder "
        "(neither h264_nvenc nor libx264). Please install an ffmpeg build "
        "with libx264."
    )


def h264_codec_args(encoder, crf=23, preset="superfast"):
    """Build the ffmpeg ``-c:v`` codec arguments for the given H.264 encoder.

    Parameters
    ----------
    encoder : str
        ``'h264_nvenc'`` or ``'libx264'`` (as returned by
        :func:`detect_h264_encoder`).
    crf : int
        Quality target (0–51, lower is better). For libx264 this is passed as
        ``-crf``; for h264_nvenc, which has no CRF mode, it is passed as the
        constant-quality ``-cq`` value.
    preset : str
        libx264 preset name (e.g. ``'superfast'``, ``'fast'``). Ignored for
        h264_nvenc, whose x264-style preset names are invalid; nvenc always
        uses its own ``p4`` preset.

    Returns
    -------
    list of str
        The ``-c:v ...`` argument list to splice into an ffmpeg command.
    """
    if encoder == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(crf)]
    return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
