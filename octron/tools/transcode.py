"""
OCTRON video transcoding tool.

Transcodes video files and multi-frame TIFF stacks to MP4 (H.264) using ffmpeg.

The per-input work lives in the GUI-free :func:`transcode_one` helper so the CLI
(:func:`run_transcode`) and the napari reader dialog (``octron/reader.py``) share
one implementation. Encoder selection, codec arguments, and the even-dimension
filter come from :mod:`octron.tools._ffmpeg`, which is also used by the render
pipeline.
"""

import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger

from octron.tools._ffmpeg import (
    EVEN_DIM_YUV420P,
    _FfmpegWriter,
    detect_h264_encoder,
    h264_codec_args,
)


VIDEO_EXTENSIONS = {
    ".avi", ".mov", ".mj2", ".mpg", ".mpeg", ".mjpeg", ".mjpg",
    ".wmv", ".mp4", ".mkv", ".mts", ".tif", ".tiff",
}

TIFF_EXTENSIONS = {".tif", ".tiff"}


class _TiffPlan:
    """A memory-bounded plan for streaming a multi-frame TIFF as RGB frames.

    ``arr`` is a lazy/memmap view of the stack reordered to canonical
    ``(frames, [extra...,] [C,] Y, X)`` order (a transpose is a cheap view, so
    no data is copied). Frames are materialised one at a time by
    :func:`_iter_rgb_frames`; ``gmin``/``gmax`` are the stack-global intensity
    bounds used to normalise every frame to uint8 consistently. Call
    :meth:`close` to release the underlying file when done.
    """

    def __init__(self, tif, arr, n_leading, n_c, frame_count, height, width, gmin, gmax):
        self._tif = tif
        self.arr = arr
        self.n_leading = n_leading
        self.n_c = n_c
        self.frame_count = frame_count
        self.height = height
        self.width = width
        self.gmin = gmin
        self.gmax = gmax

    def close(self):
        _safe_close(self._tif)


def _safe_close(tif):
    try:
        tif.close()
    except Exception:
        pass


def _to_uint8_frame(arr, gmin, gmax, np):
    """Normalise one frame to uint8 using the stack-global ``gmin``/``gmax``.

    Values are already within ``[gmin, gmax]`` (global bounds), so no clipping
    is needed. uint8 input is passed through unchanged (``gmin``/``gmax`` are
    ``None`` in that case).
    """
    if arr.dtype == np.uint8:
        return arr
    if gmax is not None and gmax > gmin:
        return ((arr.astype(np.float32) - gmin) / (gmax - gmin) * 255.0).astype(np.uint8)
    return np.zeros(arr.shape, dtype=np.uint8)


def _frame_to_rgb(frame, n_c, gmin, gmax, np):
    """Map one raw frame to an ``(Y, X, 3)`` uint8 RGB array.

    ``frame`` is ``(Y, X)`` when ``n_c == 0`` and ``(C, Y, X)`` otherwise. The
    channel mapping mirrors the previous whole-array behaviour: grayscale/1ch
    are broadcast to RGB, 2ch → R/G (B=0), 3ch → RGB, 4+ch use the first 3.
    """
    if n_c == 0:
        g = _to_uint8_frame(frame, gmin, gmax, np)            # (Y, X)
        return np.repeat(g[..., np.newaxis], 3, axis=-1)
    frame = np.moveaxis(frame, 0, -1)                         # (C, Y, X) -> (Y, X, C)
    if n_c == 1:
        g = _to_uint8_frame(frame[..., 0], gmin, gmax, np)
        return np.repeat(g[..., np.newaxis], 3, axis=-1)
    if n_c == 2:
        g = _to_uint8_frame(frame, gmin, gmax, np)            # (Y, X, 2)
        zeros = np.zeros((*g.shape[:2], 1), dtype=np.uint8)
        return np.concatenate([g, zeros], axis=-1)
    if n_c == 3:
        return _to_uint8_frame(frame, gmin, gmax, np)
    return _to_uint8_frame(frame[..., :3], gmin, gmax, np)    # 4+ channels: drop extras


def _read_tiff_plan(path):
    """Open a multi-frame TIFF and plan a memory-bounded RGB frame stream.

    The stack is read disk-backed (memmap) rather than into RAM, so very large
    stacks (thousands of frames) do not exhaust memory. A time axis (T) is
    preferred as the frame axis; a Z-stack is used when there is no time axis;
    a generic image-sequence axis ('I') is treated as the time axis when a Y
    and X image plane are also present. Grayscale and 1–4 channel stacks are
    mapped to RGB, with intensities normalised to uint8 using stack-global
    min/max (computed here so a streamed encode keeps consistent brightness).

    Parameters
    ----------
    path : Path
        Path to the TIFF file.

    Returns
    -------
    _TiffPlan or None
        A plan whose ``arr`` is a lazy view reordered to
        ``(frames, [extra...,] [C,] Y, X)``, or ``None`` (after logging the
        reason) for unsupported inputs (single-frame/2D TIFFs, ambiguous T+Z
        stacks, read failures, or missing numpy/tifffile). Call ``close()`` on
        the returned plan when done.
    """
    try:
        import numpy as np
        import tifffile
    except ImportError as e:
        logger.error(f"TIFF transcoding requires numpy+tifffile: {e}")
        return None

    try:
        tif = tifffile.TiffFile(str(path))
    except Exception as e:
        logger.error(f"Failed to read TIFF '{path.name}': {e}")
        return None

    try:
        series = tif.series[0]
        axes = series.axes   # e.g. "TCYX", "TYX", "TZYXC"
        # Read disk-backed so huge stacks are not materialised in RAM; fall
        # back to an in-memory read only if a memmap cannot be created.
        try:
            arr = series.asarray(out="memmap")
        except Exception as e:
            logger.debug(
                f"TIFF memmap unavailable for '{path.name}' ({e}); reading into RAM."
            )
            arr = series.asarray()
        # Build sizes from axes + shape directly.
        # series.sizes can be unreliable across tifffile versions.
        sizes = dict(zip(axes, arr.shape))
    except Exception as e:
        _safe_close(tif)
        logger.error(f"Failed to read TIFF '{path.name}': {e}")
        return None

    # tifffile labels a generic image-sequence axis 'I' (e.g. a plain stack of
    # 2D frames saved as axes='IYX'). When such an 'I' axis accompanies a
    # recognised Y and X image plane and there is no explicit time axis, treat
    # it as the time (T) / frame axis for video conversion.
    if 'T' not in sizes and 'I' in sizes and 'Y' in sizes and 'X' in sizes:
        logger.info(
            f"Interpreting generic 'I' axis (size {sizes['I']}) as the time (T) "
            f"axis for video conversion (axes='{axes}')."
        )
        axes = axes.replace('I', 'T')
        sizes = dict(zip(axes, arr.shape))

    n_t = sizes.get('T', 0)
    n_z = sizes.get('Z', 0)
    n_c = sizes.get('C', 0)

    logger.info(
        f"TIFF detected: axes='{axes}' shape={arr.shape} dtype={arr.dtype} "
        f"| T={n_t} Z={n_z} C={n_c} "
        f"H={sizes.get('Y', '?')} W={sizes.get('X', '?')} "
        f"| {path.name}"
    )

    # Reject TIFFs with both a time AND a Z axis — ambiguous for video conversion
    if n_t > 0 and n_z > 0:
        _safe_close(tif)
        logger.warning(
            f"Skipped '{path.name}': TIFF contains both a time axis "
            f"(T={n_t}) and a Z axis (Z={n_z}) (axes='{axes}'). "
            f"Cannot determine intended frame order for video conversion."
        )
        return None

    # Reject single-frame / 2D-only images
    if n_t >= 2:
        frame_key = 'T'
    elif n_z >= 2:
        frame_key = 'Z'
        logger.info(f"No time axis; treating Z-stack ({n_z} slices) as frames.")
    else:
        _safe_close(tif)
        logger.warning(
            f"Skipped '{path.name}': single-frame or 2D TIFF "
            f"(axes='{axes}'). Only multi-frame TIFFs are supported."
        )
        return None

    # Reorder axes to canonical order:
    # (frame_key, [unknown extras,] [C,] Y, X)
    # transpose() on a memmap is a view (no copy). We deliberately do NOT
    # reshape the whole array to flatten leading dims (that would force a
    # full-size copy and defeat the streaming); _iter_rgb_frames walks the
    # leading dims with np.ndindex instead.
    axes_list = list(axes)
    known = {'T', 'Z', 'C', 'Y', 'X'}
    extra = [a for a in axes_list if a not in known]

    target_order = [frame_key] + extra + (['C'] if n_c > 0 else []) + ['Y', 'X']

    perm = [axes_list.index(a) for a in target_order]
    if perm != list(range(arr.ndim)):
        arr = arr.transpose(perm)

    trailing = 3 if n_c > 0 else 2
    n_leading = arr.ndim - trailing
    frame_count = int(np.prod(arr.shape[:n_leading]))
    if n_leading > 1:
        logger.info(f"Flattened leading axes → {frame_count} frames.")
    height, width = int(arr.shape[-2]), int(arr.shape[-1])
    if n_c == 2:
        logger.info("2-channel TIFF mapped to R/G channels (B=0).")
    elif n_c > 4:
        logger.warning(f"'{path.name}': {n_c} channels detected; using first 3 as RGB.")

    # Precompute stack-global min/max over the channels that will actually be
    # used, so per-frame uint8 normalisation is consistent across a streamed
    # encode (matches the old whole-array behaviour). Reductions over a memmap
    # stream through the data without materialising it. Skipped for uint8
    # (passed through unchanged). The channel axis is at -3 once n_c > 0.
    gmin = gmax = None
    if arr.dtype != np.uint8:
        if n_c == 0:
            sel = arr
        elif n_c == 1:
            sel = arr[..., 0:1, :, :]
        elif n_c >= 4:
            sel = arr[..., 0:3, :, :]   # first 3 channels (drop alpha/extras)
        else:  # 2 or 3 channels: use all
            sel = arr
        gmin, gmax = float(sel.min()), float(sel.max())

    return _TiffPlan(tif, arr, n_leading, n_c, frame_count, height, width, gmin, gmax)


def _iter_rgb_frames(plan):
    """Yield successive ``(Y, X, 3)`` uint8 RGB frames from a planned TIFF.

    One frame is materialised at a time (the source stays memmapped), so peak
    memory is bounded by a single frame regardless of stack length.
    """
    import numpy as np

    arr = plan.arr
    leading_shape = arr.shape[: plan.n_leading]
    for idx in np.ndindex(*leading_shape):
        frame = np.asarray(arr[idx])
        yield _frame_to_rgb(frame, plan.n_c, plan.gmin, plan.gmax, np)


def _load_tiff_as_rgb(path):
    """Read a multi-frame TIFF into an RGB ``(frames, Y, X, 3)`` uint8 array.

    Thin convenience wrapper over :func:`_read_tiff_plan` + :func:`_iter_rgb_frames`
    that materialises the whole stack in memory. :func:`transcode_one` streams
    frames instead (bounded memory) and does not use this; it remains for
    small-stack/programmatic/test callers.

    Returns
    -------
    tuple or None
        ``(stack, frame_count, height, width)`` on success, or ``None`` for
        unsupported inputs (see :func:`_read_tiff_plan`).
    """
    plan = _read_tiff_plan(path)
    if plan is None:
        return None
    try:
        import numpy as np
        stack = np.stack(list(_iter_rgb_frames(plan)), axis=0)
    finally:
        plan.close()
    return stack, plan.frame_count, plan.height, plan.width


def transcode_one(
    input_path,
    output_path,
    *,
    crf=23,
    overwrite=False,
    fps=None,
    keep_audio=True,
    encoder=None,
):
    """Transcode a single video file or multi-frame TIFF stack to MP4 (H.264).

    This is the GUI-free core shared by the CLI and the napari reader dialog.
    It builds and runs one ffmpeg command; it does not handle directory
    expansion or skip-if-exists (callers decide that).

    Parameters
    ----------
    input_path : str or Path
        Source video file or multi-frame TIFF.
    output_path : str or Path
        Destination ``.mp4`` file.
    crf : int
        Constant Rate Factor (0–51). Lower means better quality. Default 23.
    overwrite : bool
        Pass ffmpeg's ``-y`` so it overwrites an existing output. Default False.
    fps : float, optional
        Output framerate. For videos this reinterprets the source timestamps
        (changing playback speed); for TIFF stacks it sets the playback fps.
        Defaults to source fps for videos and 20 fps for TIFFs.
    keep_audio : bool
        Re-encode audio to AAC (128k) for video inputs. Default True. TIFF
        inputs are raw frames with no audio, so this is ignored for them.
    encoder : str, optional
        H.264 encoder to use (``'libx264'`` or ``'h264_nvenc'``). When not
        provided, libx264 is selected (preferred over hardware nvenc) for
        reproducible, broadly-compatible output.

    Returns
    -------
    bool
        True if the output was written, False if the input was skipped
        (unsupported TIFF) or ffmpeg failed.

    Raises
    ------
    RuntimeError
        If no usable H.264 encoder is available and ``encoder`` is not given.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if encoder is None:
        # Transcode standardises on libx264 (-preset superfast) for reproducible,
        # widely-compatible output; nvenc is only used if libx264 is unavailable.
        encoder = detect_h264_encoder(prefer_hardware=False)
    codec_args = h264_codec_args(encoder, crf=crf, preset="superfast")
    is_tiff = input_path.suffix.lower() in TIFF_EXTENSIONS

    if is_tiff:
        # TIFF stacks are streamed frame-by-frame (bounded memory) so very
        # large stacks do not exhaust RAM.
        return _transcode_tiff(
            input_path,
            output_path,
            codec_args,
            encoder,
            overwrite=overwrite,
            fps=fps,
        )

    # --- Video inputs -------------------------------------------------------
    if fps is not None:
        logger.info(
            f"Transcoding video: source fps reinterpreted as {fps} fps "
            f"(faster playback) | '{input_path.name}'"
        )
    else:
        logger.info(f"Transcoding video: keeping source fps | '{input_path.name}'")
    cmd = ["ffmpeg"]
    if overwrite:
        cmd.append("-y")
    # -r before -i reinterprets the source timestamps at the given fps,
    # changing playback speed without duplicating frames.
    if fps is not None:
        cmd += ["-r", str(fps)]
    cmd += ["-i", str(input_path), *codec_args]
    if keep_audio:
        # -map 0:a? makes audio optional: ffmpeg silently skips audio when
        # the input has no audio stream (e.g. OCTRON-rendered MP4s), but
        # preserves and re-encodes audio when it is present.
        cmd += ["-map", "0:a?", "-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += ["-vf", EVEN_DIM_YUV420P, str(output_path)]

    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.decode("utf-8", errors="ignore").strip() if e.stderr else ""
        if stderr_msg:
            logger.error(f"Failed to transcode '{input_path.name}': {stderr_msg.splitlines()[-1]}")
        else:
            logger.error(f"Failed to transcode '{input_path.name}': {e}")
        return False

    _log_transcode_success(input_path, output_path, time.time() - t0)
    return True


def _log_transcode_success(input_path, output_path, elapsed):
    """Log a one-line transcode summary (elapsed time + size reduction)."""
    in_mb = input_path.stat().st_size / 1_048_576
    out_mb = output_path.stat().st_size / 1_048_576
    reduction = 100 * (1 - out_mb / in_mb) if in_mb > 0 else 0
    logger.info(
        f"Transcoded '{input_path.name}' in {elapsed:.1f}s | "
        f"{in_mb:.1f} MB \u2192 {out_mb:.1f} MB ({reduction:.0f}% smaller)"
    )


def _transcode_tiff(input_path, output_path, codec_args, encoder, *, overwrite, fps):
    """Stream a multi-frame TIFF to MP4, one RGB frame at a time.

    Frames are read disk-backed and piped to ffmpeg incrementally (via
    :class:`_FfmpegWriter`), so peak memory stays at a single frame -- this is
    what lets very large stacks (thousands of frames) transcode without
    exhausting RAM. Returns True on success, or False if the TIFF is
    unsupported or ffmpeg fails.
    """
    plan = _read_tiff_plan(input_path)
    if plan is None:
        return False
    out_fps = fps if fps is not None else 20.0
    logger.info(
        f"Transcoding TIFF: {plan.frame_count} frames "
        f"({plan.width}\u00d7{plan.height}) @ {out_fps} fps | '{input_path.name}'"
    )
    cmd = ["ffmpeg"]
    if overwrite:
        cmd.append("-y")
    cmd += [
        "-f", "rawvideo",
        "-pixel_format", "rgb24",
        "-video_size", f"{plan.width}x{plan.height}",
        "-framerate", str(out_fps),
        "-i", "-",
        *codec_args,
        "-an",  # raw RGB frames carry no audio
        "-vf", EVEN_DIM_YUV420P,
        str(output_path),
    ]
    t0 = time.time()
    stderr_file = tempfile.TemporaryFile(mode="w+b")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=stderr_file)
    except FileNotFoundError:
        stderr_file.close()
        plan.close()
        logger.error(
            f"Failed to transcode '{input_path.name}': ffmpeg not found on PATH."
        )
        return False

    # Pipe frames incrementally; peak memory is one frame, not the whole stack.
    writer = _FfmpegWriter(proc, stderr_file, encoder, output_path)
    try:
        for rgb in _iter_rgb_frames(plan):
            writer.write(rgb.tobytes())
        writer.close()
    except Exception as e:
        # A dead ffmpeg pipe surfaces as RuntimeError from the writer; a
        # mid-stream frame read/decode error surfaces here too. Kill ffmpeg
        # (close(check=False)) and report a skipped input rather than crash.
        writer.close(check=False)
        logger.error(f"Failed to transcode '{input_path.name}': {e}")
        return False
    finally:
        plan.close()

    _log_transcode_success(input_path, output_path, time.time() - t0)
    return True


def run_transcode(
    videos,
    output_path=None,
    crf=23,
    overwrite=False,
    fps=None,
    keep_audio=True,
):
    """
    Transcode one or more video files (or multi-frame TIFF stacks) to MP4 (H.264).

    Parameters
    ----------
    videos : str, Path, or list
        One or more video/TIFF file paths, or a directory containing them.
    output_path : str or Path, optional
        Output directory. Defaults to a ``mp4_transcoded/`` subfolder next to
        the first input file (or inside the input directory).
    crf : int
        Constant Rate Factor (0–51). Lower means better quality. Default 23.
    overwrite : bool
        Overwrite existing output files. Default False.
    fps : float, optional
        Output framerate (see :func:`transcode_one`). Defaults to source fps
        for videos and 20 fps for TIFF stacks.
    keep_audio : bool
        Keep (re-encode to AAC) the audio track of video inputs. Default True.
    """
    if not isinstance(videos, list):
        videos = [videos]
    videos = [Path(v) for v in videos]

    # Expand any directories
    expanded = []
    for v in videos:
        if v.is_dir():
            found = sorted(
                f for f in v.iterdir()
                if f.suffix.lower() in VIDEO_EXTENSIONS
            )
            if not found:
                print(f"No video files found in directory: {v}")
            else:
                print(f"Found {len(found)} video(s) in {v}")
                expanded.extend(found)
        else:
            expanded.append(v)
    videos = expanded

    if not videos:
        print("No videos to transcode.")
        return

    # Resolve output directory
    if output_path is None:
        # Place alongside the first input
        first = videos[0]
        base = first.parent if first.is_file() else first
        output_dir = base / "mp4_transcoded"
    else:
        output_dir = Path(output_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Transcoding {len(videos)} video(s) → {output_dir}  (CRF={crf})")

    # Detect the encoder once up front so a missing ffmpeg fails with one clear
    # message instead of per-file errors.  Transcode standardises on libx264
    # (see transcode_one) rather than hardware nvenc.
    try:
        encoder = detect_h264_encoder(prefer_hardware=False)
    except RuntimeError as e:
        print(f"  {e}")
        return

    successful = 0
    for i, video in enumerate(videos, 1):
        out = output_dir / f"{video.stem}.mp4"
        print(f"  [{i}/{len(videos)}] {video.name} → {out.name}")

        if not overwrite and out.exists():
            print("(skipped — already exists)")
            continue

        if transcode_one(
            video,
            out,
            crf=crf,
            overwrite=overwrite,
            fps=fps,
            keep_audio=keep_audio,
            encoder=encoder,
        ):
            successful += 1

    print(f"Done. {successful}/{len(videos)} transcoded successfully.")
