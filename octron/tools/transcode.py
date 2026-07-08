"""OCTRON video transcoding tool.

Transcodes video files and multi-frame TIFF stacks to MP4 (H.264) using ffmpeg.

The per-input work lives in the GUI-free :func:`transcode_one` helper so the CLI
(:func:`run_transcode`) and the napari reader dialog (``octron/reader.py``) share
one implementation. Encoder selection, codec arguments, and the even-dimension
filter come from :mod:`octron.tools._ffmpeg`, which is also used by the render
pipeline.
"""

import subprocess
import time
from pathlib import Path

from loguru import logger

from octron.tools._ffmpeg import (
    EVEN_DIM_YUV420P,
    detect_h264_encoder,
    h264_codec_args,
)

VIDEO_EXTENSIONS = {
    ".avi",
    ".mov",
    ".mj2",
    ".mpg",
    ".mpeg",
    ".mjpeg",
    ".mjpg",
    ".wmv",
    ".mp4",
    ".mkv",
    ".mts",
    ".tif",
    ".tiff",
}

TIFF_EXTENSIONS = {".tif", ".tiff"}


def _load_tiff_as_rgb(path):
    """Read a multi-frame TIFF into an RGB ``(frames, Y, X, 3)`` uint8 array.

    Grayscale and 1–4 channel stacks are mapped to RGB; intensities are
    normalised to uint8 while preserving relative scale. A time axis (T) is
    preferred as the frame axis; a Z-stack is used when there is no time axis.

    Parameters
    ----------
    path : Path
        Path to the TIFF file.

    Returns
    -------
    tuple or None
        ``(stack, frame_count, height, width)`` on success, or ``None`` (after
        logging the reason) for unsupported inputs (single-frame/2D TIFFs,
        ambiguous T+Z stacks, read failures, or missing numpy/tifffile).

    """
    try:
        import numpy as np
        import tifffile
    except ImportError as e:
        logger.error(f"TIFF transcoding requires numpy+tifffile: {e}")
        return None

    def _to_uint8(arr):
        """Normalise array to uint8, preserving relative intensities."""
        if arr.dtype == np.uint8:
            return arr
        smin, smax = float(arr.min()), float(arr.max())
        if smax > smin:
            return (
                (arr.astype(np.float32) - smin) / (smax - smin) * 255
            ).astype(np.uint8)
        return np.zeros_like(arr, dtype=np.uint8)

    try:
        with tifffile.TiffFile(str(path)) as tif:
            series = tif.series[0]
            axes = series.axes  # e.g. "TCYX", "TYX", "TZYXC"
            stack = series.asarray()
            # Build sizes from axes + shape directly.
            # series.sizes can be unreliable across tifffile versions.
            sizes = dict(zip(axes, stack.shape))
    except Exception as e:
        logger.error(f"Failed to read TIFF '{path.name}': {e}")
        return None

    n_t = sizes.get("T", 0)
    n_z = sizes.get("Z", 0)
    n_c = sizes.get("C", 0)

    logger.info(
        f"TIFF detected: axes='{axes}' shape={stack.shape} dtype={stack.dtype} "
        f"| T={n_t} Z={n_z} C={n_c} "
        f"H={sizes.get('Y', '?')} W={sizes.get('X', '?')} "
        f"| {path.name}"
    )

    # Reject TIFFs with both a time AND a Z axis — ambiguous for video conversion
    if n_t > 0 and n_z > 0:
        logger.warning(
            f"Skipped '{path.name}': TIFF contains both a time axis "
            f"(T={n_t}) and a Z axis (Z={n_z}) (axes='{axes}'). "
            f"Cannot determine intended frame order for video conversion."
        )
        return None

    # Reject single-frame / 2D-only images
    if n_t >= 2:
        frame_key = "T"
    elif n_z >= 2:
        frame_key = "Z"
        logger.info(
            f"No time axis; treating Z-stack ({n_z} slices) as frames."
        )
    else:
        logger.warning(
            f"Skipped '{path.name}': single-frame or 2D TIFF "
            f"(axes='{axes}'). Only multi-frame TIFFs are supported."
        )
        return None

    # Reorder axes to canonical order:
    # (frame_key, [unknown extras,] [C,] Y, X)
    axes_list = list(axes)
    known = {"T", "Z", "C", "Y", "X"}
    extra = [a for a in axes_list if a not in known]

    target_order = (
        [frame_key] + extra + (["C"] if n_c > 0 else []) + ["Y", "X"]
    )

    perm = [axes_list.index(a) for a in target_order]
    if perm != list(range(stack.ndim)):
        stack = stack.transpose(perm)

    # Flatten any extra leading dims into the frames axis.
    # The last `trailing` dims are always (C,) Y, X.
    trailing = 3 if n_c > 0 else 2
    n_leading = stack.ndim - trailing
    if n_leading > 1:
        n_frames = int(np.prod(stack.shape[:n_leading]))
        stack = stack.reshape(n_frames, *stack.shape[n_leading:])
        logger.info(f"Flattened leading axes → {n_frames} frames.")

    # Move C from position 1 to last → (frames, Y, X, C)
    if n_c > 0:
        stack = np.moveaxis(stack, 1, -1)

    # Map channels to RGB (frames, Y, X, 3)
    if n_c == 0:
        # Grayscale: (frames, Y, X) → (frames, Y, X, 3)
        stack = _to_uint8(stack)
        stack = np.repeat(stack[..., np.newaxis], 3, axis=-1)
    elif n_c == 1:
        stack = _to_uint8(stack[..., 0])
        stack = np.repeat(stack[..., np.newaxis], 3, axis=-1)
    elif n_c == 2:
        # Map to R/G channels, B = 0
        stack = _to_uint8(stack)
        zeros = np.zeros((*stack.shape[:3], 1), dtype=np.uint8)
        stack = np.concatenate([stack, zeros], axis=-1)
        logger.info("2-channel TIFF mapped to R/G channels (B=0).")
    elif n_c == 3:
        stack = _to_uint8(stack)
    elif n_c == 4:
        stack = _to_uint8(stack[..., :3])  # drop alpha
    else:
        logger.warning(
            f"'{path.name}': {n_c} channels detected; using first 3 as RGB."
        )
        stack = _to_uint8(stack[..., :3])

    frame_count, height, width, _ = stack.shape
    return stack, frame_count, height, width


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

    stack_bytes = None
    if is_tiff:
        loaded = _load_tiff_as_rgb(input_path)
        if loaded is None:
            return False
        stack, frame_count, height, width = loaded
        out_fps = fps if fps is not None else 20.0
        logger.info(
            f"Transcoding TIFF: {frame_count} frames ({width}\u00d7{height}) "
            f"@ {out_fps} fps | '{input_path.name}'"
        )
        cmd = ["ffmpeg"]
        if overwrite:
            cmd.append("-y")
        cmd += [
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(out_fps),
            "-i",
            "-",
            *codec_args,
            "-an",  # raw RGB frames carry no audio
            "-vf",
            EVEN_DIM_YUV420P,
            str(output_path),
        ]
        stack_bytes = stack.tobytes()
    else:
        if fps is not None:
            logger.info(
                f"Transcoding video: source fps reinterpreted as {fps} fps "
                f"(faster playback) | '{input_path.name}'"
            )
        else:
            logger.info(
                f"Transcoding video: keeping source fps | '{input_path.name}'"
            )
        cmd = ["ffmpeg"]
        if overwrite:
            cmd.append("-y")
        # -r before -i reinterprets the source timestamps at the given fps,
        # changing playback speed without duplicating frames.
        if fps is not None:
            cmd += ["-r", str(fps)]
        cmd += ["-i", str(input_path), *codec_args]
        if keep_audio:
            cmd += ["-c:a", "aac", "-b:a", "128k"]
        else:
            cmd += ["-an"]
        cmd += ["-vf", EVEN_DIM_YUV420P, str(output_path)]

    t0 = time.time()
    try:
        subprocess.run(
            cmd,
            input=stack_bytes,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        stderr_msg = (
            e.stderr.decode("utf-8", errors="ignore").strip()
            if e.stderr
            else ""
        )
        if stderr_msg:
            logger.error(
                f"Failed to transcode '{input_path.name}': {stderr_msg.splitlines()[-1]}"
            )
        else:
            logger.error(f"Failed to transcode '{input_path.name}': {e}")
        return False

    elapsed = time.time() - t0
    in_mb = input_path.stat().st_size / 1_048_576
    out_mb = output_path.stat().st_size / 1_048_576
    reduction = 100 * (1 - out_mb / in_mb) if in_mb > 0 else 0
    logger.info(
        f"Transcoded '{input_path.name}' in {elapsed:.1f}s | "
        f"{in_mb:.1f} MB → {out_mb:.1f} MB ({reduction:.0f}% smaller)"
    )
    return True


def run_transcode(
    videos,
    output_path=None,
    crf=23,
    overwrite=False,
    fps=None,
    keep_audio=True,
):
    """Transcode one or more video files (or multi-frame TIFF stacks) to MP4 (H.264).

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
                f for f in v.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
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
