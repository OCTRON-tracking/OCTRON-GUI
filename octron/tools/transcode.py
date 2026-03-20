"""
OCTRON video transcoding tool.

Transcodes video files to MP4 (H.264/libx264) using ffmpeg.
"""

import subprocess
import time
from pathlib import Path

from loguru import logger


VIDEO_EXTENSIONS = {
    ".avi", ".mov", ".mj2", ".mpg", ".mpeg", ".mjpeg", ".mjpg",
    ".wmv", ".mp4", ".mkv", ".mts",
}


def run_transcode(
    videos,
    output_path=None,
    crf=23,
    overwrite=False,
):
    """
    Transcode one or more video files to MP4 (H.264).

    Parameters
    ----------
    videos : str, Path, or list
        One or more video file paths, or a directory containing video files.
    output_path : str or Path, optional
        Output directory. Defaults to a ``mp4_transcoded/`` subfolder next to
        the first input file (or inside the input directory).
    crf : int
        Constant Rate Factor (0–51). Lower means better quality. Default 23.
    overwrite : bool
        Overwrite existing output files. Default False.
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
                logger.warning(f"No video files found in directory: {v}")
            else:
                logger.info(f"Found {len(found)} video(s) in {v}")
                expanded.extend(found)
        else:
            expanded.append(v)
    videos = expanded

    if not videos:
        logger.info("No videos to transcode.")
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

    logger.info(f"Transcoding {len(videos)} video(s) → {output_dir}  (CRF={crf})")
    successful = 0
    for i, video in enumerate(videos, 1):
        out = output_dir / f"{video.stem}.mp4"
        logger.info(f"  [{i}/{len(videos)}] {video.name} → {out.name}")

        if not overwrite and out.exists():
            logger.info("(skipped — already exists)")
            continue

        cmd = [
            "ffmpeg",
            "-i", str(video),
            "-c:v", "libx264",
            "-preset", "superfast",
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            str(out),
        ]
        if overwrite:
            cmd.append("-y")

        t0 = time.time()
        try:
            subprocess.run(cmd, check=True)
            elapsed = time.time() - t0
            in_mb = video.stat().st_size / 1_048_576
            out_mb = out.stat().st_size / 1_048_576
            reduction = 100 * (1 - out_mb / in_mb) if in_mb > 0 else 0
            logger.info(f"  Done in {elapsed:.1f}s | {in_mb:.1f} MB → {out_mb:.1f} MB ({reduction:.0f}% smaller)")
            successful += 1
        except subprocess.CalledProcessError:
            logger.error(f"  FAILED (see ffmpeg output above)")

    logger.info(f"Done. {successful}/{len(videos)} transcoded successfully.")
