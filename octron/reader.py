from pathlib import Path
from typing import Union, Sequence, Callable, List, Optional
from napari.types import LayerData
from napari.utils.notifications import (
    show_error,
)
# Define some types
PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], List[LayerData]]

import warnings
warnings.simplefilter("once")
from loguru import logger


def octron_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    """
    OCTRON napari reader.
    Accepts OCTRON project folders.
    
    Parameters
    ----------
    path : str or list of str
        Path to a file or folder.
    
    Returns
    -------
    function : Callable
        Function to read the file or folder.
        
    """
    
    path = Path(path)
    if path.is_dir() and path.exists():
        return read_octron_folder
        
    if path.is_file() and path.exists():
        return read_octron_file

def read_octron_file(path: "PathOrPaths") -> List["LayerData"]:
    """
    Single file reads that are dropped in the main window are not supported.
    """
    show_error(
        f"Single file drops to main window are not supported"
    )
    return [(None,)]

def read_octron_folder(path: "Path") -> List["LayerData"]:
    path = Path(path)
    # Check what kind of folder you are dealing with.
    # There are three options:
    # A. Octron project folder
    # B. Octron video (annotation) folder
    # C. Octron prediction (results) folder
    # D. Video folder to transcribe to mp4
    
    # Case A 
    
    # This is currently not implemented yet ... I am forcing people to  load 
    # the project through the load project button in the project manager tab.
    
    
    # Case C 
    # Check if the folder has .csv files AND a prediction_metadata.json
    csvs = list(path.glob("*.csv"))
    prediction_metadata = path / "prediction_metadata.json"
    if csvs and prediction_metadata.exists():
        logger.info(f"Detected OCTRON prediction folder: {path}")
        # Load predictions
        from octron.yolo_octron.yolo_octron import YOLO_octron
        yolo_octron = YOLO_octron()
        for label, track_id, _, _, _, _ in yolo_octron.load_predictions(
            save_dir = path,
            sigma_tracking_pos = 2, # Fixed for now
        ):
            logger.debug(f"Adding tracking result to viewer | Label: {label}, Track ID: {track_id}")     
        return [(None,)]
    
    
    # Case D 
    # Check if the folder has any kind of video or multi-frame TIFF files
    video_formats = [".avi", ".mov", ".mj2", ".mpg", ".mpeg", ".mjpeg", ".mjpg", ".wmv", ".mp4", ".mkv", ".mts", ".tif", ".tiff"]
    video_formats.extend([fmt.upper() for fmt in video_formats])
    
    # Find all video files in the folder
    video_files = []
    for fmt in video_formats:
        video_files.extend(list(path.glob(f"*{fmt}")))
    if video_files:
        video_files = sorted(list(set(video_files))) # Get rid of any duplicates due to case-sensitivity! 

    # If we found video files, offer to transcode them
    if video_files:
        logger.info(f"Found {len(video_files)} transcodable files in {path}")
        
        # Create a dialog for transcoding options
        from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                                   QCheckBox, QSpinBox, QDoubleSpinBox, QPushButton, QListWidget,
                                   QDialogButtonBox, QAbstractItemView, QListWidgetItem,
                                   )
        from qtpy.QtCore import QSize, Qt
        
        dialog = QDialog()
        dialog.setWindowTitle("Transcode videos to mp4")
        dialog.resize(300, 400)  # Slightly larger dialog for better visibility
        layout = QVBoxLayout()
        
        # Add description
        layout.addWidget(QLabel(f"Found {len(video_files)} inputs. Select which to transcode to mp4:"))
        
        # Add file list with multi-selection
        file_list = QListWidget()
        file_list.setSelectionMode(QAbstractItemView.MultiSelection)  # Allow multiple selection
        for video in video_files:
            item = QListWidgetItem(video.name)
            # Store the full path object on the item for later retrieval
            item.setData(Qt.UserRole, video)
            file_list.addItem(item)
            # Pre-select all videos by default
            item.setSelected(True)
        layout.addWidget(file_list)
        
        # Add selection helpers
        selection_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        selection_layout.addWidget(select_all_btn)
        selection_layout.addWidget(deselect_all_btn)
        layout.addLayout(selection_layout)
        
        # Add options
        options_layout = QHBoxLayout()
        
        # Create subfolder option
        subfolder_check = QCheckBox("Create subfolder")
        subfolder_check.setChecked(True)
        options_layout.addWidget(subfolder_check)
        
        # Overwrite existing files? 
        overwrite_check = QCheckBox("Overwrite existing")
        overwrite_check.setChecked(False) 
        options_layout.addWidget(overwrite_check)
        
        # CRF value option
        crf_layout = QHBoxLayout()
        crf_layout.addWidget(QLabel(" CRF (lower is better):"))
        crf_spin = QSpinBox()
        crf_spin.setRange(0, 51)
        crf_spin.setValue(23)  # Default CRF value
        crf_spin.setSingleStep(1)
        crf_spin.setMinimumSize(QSize(60, 25))
        crf_spin.setMaximumSize(QSize(60, 25))
        crf_layout.addWidget(crf_spin)
        options_layout.addLayout(crf_layout)
        
        layout.addLayout(options_layout)

        # Framerate option
        fps_layout = QHBoxLayout()
        fps_check = QCheckBox("Set output framerate (fps):")
        fps_check.setChecked(False)
        fps_check.setToolTip(
            "Set output framerate.\n"
            "Videos: reinterprets source frames at this fps, changing playback speed\n"
            "  (e.g. 10 fps source → 100 fps = plays 10x faster).\n"
            "  Leave unchecked to keep original playback speed.\n"
            "TIFFs: sets the playback fps of the output (default 20 fps)."
        )
        fps_layout.addWidget(fps_check)
        fps_spin = QDoubleSpinBox()
        fps_spin.setRange(1.0, 240.0)
        fps_spin.setValue(20.0)
        fps_spin.setSingleStep(1.0)
        fps_spin.setDecimals(3)
        fps_spin.setMinimumSize(QSize(80, 25))
        fps_spin.setMaximumSize(QSize(80, 25))
        fps_spin.setEnabled(False)
        fps_check.toggled.connect(fps_spin.setEnabled)
        fps_layout.addWidget(fps_spin)
        layout.addLayout(fps_layout)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # Connect buttons
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        # Select/deselect all helpers
        def select_all():
            for i in range(file_list.count()):
                file_list.item(i).setSelected(True)
        
        def deselect_all():
            for i in range(file_list.count()):
                file_list.item(i).setSelected(False)
        
        select_all_btn.clicked.connect(select_all)
        deselect_all_btn.clicked.connect(deselect_all)
        
        # Show dialog and wait for user input
        if dialog.exec_():
            # User clicked OK, process videos
            import subprocess
            import time
            
            # Get options
            create_subfolder = subfolder_check.isChecked()
            crf_value = crf_spin.value()
            overwrite_existing = overwrite_check.isChecked()
            use_custom_fps = fps_check.isChecked()
            fps_value = fps_spin.value() if use_custom_fps else None
            
            # Get selected videos using selectedItems() for reliability
            selected_items = file_list.selectedItems()
            selected_videos = [item.data(Qt.UserRole) for item in selected_items]
            
            if not selected_videos:
                logger.info("No inputs selected for transcoding.")
                return [(None,)]
            
            # Create output folder if needed
            if create_subfolder:
                output_folder = path / "mp4_transcoded"
                output_folder.mkdir(exist_ok=True)
            else:
                output_folder = path
                
            logger.info(f"Transcoding {len(selected_videos)} inputs to MP4 (CRF: {crf_value})...")
            
            # Process one input at a time
            successful = 0
            for i, video_path in enumerate(selected_videos, 1):
                is_tiff = video_path.suffix.lower() in {".tif", ".tiff"}
                input_label = video_path.name
                output_path = output_folder / f"{video_path.stem}.mp4"

                logger.info(f"Processing {i}/{len(selected_videos)}: {input_label}")
                
                # Check if file exists and overwrite is not selected
                if not overwrite_existing and output_path.exists():
                    logger.info(f"Skipped: '{output_path.name}' already exists and overwrite is disabled.")
                    continue

                # Define FFmpeg command
                if is_tiff:
                    try:
                        import numpy as np
                        import tifffile
                    except ImportError as e:
                        logger.error(f"TIFF transcoding requires numpy+tifffile: {e}")
                        continue

                    def _to_uint8(arr: "np.ndarray") -> "np.ndarray":
                        """Normalise array to uint8, preserving relative intensities."""
                        if arr.dtype == np.uint8:
                            return arr
                        smin, smax = float(arr.min()), float(arr.max())
                        if smax > smin:
                            return ((arr.astype(np.float32) - smin) / (smax - smin) * 255).astype(np.uint8)
                        return np.zeros_like(arr, dtype=np.uint8)

                    try:
                        with tifffile.TiffFile(str(video_path)) as tif:
                            series = tif.series[0]
                            axes  = series.axes   # e.g. "TCYX", "TYX", "TZYXC"
                            sizes = series.sizes  # e.g. {'T':100, 'C':2, 'Y':512, 'X':512}
                            stack = series.asarray()
                    except Exception as e:
                        logger.error(f"Failed to read TIFF '{video_path.name}': {e}")
                        continue

                    n_t = sizes.get('T', 0)
                    n_z = sizes.get('Z', 0)
                    n_c = sizes.get('C', 0)

                    logger.info(
                        f"TIFF detected: axes='{axes}' shape={stack.shape} dtype={stack.dtype} "
                        f"| T={n_t} Z={n_z} C={n_c} "
                        f"H={sizes.get('Y', '?')} W={sizes.get('X', '?')} "
                        f"| {video_path.name}"
                    )

                    # Reject TIFFs with both a time AND a Z axis — ambiguous for video conversion
                    if n_t > 0 and n_z > 0:
                        logger.warning(
                            f"Skipped '{video_path.name}': TIFF contains both a time axis "
                            f"(T={n_t}) and a Z axis (Z={n_z}) (axes='{axes}'). "
                            f"Cannot determine intended frame order for video conversion."
                        )
                        continue

                    # Reject single-frame / 2D-only images
                    if n_t >= 2:
                        frame_key = 'T'
                        n_frames = n_t
                    elif n_z >= 2:
                        frame_key = 'Z'
                        n_frames = n_z
                        logger.info(f"No time axis; treating Z-stack ({n_z} slices) as frames.")
                    else:
                        logger.warning(
                            f"Skipped '{video_path.name}': single-frame or 2D TIFF "
                            f"(axes='{axes}'). Only multi-frame TIFFs are supported."
                        )
                        continue

                    # Reorder axes to canonical order:
                    # (frame_key, [unknown extras,] [C,] Y, X)
                    axes_list = list(axes)
                    known = {'T', 'Z', 'C', 'Y', 'X'}
                    extra = [a for a in axes_list if a not in known]

                    target_order = [frame_key] + extra + (['C'] if n_c > 0 else []) + ['Y', 'X']

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
                        logger.warning(f"'{video_path.name}': {n_c} channels detected; using first 3 as RGB.")
                        stack = _to_uint8(stack[..., :3])

                    frame_count, height, width, _ = stack.shape
                    logger.info(f"Transcoding {frame_count} frames ({width}×{height}) from '{video_path.name}'")

                    cmd = [
                        "ffmpeg",
                        "-f", "rawvideo",
                        "-pixel_format", "rgb24",
                        "-video_size", f"{width}x{height}",
                        "-framerate", str(fps_value) if fps_value is not None else "20",
                        "-i", "-",
                        "-c:v", "libx264", "-preset", "superfast",
                        "-crf", str(crf_value),
                        "-an",
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                        str(output_path),
                    ]
                else:
                    # -r before -i reinterprets the source timestamps at the given fps,
                    # changing playback speed without duplicating frames.
                    cmd = ["ffmpeg"]
                    if fps_value is not None:
                        cmd += ["-r", str(fps_value)]
                    cmd += [
                        "-i", str(video_path),
                        "-c:v", "libx264", "-preset", "superfast",
                        "-crf", str(crf_value),
                        "-an",  # Disable audio for all outputs
                        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
                        str(output_path),
                    ]
                if overwrite_existing:
                    cmd.append("-y") # Overwrite output if it exists
                
                # Time the transcoding process
                start_time = time.time()
                try:
                    if is_tiff:
                        result = subprocess.run(
                            cmd,
                            input=stack.tobytes(),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                    else:
                        result = subprocess.run(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=True,
                        )
                    elapsed_time = time.time() - start_time
                    # Calculate file sizes for comparison
                    input_bytes = video_path.stat().st_size
                    input_size = input_bytes / (1024 * 1024)  # MB
                    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
                    size_reduction = 100 * (1 - output_size / input_size) if input_size > 0 else 0
                    
                    logger.info(f"Successfully transcoded in {elapsed_time:.2f} seconds | Input: {input_size:.2f} MB, Output: {output_size:.2f} MB ({size_reduction:.1f}%)")
                    successful += 1
                except subprocess.CalledProcessError as e:
                    stderr_msg = e.stderr.decode("utf-8", errors="ignore").strip()
                    if stderr_msg:
                        logger.error(f"Failed: {stderr_msg.splitlines()[-1]}")
                    else:
                        logger.error(f"Failed: {str(e)}")
            
            # Report final results
            logger.info(f"Successfully transcoded {successful}/{len(selected_videos)} inputs")
        
        return [(None,)]
    
    return [(None,)]


