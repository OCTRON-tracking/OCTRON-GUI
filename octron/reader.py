from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Optional, Union

from napari.types import LayerData
from napari.utils.notifications import (
    show_error,
)

# Define some types
PathLike = str
PathOrPaths = Union[PathLike, Sequence[PathLike]]
ReaderFunction = Callable[[PathOrPaths], list[LayerData]]

import warnings

warnings.simplefilter("once")
from loguru import logger


def octron_reader(path: "PathOrPaths") -> Optional["ReaderFunction"]:
    """OCTRON napari reader.
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


def read_octron_file(path: "PathOrPaths") -> list["LayerData"]:
    """Single file reads that are dropped in the main window are not supported."""
    show_error("Single file drops to main window are not supported")
    return [(None,)]


def read_octron_folder(path: "Path") -> list["LayerData"]:
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
            save_dir=path,
            sigma_tracking_pos=2,  # Fixed for now
        ):
            logger.debug(
                f"Adding tracking result to viewer | Label: {label}, Track ID: {track_id}"
            )
        return [(None,)]

    # Case D
    # Check if the folder has any kind of video or multi-frame TIFF files
    video_formats = [
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
    ]
    video_formats.extend([fmt.upper() for fmt in video_formats])

    # Find all video files in the folder
    video_files = []
    for fmt in video_formats:
        video_files.extend(list(path.glob(f"*{fmt}")))
    if video_files:
        video_files = sorted(
            list(set(video_files))
        )  # Get rid of any duplicates due to case-sensitivity!

    # If we found video files, offer to transcode them
    if video_files:
        logger.info(f"Found {len(video_files)} transcodable files in {path}")

        # Create a dialog for transcoding options
        from qtpy.QtCore import QSize, Qt
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QCheckBox,
            QDialog,
            QDialogButtonBox,
            QDoubleSpinBox,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QListWidgetItem,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
        )

        dialog = QDialog()
        dialog.setWindowTitle("Transcode videos to mp4")
        dialog.resize(300, 400)  # Slightly larger dialog for better visibility
        layout = QVBoxLayout()

        # Add description
        layout.addWidget(
            QLabel(
                f"Found {len(video_files)} inputs. Select which to transcode to mp4:"
            )
        )

        # Add file list with multi-selection
        file_list = QListWidget()
        file_list.setSelectionMode(
            QAbstractItemView.MultiSelection
        )  # Allow multiple selection
        for video in video_files:
            item = QListWidgetItem(video.name)
            # Store the full path object on the item for later retrieval
            item.setData(Qt.UserRole, video)
            file_list.addItem(item)
            # Preselect all videos by default
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
        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
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
            # User clicked OK, process the selected inputs through the shared
            # GUI-free transcode helper (same code path as the `octron transcode` CLI).
            from octron.tools.transcode import (
                detect_h264_encoder,
                transcode_one,
            )

            # Get options
            create_subfolder = subfolder_check.isChecked()
            crf_value = crf_spin.value()
            overwrite_existing = overwrite_check.isChecked()
            use_custom_fps = fps_check.isChecked()
            fps_value = fps_spin.value() if use_custom_fps else None

            # Get selected videos using selectedItems() for reliability
            selected_items = file_list.selectedItems()
            selected_videos = [
                item.data(Qt.UserRole) for item in selected_items
            ]

            if not selected_videos:
                logger.info("No inputs selected for transcoding.")
                return [(None,)]

            # Create output folder if needed
            if create_subfolder:
                output_folder = path / "mp4_transcoded"
                output_folder.mkdir(exist_ok=True)
            else:
                output_folder = path

            fps_info = (
                f"{fps_value} fps"
                if fps_value is not None
                else "source fps (videos) / 20 fps (TIFFs)"
            )
            logger.info(
                f"Transcoding {len(selected_videos)} inputs to MP4 | CRF: {crf_value} | Framerate: {fps_info}"
            )

            # Detect the encoder once up front (raises if ffmpeg/H.264 is missing).
            # Transcode standardises on libx264 for reproducible, compatible output.
            try:
                encoder = detect_h264_encoder(prefer_hardware=False)
            except RuntimeError as e:
                logger.error(str(e))
                return [(None,)]

            # Process one input at a time through the shared helper.  Audio is
            # kept (re-encoded to AAC) for videos; TIFF stacks have no audio.
            successful = 0
            for i, video_path in enumerate(selected_videos, 1):
                output_path = output_folder / f"{video_path.stem}.mp4"
                logger.info(
                    f"Processing {i}/{len(selected_videos)}: {video_path.name}"
                )

                # Check if file exists and overwrite is not selected
                if not overwrite_existing and output_path.exists():
                    logger.info(
                        f"Skipped: '{output_path.name}' already exists and overwrite is disabled."
                    )
                    continue

                if transcode_one(
                    video_path,
                    output_path,
                    crf=crf_value,
                    overwrite=overwrite_existing,
                    fps=fps_value,
                    keep_audio=True,
                    encoder=encoder,
                ):
                    successful += 1

            # Report final results
            logger.info(
                f"Successfully transcoded {successful}/{len(selected_videos)} inputs"
            )

        return [(None,)]

    return [(None,)]
