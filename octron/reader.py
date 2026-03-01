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
        print(
            f"🐙 Detected OCTRON prediction folder: {path}"
        )
        # Load predictions
        from octron.yolo_octron.yolo_octron import YOLO_octron
        yolo_octron = YOLO_octron()
        for label, track_id, _, _, _, _ in yolo_octron.load_predictions(
            save_dir = path,
            sigma_tracking_pos = 2, # Fixed for now 
        ):
            print(f"Adding tracking result to viewer | Label: {label}, Track ID: {track_id}")     
        return [(None,)]
    
    
    # Case D 
    # Check if the folder has any kind of video or mj2 files 
    video_formats = [".avi", ".mov", ".mj2", ".mpg", ".mpeg", ".mjpeg", ".mjpg", ".wmv", ".mp4", ".mkv"]
    video_formats.extend([fmt.upper() for fmt in video_formats])
    
    # Find all video files in the folder
    video_files = []
    for fmt in video_formats:
        video_files.extend(list(path.glob(f"*{fmt}")))
    if video_files:
        video_files = sorted(list(set(video_files))) # Get rid of any duplicates due to case-sensitivity! 

    # If we found video files, offer to transcode them
    if video_files:
        print(f"🎬 Found {len(video_files)} video files in {path}")
        
        # Create a dialog for transcoding options
        from qtpy.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                                   QCheckBox, QSpinBox, QPushButton, QListWidget,
                                   QDialogButtonBox, QAbstractItemView, QListWidgetItem,
                                   )
        from qtpy.QtCore import QSize, Qt
        
        dialog = QDialog()
        dialog.setWindowTitle("Transcode videos to mp4")
        dialog.resize(300, 400)  # Slightly larger dialog for better visibility
        layout = QVBoxLayout()
        
        # Add description
        layout.addWidget(QLabel(f"Found {len(video_files)} videos. Select which to transcode to mp4:"))
        
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
            
            # Get selected videos using selectedItems() for reliability
            selected_items = file_list.selectedItems()
            selected_videos = [item.data(Qt.UserRole) for item in selected_items]
            
            if not selected_videos:
                print("No videos selected for transcoding.")
                return [(None,)]
            
            # Create output folder if needed
            if create_subfolder:
                output_folder = path / "mp4_transcoded"
                output_folder.mkdir(exist_ok=True)
            else:
                output_folder = path
                
            print(f"🔄 Transcoding {len(selected_videos)} videos to MP4 (CRF: {crf_value})...")
            
            # Process one video at a time
            successful = 0
            for i, video_path in enumerate(selected_videos, 1):
                print(f"Processing {i}/{len(selected_videos)}: {video_path.name}")
                output_path = output_folder / f"{video_path.stem}.mp4"
                
                # Check if file exists and overwrite is not selected
                if not overwrite_existing and output_path.exists():
                    print(f"⏩ Skipped: '{output_path.name}' already exists and overwrite is disabled.")
                    continue

                # Define FFmpeg command
                cmd = [
                    "ffmpeg", "-i", str(video_path), 
                    "-c:v", "libx264", "-preset", "superfast", 
                    "-crf", str(crf_value),
                    "-c:a", "aac", "-b:a", "128k",  # Audio settings
                    str(output_path),
                ]
                if overwrite_existing:
                    cmd.append("-y") # Overwrite output if it exists
                
                # Time the transcoding process
                start_time = time.time()
                try:
                    subprocess.run(
                        cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE, 
                        check=True
                    )
                    elapsed_time = time.time() - start_time
                    # Calculate file sizes for comparison
                    input_size = video_path.stat().st_size / (1024 * 1024)  # MB
                    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
                    size_reduction = 100 * (1 - output_size / input_size) if input_size > 0 else 0
                    
                    print(f"✅ Successfully transcoded in {elapsed_time:.2f} seconds")
                    print(f"   Input: {input_size:.2f} MB, Output: {output_size:.2f} MB ({size_reduction:.1f}%)")
                    successful += 1
                except subprocess.CalledProcessError as e:
                    print(f"❌ Failed: {str(e)}")
            
            # Report final results
            print(f"✅ Successfully transcoded {successful}/{len(selected_videos)} videos")
        
        return [(None,)]
    
    return [(None,)]


