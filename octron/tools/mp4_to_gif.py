#!/usr/bin/env python

import subprocess
import time
from pathlib import Path
from typing import List
from loguru import logger

from qtpy.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QFileDialog, QListWidget, QSpinBox, 
    QProgressBar, QAbstractItemView, QComboBox, QCheckBox, QFrame
)
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QDragEnterEvent, QDropEvent, QPalette, QColor, QIcon, QPainter, QPixmap, QPainterPath, QPen
from qtpy.QtCore import QRectF


class DropArea(QFrame): 
    """Widget that accepts file drops."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setMinimumSize(100, 150)
        
        # Set up a frame with a border - this approach works more reliably
        # than just using stylesheets on custom widgets
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(7)  # Border width
        
        # Set background color
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor("#323232"))
        palette.setColor(QPalette.WindowText, Qt.white)
        palette.setColor(QPalette.Light, Qt.white)  # Border color when raised
        self.setPalette(palette)
        self.setAutoFillBackground(True)
        
        layout = QVBoxLayout()
        self.label = QLabel("Drop MP4/MOV/AVI files here")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("color: white; font-size: 14px;")
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            # Accept MP4, MOV and AVI files
            if all(url.toLocalFile().lower().endswith(('.mp4', '.mov', '.avi')) for url in urls):
                event.acceptProposedAction()
                
                # Change border color to green when active using palette
                palette = self.palette()
                palette.setColor(QPalette.Light, QColor("#259e62"))
                palette.setColor(QPalette.Dark, QColor("#259e62"))
                self.setPalette(palette)
                
                self.label.setText("Drop to add files")
    
    def dragLeaveEvent(self, event):
        """Handle drag leave events."""
        # Reset border color to white
        palette = self.palette()
        palette.setColor(QPalette.Light, Qt.white)
        palette.setColor(QPalette.Dark, Qt.white)
        self.setPalette(palette)
        
        self.label.setText("Drop MP4 files here")
    
    def dropEvent(self, event: QDropEvent):
        """Handle drop events."""
        file_paths = []
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.mp4', '.mov', '.avi')):
                file_paths.append(file_path)
        
        # Make sure we access the main window properly
        parent = self.parent()
        while parent and not hasattr(parent, 'add_files'):
            parent = parent.parent()
        
        if parent and hasattr(parent, 'add_files') and file_paths:
            parent.add_files(file_paths)
        
        # Reset border color to white
        palette = self.palette()
        palette.setColor(QPalette.Light, Qt.white)
        palette.setColor(QPalette.Dark, Qt.white)
        self.setPalette(palette)
        
        self.label.setText("Drop MP4/MOV/AVI files here")


class MP4ToGifConverter(QMainWindow):
    """
    Main application window.
    The goal is to provide a simple interface to convert MP4, MOV, and AVI files to GIFs.
    This is a task that is often useful when compressing / sharing short video clips.
       
    
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video to GIF Converter") 
        self.setMinimumSize(300, 700)  # Increased from 300, 500 to 500, 650
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Add drop area
        self.drop_area = DropArea(self)
        main_layout.addWidget(self.drop_area)
        
        # Add file list
        main_layout.addWidget(QLabel("Files to convert:"))
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.file_list.setMinimumHeight(120)
        main_layout.addWidget(self.file_list)
        
        # File controls
        file_controls = QHBoxLayout()
        self.add_button = QPushButton(u"\u2295 Add")
        self.remove_button = QPushButton(u"\u2296 Remove")
        self.clear_button = QPushButton("Clear All")
        file_controls.addWidget(self.add_button)
        file_controls.addWidget(self.remove_button)
        file_controls.addWidget(self.clear_button)
        main_layout.addLayout(file_controls)
        
        # Add options
        options_layout = QVBoxLayout()
        main_layout.addLayout(options_layout)
        
        # FPS settings
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Frame Rate:"))
        self.fps_spinner = QSpinBox()
        self.fps_spinner.setRange(1, 60)
        self.fps_spinner.setValue(20)  
        self.fps_spinner.setSuffix(" fps")
        fps_layout.addWidget(self.fps_spinner)
        options_layout.addLayout(fps_layout)
        
        # Quality settings
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High", "Medium", "Low"])
        self.quality_combo.setCurrentIndex(1)  # Default to Medium
        quality_layout.addWidget(self.quality_combo)
        options_layout.addLayout(quality_layout)
        
        # Frame skip settings 
        skip_layout = QHBoxLayout()
        skip_layout.addWidget(QLabel("Skip Frames:"))
        self.skip_spinner = QSpinBox()
        self.skip_spinner.setRange(0, 10000)
        self.skip_spinner.setValue(0)  # Default: don't skip any frames
        self.skip_spinner.setToolTip("Skip N frames between each captured frame (0 = no skipping).\n"
                                     "Higher values create smaller files but choppier animations.")
        # Remove the explanatory text label
        skip_layout.addWidget(self.skip_spinner)
        # Remove this line:
        # skip_layout.addWidget(skip_desc)
        options_layout.addLayout(skip_layout)
        
        # Resize settings
        resize_layout = QHBoxLayout()
        resize_layout.addWidget(QLabel("Max Width:"))
        self.resize_spinner = QSpinBox()
        self.resize_spinner.setRange(0, 3840)  # 0 = no resize, up to 4K
        self.resize_spinner.setValue(0)        # Default: no resizing
        self.resize_spinner.setSuffix(" px")
        self.resize_spinner.setSpecialValueText("No resize")  # Show text when value is 0
        self.resize_spinner.setToolTip("Resize output GIF to this maximum width (0 = original size).\n"
                                       "Height will be adjusted to maintain aspect ratio.")
        resize_layout.addWidget(self.resize_spinner)
        options_layout.addLayout(resize_layout)
        
        # Output folder
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("Output:"))
        self.same_folder_check = QCheckBox("Same as video")
        self.same_folder_check.setChecked(True)
        folder_layout.addWidget(self.same_folder_check)
        self.output_folder_button = QPushButton("Select Folder ...")
        self.output_folder_button.setEnabled(False)
        folder_layout.addWidget(self.output_folder_button)
        options_layout.addLayout(folder_layout)
        
        # Progress bar
        main_layout.addWidget(QLabel("Progress:"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.start_button = QPushButton("Start Conversion")
        self.start_button.setMinimumSize(QSize(100, 50))
        control_layout.addWidget(self.start_button)
        main_layout.addLayout(control_layout)
        
        # Connect signals
        self.add_button.clicked.connect(self.browse_files)
        self.remove_button.clicked.connect(self.remove_selected_files)
        self.clear_button.clicked.connect(self.clear_files)
        self.start_button.clicked.connect(self.start_conversion)
        self.same_folder_check.toggled.connect(self.toggle_output_folder)
        self.output_folder_button.clicked.connect(self.browse_output_folder)
        
        # Initialize variables
        self.output_folder = None
        self.files_to_process = []
    
    def add_files(self, file_paths: List[str]):
        """Add files to the list."""
        # Filter to only include supported video files
        video_files = [path for path in file_paths if path.lower().endswith(('.mp4', '.mov', '.avi'))]
        
        for file_path in video_files:
            # Check if file already exists in the list
            existing_items = self.file_list.findItems(file_path, Qt.MatchExactly)
            if not existing_items:
                self.file_list.addItem(file_path)
                self.files_to_process.append(file_path)
        
        self.status_label.setText(f"Added {len(video_files)} files. Ready to convert.")
    
    def browse_files(self):
        """Open a file dialog to select video files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "", "Video Files (*.mp4 *.mov *.avi)"
        )
        if files:
            self.add_files(files)
    
    def remove_selected_files(self):
        """Remove selected files from the list."""
        selected_items = self.file_list.selectedItems()
        for item in selected_items:
            row = self.file_list.row(item)
            self.file_list.takeItem(row)
            file_path = item.text()
            if file_path in self.files_to_process:
                self.files_to_process.remove(file_path)
        
        self.status_label.setText(f"Removed {len(selected_items)} files.")
    
    def clear_files(self):
        """Clear all files from the list."""
        self.file_list.clear()
        self.files_to_process = []
        self.status_label.setText("File list cleared.")
    
    def toggle_output_folder(self, checked):
        """Enable/disable output folder selection based on checkbox."""
        self.output_folder_button.setEnabled(not checked)
    
    def browse_output_folder(self):
        """Open a dialog to select output folder."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", ""
        )
        if folder:
            self.output_folder = folder
            self.output_folder_button.setText(f"...{folder[-20:]}")
    
    def start_conversion(self):
        """Start the conversion process."""
        # Check if we have files to convert
        if not self.files_to_process:
            self.status_label.setText("No files to convert!")
            return
        
        # Get options
        fps = self.fps_spinner.value()
        quality_index = self.quality_combo.currentIndex()
        skip_frames = self.skip_spinner.value()
        resize_width = self.resize_spinner.value()
        
        # Map quality to ffmpeg settings - use multiple parameters for better control
        quality_settings = {
            0: {"bayer_scale": 1, "dither": "bayer", "diff_mode": "rectangle"},  # High quality
            1: {"bayer_scale": 3, "dither": "bayer", "diff_mode": "rectangle"},  # Medium quality  
            2: {"bayer_scale": 5, "dither": "floyd_steinberg", "diff_mode": "rectangle"}  # Low quality - use different dithering
        }
        quality_params = quality_settings[quality_index]
        
        # Disable UI during conversion
        self.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting conversion...")
        QApplication.processEvents()
        
        # Process each file
        total_files = len(self.files_to_process)
        successful = 0
        
        for index, file_path in enumerate(self.files_to_process):
            input_path = Path(file_path)
            
            # Determine output path
            if self.same_folder_check.isChecked() or not self.output_folder:
                output_folder = input_path.parent
            else:
                output_folder = Path(self.output_folder)
            
            output_path = output_folder / f"{input_path.stem}.gif"
            
            # Update status
            self.status_label.setText(f"Converting {index+1}/{total_files}: {input_path.name}")
            self.progress_bar.setValue(int((index / total_files) * 100))
            QApplication.processEvents()
            
            try:
                # First pass - extract frames to a temporary directory
                start_time = time.time()
                
                # Create a temporary directory for frames
                import tempfile, shutil
                temp_dir = tempfile.mkdtemp(prefix="mp4_to_gif_")
                
                # Update status
                self.status_label.setText(f"Extracting frames from {input_path.name}...")
                QApplication.processEvents()
                
                # Get video info to calculate frame count and duration
                probe_cmd = [
                    "ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=nb_frames,duration",
                    "-of", "default=noprint_wrappers=1:nokey=1", str(input_path)
                ]
                
                try:
                    # Get frame count and duration
                    probe_result = subprocess.run(
                        probe_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    probe_output = probe_result.stdout.strip().split('\n')
                    
                    if len(probe_output) >= 2:
                        try:
                            total_frames = int(float(probe_output[0]))
                            duration = float(probe_output[1])
                        except ValueError:
                            # Some videos don't report frame count correctly
                            total_frames = 0
                            duration = float(probe_output[0])
                    else:
                        total_frames = 0
                        duration = 0
                    
                    logger.debug(f"Video duration {duration}s")
                except Exception as e:
                    logger.warning(f"Error getting video info: {e}")
                    total_frames = 0
                
                # Extract frames based on skip_frames setting
                if skip_frames > 0:
                    # Extract every Nth frame
                    framestep = skip_frames + 1
                    
                    # If we have total frames, we can estimate how many we'll extract
                    if total_frames > 0:
                        estimated_frames = total_frames // framestep
                    
                    # Build filter for selection and optional resizing
                    vf_filter = f"select='not(mod(n\\,{framestep}))'"
                    if resize_width > 0:
                        vf_filter += f",scale={resize_width}:-1:flags=lanczos"
                    
                    # Extract frames with framestep
                    extract_cmd = [
                        "ffmpeg", "-y", "-i", str(input_path),
                        "-vf", vf_filter,
                        "-vsync", "0",  # Important: don't add duplicate frames
                        f"{temp_dir}/frame_%04d.png"
                    ]
                else:
                    # Build filter for fps and optional resizing
                    vf_filter = f"fps={fps}"
                    if resize_width > 0:
                        vf_filter += f",scale={resize_width}:-1:flags=lanczos"
                    
                    # Extract all frames at desired fps
                    extract_cmd = [
                        "ffmpeg", "-y", "-i", str(input_path),
                        "-vf", vf_filter,
                        f"{temp_dir}/frame_%04d.png"
                    ]
                
                
                
                # Execute frame extraction
                extract_result = subprocess.run(
                    extract_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Count extracted frames
                extracted_frames = sorted(list(Path(temp_dir).glob("*.png")))
                num_frames = len(extracted_frames)
                
                if num_frames == 0:
                    raise Exception("No frames were extracted")
                
                logger.info(f"Successfully extracted {num_frames} frames to {temp_dir}")
                
                # Update status
                self.status_label.setText(f"Creating GIF from {num_frames} frames...")
                QApplication.processEvents()
                
                # Create palette from extracted frames
                palette_path = f"{temp_dir}/palette.png"
                palette_cmd = [
                    "ffmpeg", "-y", 
                    "-i", f"{temp_dir}/frame_%04d.png",
                    "-vf", "palettegen=stats_mode=diff",
                    palette_path
                ]
                
                
                
                # Execute palette generation
                palette_result = subprocess.run(
                    palette_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Create GIF from frames using the palette
                gif_cmd = [
                    "ffmpeg", "-y",
                    "-framerate", str(fps),  # Set input framerate
                    "-i", f"{temp_dir}/frame_%04d.png",
                    "-i", palette_path,
                    "-lavfi", f"paletteuse=dither={quality_params['dither']}:bayer_scale={quality_params['bayer_scale']}:diff_mode={quality_params['diff_mode']}",
                    str(output_path)
                ]
                
                
                
                # Execute GIF creation
                gif_result = subprocess.run(
                    gif_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=True
                )
                
                # Clean up temporary directory
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Error cleaning up temp directory: {e}")
                
                # Check if output file exists and has size
                if output_path.exists() and output_path.stat().st_size > 0:
                    elapsed_time = time.time() - start_time
                    input_size = input_path.stat().st_size / (1024 * 1024)  # MB
                    output_size = output_path.stat().st_size / (1024 * 1024)  # MB
                    logger.info(f"Successfully created GIF: {output_path}")
                    logger.info(f"From {input_size:.1f}MB to {output_size:.1f}MB")
                    self.status_label.setText(
                        f"Converted {input_path.name} in {elapsed_time:.1f}s - "
                        f"From {input_size:.1f}MB to {output_size:.1f}MB"
                    )
                    successful += 1
                else:
                    self.status_label.setText(f"Error: Output file is empty or missing for {input_path.name}")
                
            except subprocess.CalledProcessError as e:
                logger.exception(e)
                self.status_label.setText(f"Error converting {input_path.name}: {str(e)}")
            
            # Update progress
            self.progress_bar.setValue(int(((index + 1) / total_files) * 100))
            QApplication.processEvents()
        
        # Re-enable UI
        self.setEnabled(True)
        self.status_label.setText(f"Completed! Successfully converted {successful}/{total_files} files.")
    
    def closeEvent(self, event):
        """Handle application close."""
        event.accept()


# Fix the create_emoji_icon function
def create_emoji_icon(emoji, size=64, bg_color=None):
    """Create an icon from an emoji character."""
    pixmap = QPixmap(size, size)
    
    # Make transparent or use background color
    if bg_color:
        pixmap.fill(QColor(bg_color))
    else:
        pixmap.fill(Qt.transparent)
    
    # Create painter
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    
    # Set font for emoji - use int() for point size
    font = painter.font()
    font.setPointSize(int(size * 0.75))  # Convert to int to fix the error
    painter.setFont(font)
    
    # Draw emoji centered in the pixmap
    painter.drawText(
        QRectF(0, 0, size, size),
        Qt.AlignCenter,
        emoji
    )
    
    painter.end()
    return QIcon(pixmap)

def main():
    """Entry point for the converter tool."""
    import sys
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import Qt
    
    # Enable high DPI support
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, 'AA_UseHighDpiPixmaps'):
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    
    # Set application icon using octopus emoji
    app_icon = create_emoji_icon("🐙", size=128)
    app.setWindowIcon(app_icon)
    
    window = MP4ToGifConverter()
    window.setWindowIcon(app_icon)  # Also set it on the main window
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()