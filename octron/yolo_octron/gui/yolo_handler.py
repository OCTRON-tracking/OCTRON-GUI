import time
import shutil
from qtpy.QtCore import QObject
from qtpy.QtWidgets import QMessageBox
from napari.qt import create_worker
from napari.utils.notifications import show_info, show_warning, show_error

from pathlib import Path           
from octron.sam_octron.helpers.video_loader import probe_video  
from napari_pyav._reader import FastVideoReader              
from qtpy.QtWidgets import QDialog                            
from octron.gui_dialog_elements import remove_video_dialog     

# Tracker handling
from octron.tracking.helpers.tracker_checks import load_boxmot_tracker_config
from octron.tracking.tracker_config_ui import open_boxmot_tracker_config_dialog

import yaml
import torch
from octron.tracking.helpers.tracker_vis import create_color_icon
from octron.yolo_octron.constants import TASK_COLORS

class YoloHandler(QObject):
    def __init__(self, parent_widget, yolo_octron):
        super().__init__()
        self.w = parent_widget # main.py -> octron_widget
        self.yolo = yolo_octron
        
         # Device label?
        if torch.cuda.is_available():
            self.device_label = "cuda" # torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device_label = "mps" #"mps" # torch.device("mps")
            #print(f'MPS is available, but not yet supported. Using CPU instead.')
        else:
            self.device_label = "cpu" #torch.device("cpu")
        print(f'Using YOLO device: "{self.device_label}"')
        
        
        # Set up variables
        self.bbox_or_polygon_interrupt  = False # Training data generation interrupt
        self.bbox_or_polygon_generated = False
        self.training_data_interrupt  = False # Training data generation interrupt
        self.training_data_generated = False
        self.training_finished = False # YOLO training
        self.trained_models = {}
        self.videos_to_predict = {}
        
    def connect_signals(self):
        # Wire buttons/spinboxes to handler entrypoints 
        # These are all in the parent widget in main.py
        self.w.generate_training_data_btn.clicked.connect(self.init_training_data_threaded)
        self.w.start_stop_training_btn.clicked.connect(self.init_yolo_training_threaded)
        self.w.predict_start_btn.clicked.connect(self.init_yolo_prediction_threaded)
        self.w.predict_iou_thresh_spinbox.valueChanged.connect(self.on_iou_thresh_change)
        self.w.predict_video_drop_widget.callback = self.on_mp4_predict_dropped_area
        self.w.videos_for_prediction_list.currentIndexChanged.connect(self.on_video_prediction_change)
        self.w.yolomodel_tracker_list.currentIndexChanged.connect(self.on_tracker_selection_change)
        self.w.tune_tracker_btn.clicked.connect(self.on_tune_tracker_clicked)
        self.w.single_subject_checkBox.clicked.connect(self.on_one_object_per_label_clicked)
        self.w.train_resume_checkBox.toggled.connect(self.on_resume_toggled)
        self.w.yolomodel_trained_list.currentIndexChanged.connect(self.on_trained_model_changed)
        
    def on_resume_toggled(self, checked):
        """
        When the resume checkbox is toggled, disable/enable model and image size
        dropdowns to signal that these options are ignored when resuming.
        """
        self.w.yolomodel_list.setEnabled(not checked)
        self.w.yoloimagesize_list.setEnabled(not checked)

    def on_trained_model_changed(self, index):
        """
        When the user selects a trained model, check its task type and
        enable/disable mask-related prediction options accordingly.
        """
        if index <= 0:
            # Header item — reset to enabled
            self.w.predict_mask_opening_spinbox.setEnabled(True)
            self.w.prediction_mask_opening_label.setEnabled(True)
            self.w.detailed_extraction_checkBox.setEnabled(True)
            self.w.yolomodel_trained_list.setToolTip('')
            return

        model_name = self.w.yolomodel_trained_list.currentText()
        self.w.yolomodel_trained_list.setToolTip(model_name)
        model_path = self.trained_models.get(model_name)
        if model_path is None:
            return

        task = self.yolo.get_model_task(model_path)
        is_segment = (task == 'segment')

        # Opening and detailed extraction only apply to segmentation models
        # IOU remains enabled — it controls NMS for both detect and segment
        self.w.predict_mask_opening_spinbox.setEnabled(is_segment)
        self.w.prediction_mask_opening_label.setEnabled(is_segment)
        self.w.detailed_extraction_checkBox.setEnabled(is_segment)

        if not is_segment:
            self.w.predict_mask_opening_spinbox.setValue(0)
            self.w.detailed_extraction_checkBox.setChecked(False)

    def refresh_trained_model_list(self):
        """
        Refresh the trained model list combobox with the current models in the project directory.
        Each entry gets a colored square indicator showing whether the model is a
        segmentation (purple) or detection (blue) model.
        """
        # Clear the old list, and re-instantiate
        self.w.yolomodel_trained_list.clear()
        self.w.yolomodel_trained_list.addItem('Model ...')
        trained_models = self.yolo.find_trained_models(search_path=self.w.project_path)
        if not trained_models:
            self.w.main_toolbox.widget(3).setEnabled(False)
            return
        
        # Write the trained models to yolomodel_trained_list one by one
        for model in trained_models:
            # This is to clearly identify the model
            # in the list, since the model name is not unique
            model_name = '/'.join(model.parts[-5:])
            if model_name not in self.trained_models:
                self.trained_models[model_name] = model
            # Add colored indicator square based on model task
            task = self.yolo.get_model_task(model)
            color = TASK_COLORS.get(task)
            if color:
                icon = create_color_icon(color)
                self.w.yolomodel_trained_list.addItem(icon, model_name)
            else:
                self.w.yolomodel_trained_list.addItem(model_name)
        # Enable prediction tab if trained models are available
        self.w.main_toolbox.widget(3).setEnabled(True)
        self.w.predict_video_drop_groupbox.setEnabled(True)
        self.w.predict_video_predict_groupbox.setEnabled(True)
        self.w.predict_start_btn.setEnabled(True)
        self.w.predict_start_btn.setStyleSheet('')
        self.w.predict_start_btn.setText(f'▷ Predict')



    #######################################################################################################
    # TRAINING‐DATA GENERATION PIPELINE 
    #######################################################################################################
    
    def init_training_data_threaded(self):
        """
        This function manages the creation of polygon data and the subsequent
        creation / the export of training data.
        It kicks off the pipeline by calling the polygon generation function.
        
        """
        # Whenever the button "Generate" is clicked, 
        # the training data generation pipeline is started anew.
        if not hasattr(self, 'polygon_worker') and not hasattr(self, 'bbox_worker') and not hasattr(self, 'training_data_worker'):   
            self.bbox_or_polygon_generated = False
            self.training_data_generated = False   
        # Sanity check 
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not self.yolo:
            show_warning("Please load a YOLO first.")
            return
        # Check status of "Prune" checkbox
        prune = self.w.train_prune_checkBox.isChecked()
        # Check whether training folder should be overwritten or not 
        self.yolo.clean_training_dir = self.w.train_data_overwrite_checkBox.isChecked()
        # Inform YOLO of the current train mode before project_path triggers directory setup
        self.yolo.train_mode = self.w.train_mode

        # --- Overwrite: warn user before deleting anything ---
        if self.yolo.clean_training_dir:
            training_path = self.w.project_path / 'model'
            data_path = training_path / 'training_data'
            model_subdir = training_path / 'training'

            if data_path.exists():
                # Check for train-mode mismatch
                mode_mismatch = False
                existing_mode = None
                existing_config_path = data_path / 'yolo_config.yaml'
                if existing_config_path.exists():
                    with open(existing_config_path, 'r') as f:
                        existing_config = yaml.safe_load(f)
                    existing_mode = existing_config.get('train_mode', 'segment')
                    mode_mismatch = (existing_mode != self.w.train_mode)

                if mode_mismatch and model_subdir.exists():
                    # Training data AND model checkpoints will be deleted
                    warning_dialog = QMessageBox()
                    warning_dialog.setIcon(QMessageBox.Warning)
                    warning_dialog.setWindowTitle("Overwrite Training Data")
                    warning_dialog.setText("You are about to delete existing training data and model checkpoints.")
                    warning_dialog.setInformativeText(
                        f"Train mode has changed from '{existing_mode}' to '{self.w.train_mode}'. "
                        "The training data will be regenerated and model checkpoints will be removed.\n\n"
                        "Do you want to proceed?"
                    )
                    warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    warning_dialog.setDefaultButton(QMessageBox.No)
                    if warning_dialog.exec_() == QMessageBox.No:
                        return
                    shutil.rmtree(model_subdir)
                    print(f"Removed model checkpoint directory '{model_subdir.as_posix()}'")
                else:
                    # Only training data will be deleted
                    warning_dialog = QMessageBox()
                    warning_dialog.setIcon(QMessageBox.Warning)
                    warning_dialog.setWindowTitle("Overwrite Training Data")
                    warning_dialog.setText("You are about to overwrite existing training data.")
                    warning_dialog.setInformativeText(
                        "The training data directory will be removed and regenerated. "
                        "Model checkpoints will be preserved.\n\n"
                        "Do you want to proceed?"
                    )
                    warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                    warning_dialog.setDefaultButton(QMessageBox.No)
                    if warning_dialog.exec_() == QMessageBox.No:
                        return
                # Remove training data
                shutil.rmtree(data_path)
                print(f'Removed training data directory "{data_path.as_posix()}"')

            # Cleanup handled here; prevent _setup_training_directories from cleaning again
            self.yolo.clean_training_dir = False

        # Set the project_path (which also takes care of setting up training subfolders)
        if not self.yolo.project_path:
            self.yolo.project_path = self.w.project_path
        elif self.yolo.project_path != self.w.project_path: 
            # Assuming that the user wants to change the project path
            self.yolo.project_path = self.w.project_path

        status = self.w.save_object_organizer() # This is safe, since it checks whether a video was loaded
        # TODO: Could make `prepare_labels` async as well ... 
        try:
            # After saving the object organizer, extract info from all 
            # available .json files in the project directory
            self.yolo.prepare_labels(
                        prune_empty_labels=prune, 
                        min_num_frames=5, # hardcoded ... less than 5 frames are not useful
                        verbose=True, 
            )
        except AssertionError as e:
            print(f"😵 Error when preparing labels: {e}")
            return
        
        if not self.yolo.clean_training_dir:
            # Check if the training folder already exists
            # If it does, we can skip everything after this step
            if self.yolo.data_path is not None and self.yolo.data_path.exists():
                # Check if the existing config was generated with the same train_mode
                existing_config_path = self.yolo.data_path / 'yolo_config.yaml'
                if existing_config_path.exists():
                    with open(existing_config_path, 'r') as f:
                        existing_config = yaml.safe_load(f)
                    existing_mode = existing_config.get('train_mode', 'segment')
                    if existing_mode != self.w.train_mode:
                        msg = (f"Train mode mismatch: existing training data was generated for '{existing_mode}' "
                               f"but current mode is '{self.w.train_mode}'. "
                               f"Please enable 'Overwrite' to regenerate training data or switch mode.")
                        print(msg)
                        show_error(msg)
                        return
                # TODO: Since we just generated the labels_dict (in prepare_labels above),
                # a rudimentary check is actually possible, comparing the total number of expected labeled 
                # frames and the number of images in the training folder. I am skipping any checks for now.
                # Show a warning dialog that user must dismiss
                warning_dialog = QMessageBox()
                warning_dialog.setIcon(QMessageBox.Warning)
                warning_dialog.setWindowTitle("Existing Training Data")
                warning_dialog.setText("Training data directory already exists.")
                warning_dialog.setInformativeText("The existing training data will be used without regeneration. "
                                                  "No checks are performed on the training data folder. "
                                                  "If you want to regenerate the data, please check the 'Overwrite' option.")
                warning_dialog.setStandardButtons(QMessageBox.Ok)
                warning_dialog.exec_()
                self.bbox_or_polygon_generated = True
                self.training_data_generated = True
                self._on_training_data_finished()
                
                print(f"Training data path '{self.yolo.data_path.as_posix()}' already exists. Using existing directory.")
                return

        # Else ... continue the training data generation pipeline
        # Kick off polygon or bbox generation based on train mode
        if self.w.train_mode == 'detect':
            self._bbox_generation()
        else:
            self._polygon_generation()
        return

    def _polygon_generation(self):
        """
        MAIN MANAGER FOR POLYGON GENERATION
        polygon_worker()
        Manages thread worker for generating polygons
        """
        # Check if the worker has already run and was not interrupted.
        # If so, do not create a new worker, but just call the callback function.
        if not self.bbox_or_polygon_interrupt and self.bbox_or_polygon_generated:
            self._on_polygon_finished()
            return
        # Otherwise, create a new worker and manage interruptions
        if not hasattr(self, 'polygon_worker'):
            self._create_worker_polygons()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText(f'🅧 Interrupt')
            self.polygon_worker.start()
            self.bbox_or_polygon_interrupt = False
        elif hasattr(self, 'polygon_worker') and not self.polygon_worker.is_running:
            # Worker exists but is not running - clean up and create a new one
            self._uncouple_worker_polygons()
            self._create_worker_polygons()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText(f'🅧 Interrupt')
            self.polygon_worker.start()
            self.bbox_or_polygon_interrupt = False
        elif hasattr(self, 'polygon_worker') and self.polygon_worker.is_running:
            self.polygon_worker.quit()
            self.w.generate_training_data_btn.setStyleSheet('')
            self.w.generate_training_data_btn.setText(f'▷ Generate')
            self.bbox_or_polygon_interrupt = True

    def _create_worker_polygons(self):
        # Create a new worker for polygon generation        
        # Watershed? 
        enable_watershed = self.w.train_data_watershed_checkBox.isChecked()
        self.yolo.enable_watershed = enable_watershed
        self.polygon_worker = create_worker(self.yolo.prepare_polygons)
        self.polygon_worker.setAutoDelete(True) # auto destruct !!
        self.polygon_worker.yielded.connect(self._polygon_yielded)
        self.polygon_worker.finished.connect(self._on_polygon_finished)
        self.w.train_polygons_overall_progressbar.setEnabled(True)    
        self.w.train_polygons_frames_progressbar.setEnabled(True)
        self.w.train_polygons_label.setEnabled(True)

    def _polygon_yielded(self, value):
        """
        polygon_worker()
        Called upon yielding from the batch polygon generation thread worker.
        Updates the progress bar and label text next to it.
        """
        no_entry, total_label_dict, label, frame_no, total_frames = value
        self.w.train_polygons_overall_progressbar.setMaximum(total_label_dict)
        self.w.train_polygons_overall_progressbar.setValue(no_entry-1)
        self.w.train_polygons_frames_progressbar.setMaximum(total_frames) 
        self.w.train_polygons_frames_progressbar.setValue(frame_no)   
        self.w.train_polygons_label.setText(label)  

    def _on_polygon_finished(self):
        """
        polygon_worker()
        Callback for when polygon generation worker has finished executing. 
        """
        self.w.train_polygons_overall_progressbar.setValue(0)
        self.w.train_polygons_frames_progressbar.setValue(0)
        self.w.train_polygons_overall_progressbar.setEnabled(False)    
        self.w.train_polygons_frames_progressbar.setEnabled(False)
        self.w.train_polygons_label.setText('')
        self.w.train_polygons_label.setEnabled(False) 
        
        if self.bbox_or_polygon_interrupt:
            show_warning("Polygon generation interrupted.")  
            self.bbox_or_polygon_generated = False
            self.w.generate_training_data_btn.setStyleSheet('')
            self.w.generate_training_data_btn.setText(f'▷ Generate')
        else:
            self.bbox_or_polygon_generated = True
            
        # If self.bbox_or_polygon_generated is True, then start the 
        # training data export worker right after ...
        if self.bbox_or_polygon_generated and not self.training_data_generated:
            # split train/val/test then kick off data export
            self.yolo.prepare_split()
            self._training_data_export()
        else:
            pass

    def _bbox_generation(self):
        """
        MAIN MANAGER FOR BBOX GENERATION
        bbox_worker()
        Manages thread worker for generating bounding boxes (detect mode)
        """
        # Check if the worker has already run and was not interrupted.
        if not self.bbox_or_polygon_interrupt and self.bbox_or_polygon_generated:
            self._on_bbox_finished()
            return
        # Otherwise, create a new worker and manage interruptions
        if not hasattr(self, 'bbox_worker'):
            self._create_worker_bboxes()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText(f'🅧 Interrupt')
            self.bbox_worker.start()
            self.bbox_or_polygon_interrupt = False
        elif hasattr(self, 'bbox_worker') and not self.bbox_worker.is_running:
            self._uncouple_worker_bboxes()
            self._create_worker_bboxes()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText(f'🅧 Interrupt')
            self.bbox_worker.start()
            self.bbox_or_polygon_interrupt = False
        elif hasattr(self, 'bbox_worker') and self.bbox_worker.is_running:
            self.bbox_worker.quit()
            self.w.generate_training_data_btn.setStyleSheet('')
            self.w.generate_training_data_btn.setText(f'▷ Generate')
            self.bbox_or_polygon_interrupt = True

    def _create_worker_bboxes(self):
        # Create a new worker for bbox generation
        # Watershed?
        enable_watershed = self.w.train_data_watershed_checkBox.isChecked()
        self.yolo.enable_watershed = enable_watershed
        self.bbox_worker = create_worker(self.yolo.prepare_bboxes)
        self.bbox_worker.setAutoDelete(True)
        self.bbox_worker.yielded.connect(self._bbox_yielded)
        self.bbox_worker.finished.connect(self._on_bbox_finished)
        self.w.train_polygons_overall_progressbar.setEnabled(True)
        self.w.train_polygons_frames_progressbar.setEnabled(True)
        self.w.train_polygons_label.setEnabled(True)

    def _bbox_yielded(self, value):
        """
        bbox_worker()
        Called upon yielding from the batch bbox generation thread worker.
        Updates the progress bar and label text next to it.
        """
        no_entry, total_label_dict, label, frame_no, total_frames = value
        self.w.train_polygons_overall_progressbar.setMaximum(total_label_dict)
        self.w.train_polygons_overall_progressbar.setValue(no_entry-1)
        self.w.train_polygons_frames_progressbar.setMaximum(total_frames)
        self.w.train_polygons_frames_progressbar.setValue(frame_no)
        self.w.train_polygons_label.setText(label)

    def _on_bbox_finished(self):
        """
        bbox_worker()
        Callback for when bbox generation worker has finished executing.
        """
        self.w.train_polygons_overall_progressbar.setValue(0)
        self.w.train_polygons_frames_progressbar.setValue(0)
        self.w.train_polygons_overall_progressbar.setEnabled(False)
        self.w.train_polygons_frames_progressbar.setEnabled(False)
        self.w.train_polygons_label.setText('')
        self.w.train_polygons_label.setEnabled(False)

        if self.bbox_or_polygon_interrupt:
            show_warning("Bbox generation interrupted.")
            self.bbox_or_polygon_generated = False
            self.w.generate_training_data_btn.setStyleSheet('')
            self.w.generate_training_data_btn.setText(f'▷ Generate')
        else:
            self.bbox_or_polygon_generated = True

        if self.bbox_or_polygon_generated and not self.training_data_generated:
            self.yolo.prepare_split()
            self._training_data_export()
        else:
            pass

    def _training_data_export(self):
        """
        MAIN MANAGER FOR TRAINING DATA EXPORT
        """
        if not self.training_data_interrupt and self.training_data_generated:
            self._on_training_data_finished()
            return

        if not hasattr(self, 'training_data_worker'):
            self._create_worker_training_data()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText('🅧 Interrupt')
            self.training_data_worker.start()
            self.training_data_interrupt = False

        elif not self.training_data_worker.is_running:
            self._uncouple_worker_training_data()
            self._create_worker_training_data()
            self.w.generate_training_data_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.generate_training_data_btn.setText('🅧 Interrupt')
            self.training_data_worker.start()
            self.training_data_interrupt = False

        else:
            self.training_data_worker.quit()
            self.w.generate_training_data_btn.setStyleSheet('')
            self.w.generate_training_data_btn.setText('▷ Generate')
            self.training_data_interrupt = True
            self.bbox_or_polygon_generated = False

        self.w.training_data_folder_label.setEnabled(True)
        self.w.training_data_folder_label.setText(f'→{self.yolo.training_path.as_posix()[-38:]}')
        self.w.training_data_folder_label.setToolTip(self.yolo.training_path.as_posix())

    def _create_worker_training_data(self):
        # Create a new worker for training data generation / export
        # Route to the correct export function based on train_mode
        if self.w.train_mode == 'detect':
            export_func = self.yolo.create_training_data_detect
        else:
            export_func = self.yolo.create_training_data_segment
        self.training_data_worker = create_worker(export_func)
        self.training_data_worker.setAutoDelete(True) # auto destruct!
        self.training_data_worker.yielded.connect(self._training_data_yielded)
        self.training_data_worker.finished.connect(self._on_training_data_finished)
        self.w.train_export_overall_progressbar.setEnabled(True)    
        self.w.train_export_frames_progressbar.setEnabled(True)
        self.w.train_polygons_label.setEnabled(True)

    def _training_data_yielded(self, value):
        """
        training_data_worker()
        Called upon yielding from the batch training data generation thread worker.
        Updates the progress bar and label text next to it.
        """
        no_entry, total_label_dict, label, split, frame_no, total_frames = value
        self.w.train_export_overall_progressbar.setMaximum(total_label_dict)
        self.w.train_export_overall_progressbar.setValue(no_entry-1)
        self.w.train_export_frames_progressbar.setMaximum(total_frames) 
        self.w.train_export_frames_progressbar.setValue(frame_no)   
        self.w.train_export_label.setText(f'{label} ({split})')   

    def _on_training_data_finished(self):
        """
        training_data_worker()
        Callback for when training data generation worker has finished executing. 
        """
        self.w.generate_training_data_btn.setStyleSheet('')
        self.w.generate_training_data_btn.setText(f'▷ Generate')
        self.w.train_export_overall_progressbar.setValue(0)
        self.w.train_export_frames_progressbar.setValue(0)
        self.w.train_export_overall_progressbar.setEnabled(False)    
        self.w.train_export_frames_progressbar.setEnabled(False)
        self.w.train_export_label.setText('')
        self.w.train_export_label.setEnabled(False) 
        
        if self.training_data_interrupt:
            show_warning("Training data generation interrupted.")  
            self.training_data_generated = False
        else:
            show_info("Training data generation finished.")
            self.training_data_generated = True
            self.w.generate_training_data_btn.setText(f'✓ Done.')
            self.w.generate_training_data_btn.setEnabled(False)   
            self.w.train_data_overwrite_checkBox.setEnabled(False)
            self.w.train_prune_checkBox.setEnabled(False)
            self.w.train_data_watershed_checkBox.setEnabled(False)
            self.w.segmentation_bbox_decision_groupbox.setEnabled(False)
            # Write the YOLO config file
            self.yolo.write_yolo_config(train_mode=self.w.train_mode)
            # Enable next part (YOLO training) of the pipeline 
            self.w.train_train_groupbox.setEnabled(True)
            self.w.launch_tensorboard_checkBox.setEnabled(True)
            self.w.train_resume_checkBox.setEnabled(True)
            self.w.start_stop_training_btn.setStyleSheet('')
            self.w.start_stop_training_btn.setText(f'▷ Train')

    #######################################################################################################
    # YOLO TRAINING PIPELINE
    #######################################################################################################
    
    def init_yolo_training_threaded(self):
        """
        This function manages the training of the YOLO model.
        """
        if self.training_finished:
            return
        # Sanity check 
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not hasattr(self, 'yolo'):
            show_warning("Please load YOLO first.")
            return
        
        if not self.yolo.config_path and self.yolo.config_path.exists():
            show_warning(f"No YOLO config .yaml was found under '{self.yolo.config_path}'")
            
        # Verify that train_mode in the config matches the current GUI selection
        with open(self.yolo.config_path, 'r') as f:
            config = yaml.safe_load(f)
        config_mode = config.get('train_mode', 'segment')
        if config_mode != self.w.train_mode:
            msg = (f"Train mode mismatch: training data was generated for '{config_mode}' "
                    f"but current mode is '{self.w.train_mode}'. "
                    f"Please regenerate training data or switch mode.")
            print(msg)
            show_error(msg)
            return
        
        # Check status of "Launch Tensorboard" checkbox
        self.launch_tensorbrd = self.w.launch_tensorboard_checkBox.isChecked()   
        
        self.num_epochs_yolo = int(self.w.num_epochs_input.value())   
        if self.num_epochs_yolo <= 1:
            show_warning("Please select # epochs > 1")
            return
        self.save_period = int(self.w.save_period_input.value())
        
        # Check for resume first: model/image-size selections are irrelevant when resuming
        self.resume_training = False
        if self.w.train_resume_checkBox.isChecked():
            checkpoint_path = self.yolo.training_path / 'training' / 'weights' / 'last.pt'
            if checkpoint_path.exists():
                # Check whether the previous training run already completed
                ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                ckpt_epoch = ckpt.get('epoch', -1)
                if ckpt_epoch == -1:
                    show_warning(
                        "The previous training run already completed all epochs. "
                        "Nothing to resume. Uncheck 'Resume' to start a new training run."
                    )
                    return
                print(f"Resuming training from checkpoint: {checkpoint_path} (epoch {ckpt_epoch})")
                yolo_model = self.yolo.load_model(checkpoint_path, train_mode=self.w.train_mode)
                if not yolo_model:
                    show_warning("Could not load checkpoint model.")
                    return
                self.resume_training = True
                self.image_size_yolo = 0  # Ignored by YOLO when resume=True
            else:
                print("No checkpoint found (last.pt), starting fresh training.")
                show_info("No checkpoint found — starting fresh.")
        
        if not self.resume_training:
            index_model_list = self.w.yolomodel_list.currentIndex()
            if index_model_list == 0:
                show_warning("Please select a YOLO model")
                return
            model_name = self.w.yolomodel_list.currentText()
            # Reverse lookup model_id
            for model_id, model in self.w.yolomodels_dict.items():
                if model['name'] == model_name:
                    break        
            index_imagesize_list = self.w.yoloimagesize_list.currentIndex()
            if index_imagesize_list == 0:
                show_warning("Please select an image size")
                return 
            self.image_size_yolo = int(self.w.yoloimagesize_list.currentText())   
            if self.image_size_yolo % 32 != 0:
                show_warning(f'Training image size must be divisible by 32')
                return
            # If a previous model folder exists, warn and delete before fresh training
            model_subdir = self.yolo.training_path / 'training'
            if model_subdir.exists():
                warning_dialog = QMessageBox()
                warning_dialog.setIcon(QMessageBox.Warning)
                warning_dialog.setWindowTitle("Existing Model Found")
                warning_dialog.setText("You are about to delete a previous model and its checkpoints.")
                warning_dialog.setInformativeText(
                    f"A previous training run exists at:\n'{model_subdir.as_posix()}'\n\n"
                    "Starting fresh training will remove it. Do you want to proceed?"
                )
                warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
                warning_dialog.setDefaultButton(QMessageBox.No)
                if warning_dialog.exec_() == QMessageBox.No:
                    return
                shutil.rmtree(model_subdir)
                print(f"Removed previous model directory '{model_subdir.as_posix()}'")
            # LOAD YOLO MODEL (select seg or detect variant based on current train_mode)
            print(f"Loading YOLO model {model_id} (mode: {self.w.train_mode})")
            yolo_model = self.yolo.load_model(model_id, train_mode=self.w.train_mode)
            if not yolo_model:
                show_warning("Could not load YOLO model.")
                return
        
        # Deactivate the training data generation box 
        self.w.segmentation_bbox_decision_groupbox.setEnabled(False)
        self.w.train_generate_groupbox.setEnabled(False)
        # Otherwise, create a new worker and manage interruptions
        if not hasattr(self, 'yolo_trainer_worker'):
            self._create_yolo_trainer()
            self.w.start_stop_training_btn.setStyleSheet('QPushButton { color: #e7a881;}')
            self.w.start_stop_training_btn.setText(f'↯ Training')
            self.yolo_trainer_worker.start()
            self.w.start_stop_training_btn.setEnabled(False)
            # Disable training controls during training
            self.w.main_toolbox.widget(1).setEnabled(False) # Annotation
            self.w.yolomodel_list.setEnabled(False)
            self.w.yoloimagesize_list.setEnabled(False)
            self.w.train_resume_checkBox.setEnabled(False)
            self.w.train_training_overwrite_checkBox.setEnabled(False)
            self.w.launch_tensorboard_checkBox.setEnabled(False)
            self.w.num_epochs_input.setEnabled(False)
            self.w.save_period_input.setEnabled(False)

    def _create_yolo_trainer(self):
        # Create a new worker for YOLO training 
        self.yolo_trainer_worker = create_worker(self._yolo_trainer)
        self.yolo_trainer_worker.setAutoDelete(True)  # auto destruct!
        self.yolo_trainer_worker.yielded.connect(self._update_training_progress)
        self.w.train_epochs_progressbar.setEnabled(True)    
        self.w.train_finishtime_label.setEnabled(True)
        self.w.train_finishtime_label.setText('↬ ... wait one epoch')    
        if self.launch_tensorbrd:
            self.yolo.quit_tensorboard()
            self.yolo.launch_tensorboard()

    def _yolo_trainer(self):
        if not self.device_label:
            show_error("No device label found for YOLO.")
            return
        else:
            show_info(f"Training on device: {self.device_label}")
        
        # Call the training function which yields progress info
        for progress_info in self.yolo.train(
                                            device=self.device_label, 
                                            imagesz=self.image_size_yolo,
                                            epochs=self.num_epochs_yolo,
                                            save_period=self.save_period,
                                            train_mode=self.w.train_mode,
                                            resume=self.resume_training,
                                        ):
            # Yield the progress info back to the GUI thread
            yield progress_info

    def _update_training_progress(self, progress_info):
        """
        Handle training progress updates from the worker thread.
        When finished, enable the next step.
        
        Parameters
        ----------
        progress_info : dict
            Dictionary containing training progress information
        """
        current_epoch = progress_info['epoch']
        total_epochs = progress_info['total_epochs']
        epoch_time = progress_info['epoch_time']
        remaining_time = progress_info['remaining_time']
        finish_time = progress_info['finish_time']
        
        # Format the finish time (without year as requested)
        finish_time_str = ' '.join(time.ctime(finish_time).split()[:-1])
   
        self.w.train_epochs_progressbar.setMaximum(total_epochs)
        self.w.train_epochs_progressbar.setValue(current_epoch)        
        self.w.train_finishtime_label.setText(f'↬ {finish_time_str}')
        
        print(f"Epoch {current_epoch}/{total_epochs} - Time for epoch: {epoch_time:.1f}s")
        print(f"Estimated time remaining: {remaining_time:.1f} seconds")    
        print(f"Estimated finish time: {finish_time_str}")  

        if current_epoch == total_epochs: 
            self.training_finished = True
            self.w.start_stop_training_btn.setStyleSheet('')
            self.w.start_stop_training_btn.setText(f'✓ Done.')
            self.w.train_epochs_progressbar.setEnabled(False)  
            self.w.train_finishtime_label.setEnabled(False)
            # Refresh the trained model list and enable the prediction tab
            self.refresh_trained_model_list()
            

    #######################################################################################################
    # YOLO PREDICTION PIPELINE
    #######################################################################################################
    
    # Boxmot tracker selection / tuning handling
    def on_tracker_selection_change(self, index):
        """
        Handle tracker selection change in the dropdown list
        Enable/disable tune button based on selection
        """
        if index > 0:  # Any tracker selected (not the header item)
            self.w.tune_tracker_btn.setEnabled(True)
            self.w.tune_tracker_btn.setText("Tune")
            self.w.tune_tracker_btn.setStyleSheet("")
            self.w.yolomodel_tracker_list.setToolTip(self.w.yolomodel_tracker_list.currentText().strip())
        else:
            # First item selected (header/placeholder)
            self.w.tune_tracker_btn.setEnabled(False)
            self.w.tune_tracker_btn.setText("")
            self.w.tune_tracker_btn.setStyleSheet("")
            self.w.yolomodel_tracker_list.setToolTip('')
        
    def on_tune_tracker_clicked(self):
        """
        This is the "Tune" button displayed next to the tracker selection list.
        Open configuration dialog for the selected tracker
        """
        index = self.w.yolomodel_tracker_list.currentIndex()
        if index <= 0:
            return  # Should not happen as button should be disabled
        
        tracker_name = self.w.yolomodel_tracker_list.currentText().strip()
        
        # Find tracker ID from name
        tracker_id = None
        for tid, tracker_info in self.w.trackers_dict.items():
            if tracker_info['name'] == tracker_name and tracker_info['available']:
                tracker_id = tid
                break
                
        if not tracker_id:
            return
        
        config_path = self.w.base_path / self.w.trackers_dict[tracker_id]['config_path']
        tracker_config = load_boxmot_tracker_config(config_path)
        
        if tracker_config:
            # Open the configuration dialog
            updated_config = open_boxmot_tracker_config_dialog(
                self.w,  # That's the parent GUI widget
                tracker_id, 
                tracker_config, 
                config_path
            )
            if updated_config:
                print(f"Configuration for {tracker_name} updated and saved")
                
    def on_one_object_per_label_clicked(self):
        """
        When the user clicks on one_object_per_label ("1 Subject") in the GUI: 
        - Select the first tracker in the list of Trackers
        - disable the tracker selection dropdown 
        - disable the tune (tracker) button 

        """
        is_checked = self.w.single_subject_checkBox.isChecked() 
        
        if is_checked:
            # When "1 Subject" is checked
            # Set to first actual tracker (index 1, not the header at index 0)
            self.w.yolomodel_tracker_list.setCurrentIndex(1)
            self.w.yolomodel_tracker_list.setEnabled(False)
            self.w.tune_tracker_btn.setEnabled(False)
            self.w.tune_tracker_btn.setText("")
        else:
            self.w.yolomodel_tracker_list.setEnabled(True)
            self.w.yolomodel_tracker_list.setCurrentIndex(0)
            self.w.tune_tracker_btn.setEnabled(False)
        
    # YOLO Prediction handling 
    def on_iou_thresh_change(self, value):
        """
        Callback for self.w.predict_iou_thresh_spinbox.
        If IOU threshold is < 0.01, 
        check and disable the 'single_subject_checkBox'.
        This is because at IOU < 0.01, only one object will be tracked
        by fusing all detections into 1 -> so it has the same effect. 
        """
        if value < 0.01:
            self.w.single_subject_checkBox.setChecked(True)
            self.w.single_subject_checkBox.setEnabled(False)
        else:
            self.w.single_subject_checkBox.setEnabled(True)
            self.w.single_subject_checkBox.setChecked(False)

    def on_mp4_predict_dropped_area(self, video_paths):
        """
        Adds .mp4 files for YOLO video prediction. 
        Callback for the prediction drop area.
        """
        vdict = self.videos_to_predict

        for v in video_paths:
            p = Path(v)
            if p.name in vdict:
                print(f"Video {p.name} already in prediction list.")
                continue
            if not p.exists():
                print(f"File {p} does not exist.")
                continue
            if p.suffix.lower() != '.mp4':
                print(f"File {p} is not an mp4 file.")
                continue

            # probe video metadata & store reader
            meta = probe_video(p)
            reader = FastVideoReader(p, read_format='rgb24')
            vdict[p.name] = {**meta, 'video': reader}

            # update the combo/list in the GUI
            lst = self.w.videos_for_prediction_list
            lst.addItem(p.name)
            n = len(vdict)
            lst.setItemText(0, f"Videos (n={n})" if n else "List of videos to be analyzed ...")
            print(f"Added video {p.name} to prediction list.")

    def on_video_prediction_change(self):
        """
        Handles removal of one or more videos from the YOLO prediction list.
        Callback for the currentIndexChanged signal.
        """
        lst = self.w.videos_for_prediction_list
        idx = lst.currentIndex()
        entries = [lst.itemText(i) for i in range(lst.count())]
        # zero = header, one = "remove" action, else do nothing
        if idx == 0:
            return
        elif idx == 1 and len(entries) > 2:
            dlg = remove_video_dialog(self.w, entries[2:])
            dlg.list_widget.setSelectionMode(dlg.list_widget.ExtendedSelection)
            dlg.exec_()
            if dlg.result() == QDialog.Accepted:
                selected_items = dlg.list_widget.selectedItems()
                if not selected_items:
                    # No items selected, reset and return
                    lst.setCurrentIndex(0)
                    return
                # Remove all selected videos
                for item in selected_items:
                    video_name = item.text()
                    lst.removeItem(lst.findText(video_name))
                    self.videos_to_predict.pop(video_name, None)
                    print(f'Removed video "{video_name}"')
                # Refresh header count
                n = len(self.videos_to_predict)
                lst.setItemText(0, f"Videos (n={n})" if n else "List of videos to be analyzed ...")
            lst.setCurrentIndex(0)
        else:
            lst.setCurrentIndex(0)

    def init_yolo_prediction_threaded(self):
        """
        This function manages the prediction of videos
        with custom trained YOLO models
        """
        if not self.w.project_path:
            show_warning("Please select a project directory first.")
            return
        if not hasattr(self, 'yolo'):
            show_warning("Please load YOLO first.")
            return
        
        index_model_list = self.w.yolomodel_trained_list.currentIndex()
        if index_model_list == 0:
            show_warning("Please select a YOLO model")
            return
        model_name = self.w.yolomodel_trained_list.currentText()
        # The self.trained_models dictionary contains the model name as last 5 folder names
        # in the project path as key, and the model path as value
        assert model_name in self.trained_models, \
            f"Model {model_name} not found in trained models: {self.trained_models}"
        self.model_predict_path = self.trained_models[model_name]
        # Tracker
        index_tracker_list = self.w.yolomodel_tracker_list.currentIndex()
        if index_tracker_list == 0:
            show_warning("Please select a tracker")
            return 
        # Check if there are any videos to predict 
        if not self.videos_to_predict:
            show_warning("Please select a video to predict.")
            return
        
        # Collect selected options         
        self.yolo_tracker_name = self.w.yolomodel_tracker_list.currentText().strip()                            
        self.view_prediction_results = self.w.open_when_finish_checkBox.isChecked()   
        self.one_object_per_label = self.w.single_subject_checkBox.isChecked()
        self.region_details = self.w.detailed_extraction_checkBox.isChecked()
        self.overwrite_predictions = self.w.overwrite_prediction_checkBox.isChecked()    
        # ... floating point selectors 
        self.mask_opening = int(round(self.w.predict_mask_opening_spinbox.value()))
        self.conf_thresh = float(self.w.predict_conf_thresh_spinbox.value())
        self.iou_thresh = float(self.w.predict_iou_thresh_spinbox.value())
        self.skip_frames = self.w.skip_frames_analysis_spinBox.value()
        
        # Deactivate the training data generation box 
        self.w.segmentation_bbox_decision_groupbox.setEnabled(False)
        self.w.train_generate_groupbox.setEnabled(False)
        # Create new prediction worker
        self._create_yolo_predictor()
        self.w.predict_start_btn.setStyleSheet('QPushButton { color: #e7a881;}')
        self.w.predict_start_btn.setText(f'↯ Predicting')
        self.w.predict_start_btn.setEnabled(False)
        self.yolo_prediction_worker.start()
        # Disable the annotation + training data generation tabs
        self.w.main_toolbox.widget(1).setEnabled(False) # Annotation
        self.w.main_toolbox.widget(2).setEnabled(False) # Training
        # Disable prediction controls during batch prediction
        self.w.predict_video_drop_groupbox.setEnabled(False)
        self.w.yolomodel_trained_list.setEnabled(False)
        self.w.yolomodel_tracker_list.setEnabled(False)
        self.w.tune_tracker_btn.setEnabled(False)
        self.w.open_when_finish_checkBox.setEnabled(False)
        self.w.single_subject_checkBox.setEnabled(False)
        self.w.overwrite_prediction_checkBox.setEnabled(False)
        self.w.predict_conf_thresh_spinbox.setEnabled(False)
        self.w.skip_frames_analysis_spinBox.setEnabled(False)

    def _create_yolo_predictor(self):
        # Create a new worker for YOLO prediction 
        self.yolo_prediction_worker = create_worker(self._yolo_predictor)
        self.yolo_prediction_worker.setAutoDelete(True) # auto destruct!
        self.yolo_prediction_worker.yielded.connect(self._update_prediction_progress)
        self.w.predict_overall_progressbar.setEnabled(True)  
        self.w.predict_current_video_progressbar.setEnabled(True)   
        self.w.predict_current_videoname_label.setEnabled(True)
        self.w.train_finishtime_label.setText('↬ ... waiting for estimate')    
        self.w.predict_finish_time_label.setEnabled(True)

    def _yolo_predictor(self):
        if not self.device_label:
            show_error("No device label found for YOLO.")
            return
        else:
            show_info(f"Predicting on device: '{self.device_label}'")
        
        # Call the training function which yields progress info
        for progress_info in self.yolo.predict_batch(
                                            videos_dict=self.videos_to_predict,
                                            model_path=self.model_predict_path,
                                            device=self.device_label,
                                            tracker_name=self.yolo_tracker_name,
                                            skip_frames=self.skip_frames,
                                            one_object_per_label=self.one_object_per_label,
                                            region_details=self.region_details,
                                            iou_thresh=self.iou_thresh,
                                            conf_thresh=self.conf_thresh,
                                            opening_radius=self.mask_opening,
                                            overwrite=self.overwrite_predictions, 
                                        ):

            # Yield the progress info back to the GUI thread
            yield progress_info

    def _update_prediction_progress(self, progress_info):
        """
        Handle prediction progress updates from the worker thread.
        Updates progress bars and displays timing information.
        
        Parameters
        ----------
        progress_info : dict
            Dictionary containing prediction progress information
        """
        stage = progress_info.get('stage', '')
        
        if stage == 'processing':
            # Update UI for video processing
            video_name = progress_info.get('video_name', '')
            video_index = progress_info.get('video_index', 0)
            total_videos = progress_info.get('total_videos', 1)
            frame = progress_info.get('frame', 0)
            total_frames = progress_info.get('total_frames', 1)
            frame_time = progress_info.get('frame_time', 0) 
            
            remaining_time = (total_frames * frame_time) - (frame * frame_time)
            finish_time = time.time() + remaining_time
            finish_time_str = ' '.join(time.ctime(finish_time).split()[:-1])

            # Update labels
            if len(video_name) > 21:
                prefix = '...'
            else:
                prefix = ''
            shortened_video_name = f'{prefix}{video_name[-21:]}'
            
            self.w.predict_current_videoname_label.setText(f"{shortened_video_name}")
            self.w.predict_finish_time_label.setText(f"{frame_time:.2f}s per frame | ~ {finish_time_str}")
            
            # Update progress bars
            self.w.predict_overall_progressbar.setMaximum(total_videos)
            self.w.predict_current_video_progressbar.setMaximum(total_frames)
            self.w.predict_overall_progressbar.setValue(video_index)
            self.w.predict_current_video_progressbar.setValue(frame)
            
        elif stage == 'video_complete':
            # Mark video as complete with checkmark in list
            video_name = progress_info.get('video_name', '')
            if video_name:
                lst = self.w.videos_for_prediction_list
                # Find the video in the list and add checkmark
                for i in range(lst.count()):
                    if lst.itemText(i) == video_name:
                        lst.setItemText(i, f"{video_name} ✓")
                        break
            
            # Show results? 
            save_dir = progress_info.get('save_dir', '')
            if self.view_prediction_results: 
                for label, track_id, _, _, _, _  in self.yolo.load_predictions(save_dir=save_dir):
                    print(f"Adding tracking result to viewer | Label: {label}, Track ID: {track_id}")
            
        elif stage == 'complete':
            self.yolo_prediction_worker.quit()
            # Reset video progress bar for next video
            self.w.predict_current_video_progressbar.setValue(0)
            self.w.predict_overall_progressbar.setValue(0)
            self.w.predict_current_video_progressbar.setEnabled(False)
            self.w.predict_overall_progressbar.setEnabled(False)
            
            self.w.predict_current_videoname_label.setText('')
            self.w.predict_finish_time_label.setText('')
            self.w.predict_current_videoname_label.setEnabled(False)
            self.w.predict_finish_time_label.setEnabled(False)
            
            # Re-enable UI elements
            self.w.predict_start_btn.setStyleSheet('')
            self.w.predict_start_btn.setText('▷ Predict')
            self.w.predict_start_btn.setEnabled(True)
            self.w.main_toolbox.widget(1).setEnabled(True)  # Re-enable Annotation tab
            self.w.main_toolbox.widget(2).setEnabled(True)  # Re-enable Training tab
            self.w.predict_video_drop_groupbox.setEnabled(True)
            self.w.yolomodel_trained_list.setEnabled(True)
            self.w.yolomodel_tracker_list.setEnabled(True)
            self.w.tune_tracker_btn.setEnabled(True)
            self.w.open_when_finish_checkBox.setEnabled(True)
            self.w.single_subject_checkBox.setEnabled(True)
            self.w.overwrite_prediction_checkBox.setEnabled(True)
            self.w.predict_conf_thresh_spinbox.setEnabled(True)
            self.w.skip_frames_analysis_spinBox.setEnabled(True)


    # Worker uncoupling functions
    def _uncouple_worker_polygons(self):
        try:
            self.polygon_worker.yielded.disconnect(self._polygon_yielded)
            self.polygon_worker.finished.disconnect(self._on_polygon_finished)
            self.polygon_worker.quit()
        except Exception as e:
            print(f"Error when uncoupling polygon worker: {e}")

    def _uncouple_worker_bboxes(self):
        try:
            self.bbox_worker.yielded.disconnect(self._bbox_yielded)
            self.bbox_worker.finished.disconnect(self._on_bbox_finished)
            self.bbox_worker.quit()
        except Exception as e:
            print(f"Error when uncoupling bbox worker: {e}")

    def _uncouple_worker_training_data(self):
        try:
            self.training_data_worker.yielded.disconnect(self._training_data_yielded)
            self.training_data_worker.finished.disconnect(self._on_training_data_finished)
            self.training_data_worker.quit()
        except Exception as e:
            print(f"Error when uncoupling training data worker: {e}")