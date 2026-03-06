#### Changelog

## vers. 0.2
- Date: 2026-03-06
- 98 commits since v0.1

### Major Features
- SAM3 support — Initial integration of SAM3 segmentation models alongside SAM2, including new GUI elements, logit thresholding, and predictor management. Extensive renaming of folders and imports to remove SAM2-specific biases.
- Detection (bounding box) mode — Full support for YOLO detection training, prediction, and result loading in bounding-box-only mode. Includes dedicated bbox workers, watershed segmentation for bboxes (same as for segmentation), and `bbox_aspect_ratio` as a new standard parameter in prediction output
- Region properties dialog — New streamlined region properties selection UI (`region_props_dialog.py`) triggered when the user clicks "Detailed" in the Analysis tab.
- Batch prediction CLI improvements — Prepared `batch_predict` for easier command-line usage, with usability improvements for `videos_for_prediction_list` handling.

### Features & Enhancements
- Video management — New "delete" option when right-clicking on a video in the existing data table; checkmark indicator when prediction finishes.
- Model management — Trained model type detection with color coding; automatic model list refresh after training; simplified SAM2/YOLO model download prompts.
- Training improvements — Increased default patience from 50 to 100 epochs; detection and resume training support; re-enable buttons after training/dataset generation finishes; disable key dropdowns/buttons during training.
- GUI polish — Various layout, tooltip, and usability improvements; shorter track display (max 200); IOU spinbox behavior bugfix; improved annotated frame detection speed.
- SAM annotation loading via dedicated class. This makes it easier to load the SAM annotation results if someone needs it (and hasn't trained a YOLO model on those data).
- Accept `.mts` files for video conversion.
- GPU cache cleanup when videos/predictor change. This has the nice benefit that users don't have to quit the app anymore.

### Bug Fixes
- Fixed `float16`/`bfloat16` errors on CUDA.
- Fixed MPS image size and dtype issues.
- Fixed `ModuleNotFoundError: 'pkg_resources'` (tensorboard).
- Fixed prediction loading for detection mode.
- Fixed project folder change state handling. This now clears (refreshes) menus correctly.
- Fixed bugs in prediction boxes / dialogues when all predictions are skipped.
- Fixed frame index validation when loading results. It is now not anymore implicitly assumed that the csv and zarr ouput match.
- Fixed print statement for skipping frames. Now the GUI and print statement show the same info.
- Fixed SAM2/SAM3 annotation incompatibility detection. The user is now correctly informed that SAM2 and SAM3 annotation projects are incompatible (they can either annotate with SAM2 or SAM3 on a single video but cannot switch once started on that particular video).
- Bugfixes for multi-label case.

### Dependencies
- Bumped napari to 0.7.0a3 (with PyQt6 support).
- Bumped ultralytics to >=8.4.19.
- Added `timm`, `clip`, `huggingface_hub`, and `transformers` to dependencies.
- Updated SAM2 and boxmot wheel URLs.
- Set semantic model chunk size to 6 (from 15). This is because SAM3 multi (semantic) is very heavy and prediction latencies were a bit too long for my taste.

### Documentation & Build
- Updated README and notebooks.
- Updated YOLO model download URLs.
- Cleaned up `__init__` imports.

## vers. 0.1 (biorxiv publication)
- Date: 2025-12-20
- 37 commits since v0.0.5

### Features & Enhancements
- New "Detailed" analysis mode — Added `detailed_extraction_checkBox` in the Analysis tab. When active, extracts full regionprops; otherwise only bounding box information is used (`region_details` keyword for `batch_predict`).
- Mask buffering (up to 10x speedup) — Masks are now buffered in memory and flushed to zarr on disk in batches via a new `buffer_size` parameter in `predict_batch`.
- Improved tooltips — More explicit tooltips throughout the GUI.
- Better single-subject handling — Improved UX when "1 subject" is selected.
- Homogenized D-(Deep)OcSort naming across the codebase.

### Tracker Updates
- Fixed HybridSort object association — A bug in HybridSort prevented extraction of objects via `det_ind`. Object association is now done by proper index matching instead of unreliable bounding box similarity.
- Adapted to newest HybridSort fixes and updated `hybridsort.yaml` config.
- Adapted to newest boxmot tracker API changes.
- Switched BoostTrack `per_class` default to `False`.
- Increased tolerance (to 1%) for re-associating box coordinates; reorganized box coord sorting.
- Fixed `cmc_method` rejecting `"none"` as a value.
- Ensured stripped tracker names are passed correctly.

### Bug Fixes
- Fixed single object tracking.
- Fixed ID finding when `per_class` is `True`.
- Fixed `tracking_has_started` state management.
- Fixed CSV format parsing (header lines = 7).
- Fixed `remove_video_dialog` issues.
- Fixed bugs in video plugin and video transcoder plugin.
- Fixed a GUI element layout bug.
- Improved segmentation fault error verbosity.
- Tooltip and typo corrections.

### Dependencies
- Bumped napari from 0.6.5 → 0.6.6.
- Updated SAM2 requirements.
- Pinned boxmot to `main` branch (instead of release).

### Documentation & Build
- Added BoxMOT to README acknowledgments.
- Added sponsor information.
- Removed `post_install` from scripts section in `pyproject.toml`.
- Updated `post_install.py` and `pyproject.toml`.

## vers. 0.0.5
- Date: 2025-09-26
- 55 commits since v0.0.4

### Features & Enhancements
- Boxmot integration — Full integration of the boxmot library for multi-object tracking, replacing the previous tracking architecture. Includes support for BoostTrack, BotSort, ByteTrack, DeepOcSort, HybridSort, OcSort, and StrongSort trackers.
- ReID model selection in GUI — Users can now select re-identification models for tracking directly from the GUI.
- Tracker tuning dialog — New "Tune (Tracker)" button and per-tracker configuration UI (`tracker_config_ui.py`) for fine-tuning tracker parameters via YAML configs.
- `octron-test-gpu` CLI command — New utility function to test GPU access.
- Option to close holes in mask results via `YOLO_results`.
- Option to create annotation layers from notebook.
- Dynamic `rect=True/False` determination during prediction, with additional prediction metadata output.
- Configurable `max_det` parameter (default 2000) for both training and prediction.
- Editable training image size field in the GUI. Users can choose whatever image size they want for training the models.
- Overwrite checkbox added to the analysis tab.
- `set_project_folder()` method for programmatic project setup.
- Unlimited color labels/suffixes — The GUI no longer runs out of colors.

### Bug Fixes
- Fixed `load_sam2model` initialization.
- Fixed result loading after prediction (`_ensure_track_ids_loaded`, later removed in favor of a cleaner approach).
- Fixed `check_gpu_access`.
- Fixed label, track ID, and related field initialization.
- Fixed `rect=True/False` handling for prediction.
- Fixed Windows-specific blocking issue (reverted to single image sampling).
- Video names with whitespace are now handled correctly.

### Dependencies
- Bumped napari to 0.6.3.
- Bumped ultralytics and zarr versions.
- New `napari-pyav` dependency (0.0.10.1); removed direct AV requirement.
- Added `horsto/boxmot` release to dependencies.

### Other
- Removed solidity and all LAB feature functions.
- Disabled table view resizing.
- Added `CITATION.cff` for OCTRON citation guidelines.
- Constrained random image sampling during prediction for faster execution.
- Added boilerplate visualization code for polygons.
- Changed minimum size and ratio requirements for training data.

## vers. 0.0.4 
- Date 2025-05-08
- High level access to `YOLO_octron` and `YOLO_results` classes via, for example, `from octron import YOLO_results` 
- New notebook that explains the usage of these classes under "octron/notebooks/OCTRON_results_loading.ipynb"
- New `get_frame_indices` in `YOLO_results`: Helper to just return the valid frame indices for all track IDs.
- Enable `YOLO_results` to access results regardless of whether the original video was found or not. 
- Mask prediction results are now directly read from masks created in YOLO, instead of going through polygon-mask conversion steps. This is more efficient and less error prone. The Gaussian smoothing sigma parameter for prediction polygons (GUI and code) has been replaced with an `opening` (binary opening of masks) parameter. Morphological opening is (optionally) applied, which, similarly to the original smoothing sigma, can help to improve mask results. 
- Feature columns (eccentricity, area, ...) are now also interpolated with `interpolate=True` alongside position data
- During prediction frame and mask CIE LAB average values are extracted and saved in the .csv output. The experimenter thereby has access to color and brightness information for every frame and extracted mask after prediction completed. These values are new additions to the features columns, alongside eccentricity, area, etc. 
- Major update of ultralytics (8.3.152) that gets rid of an offest of mask vs. frame data introduced when the masks are scaled back to the original image size after prediction. See [PR 20957](https://github.com/ultralytics/ultralytics/pull/20957). 
- Created wheels for quick installation of py-av, sam2, and sam2-hq

## vers. 0.0.3
- Date: 2025-05-25
- Added buttons in annotation tab that allows users to skip to next / previous annotated frame. 
- Added `YOLO_results` class for unified access to OCTRON prediction results. Do `from octron import YOLO_results` to use this class. 
- Added skip frame functionality to analysis (prediction) of new videos in OCTRON which allows to analyze only a subset of the video frames in each video. 
- Added metadata export for each prediction that saves all parameters that the prediction has been run with.
- When loading prediction results via drag-n-drop on the OCTRON main window, the masks are now shown automatically and skipped frames are interpolated over. 
- MPS is now engaged on capable systems when training / predicting with YOLO.
- Retired YOLO Model X(tra large) from available model list for training since it is unnecessary.
- Shortened pruning method when creating training data. This needs to be checked for edge cases at some point, but runs much faster now. 

## vers. 0.0.2
- Date: 2025-04-15
- Initial working release.
- Implemented programmatic version update and tagging.