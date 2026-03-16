"""
OCTRON prediction pipeline.

Wraps YOLO_octron.predict_batch() into a single callable function
that can be used from the CLI or called programmatically.
"""

from pathlib import Path


def run_predict(
    videos,
    model_path,
    device='auto',
    tracker_name=None,
    tracker_cfg_path=None,
    tracker_params=None,
    skip_frames=0,
    one_object_per_label=False,
    iou_thresh=0.7,
    conf_thresh=0.5,
    opening_radius=0,
    overwrite=True,
    buffer_size=500,
):
    """
    Run YOLO prediction and tracking on one or more videos.

    Parameters
    ----------
    videos : str, Path, or list
        Single video path, or a list of video paths.
    model_path : str or Path
        Path to the trained YOLO model (.pt file).
    device : str
        Device to run inference on ('auto', 'cpu', 'cuda', 'mps'). 'auto'
        selects CUDA if available, then MPS, then CPU.
    tracker_name : str, optional
        Tracker to use (e.g. 'ByteTrack', 'BotSort'). Either this or
        tracker_cfg_path must be provided.
    tracker_cfg_path : str or Path, optional
        Path to a custom boxmot tracker config YAML. Takes priority over
        tracker_name when provided.
    tracker_params : dict, optional
        Parameter overrides applied on top of the resolved tracker config
        (e.g. {'det_thresh': 0.5, 'max_age': 100}).
    skip_frames : int
        Number of frames to skip between predictions.
    one_object_per_label : bool
        Track only the highest-confidence detection per label per frame.
    iou_thresh : float
        IOU threshold for detection.
    conf_thresh : float
        Confidence threshold for detection.
    opening_radius : int
        Morphological opening radius applied to masks to remove noise.
    overwrite : bool
        Overwrite existing prediction results.
    buffer_size : int
        Number of frames buffered before writing to zarr.
    """
    from octron.yolo_octron.yolo_octron import YOLO_octron
    from octron.test_gpu import auto_device
    if device == 'auto':
        device = auto_device()

    yolo = YOLO_octron()

    print(f'Running prediction with model: {model_path}')
    for progress in yolo.predict_batch(
        videos=videos,
        model_path=model_path,
        device=device,
        tracker_name=tracker_name,
        tracker_cfg_path=tracker_cfg_path,
        tracker_params=tracker_params,
        skip_frames=skip_frames,
        one_object_per_label=one_object_per_label,
        iou_thresh=iou_thresh,
        conf_thresh=conf_thresh,
        opening_radius=opening_radius,
        overwrite=overwrite,
        buffer_size=buffer_size,
    ):
        stage = progress.get('stage', '')
        video = progress.get('video_name', '')
        vidx = progress.get('video_index', '?')
        total_v = progress.get('total_videos', '?')
        frame = progress.get('frame', '?')
        total_f = progress.get('total_frames', '?')
        pct = progress.get('overall_progress', 0)
        eta = progress.get('eta', 0)
        print(
            f'  [{stage}] video {vidx}/{total_v} ({video}): '
            f'frame {frame}/{total_f} | {pct:.1f}% | ETA: {eta:.0f}s',
            end='\r',
        )
    print()
    print('Prediction complete.')
