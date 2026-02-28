# This file contains helper functions to add layers to the napari viewer through OCTRON
from pathlib import Path
import numpy as np
from napari.utils import Colormap
from napari.utils.notifications import show_info, show_error
from octron.sam2_octron.helpers.sam2_zarr import create_image_zarr, load_image_zarr
import warnings 
warnings.simplefilter("ignore")

def add_sam2_mask_layer(viewer,
                        video_layer,
                        name,
                        project_path,
                        color,
                        video_hash_abrrev=None,
                        ):
    """
    Generic mask layer for napari and SAM2.
    Initiates the mask layer, a napari labels layer instance,
    and fixes it's color to "base_color'.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new mask layer.
    project_path : str or Path
        Path to the project directory.
    color : str or list
        Color of the mask layer. w
    video_hash_abbrev : str, optional
        Abbreviated hash of the video file. This is used as 
        a unique identifier for the corresponding video file throughout.
        
    Returns
    -------
    labels_layer : napari.layers.Labels
        Labels layer object.
    layer_data : zarr.core.Array
        Zarr array object.
    zarr_file_path : Path
        Absolute path to the zarr file/folder.

    """
    project_path = Path(project_path)
    
    assert project_path.exists(), f"Project path {project_path.as_posix()} does not exist."  

    # Check if required metadata exists before creating the dummy mask
    required_keys = ['num_frames', 'height', 'width']
    if all(k in video_layer.metadata for k in required_keys):        
        # Create a zarr array for the mask (prediction) data
        num_frames = video_layer.metadata['num_frames']
        height = video_layer.metadata['height']
        width = video_layer.metadata['width']
        zarr_file_path = project_path / f"{name}.zarr"
        status = False
        if zarr_file_path.exists():
            layer_data, status = load_image_zarr(zarr_file_path, 
                                                num_frames=num_frames, 
                                                image_height=height, 
                                                image_width=width, 
                                                chunk_size=20,
                                                video_hash_abrrev=video_hash_abrrev,
                                                )
            if status:
                print(f"Prediction (mask) layer data found at {zarr_file_path.as_posix()}")
            else:
                show_error(f"Failed to load Zarr array from {zarr_file_path.as_posix()}")
        if not zarr_file_path.exists() or not status:
            layer_data = create_image_zarr(zarr_file_path, 
                                        num_frames=num_frames, 
                                        image_height=height, 
                                        image_width=width, 
                                        chunk_size=20,
                                        fill_value=-1,
                                        dtype='int8',
                                        video_hash_abbrev=video_hash_abrrev,
                                        )
    else:
        show_error("Video layer metadata incomplete; dummy mask not created.")
        return None, None, None
    
    # Add the labels layer to the viewer
    labels_layer = viewer.add_labels(
        layer_data,
        name=name,  
        opacity=0.4,  
        blending='additive',  
        colormap=color, 
    )

    # Hide buttons that you don't want the user to access
    # TODO: This will be deprecated in future versions of napari.
    qctrl = viewer.window.qt_viewer.controls.widgets[labels_layer]
    buttons_to_hide =  ['erase_button',
                        'fill_button',
                        'paint_button',
                        'pick_button',
                        'polygon_button',
                        'transform_button',
                        ]
    for btn in buttons_to_hide: 
        getattr(qctrl, btn).hide() 
        
    return labels_layer, layer_data, zarr_file_path


def add_sam2_shapes_layer(
    viewer,
    name,
    color,
    semantic_mode=False,
    ):
    """
    Generic shapes layer for napari and SAM2.
    Initiates the shapes layer, a napari shapes layer instance,
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new shapes layer.
    base_color : str or list
        Color of the shapes layer.
    
    Returns
    -------
    shapes_layer : napari.layers.Shapes
        Shapes layer object.    
    
    """
    
    
    shapes_layer = viewer.add_shapes(None, 
                                 ndim=3,
                                 name=name, 
                                 scale=(1,1),
                                 edge_width=1,
                                 edge_color=color,
                                 face_color=[1,1,1,0],
                                 opacity=.4,
                                 )

    # Hide buttons that you don't want the user to access       
    # TODO: This will be deprecated in future versions of napari.
    qctrl = viewer.window.qt_viewer.controls.widgets[shapes_layer]
    if semantic_mode:
        buttons_to_hide = [
                        'select_button',
                        'direct_button',
                        'ellipse_button',
                        'line_button',
                        'path_button',
                        'polyline_button',
                        'polygon_button',
                        'polygon_lasso_button',
                        'vertex_insert_button',
                        'vertex_remove_button',
                        'move_front_button',
                        'move_back_button',
                        'delete_button',
                        ]
    else:
        buttons_to_hide = [
                        'line_button',
                        'path_button',
                        'polyline_button',
                        ]
    for btn in buttons_to_hide:
        attr = getattr(qctrl, btn)
        attr.hide()
        
    # Select the shapes layer and activate the rectangle tool
    viewer.layers.selection.active = shapes_layer
    viewer.layers.selection.active.mode = 'add_rectangle'
    return shapes_layer


def add_sam2_points_layer(    
    viewer,
    name,
    ):
    """
    Generic points layer for napari and SAM2.
    Initiates the points layer, a napari points layer instance,
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    video_layer : napari.layers.Image
        Video layer = video layer object
    name : str
        Name of the new points layer.
        
    Returns
    -------
    points_layer : napari.layers.Points
        Points layer object.
        
    """
    points_layer = viewer.add_points(None, 
                                 ndim=3,
                                 name=name, 
                                 scale=(1,1),
                                 border_color=[.7, .7, .7, 1],
                                 border_width=.2,
                                 opacity=.6,
                                 )

    # Select the current, add tool for the points layer
    viewer.layers.selection.active = points_layer
    viewer.layers.selection.active.mode = 'add'
    return points_layer


def add_annotation_projection(    
    viewer,
    object_organizer,
    label,
    ):
    """
    Creates a average projection of all masks for a given label.
    This visualizes the current annotation state for a given label 
    and lets the user decide on the quality of the annotation.
    
    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer object.
    object_organizer : octron.object_organizer.ObjectOrganizer
        Object organizer instance.
    label : str
        Label for which to create the projection.
    name : str

    """
    
    # Retrieve colors which are saved as part of the object organizer
    # since there they are used to assign unique colors to newly created label suffix combinations
    (label_colors, indices_max_diff_labels, _) = object_organizer.all_colors()
    
    collected_mask_data = [] # Data from prediction layers
    for entry in object_organizer.get_entries_by_label(label):
        # There might be multiple entries (with suffixes) for the same label
        # This is why we loop here over all entries for that label ... 
        
        prediction_layer_data = entry.prediction_layer.data # Mask data
        annotation_layer = entry.annotation_layer
        # Get color and make map 
        colors = label_colors[indices_max_diff_labels[entry.label_id % object_organizer.n_labels_max]]
        colors.insert(0, [0.,0.,0.,0.]) # Add transparent color for background
        cm = Colormap(colors, name=label, display_name=label)
        # Filter by prediction indices. The fill value is -1, so we can filter by >= 0
        predicted_indices = np.where(prediction_layer_data[:,0,0] >= 0)[0]
        if len(predicted_indices):
            prediction_layer_data = prediction_layer_data[predicted_indices]
            collected_mask_data.append(prediction_layer_data)
            annotation_layer.visible = False
            
    if not collected_mask_data:
        show_error(f"No masks found for label '{label}'.")
        return
    collected_mask_data = np.vstack(collected_mask_data)
    collected_mask_data_mean = np.mean(collected_mask_data, axis=0)
    viewer.add_image(collected_mask_data_mean, 
                    rgb=False, 
                    blending='additive',
                    opacity=0.75, 
                    interpolation2d='cubic', 
                    colormap=cm, 
                    name=f'Projection for {label} (n={collected_mask_data.shape[0]})',
                    )            
    
    return