# OCTRON SAM2 related callbacks
import time
import numpy as np
from octron.sam2_octron.helpers.sam2_octron import (
    SAM2_octron,
    run_new_pred,
)
from napari.utils.notifications import (
    show_warning,
    show_info,
    show_error,
)

import warnings 
warnings.simplefilter("ignore")

class sam2_octron_callbacks():
    """
    Callback for octron and SAM2.
    """
    def __init__(self, octron):
        # Store the reference to the main OCTRON widget
        self.octron = octron
        self.viewer = octron._viewer
        self.left_right_click = None
            
    def on_mouse_press(self, layer, event):
        """
        Generic function to catch left and right mouse clicks
        """
        if event.type == 'mouse_press':
            if event.button == 1:  # Left-click
                self.left_right_click = 'left'
            elif event.button == 2:  # Right-click
                self.left_right_click = 'right'     
        
    
    def on_shapes_changed(self, event):
        """
        Callback function for napari annotation "Shapes" layer.
        This function is called whenever changes are made to annotation shapes.
        It extracts the mask from the shapes layer and runs the predictor on it.
        
        There is one special case for the "rectangle tool", which acts as "box input" 
        to SAM2 instead of creating an input mask.
        
        """
        shapes_layer = event.source
        if not len(shapes_layer.data):
            # This happens when the SAM2 predictor is reset by the user
            return
        if not self.octron.predictor:
            show_warning('No model loaded.')
            return
        if not self.octron.prefetcher_worker:
            show_warning('Prefetcher worker not initialized.')
            return
        
        action = event.action
        if action in ['added','removed','changed']:
            frame_idx = self.viewer.dims.current_step[0] 
            obj_id = shapes_layer.metadata['_obj_id']
            
            # Get the corresponding mask layer 
            organizer_entry = self.octron.object_organizer.entries[obj_id]
            prediction_layer = organizer_entry.prediction_layer
            if prediction_layer is None:
                # That should actually never happen 
                print('No corresponding prediction layer found.')
                return   
            
            video_height = self.octron.video_layer.metadata['height']    
            video_width = self.octron.video_layer.metadata['width']   
            predictor = self.octron.predictor
            
            ############################################################    
            
            if shapes_layer.mode == 'add_rectangle':
                if action == 'removed':
                    return
                # Take care of box input first. 
                # If the rectangle tool is selected, extract "box" coordinates
                box = shapes_layer.data[-1]
                if len(box) > 4:
                    box = box[-4:]
                # Find out what the top left and bottom right coordinates are
                box = np.stack(box)[:,1:]
                box_sum = np.sum(box, axis=1)
                top_left_idx = np.argmin(box_sum, axis=0)
                bottom_right_idx = np.argmax(box_sum, axis=0)
                top_left, bottom_right = box[top_left_idx,:], box[bottom_right_idx,:]
                
                # Check if Mode B (semantic detection) is active
                from octron.sam2_octron.helpers.sam3_octron import SAM3_semantic_octron
                if isinstance(predictor, SAM3_semantic_octron):
                    # Mode B: detect ALL similar objects, then add each to tracker
                    mask = self._handle_semantic_box_detection(
                        predictor=predictor,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        organizer_entry=organizer_entry,
                        prediction_layer=prediction_layer,
                        top_left=top_left,
                        bottom_right=bottom_right,
                    )
                else:
                    # Mode A / SAM2: single-instance box prompt
                    mask = run_new_pred(predictor=predictor,
                                        frame_idx=frame_idx,
                                        obj_id=obj_id,
                                        labels=[1],
                                        box=[top_left[1],
                                             top_left[0],
                                             bottom_right[1],
                                             bottom_right[0]
                                             ],
                                        )
                shapes_layer.data = shapes_layer.data[:-1]
                shapes_layer.refresh()  
                
            else:
                # In all other cases, just treat shapes as masks 
                shape_masks = np.stack(shapes_layer.to_masks((video_height, video_width)))
                if len(shape_masks) == 1: 
                    shape_mask = shape_masks[0]
                else:
                    frame_indices = np.array([s[0][0] for s in shapes_layer.data]).astype(int)
                    valid_indices = np.argwhere(frame_indices == frame_idx)
                    valid_masks = shape_masks[valid_indices].squeeze()
                    if valid_masks.ndim == 3:
                        shape_mask = np.sum(valid_masks, axis=0)
                    else:
                        shape_mask = valid_masks
                shape_mask[shape_mask > 0] = 1
                shape_mask = shape_mask.astype(np.uint8)
            
                label = 1 # Always positive for now
                mask = run_new_pred(predictor=predictor,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    labels=label,
                                    masks=shape_mask,
                                    )
            if mask is not None:
                # mask can be None,
                # when new objects are added after tracking starts 
                # for the SAM2-HQ model. See comments in 
                # sam2hq_octron.add_new_mask / add_new_points_or_box
                prediction_layer.data[frame_idx] = mask
                prediction_layer.refresh()
  
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return
    
    
    def _handle_semantic_box_detection(
        self,
        predictor,
        frame_idx,
        obj_id,
        organizer_entry,
        prediction_layer,
        top_left,
        bottom_right,
    ):
        """
        Handle Mode B (SAM3 semantic detection) when the user draws a box.
        
        Runs detection to find ALL similar objects in the frame, then:
        - Uses the existing organizer entry for the first detected object
        - Creates new organizer entries for each additional detection
        
        Parameters
        ----------
        predictor : SAM3_semantic_octron
            The semantic predictor.
        frame_idx : int
            Current frame index.
        obj_id : int
            Object ID of the entry the user drew the box on.
        organizer_entry : Obj
            The organizer entry for the current object.
        prediction_layer : napari.layers.Labels
            The prediction layer for the current object.
        top_left : np.ndarray
            Top-left corner [y, x] in video pixel coordinates.
        bottom_right : np.ndarray
            Bottom-right corner [y, x] in video pixel coordinates.
            
        Returns
        -------
        mask : np.ndarray or None
            The mask for the first detected object (the one the user drew on).
        """
        # Convert to xyxy format: [x1, y1, x2, y2]
        box_xyxy = [
            float(top_left[1]),    # x1
            float(top_left[0]),    # y1
            float(bottom_right[1]),# x2 
            float(bottom_right[0]),# y2
        ]
        
        # Helper function to calculate IoU (Intersection over Union) between two boxes
        def calculate_iou(box1, box2):
            """Calculate IoU between two boxes in [x1, y1, x2, y2] format."""
            x1_inter = max(box1[0], box2[0])
            y1_inter = max(box1[1], box2[1])
            x2_inter = min(box1[2], box2[2])
            y2_inter = min(box1[3], box2[3])
            
            if x2_inter < x1_inter or y2_inter < y1_inter:
                return 0.0
            
            inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
            box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = box1_area + box2_area - inter_area
            
            return inter_area / union_area if union_area > 0 else 0.0
        
        # Buffer box prompts - accumulate all boxes for this frame/object
        if not hasattr(organizer_entry, '_semantic_box_prompts'):
            organizer_entry._semantic_box_prompts = {}
        if frame_idx not in organizer_entry._semantic_box_prompts:
            organizer_entry._semantic_box_prompts[frame_idx] = []
        
        # Check if new box overlaps significantly with existing boxes (IoU > 0.5)
        existing_boxes = organizer_entry._semantic_box_prompts[frame_idx]
        is_redundant = False
        iou_threshold = 0.5
        
        for existing_box in existing_boxes:
            iou = calculate_iou(box_xyxy, existing_box)
            if iou > iou_threshold:
                is_redundant = True
                show_warning(
                    f'SAM3 Mode B: New box overlaps {iou:.2f} with existing box. '
                    f'Skipping to avoid redundancy. Draw boxes around DIFFERENT objects.'
                )
                break
        
        if is_redundant:
            # Don't add the box, just return current mask without re-detecting
            current_mask = prediction_layer.data[frame_idx]
            return current_mask if current_mask is not None else np.zeros(
                (self.octron.video_layer.metadata['height'], 
                 self.octron.video_layer.metadata['width']),
                dtype=np.uint8
            )
        
        # Add the new box to the buffer
        organizer_entry._semantic_box_prompts[frame_idx].append(box_xyxy)
        all_boxes = organizer_entry._semantic_box_prompts[frame_idx]
        
        # Always reset text embeddings before detection to prevent state corruption
        # This is especially important after propagation, which can leave stale embeddings
        if hasattr(predictor, 'detector'):
            if hasattr(predictor.detector, 'text_embeddings'):
                predictor.detector.text_embeddings = {}
            if hasattr(predictor.detector, 'names'):
                predictor.detector.names = []
        
        show_info(f'SAM3 Mode B: Running detection with {len(all_boxes)} box prompt(s)...')
        
        # Read detection threshold from GUI input
        thresh_text = self.octron.sam3detect_thresh.text().strip()
        if not thresh_text:
            conf_threshold = 0.5
        else:
            try:
                conf_threshold = float(thresh_text)
            except (ValueError, TypeError):
                show_error(f'Invalid detection threshold: "{thresh_text}". Must be a number between 0 and 1.')
                return None
            if not (0.0 <= conf_threshold <= 1.0):
                show_error(f'Detection threshold {conf_threshold} out of range. Must be between 0 and 1.')
                return None
        
        # Run detection with ALL accumulated boxes together
        # This allows the model to see all examples simultaneously
        pred_masks, pred_scores, _ = predictor.detect(
            frame_idx=frame_idx,
            bboxes=all_boxes,  # Pass all boxes, not just the latest one
            conf_threshold=conf_threshold,
        )
        
        if pred_masks is None or pred_masks.shape[0] == 0:
            show_warning('SAM3 Mode B: No objects detected.')
            return None
        
        n_detections = pred_masks.shape[0]
        
        # Merge all detected masks into a single combined mask
        combined_mask = np.zeros_like(
            pred_masks[0].cpu().numpy(), dtype=np.uint8
        )
        for i in range(n_detections):
            combined_mask |= (pred_masks[i] > 0).cpu().numpy().astype(np.uint8)
        
        # Get number of box prompts used
        n_boxes = len(organizer_entry._semantic_box_prompts.get(frame_idx, []))
        max_score = pred_scores.max().item() if pred_scores is not None and pred_scores.numel() > 0 else 0.0
        show_info(
            f'SAM3 Mode B: Detected {n_detections} objects using {n_boxes} box prompt(s). '
            f'Max score: {max_score:.3f}'
        )
        
        # Update visual layer
        prediction_layer.data[frame_idx] = combined_mask
        prediction_layer.refresh()
        
        # DO NOT call add_new_mask here - it corrupts detector state!
        # The tracker will be updated later when needed (before propagation)
        # Store the accumulated mask for later
        if not hasattr(organizer_entry, '_semantic_accumulated_masks'):
            organizer_entry._semantic_accumulated_masks = {}
        organizer_entry._semantic_accumulated_masks[frame_idx] = combined_mask
        
        return combined_mask
    
    
    def on_points_changed(self, event):
        """
        Callback function for napari annotation "Points" layer.
        This function is called whenever changes are made to annotation points.

        """
        points_layer = event.source
        if not len(points_layer.data):
            # This happens when the SAM2 predictor is reset by the user
            return
        if not self.octron.predictor:
            show_warning('No model loaded.')
            return
        if not self.octron.prefetcher_worker:
            show_warning('Prefetcher worker not initialized.')
            return
        
        action = event.action
        predictor = self.octron.predictor
        frame_idx  = self.viewer.dims.current_step[0] 
        obj_id = points_layer.metadata['_obj_id']
        
        # Check if using SAM3 semantic mode with points (not supported)
        from octron.sam2_octron.helpers.sam3_octron import SAM3_semantic_octron
        if isinstance(predictor, SAM3_semantic_octron) and action == 'added':
            show_warning(
                'SAM3 semantic mode does not support point prompts for detection. '
                'Point prompts will perform single-object segmentation only. '
                'Use the rectangle tool (box prompt) for semantic detection of all similar objects.'
            )
        
        # Get the corresponding mask layer 
        organizer_entry = self.octron.object_organizer.entries[obj_id]
        prediction_layer = organizer_entry.prediction_layer
        color = organizer_entry.color
        
        if prediction_layer is None:
            # That should actually never happen 
            print('No corresponding prediction (mask) layer found.')
            return    
        
        if action == 'added':
            # A new point has just been added. 
            # Find out if you are dealing with a left or right click    
            if self.left_right_click == 'left':
                label = 1
                points_layer.face_color[-1] = color
                points_layer.border_color[-1] = [.7, .7, .7, 1]
                points_layer.symbol[-1] = 'o'
            elif self.left_right_click == 'right':
                label = 0
                points_layer.face_color[-1] = [.7, .7, .7, 1]
                points_layer.border_color[-1] = color 
                points_layer.symbol[-1] = 'x'
            points_layer.refresh() # THIS IS IMPORTANT
            # Prefetch next batch of images
            if not self.octron.prefetcher_worker.is_running:
                self.octron.prefetcher_worker.run()
            
        # Loop through all the data and create points and labels
        if action in ['added','removed','changed']:
            labels = []
            point_data = []
            for pt_no, pt in enumerate(points_layer.data):
                if pt[0] != frame_idx:  
                    continue
                # Find out which label was attached to the point
                # by going through the symbol lists
                cur_symbol = points_layer.symbol[pt_no]
                if cur_symbol in ['o','disc']:
                    label = 1
                else:
                    label = 0
                labels.append(label)
                point_data.append(pt[1:][::-1]) # index 0 is the frame number
            
            if point_data:
                # Then run the actual prediction
                mask = run_new_pred(predictor=predictor,
                                    frame_idx=frame_idx,
                                    obj_id=obj_id,
                                    labels=labels,
                                    points=point_data,
                                    )
                if mask is not None:
                    # mask can be None,
                    # when new objects are added after tracking starts 
                    # for the SAM2-HQ model. See comments in 
                    # sam2hq_octron.add_new_mask / add_new_points_or_box
                    prediction_layer.data[frame_idx,:,:] = mask
            prediction_layer.refresh()  
        else:
            # Catching all above with ['added','removed','changed']
            pass
        return    
    
    
    def prefetch_images(self):
        """
        Thread worker for prefetching images for fast processing in the viewer
        This also initializes the SAM2 model if it is not yet initialized.
        WHY? Because we need the SAM2 model to register the fetched images in its inference state.
        TODO: There might be a more elegant way to handle this. 
        """
        predictor = self.octron.predictor
        assert predictor, "No model loaded."
        self.octron.init_sam2_model() # This initializes the model if it is not yet initialized
        
        viewer = self.octron._viewer    
        video_layer = self.octron.video_layer   
        num_frames = video_layer.metadata['num_frames']
        # Chunk size and skipping of frames
        chunk_size = self.octron.chunk_size
        skip_frames = self.octron.skip_frames   
        
        current_indices = viewer.dims.current_step
        current_frame = current_indices[0]
        
        # Create slice and check if there are enough frames to prefetch
        end_frame = min(num_frames-1, current_frame + chunk_size * skip_frames)
        image_indices = list(range(current_frame, end_frame, skip_frames))
        if not image_indices:
            return
        
        print(f'⚡️ Prefetching {len(image_indices)} images, skipping {skip_frames} frames | start: {current_frame}')
        _ = predictor.images[image_indices]

    
    def next_predict(self):
        """
        Threaded function to run the predictor forward on exactly one frame.
        Uses SAM2 => propagate_in_video function.
        
        """    

        # Prefetch images if they are not cached yet 
        # For this, reset the chunk_size to 1
        # This will ensure we are only prefetching one frame
        # At the end of the function, we will reset the chunk_size to the original value
        skip_frames = self.octron.skip_frames_spinbox.value()
        if skip_frames < 1:
            skip_frames = 1 # Just hard reset any unrealistic values here
        self.octron.skip_frames = skip_frames  
        chunk_size_real = self.octron.chunk_size
        self.octron.chunk_size = 1
        if getattr(self.octron.predictor, 'images', None) is not None:
            self.prefetch_images()
        
        current_frame = self.viewer.dims.current_step[0]         
        num_frames = self.octron.video_layer.metadata['num_frames']
        end_frame = min(num_frames-1, current_frame + self.octron.chunk_size * skip_frames)
        image_idxs = [current_frame, end_frame]
        
        start_time = time.time()    
        # Just copying routine here from the batch_predict function    
        # Loop over frames and run prediction (single frame!)
        counter = 1
        for out_frame_idx, out_obj_ids, out_mask_logits in self.octron.predictor.propagate_in_video(
            processing_order=image_idxs
        ):
            
            if counter == 1:
                last_run = True
            else:
                last_run = False
            for i, out_obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                yield counter, out_frame_idx, out_obj_id, mask.squeeze(), last_run

            counter += 1
            
        end_time = time.time()
        print(f'Start idx {current_frame} | Predicted 1 frame in {end_time-start_time:.2f} seconds')
        self.octron.chunk_size = chunk_size_real
        return

    
    def batch_predict(self):
        """
        Threaded function to run the predictor forward on a batch of frames.
        Uses SAM2 => propagate_in_video function.
        
        """

        skip_frames = self.octron.skip_frames_spinbox.value()
        if skip_frames < 1:
            skip_frames = 1 # Just hard reset any unrealistic values here
        elif skip_frames >= 1: # The user expects that the skip_frames are 1-based!
            skip_frames +=1
        self.octron.skip_frames = skip_frames  

        # Prefetch images if they are not cached yet 
        if getattr(self.octron.predictor, 'images', None) is not None:
            self.prefetch_images()
        
        current_frame = self.viewer.dims.current_step[0]         
        num_frames = self.octron.video_layer.metadata['num_frames']
        end_frame = min(num_frames-1, current_frame + self.octron.chunk_size * skip_frames)
        image_idxs = list(range(current_frame, end_frame, skip_frames)) 
        start_time = time.time()        
        # Loop over frames and run prediction (single frame!)   
        counter = 1
        for out_frame_idx, out_obj_ids, out_mask_logits in self.octron.predictor.propagate_in_video(
            processing_order=image_idxs
            ):
            
            if counter == end_frame:
                last_run = True
            else:
                last_run = False
            try:
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[i] > 0).cpu().numpy().astype(np.uint8)
                    yield counter, out_frame_idx, out_obj_id, mask.squeeze(), last_run
            except Exception as e:
                # Trying again for sam hq 
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask = out_mask_logits[0][out_mask_logits[0] == out_obj_id]
                    mask = mask.cpu().numpy().astype(np.uint8)
                    yield counter, out_frame_idx, out_obj_id, mask.squeeze(), last_run
                
            counter += 1
            
        end_time = time.time()
        print(f'Start idx {current_frame} | Predicted {self.octron.chunk_size} frames in {end_time-start_time:.2f} seconds')
        
        return