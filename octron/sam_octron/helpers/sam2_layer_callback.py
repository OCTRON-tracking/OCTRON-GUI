# OCTRON SAM2 related callbacks
import time
import numpy as np
import cmasher as cmr
from napari.utils import DirectLabelColormap
from octron.sam_octron.helpers.sam2_octron import (
    SAM2_octron,
    run_new_pred,
)
from octron.sam_octron.helpers.sam2_colors import sample_maximally_different, create_semantic_colormap
from octron.sam_octron.helpers.sam2_zarr import mark_frames_annotated
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
                from octron.sam_octron.helpers.sam3_octron import SAM3_semantic_octron
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
                try:
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
                except Exception as e:
                    import traceback
                    print(f'❌ Error processing shape mask: {e}')
                    traceback.print_exc()
                    return
            if mask is not None:
                # mask can be None,
                # when new objects are added after tracking starts 
                # for the SAM2-HQ model. See comments in 
                # sam2hq_octron.add_new_mask / add_new_points_or_box
                prediction_layer.data[frame_idx] = mask
                mark_frames_annotated(prediction_layer.data, frame_idx)
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
        
        # Buffer box prompts per frame, capped to the most recent N.
        # The text prompt (label name) provides the category-level prior;
        # the boxes just show exemplar locations.  Too many box prompts
        # dilute the transformer's attention and cause mask quality to
        # degrade (more background co-selected).  2-3 boxes is the sweet
        # spot; beyond that, the additional exemplars add noise.
        _MAX_BOX_PROMPTS = 3
        if not hasattr(organizer_entry, '_semantic_box_prompts'):
            organizer_entry._semantic_box_prompts = {}
        if frame_idx not in organizer_entry._semantic_box_prompts:
            organizer_entry._semantic_box_prompts[frame_idx] = []
        
        organizer_entry._semantic_box_prompts[frame_idx].append(box_xyxy)
        # Keep only the N most recent boxes (drop oldest)
        if len(organizer_entry._semantic_box_prompts[frame_idx]) > _MAX_BOX_PROMPTS:
            organizer_entry._semantic_box_prompts[frame_idx] = \
                organizer_entry._semantic_box_prompts[frame_idx][-_MAX_BOX_PROMPTS:]
        all_boxes = organizer_entry._semantic_box_prompts[frame_idx]
        
        # Use the label name as a text prompt so the detector knows WHAT
        # to look for, not just WHERE the examples are.  SAM3 is a
        # vision-language model — with the generic "visual" text, a single
        # box prompt often yields low-confidence results.  Passing the
        # actual category name (e.g. "fly", "screw") dramatically improves
        # single-box detection quality.
        #
        # However, if the label name is not a real word the text encoder
        # recognises (e.g. "octo", abbreviations, project codes), the
        # text embedding actually *hurts* — it pushes scores to near zero.
        # We detect this by trying with text first; if the best score is
        # very low we automatically fall back to box-only mode ("visual").
        text_prompt = organizer_entry.label  # e.g. "fly", "screw"
        
        # Always reset text embeddings before detection so that switching
        # labels forces fresh text-encoding for the new category name.
        if hasattr(predictor, 'detector'):
            if hasattr(predictor.detector, 'text_embeddings'):
                predictor.detector.text_embeddings = {}
            if hasattr(predictor.detector, 'names'):
                predictor.detector.names = []
        
        print(f'SAM3 Mode B: Running detection for "{text_prompt}" with {len(all_boxes)} box prompt(s)...')
        
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
        
        # Run detection with text + box prompts.  If the label name
        # produces a very weak response (max score < 0.25), the text
        # encoder doesn't understand the word — retry without text so
        # the detector relies purely on visual similarity from the boxes.
        _TEXT_FALLBACK_THRESH = 0.25
        pred_masks, pred_scores, _ = predictor.detect(
            frame_idx=frame_idx,
            text=text_prompt,
            bboxes=all_boxes,
            conf_threshold=conf_threshold,
        )
        
        if pred_scores is not None and pred_scores.numel() > 0:
            best_score = pred_scores.max().item()
        else:
            best_score = 0.0
        
        if best_score < _TEXT_FALLBACK_THRESH:
            # Text prompt isn't helping — fall back to box-only ("visual")
            print(f'  ↳ Text "{text_prompt}" produced low scores (max {best_score:.3f}), '
                  f'retrying with box-only mode...')
            if hasattr(predictor, 'detector'):
                if hasattr(predictor.detector, 'text_embeddings'):
                    predictor.detector.text_embeddings = {}
                if hasattr(predictor.detector, 'names'):
                    predictor.detector.names = []
            pred_masks, pred_scores, _ = predictor.detect(
                frame_idx=frame_idx,
                bboxes=all_boxes,
                conf_threshold=conf_threshold,
            )
        
        if pred_masks is None or pred_masks.shape[0] == 0:
            show_warning('SAM3 Mode B: No objects detected.')
            return None
        
        n_detections = pred_masks.shape[0]
        
        # Sort detections by confidence (descending) so highest confidence gets priority
        sorted_indices = pred_scores.argsort(descending=True)
        
        # Encode each detected mask with a unique 1-based object ID
        # (0 = background). Higher-confidence masks get priority for overlapping pixels:
        # stamp highest-confidence first, only fill pixels still at 0.
        id_mask = np.zeros_like(
            pred_masks[0].cpu().numpy(), dtype=np.int16
        )
        for rank, idx in enumerate(sorted_indices):
            mask_id = rank + 1  # 1-based IDs
            binary = (pred_masks[idx] > 0).cpu().numpy()
            id_mask[(binary) & (id_mask == 0)] = mask_id
        
        # Get number of box prompts used
        n_boxes = len(organizer_entry._semantic_box_prompts.get(frame_idx, []))
        max_score = pred_scores.max().item() if pred_scores is not None and pred_scores.numel() > 0 else 0.0
        n_objects = len(np.unique(id_mask)) - 1  # Exclude background
        print(
            f'SAM3 Mode B: Detected {n_detections} objects ({n_objects} encoded) '
            f'using {n_boxes} box prompt(s). Max score: {max_score:.3f}'
        )
        
        # Update visual layer
        prediction_layer.data[frame_idx] = id_mask
        mark_frames_annotated(prediction_layer.data, frame_idx)
        
        # Assign distinct colors so each object ID is visually separable
        prediction_layer.colormap = create_semantic_colormap(n_objects, label_id=organizer_entry.label_id)
        prediction_layer.refresh()
        
        # Persist max object ID in zarr metadata so the colormap
        # can be restored when the project is reloaded later.
        if hasattr(prediction_layer.data, 'attrs'):
            prev_max = prediction_layer.data.attrs.get('max_object_id', 0)
            prediction_layer.data.attrs['max_object_id'] = max(int(n_objects), int(prev_max))
        
        # DO NOT call add_new_mask here - it corrupts detector state!
        # The tracker will be updated later when needed (before propagation)
        # Store the ID-encoded mask for later.  Replace the entire dict so
        # each label only keeps its most recent detection frame.  Without
        # this, old detections (e.g. frame 5) accumulate alongside new ones
        # (e.g. frame 25), causing the same physical objects to be
        # registered twice with different tracker IDs.
        organizer_entry._semantic_accumulated_masks = {frame_idx: id_mask}
        
        return id_mask
    
    
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
        from octron.sam_octron.helpers.sam3_octron import SAM3_semantic_octron
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
                    mark_frames_annotated(prediction_layer.data, frame_idx)
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
        
        print(f'⚡️ Prefetching {len(image_indices)} images, skipping {skip_frames - 1} frames | start: {current_frame}')
        _ = predictor.images[image_indices]
        
        # Pre-compute backbone features so propagation only needs track_step.
        # Limit to the cache capacity to avoid OOM.
        tracker = getattr(predictor, 'tracker', predictor)
        max_bb = getattr(tracker, '_max_cached_backbone_frames', 0)
        if (max_bb > 0
                and hasattr(tracker, 'inference_state')
                and tracker.inference_state):
            prefetch_indices = image_indices[:max_bb]
            t0 = time.perf_counter()
            import torch
            # Use batched backbone when available (SAM3); fall back to
            # per-frame computation for SAM2 or other predictors.
            if hasattr(tracker, '_prefetch_backbone_batch'):
                tracker._prefetch_backbone_batch(prefetch_indices, batch_size=4)
            elif hasattr(tracker, '_get_image_feature'):
                with torch.inference_mode():
                    for idx in prefetch_indices:
                        tracker._get_image_feature(tracker.inference_state, idx, batch_size=1)
            t1 = time.perf_counter()
            print(f'⚡️ Pre-computed backbone features for {len(prefetch_indices)} frames in {t1-t0:.2f}s')

    
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
            # Single GPU→CPU transfer for all objects instead of N separate ones
            all_masks = (out_mask_logits > 0).cpu().numpy().astype(np.uint8)
            for i, out_obj_id in enumerate(out_obj_ids):
                yield counter, out_frame_idx, out_obj_id, all_masks[i].squeeze(), last_run

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
                # Single GPU→CPU transfer for all objects instead of N separate ones
                all_masks = (out_mask_logits > 0).cpu().numpy().astype(np.uint8)
                for i, out_obj_id in enumerate(out_obj_ids):
                    yield counter, out_frame_idx, out_obj_id, all_masks[i].squeeze(), last_run
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