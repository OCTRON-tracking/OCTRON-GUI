# Main SAM2 predictor class for OCTRON
# This class is a subclass of the SAM2VideoPredictor class from the SAM2 library

import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # Leaving this here out of pure desperation

from collections import OrderedDict
import torch
import numpy as np
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import concat_points    

# I am using kornia for morphological operations
# Check https://kornia.readthedocs.io/en/latest/morphology.html#kornia.morphology.closing
from kornia.morphology import closing as kornia_closing
from torch import tensor as torch_tensor
from skimage.morphology import disk

# Custom Zarr archive class
from .sam2_zarr import OctoZarr

import warnings
warnings.simplefilter("ignore")

class SAM2_octron(SAM2VideoPredictor):
    """
    Subclass of SAM2VideoPredictor that adds some additional functionality for OCTRON.
    """
    def __init__(
        self,
        **kwargs,
    ):
        
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=True
        # if `add_all_frames_to_correct_as_cond` is True, we also append to the conditioning frame list any frame that receives a later correction click
        # if `add_all_frames_to_correct_as_cond` is False, we conditioning frame list to only use those initial conditioning frames
        add_all_frames_to_correct_as_cond=True
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
        self.perform_morphological_operations = False
        super().__init__(**kwargs)
       
        
        print('Loaded SAM2VideoPredictor OCTRON')
        
    @torch.inference_mode()
    def init_state(
        self,
        video_data,
        zarr_store,
    ):

        compute_device = self.device          
        # Sanity checks on video data
        assert len(video_data.shape) == 4, f"video data should have shape (num_frames, H, W, 3), got {video_data.shape}"
        assert video_data.shape[3] == 3, f"video data should be RGB and have shape (num_frames, H, W, 3), got {video_data.shape}"

        """Initialize an inference state."""
        inference_state = {}
        self.inference_state = inference_state 
        
        # Zarr store for the image data
        # zarr_chunk_size = zarr_store.chunks[0]
        # Replace the zarr array with the custom subclass
        self.images = OctoZarr(zarr_store, video_data) 
        # Store the image data zarr in the inference state
        inference_state["images"] = self.images
        
        num_frames, video_height, video_width, _ = video_data.shape
        inference_state["num_frames"] = num_frames 
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] =  video_height
        inference_state["video_width"]  =  video_width 
        
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["frames_tracked_per_obj"] = {}
        # Keep track of centroids 
        inference_state["centroids"] = {} # -> obj_id ->  frame_idx : centroid coordinates
        inference_state["areas"] = {} # -> obj_id ->  frame_idx : area of region
        
        # Compatibility with SAM2HQ model
        inference_state["tracking_has_started"] = False 
        
        # Warm up the visual backbone and cache the image feature on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        print('🚀 Initialized SAM2 model')
        
        self.video_data = video_data            
        
                
        
        #self.fill_hole_area = 200
        # For morphological operations 
        if self.perform_morphological_operations:
            # TODO Make configurable
            self.disk_size = 2
            self.closing_kernel = torch_tensor(disk(self.disk_size).tolist()).to(compute_device)
                    
    
    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        
        
        """
        This function is called both when new points / masks are added,
        as well as for (batched) video prediction.
        
        
        Run tracking on a single frame based on current inputs and previous memory.
        """


        # Retrieve correct image features
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(self.inference_state, frame_idx, batch_size)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        # optionally offload the output to CPU memory to save GPU space
        storage_device = self.inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        
        # Introduce morphological opening here
        if self.perform_morphological_operations:
            pred_masks_gpu = kornia_closing(pred_masks_gpu, kernel=self.closing_kernel)

        # TODO: This might be useful. 
        # Leaving out for now.
        # if self.fill_hole_area > 0:
        #     pred_masks_gpu = fill_holes_in_mask_scores(
        #         pred_masks_gpu, self.fill_hole_area
        #     )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(self.inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu


    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        processing_order=None,
        reverse=False,
        ):
        """
        Propagate the tracking results in the video.
        
        
        """
        # TODO: More checks for correct inputs
        self.propagate_in_video_preflight(self.inference_state)

        obj_ids = self.inference_state["obj_ids"]
        num_frames = self.inference_state["num_frames"]
        batch_size = self._get_obj_num(self.inference_state)

        # set start index, end index, and processing order
        if processing_order is None:
            if start_frame_idx is None:
                # default: start from the earliest frame with input points
                start_frame_idx = min(
                    t
                    for obj_output_dict in self.inference_state["output_dict_per_obj"].values()
                    for t in obj_output_dict["cond_frame_outputs"]
                )
            if max_frame_num_to_track is None:
                # default: track all the frames in the video
                max_frame_num_to_track = num_frames
            if reverse:
                end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
                if start_frame_idx > 0:
                    processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
                else:
                    processing_order = []  # skip reverse tracking if starting from frame 0
            else:
                end_frame_idx = min(
                    start_frame_idx + max_frame_num_to_track, num_frames - 1
                )
                processing_order = range(start_frame_idx, end_frame_idx + 1)
                    
        try:
            for frame_idx in processing_order:
                pred_masks_per_obj = []
                for _ in range(batch_size):
                    pred_masks_per_obj.append(None)
                for obj_idx in range(batch_size):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    # We skip those frames already in consolidated outputs (these are frames
                    # that received input clicks or mask). Note that we cannot directly run
                    # batched forward on them via `_run_single_frame_inference` because the
                    # number of clicks on each object might be different.
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        storage_key = "cond_frame_outputs"
                        current_out = obj_output_dict[storage_key][frame_idx]
                        device = self.inference_state["device"]
                        pred_masks = current_out["pred_masks"].to(device, non_blocking=True)
                        
                        # TODO: Reimplement this function 
                        # if self.clear_non_cond_mem_around_input:
                            # # clear non-conditioning memory of the surrounding frames
                            # self._clear_obj_non_cond_mem_around_input(
                            #     self.inference_state, frame_idx, obj_idx
                            # )
                    else:
                        storage_key = "non_cond_frame_outputs"
                        current_out, pred_masks = self._run_single_frame_inference(
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,  # run on the slice of a single object
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        
                        obj_output_dict[storage_key][frame_idx] = current_out
                        
                        #Clear all non conditioned output frames that are older than 16 frames
                        #https://github.com/facebookresearch/sam2/issues/196#issuecomment-2286352777
                        oldest_allowed_idx = frame_idx - 16
                        all_frame_idxs = obj_output_dict[storage_key].keys()
                        old_frame_idxs = [idx for idx in all_frame_idxs if idx < oldest_allowed_idx]
                        for old_idx in old_frame_idxs:
                            obj_output_dict[storage_key].pop(old_idx)
                            for objid in self.inference_state['output_dict_per_obj'].keys():
                                if old_idx in self.inference_state['output_dict_per_obj'][objid][storage_key]:
                                    self.inference_state['output_dict_per_obj'][objid][storage_key].pop(old_idx)
                        
                        
                        
                        
                                
                    self.inference_state["frames_tracked_per_obj"][obj_idx][frame_idx] = {
                        "reverse": reverse
                    }
                    pred_masks_per_obj[obj_idx] = pred_masks

                # Resize the output mask to the original video resolution (we directly use
                # the mask scores on GPU for output to avoid any CPU conversion in between)
                if len(pred_masks_per_obj) > 1:
                    all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                else:
                    all_pred_masks = pred_masks_per_obj[0]
                _, video_res_masks = self._get_orig_video_res_output(
                    self.inference_state, all_pred_masks
                )
                yield frame_idx, obj_ids, video_res_masks
        except Exception as e:
            print(e)
            pass

            
    @torch.inference_mode()
    def reset_state(self):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results()
        # Remove all object ids
        self.inference_state["obj_id_to_idx"].clear()
        self.inference_state["obj_idx_to_id"].clear()
        self.inference_state["obj_ids"].clear()
        self.inference_state["point_inputs_per_obj"].clear()
        self.inference_state["mask_inputs_per_obj"].clear()
        self.inference_state["output_dict_per_obj"].clear()
        self.inference_state["temp_output_dict_per_obj"].clear()
        self.inference_state["frames_tracked_per_obj"].clear()


    def _reset_tracking_results(self):
        """Reset all tracking inputs and results across the videos."""
        for v in self.inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in self.inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in self.inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in self.inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in self.inference_state["frames_tracked_per_obj"].values():
            v.clear()
        self.inference_state["tracking_has_started"] = False
            
    
    @torch.inference_mode()
    def remove_object(self, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state. If strict is True, we check whether
        the object id actually exists and raise an error if it doesn't exist.
        """
        try:
            old_obj_idx_to_rm = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        except AttributeError as e:
            print(e)
            return
        updated_frames = []
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return self.inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {self.inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(self.inference_state["obj_id_to_idx"]) == 1:
            self.reset_state()
            return self.inference_state["obj_ids"], updated_frames

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            self.inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            self.inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        for frame_idx in obj_input_frames_inds:
            self.clear_all_prompts_in_frame(
                self.inference_state, frame_idx, obj_id, need_output=False
            )

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in self.inference_state)
        old_obj_ids = self.inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        self.inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        self.inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        self.inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(self.inference_state["point_inputs_per_obj"])
        _map_keys(self.inference_state["mask_inputs_per_obj"])
        _map_keys(self.inference_state["output_dict_per_obj"])
        _map_keys(self.inference_state["temp_output_dict_per_obj"])
        _map_keys(self.inference_state["frames_tracked_per_obj"])

        # Step 3: Further collect the outputs on those frames in `obj_input_frames_inds`, which
        # could show an updated mask for objects previously occluded by the object being removed
        if need_output:
            temp_output_dict_per_obj = self.inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    self.inference_state,
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    self.inference_state, consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))

        return self.inference_state["obj_ids"], updated_frames
    
    
    ######## ADDING NEW POINTS AND MASKS ################################################################
    #####################################################################################################
    
    
    
    @torch.inference_mode()
    def add_new_points_or_box(
        self,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """
        Add new points or a box to a frame.
        
        Parameters
        ----------
        frame_idx : int
            The index of the frame to add the points or box to.
        obj_id : int
            The id of the object to add the points or box to.
        points : array-like, optional
            The points to add. If not provided, a box must be provided.
        labels : array-like, optional
            The labels for the points. If not provided, a box must be provided.
        clear_old_points : bool, optional
            Whether to clear old points. Default is True.
        normalize_coords : bool, optional
            Whether to normalize the coordinates of the points. Default is True.
        box : array-like, optional
            The box to add. If not provided, points must be provided.
            
        Returns
        -------
        frame_idx : int
            The index of the frame the points or box were added to.
        obj_ids : list
            The list of object ids the points or box were added to.
        video_res_masks : torch.Tensor
            The resized mask at the original video resolution.
        
        
        """
       
        obj_idx = self._obj_id_to_idx(self.inference_state, obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if not isinstance(box, torch.Tensor):
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
            box_coords = box.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)

        if normalize_coords:
            video_H = self.inference_state["video_height"]
            video_W = self.inference_state["video_width"]
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(self.inference_state["device"])
        labels = labels.to(self.inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            device = self.inference_state["device"]
            prev_sam_mask_logits = prev_out["pred_masks"].to(device, non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            self.inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            self.inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

   
    @torch.inference_mode()
    def add_new_mask(
        self,
        frame_idx,
        obj_id,
        mask,
    ):
        """
        Add a new mask to a frame.
        
        Parameter
        ----------
        frame_idx : int
            The index of the frame to add the mask to.
        obj_id : int
            The id of the object to add the mask to.
        mask : array-like
            The mask to add. 
        
        Returns
        -------
        frame_idx : int
            The index of the frame the mask was added to.
        obj_ids : list
            The list of object ids the mask was added to.
        video_res_masks : torch.Tensor
            The resized mask at the original video resolution.


        """
        obj_idx = self._obj_id_to_idx(self.inference_state, obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None]  # add batch and channel dimension
        mask_inputs_orig = mask_inputs_orig.float().to(self.inference_state["device"])

        # resize the mask if it doesn't match the model's image size
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = torch.nn.functional.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig

        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        obj_frames_tracked = self.inference_state["frames_tracked_per_obj"][obj_idx]
        is_init_cond_frame = frame_idx not in obj_frames_tracked
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = obj_frames_tracked[frame_idx]["reverse"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            self.inference_state,
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            self.inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks


#####################################################################################################


def run_new_pred(predictor,
                 frame_idx,
                 obj_id, 
                 labels,
                 points=None,
                 masks=None,
                 box=None,
                 **kwargs,
                 ):
    """
    Run a new prediction on the SAM2 model in OCTRON.
    This is a wrapper around the SAM2_octron functions that allow 
    for adding new points or masks.
    The function returns the mask image that can be re-added to the 
    viewer. 
    
    
    Parameters
    ----------
    predictor : SAM2_octron
        The SAM2_octron predictor object.
    frame_idx : int
        The current frame index - this is the frame to run the prediction on.
    obj_id : int
        The current object id for prediction.
        Only one ID is allowed for now.
    labels : list
        A list of labels for the points. 
        These can be 0 (negative click) or 1 (positive click).
    points : list
        The points to run the prediction on.
    masks : np.array
        The masks to run the prediction on.
    box : list
        Box coordiunates: [top_left[1],top_left[0],bottom_right[1],bottom_right[0]]
    **kwargs : dict
        Additional keyword arguments.
        clear_old_points : bool
            Whether to clear old points. Default is True.
        normalize_coords : bool
            Whether to normalize the coordinates. Default is True.
    
    Returns
    -------
    mask : np.array
        The mask image that can be re-added to the viewer.
        Returns None if the frame index is None.
        This happens if SAM2 HQ tries to add new objects on existing tracking.
    
    """
    clear_old_points = kwargs.get('clear_old_points', True)
    normalize_coords = kwargs.get('normalize_coords', True)

    assert isinstance(obj_id, int), f'Object ID must be an integer, got {type(obj_id)}'
    if isinstance(labels, int):
        assert labels in [0,1], f'Label must be 0 or 1, got "{labels}"'
    elif isinstance(labels, list):
        for l_ in set(labels):
            assert l_ in [0,1], f'Labels must be 0 or 1, got {set(labels)}'
    else:
        raise ValueError(f'Labels must be an integer or a list, got {type(labels)}')
    assert points is not None or masks is not None or box is not None, \
        f'Either point, box or mask input must be provided'
        
    if points is not None:
        points_length = len(points) if isinstance(points, (list, np.ndarray)) else 1
        labels_length = len(labels) if isinstance(labels, (list, np.ndarray)) else 1
        assert points_length == labels_length, f'Number of points and labels must match,\
            got {points_length} points and {labels_length} labels'
    if box is not None:
        assert len(box) == 4, f'Box input must have 4 numbers [y1,x1,y2,x2], got {len(box)}'
        
    ########### MASK INPUT ####################################################################
    if masks is not None:
        assert len(masks.shape) == 2, f'Input masks must be 2D, got {masks.shape}'
        frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
                                                    frame_idx=frame_idx,
                                                    obj_id=obj_id,
                                                    mask=np.array(masks,dtype=bool),
                                                    )
        # Function returns None, None, None in case a new object is added for SAM2-HQ 
        # This is because SAM2-HQ does not allow adding new objects on top of existing ones.
        # See comments in sam2hq_octron.add_new_mask
        if frame_idx is None:
            return None 
        index_obj_id = obj_ids.index(obj_id)
        mask = (video_res_masks[index_obj_id] > 0).cpu().numpy().astype(np.uint8)
                
    ########### POINT INPUT ###################################################################
    if points is not None:
        frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                                    frame_idx=frame_idx,
                                                    obj_id=obj_id,
                                                    points=np.array(points,dtype=np.float32),
                                                    labels=np.array(labels, np.int32),
                                                    clear_old_points=clear_old_points,
                                                    normalize_coords=normalize_coords
                                                    )
        # Function returns None, None, None in case a new object is added for SAM2-HQ 
        # This is because SAM2-HQ does not allow adding new objects on top of existing ones.
        # See comments in sam2hq_octron.add_new_points_or_box
        if frame_idx is None:
            return None
        index_obj_id = obj_ids.index(obj_id)
        mask = (video_res_masks[index_obj_id] > 0).cpu().numpy().astype(np.uint8)
        
    ########### BOX INPUT #####################################################################
    if box is not None:
        frame_idx, obj_ids, video_res_masks = predictor.add_new_points_or_box(
                                                    frame_idx=frame_idx,
                                                    obj_id=obj_id,
                                                    box=box,
                                                    clear_old_points=clear_old_points,
                                                    normalize_coords=normalize_coords
                                                    )
        # Function returns None, None, None in case a new object is added for SAM2-HQ 
        # This is because SAM2-HQ does not allow adding new objects on top of existing ones.
        # See comments in sam2hq_octron.add_new_points_or_box
        if frame_idx is None:
            return None
        index_obj_id = obj_ids.index(obj_id)
        mask = (video_res_masks[index_obj_id] > 0).cpu().numpy().astype(np.uint8)
    
    mask = mask.squeeze() # From 4D => 2D (the first 2 dimensions are always 1)
    return mask

if __name__ == "__main__":
    pass