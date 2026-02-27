# Main SAM3 predictor class for OCTRON
# This wraps the ultralytics SAM3Model (which extends SAM2Model) with 
# the same OCTRON interface as SAM2_octron, using OctoZarr for image loading.

import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np

from kornia.morphology import closing as kornia_closing
from torch import tensor as torch_tensor
from skimage.morphology import disk

from .sam2_zarr import OctoZarr

import warnings
warnings.simplefilter("ignore")

# SAM3 constants
SAM3_IMAGE_SIZE = 1008
SAM3_BACKBONE_STRIDE = 14
SAM3_BB_FEAT_SIZES = [
    (288, 288),  # 1008 / 14 * 4
    (144, 144),  # 1008 / 14 * 2
    (72, 72),    # 1008 / 14
]
# SAM3 normalization: maps [0,1] to [-1,1]
SAM3_IMG_MEAN = (0.5, 0.5, 0.5)
SAM3_IMG_STD  = (0.5, 0.5, 0.5)

# Placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM3_octron:
    """
    OCTRON wrapper around the ultralytics SAM3Model for interactive video tracking.
    
    Provides the same interface as SAM2_octron:
        - init_state(video_data, zarr_store)
        - add_new_points_or_box(frame_idx, obj_id, points, labels, ...)
        - add_new_mask(frame_idx, obj_id, mask)
        - propagate_in_video(start_frame_idx, max_frame_num_to_track, ...)
        - reset_state()
        - remove_object(obj_id, ...)
    
    Uses the ultralytics SAM3Model.track_step which has the same signature as SAM2's
    (no image parameter), so the OctoZarr/backbone feature extraction pattern from 
    SAM2_octron carries over directly.
    """
    
    def __init__(
        self,
        model,
        device,
    ):
        self.model = model
        self.device = device
        self.inference_state = {}
        
        # Model geometry
        self.image_size = model.image_size  # 1008
        self.backbone_stride = model.backbone_stride  # 14
        self._bb_feat_sizes = SAM3_BB_FEAT_SIZES
        # pred_masks from track_step are at image_size // 4
        # (NOT _bb_feat_sizes[0] which is backbone_stride-based)
        self._pred_mask_res = (self.image_size // 4, self.image_size // 4)  # (252, 252)
        
        # Tracking config (same defaults as SAM2_octron)
        self.non_overlap_masks = False
        self.clear_non_cond_mem_around_input = True
        self.add_all_frames_to_correct_as_cond = True
        self.perform_morphological_operations = False
    
    
    @torch.inference_mode()
    def init_state(
        self,
        video_data,
        zarr_store,
    ):
        """
        Initialize an inference state for video tracking.
        
        Parameters
        ----------
        video_data : np.ndarray
            Video data with shape (num_frames, H, W, 3).
        zarr_store : zarr.core.Array
            Zarr array for caching preprocessed image features.
        """
        compute_device = self.device
        
        assert len(video_data.shape) == 4, \
            f"video data should have shape (num_frames, H, W, 3), got {video_data.shape}"
        assert video_data.shape[3] == 3, \
            f"video data should be RGB and have shape (num_frames, H, W, 3), got {video_data.shape}"
        
        inference_state = {}
        self.inference_state = inference_state
        
        # Zarr store for image data - OctoZarr handles resize + normalization
        # Override normalization constants for SAM3 (mean/std = 0.5 on [0,1] scale)
        self.images = OctoZarr(zarr_store, video_data)
        self.images.img_mean = torch.tensor(SAM3_IMG_MEAN, dtype=torch.float32)[:, None, None]
        self.images.img_std  = torch.tensor(SAM3_IMG_STD,  dtype=torch.float32)[:, None, None]
        inference_state["images"] = self.images
        
        num_frames, video_height, video_width, _ = video_data.shape
        inference_state["num_frames"]    = num_frames
        inference_state["video_height"]  = video_height
        inference_state["video_width"]   = video_width
        
        inference_state["device"] = compute_device
        inference_state["storage_device"] = compute_device
        
        # Per-object inputs
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # Visual feature cache
        inference_state["cached_features"] = {}
        # Constants (shared across frames)
        inference_state["constants"] = {}
        # Object ID ↔ index mappings
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # Shared output dict (batched across objects)
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        # Per-object output slices (sharing memory with output_dict)
        inference_state["output_dict_per_obj"] = {}
        # Temp outputs from user interactions (merged before propagation)
        inference_state["temp_output_dict_per_obj"] = {}
        # Consolidated frame indices 
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }
        # Tracking metadata
        inference_state["frames_already_tracked"] = []
        inference_state["tracking_has_started"] = False
        
        # OCTRON-specific: centroid & area tracking
        inference_state["centroids"] = {}
        inference_state["areas"] = {}
        
        # Warm up backbone on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        print('🚀 Initialized SAM3 model')
        
        self.video_data = video_data
        
        if self.perform_morphological_operations:
            self.disk_size = 2
            self.closing_kernel = torch_tensor(disk(self.disk_size).tolist()).to(compute_device)
    
    
    # ── Image feature extraction ──────────────────────────────────────────────
    
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Get backbone features for a frame, using OctoZarr for image loading.
        Caches features for recently visited frames.
        """
        image, backbone_out = inference_state["cached_features"].get(
            frame_idx, (None, None)
        )
        if backbone_out is None:
            # Load preprocessed image from OctoZarr
            image = inference_state["images"][frame_idx]
            if image.dim() == 3:
                image = image.unsqueeze(0)  # add batch dim: (1, C, H, W)
            image = image.to(self.device).float()
            backbone_out = self.model.forward_image(image)
            inference_state["cached_features"] = {frame_idx: (image, backbone_out)}
        
        # Expand features if batch_size > 1 (multi-object)
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"],
            "vision_pos_enc": backbone_out["vision_pos_enc"],
        }
        _, vis_feats, vis_pos_embed, feat_sizes = \
            self.model._prepare_backbone_features(expanded_backbone_out, batch=batch_size)
        
        return image, expanded_backbone_out, vis_feats, vis_pos_embed, feat_sizes
    
    
    # ── Single frame inference ────────────────────────────────────────────────
    
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
        Run tracking on a single frame based on current inputs and previous memory.
        Returns (compact_current_out, pred_masks_gpu) matching SAM2_octron's interface.
        """
        # Get backbone features from OctoZarr
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(self.inference_state, frame_idx, batch_size)
        
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
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
        
        storage_device = self.inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        
        pred_masks_gpu = current_out["pred_masks"]
        
        # Morphological closing on predicted masks
        if self.perform_morphological_operations:
            pred_masks_gpu = kornia_closing(pred_masks_gpu, kernel=self.closing_kernel)
        
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(current_out)
        obj_ptr = current_out["obj_ptr"]
        object_score_logits = current_out["object_score_logits"]
        
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }
        return compact_current_out, pred_masks_gpu
    
    
    # ── Memory position encoding cache ────────────────────────────────────────
    
    def _get_maskmem_pos_enc(self, current_out):
        """Cache maskmem_pos_enc since it's constant across frames."""
        out_maskmem_pos_enc = current_out.get("maskmem_pos_enc")
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in self.inference_state["constants"]:
                assert isinstance(out_maskmem_pos_enc, list)
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                self.inference_state["constants"]["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = self.inference_state["constants"]["maskmem_pos_enc"]
            # Expand to actual batch size
            batch_size = out_maskmem_pos_enc[0].shape[0]
            if batch_size > 1:
                out_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return out_maskmem_pos_enc
    
    
    # ── Object ID mapping ─────────────────────────────────────────────────────
    
    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx
        
        allow_new_object = not self.inference_state["tracking_has_started"]
        if allow_new_object:
            obj_idx = len(self.inference_state["obj_id_to_idx"])
            self.inference_state["obj_id_to_idx"][obj_id] = obj_idx
            self.inference_state["obj_idx_to_id"][obj_idx] = obj_id
            self.inference_state["obj_ids"] = list(self.inference_state["obj_id_to_idx"])
            # Initialize per-object storage
            self.inference_state["point_inputs_per_obj"][obj_idx] = {}
            self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
            self.inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            self.inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {self.inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )
    
    def _get_obj_num(self):
        return len(self.inference_state["obj_idx_to_id"])
    
    
    # ── Output consolidation (from ultralytics SAM2VideoPredictor) ────────────
    
    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        consolidate_at_video_res=False,
        run_mem_encoder=False,
    ):
        """Consolidate per-object temporary outputs into unified output."""
        batch_size = self._get_obj_num()
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(batch_size, 1, *self._pred_mask_res),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=torch.float32,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                fill_value=10.0,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        
        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            if out is None:
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out["pred_masks"]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = F.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]
        
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            consolidated_out["maskmem_features"], consolidated_out["maskmem_pos_enc"] = \
                self._run_memory_encoder(
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    high_res_masks=high_res_masks,
                    is_mask_from_pts=True,
                    object_score_logits=consolidated_out["object_score_logits"],
                )
        
        if consolidate_at_video_res:
            consolidated_out["pred_masks_video_res"] = consolidated_out["pred_masks"]
        
        return consolidated_out
    
    def _run_memory_encoder(self, frame_idx, batch_size, high_res_masks, is_mask_from_pts, object_score_logits):
        """Run memory encoder on predicted masks to create memory features for a given frame."""
        (_, _, current_vision_feats, _, feat_sizes) = \
            self._get_image_feature(self.inference_state, frame_idx=frame_idx, batch_size=batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )
        maskmem_pos_enc_out = self._get_maskmem_pos_enc({"maskmem_pos_enc": maskmem_pos_enc})
        return maskmem_features.to(
            dtype=torch.float16, device=self.device, non_blocking=True
        ), maskmem_pos_enc_out
    
    def _add_output_per_object(self, frame_idx, current_out, storage_key):
        """Split multi-object output into per-object slices (sharing tensor memory)."""
        maskmem_features = current_out["maskmem_features"]
        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        
        for obj_idx, obj_output_dict in self.inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out
    
    def _get_orig_video_res_output(self, pred_masks):
        """Resize predicted masks to original video resolution."""
        video_H = self.inference_state["video_height"]
        video_W = self.inference_state["video_width"]
        # Upscale to image size first, then to video resolution
        if pred_masks.shape[-2:] != (video_H, video_W):
            video_res_masks = F.interpolate(
                pred_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        else:
            video_res_masks = pred_masks
        return pred_masks, video_res_masks
    
    
    # ── Preflight (consolidation before propagation) ──────────────────────────
    
    @torch.inference_mode()
    def propagate_in_video_preflight(self):
        """
        Consolidate temporary outputs before tracking propagation.
        
        Follows the same pattern as the original SAM2VideoPredictor:
        encode memory per-object first (with the correct frame's backbone features),
        then move outputs into the per-object output dicts.
        """
        self.inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num()
        
        # Phase 1: Per-object memory encoding
        # For each object, encode memory features on temporary outputs
        # (these are frames that received user clicks/masks)
        for obj_idx in range(batch_size):
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    # Run memory encoder if memory features are missing
                    if out["maskmem_features"] is None:
                        high_res_masks = F.interpolate(
                            out["pred_masks"].to(self.device),
                            size=(self.image_size, self.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                            frame_idx=frame_idx,
                            batch_size=1,
                            high_res_masks=high_res_masks,
                            is_mask_from_pts=True,
                            object_score_logits=out.get("object_score_logits",
                                torch.full((1, 1), 10.0, device=self.device)),
                        )
                        out["maskmem_features"] = maskmem_features
                        out["maskmem_pos_enc"] = maskmem_pos_enc
                    
                    # Move from temp to permanent output dict
                    obj_output_dict[storage_key][frame_idx] = out
                
                # Clear temporary outputs
                obj_temp_output_dict[storage_key].clear()
            
            # If a frame is in cond, remove from non_cond (per-object)
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        
        # Phase 2: Build consolidated (batched) output dict from per-object dicts
        # This is used by the propagation loop for multi-object tracking
        output_dict = self.inference_state["output_dict"]
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        
        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            all_frame_inds = set()
            for obj_idx in range(batch_size):
                all_frame_inds.update(
                    self.inference_state["output_dict_per_obj"][obj_idx][storage_key].keys()
                )
            consolidated_frame_inds[storage_key].update(all_frame_inds)
            
            for frame_idx in all_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=False,
                )
                output_dict[storage_key][frame_idx] = consolidated_out
        
        # Cleanup: if a frame is in cond, remove from non_cond (global)
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
    
    
    # ── Adding new points and masks ───────────────────────────────────────────
    
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
            The points to add. 
        labels : array-like, optional
            The labels for the points.
        clear_old_points : bool, optional
            Whether to clear old points. Default is True.
        normalize_coords : bool, optional
            Whether to normalize the coordinates. Default is True.
        box : array-like, optional
            The box to add [y1, x1, y2, x2].
            
        Returns
        -------
        frame_idx : int
        obj_ids : list
        video_res_masks : torch.Tensor
        """
        obj_idx = self._obj_id_to_idx(obj_id)
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
            points = points.unsqueeze(0)
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        
        # Box as first two points with labels 2 and 3
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
        points = points * self.image_size
        points = points.to(self.device)
        labels = labels.to(self.device)
        
        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = self._concat_points(point_inputs, points, labels)
        
        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        
        is_init_cond_frame = frame_idx not in self.inference_state.get("frames_already_tracked", [])
        if is_init_cond_frame:
            reverse = False
        else:
            # For SAM3 there's no per-object frames_tracked, so default to False
            reverse = False
        
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        # Previous mask logits for correction clicks
        prev_sam_mask_logits = None
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
        
        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].to(self.device, non_blocking=True)
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
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
        
        Parameters
        ----------
        frame_idx : int
        obj_id : int
        mask : array-like
            2D mask array.
            
        Returns
        -------
        frame_idx : int
        obj_ids : list
        video_res_masks : torch.Tensor
        """
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]
        
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask_H, mask_W = mask.shape
        mask_inputs_orig = mask[None, None].float().to(self.device)
        
        if mask_H != self.image_size or mask_W != self.image_size:
            mask_inputs = F.interpolate(
                mask_inputs_orig,
                size=(self.image_size, self.image_size),
                align_corners=False,
                mode="bilinear",
                antialias=True,
            )
            mask_inputs = (mask_inputs >= 0.5).float()
        else:
            mask_inputs = mask_inputs_orig
        
        mask_inputs_per_frame[frame_idx] = mask_inputs
        point_inputs_per_frame.pop(frame_idx, None)
        
        is_init_cond_frame = frame_idx not in self.inference_state.get("frames_already_tracked", [])
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = False
        
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,
            frame_idx=frame_idx,
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
        )
        obj_temp_output_dict[storage_key][frame_idx] = current_out
        
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks
    
    
    # ── Propagation ───────────────────────────────────────────────────────────
    
    @torch.inference_mode()
    def propagate_in_video(
        self,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        processing_order=None,
        reverse=False,
    ):
        """
        Propagate tracking results through the video.
        Yields (frame_idx, obj_ids, video_res_masks) per frame.
        """
        self.propagate_in_video_preflight()
        
        obj_ids = self.inference_state["obj_ids"]
        num_frames = self.inference_state["num_frames"]
        batch_size = self._get_obj_num()
        output_dict = self.inference_state["output_dict"]
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        
        if processing_order is None:
            if start_frame_idx is None:
                start_frame_idx = min(
                    t
                    for obj_output_dict in self.inference_state["output_dict_per_obj"].values()
                    for t in obj_output_dict["cond_frame_outputs"]
                )
            if max_frame_num_to_track is None:
                max_frame_num_to_track = num_frames
            if reverse:
                end_frame_idx = max(start_frame_idx - max_frame_num_to_track, 0)
                if start_frame_idx > 0:
                    processing_order = range(start_frame_idx, end_frame_idx - 1, -1)
                else:
                    processing_order = []
            else:
                end_frame_idx = min(
                    start_frame_idx + max_frame_num_to_track, num_frames - 1
                )
                processing_order = range(start_frame_idx, end_frame_idx + 1)
        
        try:
            for frame_idx in processing_order:
                pred_masks_per_obj = [None] * batch_size
                
                for obj_idx in range(batch_size):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        storage_key = "cond_frame_outputs"
                        current_out = obj_output_dict[storage_key][frame_idx]
                        pred_masks = current_out["pred_masks"].to(
                            self.device, non_blocking=True
                        )
                    else:
                        storage_key = "non_cond_frame_outputs"
                        current_out, pred_masks = self._run_single_frame_inference(
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=True,
                        )
                        obj_output_dict[storage_key][frame_idx] = current_out
                        
                        # Prune old non-cond frames (keep last 16)
                        oldest_allowed_idx = frame_idx - 16
                        all_frame_idxs = list(obj_output_dict[storage_key].keys())
                        old_frame_idxs = [idx for idx in all_frame_idxs if idx < oldest_allowed_idx]
                        for old_idx in old_frame_idxs:
                            obj_output_dict[storage_key].pop(old_idx)
                            for objid in self.inference_state['output_dict_per_obj'].keys():
                                if old_idx in self.inference_state['output_dict_per_obj'][objid][storage_key]:
                                    self.inference_state['output_dict_per_obj'][objid][storage_key].pop(old_idx)
                    
                    self.inference_state["frames_already_tracked"].append(frame_idx)
                    pred_masks_per_obj[obj_idx] = pred_masks
                
                if len(pred_masks_per_obj) > 1:
                    all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                else:
                    all_pred_masks = pred_masks_per_obj[0]
                
                _, video_res_masks = self._get_orig_video_res_output(all_pred_masks)
                yield frame_idx, obj_ids, video_res_masks
        except Exception as e:
            print(e)
            pass
    
    
    # ── State management ──────────────────────────────────────────────────────
    
    @torch.inference_mode()
    def reset_state(self):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results()
        self.inference_state["obj_id_to_idx"].clear()
        self.inference_state["obj_idx_to_id"].clear()
        self.inference_state["obj_ids"].clear()
        self.inference_state["point_inputs_per_obj"].clear()
        self.inference_state["mask_inputs_per_obj"].clear()
        self.inference_state["output_dict_per_obj"].clear()
        self.inference_state["temp_output_dict_per_obj"].clear()
    
    def _reset_tracking_results(self):
        """Reset all tracking inputs and results."""
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
        self.inference_state["output_dict"]["cond_frame_outputs"].clear()
        self.inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        self.inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        self.inference_state["tracking_has_started"] = False
        self.inference_state["frames_already_tracked"].clear()
    
    
    @torch.inference_mode()
    def remove_object(self, obj_id, strict=False, need_output=True):
        """
        Remove an object id from the tracking state.
        Returns (obj_ids, updated_frames).
        """
        try:
            old_obj_idx_to_rm = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        except AttributeError as e:
            print(e)
            return
        
        updated_frames = []
        if old_obj_idx_to_rm is None:
            if not strict:
                return self.inference_state["obj_ids"], updated_frames
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {self.inference_state['obj_ids']}."
            )
        
        if len(self.inference_state["obj_id_to_idx"]) == 1:
            self.reset_state()
            return self.inference_state["obj_ids"], updated_frames
        
        # Clear inputs for the object being removed
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(
            self.inference_state["point_inputs_per_obj"][old_obj_idx_to_rm]
        )
        obj_input_frames_inds.update(
            self.inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm]
        )
        
        # Update object ID mappings
        old_obj_ids = self.inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        self.inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        self.inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        self.inference_state["obj_ids"] = new_obj_ids
        
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
        
        if need_output:
            temp_output_dict_per_obj = self.inference_state["temp_output_dict_per_obj"]
            for frame_idx in obj_input_frames_inds:
                is_cond = any(
                    frame_idx in obj_temp_output_dict["cond_frame_outputs"]
                    for obj_temp_output_dict in temp_output_dict_per_obj.values()
                )
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx,
                    is_cond=is_cond,
                    consolidate_at_video_res=True,
                )
                _, video_res_masks = self._get_orig_video_res_output(
                    consolidated_out["pred_masks_video_res"]
                )
                updated_frames.append((frame_idx, video_res_masks))
        
        return self.inference_state["obj_ids"], updated_frames
    
    
    # ── Utilities ─────────────────────────────────────────────────────────────
    
    @staticmethod
    def _concat_points(old_point_inputs, new_points, new_labels):
        """Concatenate new points/labels with existing ones (or create new)."""
        if old_point_inputs is None:
            return {"point_coords": new_points, "point_labels": new_labels}
        return {
            "point_coords": torch.cat(
                [old_point_inputs["point_coords"], new_points], dim=1
            ),
            "point_labels": torch.cat(
                [old_point_inputs["point_labels"], new_labels], dim=1
            ),
        }


#####################################################################################################


class SAM3_semantic_octron:
    """
    Semantic mode: detect all matching objects with text/box prompts on a frame,
    then track them across frames using the SAM3_octron tracker.
    
    Composes:
        - SAM3SemanticModel (detector): finds objects from text/visual prompts
        - SAM3_octron (tracker): tracks detected objects across frames
    
    Usage:
        predictor = build_sam3_octron(ckpt_path, semantic=True)
        predictor.init_state(video_data, zarr_store)
        # Detect all "cells" on frame 0 and add them to tracker
        obj_ids, masks = predictor.detect_and_track(frame_idx=0, text="cell")
        # Or detect with a box prompt
        obj_ids, masks = predictor.detect_and_track(frame_idx=0, bboxes=[[x1,y1,x2,y2]])
        # Propagate
        for frame_idx, obj_ids, masks in predictor.propagate_in_video():
            ...
        # Manual corrections still work (delegates to tracker)
        predictor.add_new_points_or_box(frame_idx=5, obj_id=0, points=..., labels=...)
    """
    
    def __init__(self, detector_model, tracker, device, detector_ckpt_path=None):
        """
        Parameters
        ----------
        detector_model : SAM3SemanticModel
            The ultralytics SAM3 semantic model for detection.
        tracker : SAM3_octron
            The OCTRON tracker instance.
        device : torch.device
        detector_ckpt_path : str, optional
            Path to detector checkpoint for reloading.
        """
        self.detector = detector_model
        self.tracker = tracker
        self.device = device
        self.detector_ckpt_path = detector_ckpt_path
        self._next_obj_id = 0
    
    # ── Delegation properties ─────────────────────────────────────────────────
    
    @property
    def inference_state(self):
        return self.tracker.inference_state
    
    @property
    def images(self):
        return self.tracker.images
    
    @property
    def image_size(self):
        return self.tracker.image_size
    
    # ── Init / Reset ──────────────────────────────────────────────────────────
    
    def init_state(self, video_data, zarr_store):
        """Initialize tracking state (delegates to tracker)."""
        self.tracker.init_state(video_data, zarr_store)
        self._next_obj_id = 0
    
    def reset_state(self):
        """Reset all tracking state."""
        self.tracker.reset_state()
        self._next_obj_id = 0
    
    def reload_detector(self):
        """Reload detector model from checkpoint to ensure clean state."""
        if self.detector_ckpt_path is None:
            return
        
        from ultralytics.models.sam.build_sam3 import build_sam3_image_model
        
        print("♻️ Reloading SAM3 detector for clean state...")
        self.detector = build_sam3_image_model(self.detector_ckpt_path)
        self.detector = self.detector.to(self.device)
        self.detector.eval()
    
    # ── Detection ────────────────────────────────────────────────────────────────────
    
    @torch.inference_mode()
    def detect(
        self,
        frame_idx,
        text=None,
        bboxes=None,
        labels=None,
        conf_threshold=0.5,
    ):
        """
        Run detection on a single frame using text and/or box prompts.
        
        Parameters
        ----------
        frame_idx : int
            Frame to run detection on.
        text : str or list[str], optional
            Text prompt(s) describing the objects to find.
        bboxes : array-like, optional
            Bounding boxes in xyxy pixel coordinates, shape (N, 4).
        labels : array-like, optional
            Labels for the bounding boxes (positive=1).
        conf_threshold : float
            Confidence threshold for detections.
            
        Returns
        -------
        pred_masks : torch.Tensor or None
            Boolean masks at video resolution, shape (N_detections, H, W).
        pred_scores : torch.Tensor or None
            Detection confidence scores, shape (N_detections,).
        pred_classes : torch.Tensor or None
            Class indices, shape (N_detections,).
        """
        from ultralytics.models.sam.sam3.geometry_encoders import Prompt
        from ultralytics.utils import ops
        
        assert text is not None or bboxes is not None, \
            "At least one of text or bboxes must be provided"
        
        # Get preprocessed image from OctoZarr (same as tracker uses)
        image = self.tracker.images[frame_idx]
        if image.dim() == 3:
            image = image.unsqueeze(0)
        image = image.to(self.device).float()
        
        # Run detector backbone
        features = self.detector.backbone.forward_image(image)
        
        # Set up text prompts
        use_text = text is not None
        if use_text:
            text_batch = [text] if isinstance(text, str) else text
        else:
            text_batch = ["visual"]
        nc = len(text_batch)
        
        if self.detector.names != text_batch:
            self.detector.set_classes(text=text_batch)
        
        # Set up geometric prompts (boxes)
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, nc, 4, device=self.device),
            box_mask=torch.zeros(nc, 0, device=self.device, dtype=torch.bool),
        )
        
        if bboxes is not None:
            bboxes_t = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            if bboxes_t.ndim == 1:
                bboxes_t = bboxes_t[None]
            # xyxy pixel coords → xywh normalized
            video_H = self.tracker.inference_state["video_height"]
            video_W = self.tracker.inference_state["video_width"]
            bboxes_xywh = ops.xyxy2xywh(bboxes_t)
            bboxes_xywh[:, 0::2] /= video_W
            bboxes_xywh[:, 1::2] /= video_H
            if labels is None:
                labels_t = torch.ones(bboxes_xywh.shape[0], dtype=torch.int32, device=self.device)
            else:
                labels_t = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            # Append each box: shape (1, 1, 4) and (1, 1)
            for i in range(len(bboxes_xywh)):
                geometric_prompt.append_boxes(
                    bboxes_xywh[i].view(1, 1, 4),
                    labels_t[i].view(1, 1),
                )
        
        # Run detection
        text_ids = torch.arange(nc, device=self.device, dtype=torch.long)
        outputs = self.detector.forward_grounding(
            backbone_out=features,
            text_ids=text_ids,
            geometric_prompt=geometric_prompt,
        )
        
        # Extract masks and scores
        pred_logits = outputs["pred_logits"]
        pred_masks = outputs["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        if "presence_logit_dec" in outputs:
            presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
            pred_scores = (pred_scores * presence_score).squeeze(-1)
        else:
            pred_scores = pred_scores.squeeze(-1)
        
        pred_cls = torch.arange(
            pred_scores.shape[0], dtype=pred_scores.dtype, device=self.device,
        )[:, None].expand_as(pred_scores)
        
        # Diagnostic output: show score distribution before filtering
        print(f"🔍 SAM3 detection: {pred_scores.numel()} raw queries, "
              f"max score={pred_scores.max().item():.3f}, "
              f"scores > 0.1: {(pred_scores > 0.1).sum().item()}, "
              f"scores > 0.25: {(pred_scores > 0.25).sum().item()}, "
              f"scores > 0.5: {(pred_scores > 0.5).sum().item()}")
        
        # Filter by confidence threshold
        keep = pred_scores > conf_threshold
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]
        pred_cls = pred_cls[keep]
        
        print(f"🔍 Kept {pred_masks.shape[0]} detections above threshold {conf_threshold}")
        
        if pred_masks.shape[0] == 0:
            return None, None, None
        
        # Upscale masks to video resolution
        video_H = self.tracker.inference_state["video_height"]
        video_W = self.tracker.inference_state["video_width"]
        pred_masks = F.interpolate(
            pred_masks.float()[None],
            size=(video_H, video_W),
            mode="bilinear",
        )[0] > 0.5
        
        return pred_masks, pred_scores, pred_cls
    
    # ── Detect + Track ────────────────────────────────────────────────────────
    
    @torch.inference_mode()
    def detect_and_track(
        self,
        frame_idx,
        text=None,
        bboxes=None,
        labels=None,
        conf_threshold=0.5,
    ):
        """
        Detect objects on a frame and add each to the tracker.
        
        Parameters
        ----------
        frame_idx : int
            Frame to detect on.
        text : str or list[str], optional
            Text prompt describing the objects.
        bboxes : array-like, optional
            Bounding boxes in xyxy pixel coordinates.
        labels : array-like, optional
            Labels for the bounding boxes.
        conf_threshold : float
            Confidence threshold for detections.
            
        Returns
        -------
        obj_ids_added : list[int]
            Object IDs that were added to the tracker.
        video_res_masks : torch.Tensor or None
            Masks at video resolution for all tracked objects after adding detections.
        """
        pred_masks, pred_scores, _ = self.detect(
            frame_idx, text=text, bboxes=bboxes, labels=labels,
            conf_threshold=conf_threshold,
        )
        
        if pred_masks is None or pred_masks.shape[0] == 0:
            return [], None
        
        obj_ids_added = []
        video_res_masks = None
        
        for i in range(pred_masks.shape[0]):
            obj_id = self._next_obj_id
            self._next_obj_id += 1
            mask_np = pred_masks[i].cpu().numpy()
            _, obj_ids, video_res_masks = self.tracker.add_new_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask_np,
            )
            obj_ids_added.append(obj_id)
        
        return obj_ids_added, video_res_masks
    
    # ── Delegated tracker methods ─────────────────────────────────────────────
    
    def propagate_in_video(self, **kwargs):
        """Propagate tracking. Yields (frame_idx, obj_ids, video_res_masks)."""
        return self.tracker.propagate_in_video(**kwargs)
    
    def add_new_points_or_box(self, *args, **kwargs):
        """Add manual point/box corrections (delegates to tracker)."""
        return self.tracker.add_new_points_or_box(*args, **kwargs)
    
    def add_new_mask(self, *args, **kwargs):
        """Add a manual mask (delegates to tracker)."""
        return self.tracker.add_new_mask(*args, **kwargs)
    
    def remove_object(self, *args, **kwargs):
        """Remove an object from tracking."""
        return self.tracker.remove_object(*args, **kwargs)


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
    Run a new prediction on the SAM3 model in OCTRON.
    Wrapper around SAM3_octron functions for adding new points or masks.
    Returns the mask image that can be re-added to the viewer.
    
    Parameters
    ----------
    predictor : SAM3_octron
        The SAM3_octron predictor object.
    frame_idx : int
        The current frame index.
    obj_id : int
        The current object id for prediction.
    labels : list or int
        Labels for the points (0=negative, 1=positive).
    points : list, optional
    masks : np.array, optional
    box : list, optional
        Box coordinates: [top_left[1],top_left[0],bottom_right[1],bottom_right[0]]
    **kwargs : dict
        clear_old_points : bool (default True)
        normalize_coords : bool (default True)
    
    Returns
    -------
    mask : np.array or None
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
        if frame_idx is None:
            return None
        index_obj_id = obj_ids.index(obj_id)
        mask = (video_res_masks[index_obj_id] > 0).cpu().numpy().astype(np.uint8)
    
    mask = mask.squeeze()
    return mask

if __name__ == "__main__":
    pass
