# Main SAM3.1 predictor class for OCTRON
# This wraps the ultralytics SAM3/SAM3.1 Model (which extends SAM2Model) with 
# the same OCTRON interface as SAM2_octron, using OctoZarr for image loading.

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from collections import OrderedDict
import torch
import torch.nn.functional as F
import numpy as np
from loguru import logger

from kornia.morphology import closing as kornia_closing
from torch import tensor as torch_tensor
from skimage.morphology import disk

from .sam_zarr import OctoZarr

import warnings
warnings.simplefilter("ignore")

# SAM3.1 constants
SAM3_IMAGE_SIZE = 1008
SAM3_BACKBONE_STRIDE = 14
_fs = SAM3_IMAGE_SIZE // SAM3_BACKBONE_STRIDE  # base feature size
SAM3_BB_FEAT_SIZES = [
    (_fs * 4, _fs * 4),
    (_fs * 2, _fs * 2),
    (_fs,     _fs),
]
# SAM3.1 normalization: maps [0,1] to [-1,1]
SAM3_IMG_MEAN = (0.5, 0.5, 0.5)
SAM3_IMG_STD  = (0.5, 0.5, 0.5)

# Placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM3_octron:
    """
    OCTRON wrapper around the ultralytics SAM3/SAM3.1 Model for interactive video tracking.
    
    Provides the same interface as SAM2_octron:
        - init_state(video_data, zarr_store)
        - add_new_points_or_box(frame_idx, obj_id, points, labels, ...)
        - add_new_mask(frame_idx, obj_id, mask)
        - propagate_in_video(start_frame_idx, max_frame_num_to_track, ...)
        - reset_state()
        - remove_object(obj_id, ...)
    
    Uses the ultralytics SAM3/SAM3.1 Model.track_step which has the same signature as SAM2's
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
        # Force the SAM decoder to actually run on mask-only inputs
        # (the default True causes mask inputs to be returned unchanged).
        self.model.use_mask_input_as_output_without_sam = False
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
        
        # Backbone feature cache: keep up to N frames of pre-computed backbone
        # features to avoid redundant forward_image calls during propagation.
        # Set to 0 to fall back to single-frame cache (original behaviour).
        self._max_cached_backbone_frames = 16
        
        # Run the model in float16 on CUDA for maximum throughput.
        # SAM2/SAM3 are trained with AMP so the weights are fp16-safe.
        # MPS is excluded: its matrix-multiply kernel requires matching
        # accumulator dtypes and crashes with pure fp16, and several ops
        # fall back to CPU making fp16 *slower* than fp32 on Apple Silicon.
        if device.type == "cuda":
            self.model = self.model.half()
        
        # How many non-conditioning frames to keep in memory per object.
        # Smaller = faster attention (fewer memory tokens) but less context.
        self._max_memory_frames = 2
        
        # Memory encoder stride: only run the memory encoder every N-th
        # propagated frame.  Frames without encoding still produce masks
        # but are NOT stored in the output dict, so they won't contribute
        # to future frames' memory attention.  This saves the full
        # _encode_new_memory cost on skipped frames (~30-40 ms / object)
        # AND reduces the number of memory tokens for attention.
        self._mem_encode_stride = 3
    
    
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
        inference_state["cached_features"] = OrderedDict()
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
        inference_state["frames_already_tracked"] = set()
        inference_state["tracking_has_started"] = False
        
        # OCTRON-specific: centroid & area tracking
        inference_state["centroids"] = {}
        inference_state["areas"] = {}
        
        # Warm up backbone on frame 0
        self._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        logger.info('🚀 Initialized SAM3.1 model')
        
        self.video_data = video_data
        
        if self.perform_morphological_operations:
            self.disk_size = 2
            self.closing_kernel = torch_tensor(disk(self.disk_size).tolist()).to(compute_device)
    
    
    # ── Image feature extraction ──────────────────────────────────────────────
    
    @torch.inference_mode()
    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """
        Get backbone features for a frame, using OctoZarr for image loading.
        Caches features in an LRU cache keyed by frame index.
        """
        cached = inference_state["cached_features"]
        result = cached.get(frame_idx)
        if result is not None:
            image, backbone_out = result
            # Move to end (most-recently-used) for LRU eviction
            cached.move_to_end(frame_idx)
        else:
            # Load preprocessed image from OctoZarr
            image = inference_state["images"][frame_idx]
            if image.dim() == 3:
                image = image.unsqueeze(0)  # add batch dim: (1, C, H, W)
            # Match the model's dtype (float16 if converted in __init__)
            model_dtype = next(self.model.parameters()).dtype
            image = image.to(device=self.device, dtype=model_dtype)
            backbone_out = self.model.forward_image(image)
            cached[frame_idx] = (image, backbone_out)
            # LRU eviction: drop oldest entries beyond the limit
            max_cached = self._max_cached_backbone_frames
            if max_cached > 0:
                while len(cached) > max_cached:
                    cached.popitem(last=False)
            else:
                # Legacy single-frame behaviour
                if len(cached) > 1:
                    oldest = next(k for k in cached if k != frame_idx)
                    del cached[oldest]
        
        # Expand features if batch_size > 1 (multi-object)
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"],
            "vision_pos_enc": backbone_out["vision_pos_enc"],
        }
        _, vis_feats, vis_pos_embed, feat_sizes = \
            self.model._prepare_backbone_features(expanded_backbone_out, batch=batch_size)
        
        return image, expanded_backbone_out, vis_feats, vis_pos_embed, feat_sizes
    
    @torch.inference_mode()
    def _prefetch_backbone_batch(self, frame_indices, batch_size=4):
        """
        Batch-compute backbone features for multiple frames at once.
        
        Processes *batch_size* frames per forward_image call, which
        reduces per-frame GPU dispatch overhead and improves utilisation.
        Already-cached frames are skipped automatically.
        """
        cached = self.inference_state["cached_features"]
        uncached = [idx for idx in frame_indices if idx not in cached]
        if not uncached:
            return
        
        model_dtype = next(self.model.parameters()).dtype
        max_cached = self._max_cached_backbone_frames
        
        for start in range(0, len(uncached), batch_size):
            batch_indices = uncached[start : start + batch_size]
            
            # Load images from OctoZarr and stack into a single tensor
            images = []
            for idx in batch_indices:
                img = self.inference_state["images"][idx]
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                images.append(img)
            batch_tensor = torch.cat(images, dim=0).to(
                device=self.device, dtype=model_dtype
            )
            
            # Single batched backbone call
            backbone_out = self.model.forward_image(batch_tensor)
            
            # Split per-frame and store in the LRU cache.
            # .clone() makes each slice independent so the batch
            # tensor can be freed after the loop iteration.
            for i, idx in enumerate(batch_indices):
                frame_backbone_out = {
                    "backbone_fpn": [
                        feat[i : i + 1].clone()
                        for feat in backbone_out["backbone_fpn"]
                    ],
                    "vision_pos_enc": [
                        pos[i : i + 1].clone()
                        for pos in backbone_out["vision_pos_enc"]
                    ],
                }
                cached[idx] = (batch_tensor[i : i + 1].clone(), frame_backbone_out)
                if max_cached > 0:
                    while len(cached) > max_cached:
                        cached.popitem(last=False)
    
    
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
        
        # The backbone may output bfloat16 on CUDA (from internal autocast)
        # while model weights are float16 (from .half()).  Cast to match.
        model_dtype = next(self.model.parameters()).dtype
        current_vision_feats = [x.to(model_dtype) for x in current_vision_feats]
        current_vision_pos_embeds = [x.to(model_dtype) for x in current_vision_pos_embeds]
        
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
            # MPS: store as float32 for accumulation stability.
            # CUDA: store as float16 to match model weights (from .half()).
            mem_dtype = torch.float32 if storage_device.type == "mps" else torch.float16
            maskmem_features = maskmem_features.to(mem_dtype)
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
            logger.warning(f"⚠️ Cannot add a new label (id={obj_id}) after batch prediction has already run.")
            logger.warning(f"You can only annotate existing labels: {self.inference_state['obj_ids']}")
            logger.warning(f"To add additional labels, reset the predictor first (click 'Reset').")
            return None
    
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
        
        # Use the model's dtype so that consolidated tensors match the
        # weights (float16 on CUDA after .half(), float32 on MPS/CPU).
        # This avoids mixed-dtype tensors when they're stored per-object
        # and later read back by track_step's memory attention.
        fill_dtype = next(self.model.parameters()).dtype
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(batch_size, 1, *self._pred_mask_res),
                fill_value=NO_OBJ_SCORE,
                dtype=fill_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=fill_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                fill_value=10.0,
                dtype=fill_dtype,
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
        # Match the model's weight dtype (float16 on CUDA after .half(),
        # float32 on MPS/CPU).  The backbone may output bfloat16 on CUDA
        # (from internal autocast) while the memory encoder conv layers
        # are float16, causing a dtype mismatch.  Explicit cast fixes this.
        model_dtype = next(self.model.parameters()).dtype
        current_vision_feats = [x.to(model_dtype) for x in current_vision_feats]
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks.to(model_dtype),
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits.to(model_dtype),
        )
        maskmem_pos_enc_out = self._get_maskmem_pos_enc({"maskmem_pos_enc": maskmem_pos_enc})
        mem_dtype = torch.float32 if self.device.type == "mps" else torch.float16
        return maskmem_features.to(
            dtype=mem_dtype, device=self.device, non_blocking=True
        ), maskmem_pos_enc_out
    
    def _add_output_per_object(self, frame_idx, current_out, storage_key, valid_obj_indices=None):
        """Split multi-object output into per-object slices (sharing tensor memory).
        
        Parameters
        ----------
        valid_obj_indices : set[int] or None
            If provided, only store results for these object indices.
            Objects not in this set are skipped to avoid creating ghost
            conditioning entries with NO_OBJ_SCORE masks and garbage
            obj_ptr that would corrupt memory attention.
        """
        maskmem_features = current_out["maskmem_features"]
        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        
        for obj_idx, obj_output_dict in self.inference_state["output_dict_per_obj"].items():
            if valid_obj_indices is not None and obj_idx not in valid_obj_indices:
                continue
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
        
        # Phase 1: Move temporary per-object outputs → permanent dicts,
        # collecting which frame indices are genuinely new (from user
        # interactions).  Only those need consolidation + memory encoding
        # in Phase 2.  Frames stored directly by the propagation loop
        # already have valid per-object memory and must NOT be re-encoded.
        new_temp_frames = {"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}
        # Track which objects actually have real outputs at each temp frame
        # so that Phase 2 can skip "ghost" entries for objects that were
        # never conditioned at a given frame.
        _real_objs_per_frame = {}  # frame_idx → set[obj_idx]
        for obj_idx in range(batch_size):
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            for is_cond in [False, True]:
                storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
                for frame_idx, out in obj_temp_output_dict[storage_key].items():
                    new_temp_frames[storage_key].add(frame_idx)
                    obj_output_dict[storage_key][frame_idx] = out
                    _real_objs_per_frame.setdefault(frame_idx, set()).add(obj_idx)
                obj_temp_output_dict[storage_key].clear()
            
            # If a frame is in cond, remove from non_cond (per-object)
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        
        # Phase 2: Consolidate + encode memory ONLY for frames that were
        # just moved from temp (i.e. new user-provided conditioning).
        output_dict = self.inference_state["output_dict"]
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        
        for is_cond in [False, True]:
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            for frame_idx in new_temp_frames[storage_key]:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = consolidated_out
                consolidated_frame_inds[storage_key].add(frame_idx)
                # Only split back to objects that actually have data at
                # this frame.  Objects without data would receive ghost
                # entries (NO_OBJ_SCORE + garbage obj_ptr) that corrupt
                # memory attention during propagation.
                self._add_output_per_object(
                    frame_idx, consolidated_out, storage_key,
                    valid_obj_indices=_real_objs_per_frame.get(frame_idx),
                )
        
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
        if obj_idx is None:
            return None, None, None
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
        if obj_idx is None:
            return None, None, None
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
        
        # SAM3.1's TwoWayTransformer decoder treats mask-only prompts too
        # literally — it follows the polygon shape instead of using visual
        # features to discriminate object vs background within the region.
        # Strategy: derive a bounding box from the polygon as the primary
        # prompt (SAM handles boxes well) and pass the polygon mask as a
        # soft prior via prev_sam_mask_logits.  The box forces the decoder
        # to rely on visual features while the polygon biases it.
        mask_2d = mask_inputs[0, 0]  # (image_size, image_size)
        ys, xs = torch.where(mask_2d > 0.5)
        
        if len(ys) > 0:
            # Bounding box of the polygon in image_size coords
            x1, y1 = xs.min().item(), ys.min().item()
            x2, y2 = xs.max().item(), ys.max().item()
            box_coords = torch.tensor(
                [[[x1, y1], [x2, y2]]], dtype=torch.float32, device=self.device
            )
            box_labels = torch.tensor(
                [[2, 3]], dtype=torch.int32, device=self.device
            )
            point_inputs = {
                "point_coords": box_coords,
                "point_labels": box_labels,
            }
        else:
            point_inputs = None
        
        point_inputs_per_frame[frame_idx] = point_inputs if point_inputs is not None else {}
        mask_inputs_per_frame.pop(frame_idx, None)
        
        is_init_cond_frame = frame_idx not in self.inference_state.get("frames_already_tracked", [])
        reverse = False
        
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        
        if point_inputs is not None:
            # Pure box prompt — let SAM3.1 use visual features to segment
            current_out, _ = self._run_single_frame_inference(
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=False,
            )
        else:
            # Fallback: mask-only
            current_out, _ = self._run_single_frame_inference(
                output_dict=obj_output_dict,
                frame_idx=frame_idx,
                batch_size=1,
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=None,
                mask_inputs=mask_inputs * 6.0 - 3.0,
                reverse=reverse,
                run_mem_encoder=False,
            )
        
        # Intersect SAM3.1's prediction with the original polygon.
        # The box prompt gives SAM3.1 freedom to find the object using
        # visual features, but may include regions outside the polygon.
        # Clamping logits to strongly negative outside the polygon
        # recovers the polygon's shape specificity.
        pred_masks = current_out["pred_masks"]
        polygon_low_res = F.interpolate(
            mask_inputs,
            size=pred_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).to(pred_masks.device) > 0.5
        current_out["pred_masks"] = torch.where(
            polygon_low_res,
            pred_masks,
            torch.clamp(pred_masks, max=-10.0),
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
        
        Uses **per-object** inference (batch_size=1 per object) rather than
        batched multi-object inference.  On MPS the batched approach creates
        enormous intermediate tensors in the memory-attention layer (N objects
        × M memory frames × spatial tokens) which is orders of magnitude
        slower than processing each object independently.
        
        Yields (frame_idx, obj_ids, video_res_masks) per frame.
        """
        import time as _time
        
        t_preflight_start = _time.perf_counter()
        try:
            self.propagate_in_video_preflight()
        except Exception as e:
            import traceback
            logger.error(f"❌ SAM3.1 propagate_in_video_preflight failed: {e}")
            traceback.print_exc()
            return
        t_preflight_end = _time.perf_counter()
        
        obj_ids = self.inference_state["obj_ids"]
        num_frames = self.inference_state["num_frames"]
        batch_size = self._get_obj_num()
        
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
        
        encode_stride = self._mem_encode_stride
        # Pruning window: keep exactly max_memory encoded frames.
        # With stride S and max M, the window is (M-1)*S positions.
        oldest_allowed_offset = max(1, (self._max_memory_frames - 1) * encode_stride)
        
        # ── Prune stale frames from previous batches ──────────────
        # When jumping to a distant position (e.g. frame 0 → 1042),
        # old non-cond frames are irrelevant and slow down memory
        # attention.  Remove any that fall outside the pruning window
        # relative to the first frame we're about to process.
        first_frame = processing_order[0] if processing_order else None
        if first_frame is not None:
            stale_cutoff = first_frame - oldest_allowed_offset
            for obj_idx in range(batch_size):
                obj_d = self.inference_state["output_dict_per_obj"][obj_idx]
                stale_keys = [k for k in obj_d["non_cond_frame_outputs"]
                              if k < stale_cutoff]
                for k in stale_keys:
                    obj_d["non_cond_frame_outputs"].pop(k)
            # Also clean global output_dict
            global_nc = self.inference_state["output_dict"]["non_cond_frame_outputs"]
            for k in [k for k in global_nc if k < stale_cutoff]:
                global_nc.pop(k)
        
        logger.info(f"🚀 SAM3.1 propagation: {batch_size} objects (per-object), "
              f"preflight {t_preflight_end - t_preflight_start:.2f}s, "
              f"backbone cache {len(self.inference_state['cached_features'])} frames, "
              f"memory bank {self._max_memory_frames} frames, "
              f"encode stride {encode_stride}")
        
        frame_count = 0
        total_inference_ms = 0.0
        t_loop_start = _time.perf_counter()
        
        try:
            for frame_idx in processing_order:
                pred_masks_per_obj = [None] * batch_size
                
                # Decide whether to run the memory encoder on this frame.
                # Skipped frames still produce masks but are NOT stored,
                # so the model's memory attention sees fewer (sparser) frames.
                encode_mem = (frame_count % encode_stride == 0)
                
                t_frame_start = _time.perf_counter()
                for obj_idx in range(batch_size):
                    obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                    
                    if frame_idx in obj_output_dict["cond_frame_outputs"]:
                        # Conditioning frame — use preflight result
                        current_out = obj_output_dict["cond_frame_outputs"][frame_idx]
                        pred_masks = current_out["pred_masks"].to(
                            self.device, non_blocking=True
                        )
                    else:
                        # Non-conditioning frame — per-object forward (batch_size=1)
                        storage_key = "non_cond_frame_outputs"
                        current_out, pred_masks = self._run_single_frame_inference(
                            output_dict=obj_output_dict,
                            frame_idx=frame_idx,
                            batch_size=1,
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=reverse,
                            run_mem_encoder=encode_mem,
                        )
                        # Only persist frames that carry memory features;
                        # the model gracefully skips missing frame indices.
                        if encode_mem:
                            obj_output_dict[storage_key][frame_idx] = current_out
                        
                        # Prune old non-cond frames for this object
                        oldest_allowed = frame_idx - oldest_allowed_offset
                        for old_idx in [k for k in list(obj_output_dict[storage_key].keys())
                                        if k < oldest_allowed]:
                            obj_output_dict[storage_key].pop(old_idx)
                    
                    pred_masks_per_obj[obj_idx] = pred_masks
                t_frame_end = _time.perf_counter()
                total_inference_ms += (t_frame_end - t_frame_start) * 1000
                
                # Combine per-object masks → (N, 1, H, W)
                # Conditioning-frame masks can come from consolidated storage at
                # self._pred_mask_res while live inference may return a different
                # decoder resolution (e.g. 288x288). Normalize before cat.
                target_hw = self._pred_mask_res
                for i, m in enumerate(pred_masks_per_obj):
                    if m.shape[-2:] != target_hw:
                        pred_masks_per_obj[i] = F.interpolate(
                            m,
                            size=target_hw,
                            mode="bilinear",
                            align_corners=False,
                        )
                all_pred_masks = torch.cat(pred_masks_per_obj, dim=0)
                self.inference_state["frames_already_tracked"].add(frame_idx)
                _, video_res_masks = self._get_orig_video_res_output(all_pred_masks)
                frame_count += 1
                yield frame_idx, obj_ids, video_res_masks
        except Exception as e:
            import traceback
            logger.error(f"❌ SAM3.1 propagate_in_video error at frame {frame_idx}: {e}")
            traceback.print_exc()
        
        t_loop_end = _time.perf_counter()
        if frame_count > 0:
            avg_total = (t_loop_end - t_loop_start) / frame_count * 1000
            avg_inf = total_inference_ms / frame_count
            logger.info(f"📊 SAM3.1 propagation done: {frame_count} frames, "
                  f"avg {avg_total:.0f}ms/frame "
                  f"(inference {avg_inf:.0f}ms, "
                  f"{avg_inf / batch_size:.0f}ms/object)")
        
        # Free GPU memory between batches to reduce fragmentation.
        if self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
    
    
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
        # Release backbone feature cache to free GPU memory
        self.inference_state["cached_features"].clear()
        # Force GC before clearing the GPU cache so that Python
        # actually releases the tensor objects first.
        import gc
        gc.collect()
        if self.device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()
    
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
            logger.exception(e)
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
        - SAM3.1 SemanticModel (detector): finds objects from text/visual prompts
        - SAM3.1 tracker (SAM3_octron): tracks detected objects across frames
    
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
        
        logger.info("♻️ Reloading SAM3.1 detector for clean state...")
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
        
        # Free GPU memory occupied by the tracker to avoid OOM / hangs
        # during detection.  After propagation the tracker holds backbone
        # cache, output dicts (maskmem_features × objects × frames), and
        # other tensors that consume significant GPU memory.
        #
        # It is safe to fully reset because the tracker is always rebuilt
        # from _semantic_accumulated_masks before the next propagation
        # (see init_prediction_threaded).  Before the first propagation
        # the tracker state is small and doesn't need clearing.
        if self.tracker.inference_state.get("tracking_has_started", False):
            self.tracker.reset_state()
            # Force Python to actually release the tensors BEFORE clearing
            # the GPU cache.  Without gc.collect(), MPS may hang because
            # empty_cache() can't reclaim blocks still referenced by
            # Python objects waiting for garbage collection.
            import gc
            gc.collect()
            if self.device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()
            elif self.device.type == "cuda":
                torch.cuda.empty_cache()
        
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
        logger.debug(f"🔍 SAM3.1 detection: {pred_scores.numel()} raw queries, "
              f"max score={pred_scores.max().item():.3f}, "
              f"scores > 0.1: {(pred_scores > 0.1).sum().item()}, "
              f"scores > 0.25: {(pred_scores > 0.25).sum().item()}, "
              f"scores > 0.5: {(pred_scores > 0.5).sum().item()}")
        
        # Filter by confidence threshold
        keep = pred_scores > conf_threshold
        pred_masks = pred_masks[keep]
        pred_scores = pred_scores[keep]
        pred_cls = pred_cls[keep]
        
        logger.debug(f"🔍 Kept {pred_masks.shape[0]} detections above threshold {conf_threshold}")
        
        if pred_masks.shape[0] == 0:
            return None, None, None
        
        # Upscale masks to video resolution
        video_H = self.tracker.inference_state["video_height"]
        video_W = self.tracker.inference_state["video_width"]
        pred_masks = F.interpolate(
            pred_masks.float()[None],
            size=(video_H, video_W),
            mode="bilinear",
        )[0] > 0
        
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
            mask_np = pred_masks[i].cpu().numpy()
            out_frame_idx, obj_ids, video_res_masks = self.tracker.add_new_mask(
                frame_idx=frame_idx,
                obj_id=obj_id,
                mask=mask_np,
            )
            if out_frame_idx is None:
                return obj_ids_added, video_res_masks
            self._next_obj_id += 1
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
        The SAM3_octron predictor object (SAM3.1).
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
