
import os 
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
from ultralytics.models.sam.build_sam3 import build_interactive_sam3


def build_sam3_octron(
    ckpt_path,
    semantic=False,
    semantic_ckpt_path=None,
    mode="eval",
    **kwargs,
):
    """
    Build the SAM3 model from a checkpoint file for OCTRON usage.
    
    Uses the ultralytics SAM3 implementation which provides the SAM2-compatible 
    track_step interface on top of SAM3 weights.
    
    Parameters
    ----------
    ckpt_path : str
        Path to the SAM3 model checkpoint file (.pt) for the interactive/tracking model.
    semantic : bool
        If True, also build the SAM3SemanticModel detector and return a
        SAM3_semantic_octron that composes detection + tracking (Mode B).
        If False (default), return a plain SAM3_octron tracker (Mode A).
    semantic_ckpt_path : str, optional
        Path to a separate checkpoint for the semantic model. If None,
        uses the same ckpt_path for both models.
    mode : str
        Mode to run the model in. Default is "eval"
    **kwargs : dict
        Additional keyword arguments.
        
    Returns
    -------
    model : SAM3_octron or SAM3_semantic_octron
        The predictor object.
    device : torch.device
        The device the model is running on.
    """
    from .sam3_octron import SAM3_octron
    
    # Find out which device to use
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "⚠️ Support for MPS devices is preliminary. SAM 3 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
        )

    # Build SAM3Model via ultralytics (handles checkpoint loading + key remapping)
    sam3_model = build_interactive_sam3(ckpt_path)
    sam3_model = sam3_model.to(device)
    if mode == "eval":
        sam3_model.eval()
    
    # Wrap in SAM3_octron (tracker – used in both modes)
    tracker = SAM3_octron(model=sam3_model, device=device)
    
    if not semantic:
        print('Loaded SAM3 OCTRON – Mode A (interactive tracking)')
        return tracker, device
    
    # --- Mode B: also build the semantic detector ---
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model
    from .sam3_octron import SAM3_semantic_octron
    
    sem_ckpt = semantic_ckpt_path or ckpt_path
    detector = build_sam3_image_model(sem_ckpt)
    detector = detector.to(device)
    if mode == "eval":
        detector.eval()
    
    predictor = SAM3_semantic_octron(
        detector_model=detector,
        tracker=tracker,
        device=device,
        detector_ckpt_path=sem_ckpt,
    )
    
    print('Loaded SAM3 OCTRON – Mode B (semantic detection + tracking)')
    return predictor, device
