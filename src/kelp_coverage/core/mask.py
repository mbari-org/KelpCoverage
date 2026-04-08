import numpy as np
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from kelp_coverage import config

def reconstruct_mask_gpu(
    masks: List[Tuple[torch.Tensor, Any]],
    slice_info: Dict,
    device: str,
    merge_logic: str = "OR",
) -> torch.Tensor:
    H, W = slice_info["original_shape"][:2]
    has_valid = any(torch.is_tensor(m[0]) and m[0].numel() > 0 for m in masks)

    if not has_valid or merge_logic == "OR":
        full_mask = torch.zeros((H, W), dtype=torch.bool, device=device)
    elif merge_logic == "AND":
        full_mask = torch.ones((H, W), dtype=torch.bool, device=device)
        processed = torch.zeros((H, W), dtype=torch.bool, device=device)
    else:
        raise ValueError(f"merge_logic must be 'OR' or 'AND', got '{merge_logic}'")

    if not has_valid:
        return full_mask

    for i, (mask_tensor, _) in enumerate(masks):
        if not torch.is_tensor(mask_tensor) or mask_tensor.numel() == 0:
            continue

        start_x, start_y = slice_info["img_starting_pts"][i]  # (col, row)
        h, w = mask_tensor.shape
        end_y = min(start_y + h, H)
        end_x = min(start_x + w, W)

        if start_y >= H or start_x >= W:
            continue

        region_h = end_y - start_y
        region_w = end_x - start_x
        region   = (slice(start_y, end_y), slice(start_x, end_x))
        s_mask   = mask_tensor[:region_h, :region_w].bool().to(device)

        if merge_logic == "OR":
            full_mask[region] |= s_mask
        elif merge_logic == "AND":
            overlap = processed[region]
            full_mask[region][overlap]  &= s_mask[overlap]
            full_mask[region][~overlap]  = s_mask[~overlap]
            processed[region]            = True

    return full_mask

def erode_mask_gpu(kelp_mask: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1

    padding = kernel_size // 2
    inverted = ~kelp_mask
    dilated_inv = (
        F.max_pool2d(
            inverted.float().unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )
        .squeeze()
        .bool()
    )
    return ~dilated_inv

def merge_hierarchical_masks(
    fine_water: torch.Tensor,
    coarse_water: torch.Tensor,
    lab_tensor: torch.Tensor,
    water_lab: torch.Tensor,
    use_erosion: bool            = config.USE_EROSION_MERGE,
    erosion_kernel: int          = config.EROSION_KERNEL_SIZE,
    use_color_validation: bool   = config.USE_COLOR_VALIDATION,
    color_threshold: float       = config.MERGE_COLOR_THRESHOLD,
    lightness_threshold: float   = config.MERGE_LIGHTNESS_THRESHOLD,
) -> torch.Tensor:
    device = fine_water.device

    fine_kelp   = ~fine_water
    coarse_kelp = ~coarse_water

    if use_erosion:
        coarse_kelp = erode_mask_gpu(coarse_kelp, erosion_kernel)

    agreed_kelp  = fine_kelp & coarse_kelp
    disagreement = fine_kelp ^ coarse_kelp

    validated_kelp = torch.zeros_like(disagreement)

    if use_color_validation and torch.any(disagreement):
        pixels_lab = lab_tensor[disagreement]

        lightness      = pixels_lab[:, 0]
        chroma         = pixels_lab[:, 1:]
        chroma_dist    = torch.linalg.norm(chroma - water_lab[1:], dim=1)

        is_kelp = (lightness < lightness_threshold) & (chroma_dist > color_threshold)
        validated_kelp[disagreement] = is_kelp

    final_kelp = agreed_kelp | validated_kelp
    return ~final_kelp

def calculate_coverage(water_mask: torch.Tensor) -> float:
    if water_mask.numel() == 0:
        return 0.0
    total       = water_mask.numel()
    kelp_pixels = total - int(torch.sum(water_mask).item())
    return (kelp_pixels / total) * 100.0
