import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from kelp_coverage import config

def get_grid_info(
    lab_tensor: torch.Tensor,
    water_lab: torch.Tensor,
    threshold: int = config.THRESHOLD,
    grid_size: int = config.GRID_SIZE,
    std_threshold: float = config.UNIFORMITY_STD_THRESHOLD,
) -> Tuple[torch.Tensor, Dict]:
    h, w, _ = lab_tensor.shape

    pixel_color_dist = torch.linalg.norm(lab_tensor - water_lab, dim=2)
    water_color_pixels = pixel_color_dist <= threshold

    l_channel = lab_tensor[:, :, 0].unsqueeze(0).unsqueeze(0)
    pooled_l_sq = F.avg_pool2d(l_channel ** 2, kernel_size=grid_size, stride=grid_size)
    pooled_l    = F.avg_pool2d(l_channel,      kernel_size=grid_size, stride=grid_size)
    grid_stds   = torch.sqrt(torch.clamp(pooled_l_sq.squeeze() - pooled_l.squeeze() ** 2, min=0))
    uniform_grids = grid_stds <= std_threshold

    a_channel = lab_tensor[:, :, 1].unsqueeze(0).unsqueeze(0)
    b_channel = lab_tensor[:, :, 2].unsqueeze(0).unsqueeze(0)
    pooled_a  = F.avg_pool2d(a_channel, kernel_size=grid_size, stride=grid_size)
    pooled_b  = F.avg_pool2d(b_channel, kernel_size=grid_size, stride=grid_size)
    avg_lab_grids = (
        torch.cat([pooled_l, pooled_a, pooled_b], dim=1).squeeze(0).permute(1, 2, 0)
    )
    grid_color_dist  = torch.linalg.norm(avg_lab_grids - water_lab, dim=2)
    water_color_grids = grid_color_dist <= threshold

    valid_water_grids = uniform_grids & water_color_grids

    grid_pixel_mask = (
        F.interpolate(
            valid_water_grids.float().unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest",
        )
        .squeeze()
        .bool()
    )

    candidates_yx = torch.nonzero(grid_pixel_mask & water_color_pixels)

    info = {
        "uniform_grids":      uniform_grids.cpu().numpy(),
        "water_color_grids":  water_color_grids.cpu().numpy(),
        "valid_water_grids":  valid_water_grids.cpu().numpy(),
    }
    return candidates_yx, info

def check_shortcut_condition(
    info: Dict,
    uniform_thresh: float = config.UNIFORM_GRID_THRESH,
    water_thresh: float   = config.WATER_GRID_THRESH,
) -> Tuple[bool, float, float]:
    uniform_grids     = info.get("uniform_grids")
    water_color_grids = info.get("water_color_grids")

    if uniform_grids is None or water_color_grids is None:
        return False, 0.0, 0.0

    total = uniform_grids.size
    if total == 0:
        return False, 0.0, 0.0

    uniform_pct = float(np.sum(uniform_grids)) / total
    water_pct   = float(np.sum(water_color_grids)) / total
    is_shortcut = uniform_pct >= uniform_thresh and water_pct >= water_thresh
    return is_shortcut, uniform_pct, water_pct

def _poisson_disk_sampling(
    points: np.ndarray,
    n_samples: int,
    slice_size: int,
    k: int = config.POISSON_DISK_K,
) -> List[int]:
    if len(points) == 0:
        return []
    if len(points) <= n_samples:
        return list(range(len(points)))

    min_dist_sq = (slice_size ** 2 * 2) / (n_samples ** 2 * 4)
    selected = [int(np.random.randint(0, len(points)))]

    while len(selected) < n_samples:
        found = False
        for _ in range(k):
            idx = int(np.random.randint(0, len(points)))
            candidate = points[idx]
            diffs    = points[selected] - candidate
            dist_sq  = np.sum(diffs ** 2, axis=1)
            if np.all(dist_sq > min_dist_sq):
                selected.append(idx)
                found = True
                break
        if not found:
            break

    return selected

def _one_per_grid(
    candidates_yx: torch.Tensor,
    grid_size: int,
    img_width: int,
) -> torch.Tensor:
    if candidates_yx.shape[0] == 0:
        return candidates_yx

    num_cols = (img_width + grid_size - 1) // grid_size
    perm     = torch.randperm(candidates_yx.shape[0], device=candidates_yx.device)
    shuffled = candidates_yx[perm]

    grid_ids = (shuffled[:, 0] // grid_size) * num_cols + (shuffled[:, 1] // grid_size)
    sorted_ids, sort_order = torch.sort(grid_ids, stable=True)

    is_first = torch.cat([
        torch.ones(1, dtype=torch.bool, device=candidates_yx.device),
        sorted_ids[1:] != sorted_ids[:-1],
    ])

    return shuffled[sort_order[is_first]]

def select_prompt_points(
    lab_tensor: torch.Tensor,
    water_lab: torch.Tensor,
    num_points: int          = config.NUM_POINTS,
    threshold: int           = config.THRESHOLD,
    grid_size: int           = config.GRID_SIZE,
    std_threshold: float     = config.UNIFORMITY_STD_THRESHOLD,
    slice_size: int          = config.SLICE_SIZE,
    return_diagnostics: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    _, w, _ = lab_tensor.shape

    candidates_yx, info = get_grid_info(
        lab_tensor, water_lab, threshold, grid_size, std_threshold
    )

    if return_diagnostics:
        info["initial_candidates"] = candidates_yx.cpu().numpy()

    if candidates_yx.shape[0] < num_points:
        if return_diagnostics:
            info["grid_filtered_candidates"] = np.array([])
        return None, info if return_diagnostics else None

    filtered_yx = _one_per_grid(candidates_yx, grid_size, w)

    if return_diagnostics:
        info["grid_filtered_candidates"] = filtered_yx.cpu().numpy()

    num_candidates = filtered_yx.shape[0]
    if num_candidates == 0:
        return None, info if return_diagnostics else None

    num_to_sample = min(num_points, num_candidates)
    candidates_np = filtered_yx.cpu().numpy()
    indices = _poisson_disk_sampling(candidates_np, num_to_sample, slice_size)
    final_yx = filtered_yx[indices] if indices else filtered_yx[:num_to_sample]

    points_xy = final_yx[:, [1, 0]].cpu().numpy().astype(np.float32)
    return points_xy, info if return_diagnostics else None
