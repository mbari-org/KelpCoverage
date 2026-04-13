import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from kelp_coverage import config as cfg
from kelp_coverage.core.color import load_image, rgb_to_lab_gpu, convert_opencv_lab
from kelp_coverage.core.slicing import slice_image
from kelp_coverage.core.candidates import select_prompt_points, check_shortcut_condition
from kelp_coverage.core.mask import (
    reconstruct_mask_gpu,
    erode_mask_gpu,
    merge_hierarchical_masks,
    calculate_coverage,
)
from kelp_coverage.models.protocol import load_model
import kelp_coverage.models.mobile_sam

def process_image(
    image_path: str,
    model,
    water_lab: torch.Tensor,
    device: str,
    slice_size: int          = cfg.SLICE_SIZE,
    slice_overlap: float     = cfg.SLICE_OVERLAP,
    padding: int             = cfg.PADDING,
    num_points: int          = cfg.NUM_POINTS,
    threshold: int           = cfg.THRESHOLD,
    grid_size: int           = cfg.GRID_SIZE,
    std_threshold: float     = cfg.UNIFORMITY_STD_THRESHOLD,
    uniform_thresh: float    = cfg.UNIFORM_GRID_THRESH,
    water_thresh: float      = cfg.WATER_GRID_THRESH,
    fallback_brightness: float = cfg.FALLBACK_BRIGHTNESS_THRESHOLD,
    fallback_distance: float = cfg.FALLBACK_DISTANCE_THRESHOLD,
    gpu_batch_size: int      = cfg.GPU_BATCH_SIZE,
    downsample_factor: float = cfg.DOWNSAMPLE_FACTOR,
    clahe: bool              = cfg.CLAHE_ENABLED,
    verbose: bool            = False,
    image: Optional[np.ndarray] = None,
    lab_tensor: Optional[torch.Tensor] = None,
) -> Tuple[List[Tuple[torch.Tensor, Any]], Dict]:
    if image is None:
        image = load_image(image_path, downsample_factor, clahe)
    slice_info = slice_image(image, slice_size, slice_overlap, padding)
    num_slices = len(slice_info["img_list"])

    if lab_tensor is None:
        lab_tensor = rgb_to_lab_gpu(image, device)

    results: List[Tuple[torch.Tensor, Any]] = [
        (torch.empty(0), np.array([])) for _ in range(num_slices)
    ]

    pending: List[Tuple[int, np.ndarray, np.ndarray]] = []

    image_name   = os.path.basename(image_path)
    num_slices   = len(slice_info["img_list"])
    slice_iter   = (
        tqdm(enumerate(slice_info["img_list"]), total=num_slices,
             desc=f"  {image_name} slices", leave=False)
        if verbose else enumerate(slice_info["img_list"])
    )

    for i, slice_img in slice_iter:
        start_x, start_y = slice_info["img_starting_pts"][i]
        h, w, _     = slice_img.shape
        content_h   = h - 2 * padding
        content_w   = w - 2 * padding
        end_y       = min(start_y + content_h, lab_tensor.shape[0])
        end_x       = min(start_x + content_w, lab_tensor.shape[1])

        lab_slice_gpu = lab_tensor[start_y:end_y, start_x:end_x]

        if padding > 0:
            lab_slice_gpu = F.pad(
                lab_slice_gpu, (0, 0, padding, padding, padding, padding), "constant", 0
            )

        points_xy, diagnostics = select_prompt_points(
            lab_slice_gpu, water_lab,
            num_points=num_points, threshold=threshold,
            grid_size=grid_size, std_threshold=std_threshold,
            slice_size=slice_size,
            return_diagnostics=True,
        )

        is_shortcut, _, _ = check_shortcut_condition(
            diagnostics, uniform_thresh, water_thresh
        )
        if is_shortcut:
            if verbose:
                print(f"    Slice {i}: uniform + water-colored, skipping SAM.")
            results[i] = (
                torch.ones((content_h, content_w), dtype=torch.bool, device=device),
                np.array([]),
            )
            continue

        if points_xy is None or len(points_xy) == 0:
            avg_brightness = lab_slice_gpu[:, :, 0].mean().item()
            avg_dist = torch.linalg.norm(
                lab_slice_gpu - water_lab, dim=2
            ).mean().item()
            is_water = avg_brightness > fallback_brightness and avg_dist < fallback_distance
            if verbose:
                if is_water:
                    print(f"    Slice {i}: no points, fallback → all water.")
                else:
                    print(f"    Slice {i}: no points, fallback → all kelp.")
            mask = torch.ones((content_h, content_w), dtype=torch.bool, device=device) \
                   if is_water else \
                   torch.zeros((content_h, content_w), dtype=torch.bool, device=device)
            results[i] = (mask, np.array([]))
            continue

        pending.append((i, slice_img, points_xy))

    if pending:
        indices   = [p[0] for p in pending]
        slices    = [p[1] for p in pending]
        pts_list  = [p[2] for p in pending]
        n_batches = (len(slices) + gpu_batch_size - 1) // gpu_batch_size

        for batch_start in range(0, len(slices), gpu_batch_size):
            batch_slices = slices[batch_start:batch_start + gpu_batch_size]
            batch_pts    = pts_list[batch_start:batch_start + gpu_batch_size]
            batch_idx    = indices[batch_start:batch_start + gpu_batch_size]

            if verbose:
                batch_num = batch_start // gpu_batch_size + 1
                print(f"    SAM batch {batch_num}/{n_batches}: {len(batch_slices)} slices...")

            batch_tensor, input_sizes = model.preprocess(batch_slices)
            embeddings = model.encode(batch_tensor)

            for j, (slice_img, pts, orig_idx) in enumerate(
                zip(batch_slices, batch_pts, batch_idx)
            ):
                mask = model.decode(
                    embeddings[j].unsqueeze(0),
                    pts,
                    slice_img.shape[:2],
                    input_sizes[j],
                )
                if padding > 0:
                    h_m, w_m = mask.shape
                    if h_m > 2 * padding and w_m > 2 * padding:
                        mask = mask[padding:-padding, padding:-padding]

                results[orig_idx] = (mask.to(device), pts)

    return results, slice_info


def _load_checkpoint(json_path: str) -> Tuple[Dict, Dict]:
    if not os.path.exists(json_path):
        return {}, {}
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        results_dict = {r["image_name"]: r for r in data.get("results", [])}
        return results_dict, data
    except (json.JSONDecodeError, KeyError):
        print("Warning: could not parse existing results.json — starting fresh.")
        return {}, {}


def _write_checkpoint(
    json_path: str,
    all_results_dict: Dict,
    command_str: str,
    run_args_dict: Dict,
    existing_top: Dict,
) -> None:
    output = {
        "command":  existing_top.get("command",  command_str),
        "run_args": existing_top.get("run_args", run_args_dict),
        "results":  list(all_results_dict.values()),
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=4)


def process_site(
    image_paths: List[str],
    model,
    water_lab: torch.Tensor,
    device: str,
    run_dir: str,
    command_str: str,
    run_args_dict: Dict,
    site_name: Optional[str]  = None,
    tator_csv: Optional[str]  = None,
    verbose: bool             = False,
    overwrite: bool           = False,
    generate_overlay: bool    = False,
    generate_slice_viz: bool  = False,
    generate_threshold_viz: bool = False,
    generate_erosion_viz: bool   = False,
    hierarchical: bool        = cfg.HIERARCHICAL,
    hierarchical_slice_size: int = cfg.HIERARCHICAL_SLICE_SIZE,
    **process_kwargs,
) -> None:
    from kelp_coverage.io.pixel_analysis import extract_location
    from kelp_coverage.viz import visualization as viz

    os.makedirs(os.path.join(run_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "visualizations"), exist_ok=True)

    json_path = os.path.join(run_dir, "results.json")
    all_results_dict, existing_top = _load_checkpoint(json_path)

    images_since_save = 0
    iterator = (
        image_paths if verbose
        else tqdm(image_paths, desc=f"Processing {site_name or 'site'}")
    )

    try:
        for image_path in iterator:
            image_name = os.path.basename(image_path)

            if not overwrite and image_name in all_results_dict:
                if verbose:
                    print(f"Skipping {image_name} (already processed)")
                continue

            try:
                image_base     = os.path.splitext(image_name)[0]
                original_image = load_image(image_path, **{
                    k: process_kwargs[k] for k in ("downsample_factor", "clahe")
                    if k in process_kwargs
                })
                lab_tensor_gpu = rgb_to_lab_gpu(original_image, device)

                _MERGE_KEYS = {"use_erosion", "erosion_kernel", "use_color_validation",
                               "color_threshold", "lightness_threshold"}
                image_kwargs = {k: v for k, v in process_kwargs.items() if k not in _MERGE_KEYS}
                coarse_kwargs = {k: v for k, v in image_kwargs.items() if k != "slice_size"}

                if hierarchical:
                    if verbose:
                        print(f"\n  [{image_name}] Fine pass (slice_size={image_kwargs.get('slice_size', cfg.SLICE_SIZE)})...")
                    fine_results, fine_slice_info = process_image(
                        image_path, model, water_lab, device,
                        verbose=verbose, image=original_image, lab_tensor=lab_tensor_gpu, **image_kwargs,
                    )
                    if verbose:
                        print(f"  [{image_name}] Coarse pass (slice_size={hierarchical_slice_size})...")
                    coarse_results, coarse_slice_info = process_image(
                        image_path, model, water_lab, device,
                        slice_size=hierarchical_slice_size,
                        verbose=verbose, image=original_image, lab_tensor=lab_tensor_gpu, **coarse_kwargs,
                    )
                    fine_water   = reconstruct_mask_gpu(fine_results,   fine_slice_info,   device, "OR")
                    coarse_water = reconstruct_mask_gpu(coarse_results, coarse_slice_info, device, "AND")
                    water_mask_gpu = merge_hierarchical_masks(
                        fine_water, coarse_water, lab_tensor_gpu, water_lab,
                        **{k: process_kwargs[k] for k in (
                            "use_erosion", "erosion_kernel", "use_color_validation",
                            "color_threshold", "lightness_threshold",
                        ) if k in process_kwargs},
                    )
                    results    = fine_results
                    slice_info = fine_slice_info
                else:
                    results, slice_info = process_image(
                        image_path, model, water_lab, device,
                        verbose=verbose, image=original_image, lab_tensor=lab_tensor_gpu, **image_kwargs,
                    )
                    water_mask_gpu = reconstruct_mask_gpu(results, slice_info, device, "OR")

                coverage = calculate_coverage(water_mask_gpu)
                full_mask_np = water_mask_gpu.cpu().numpy()

                viz.save_binary_mask(full_mask_np, image_base, os.path.join(run_dir, "masks"))

                if generate_overlay:
                    viz.save_overlay(
                        original_image, {"Final": full_mask_np},
                        f"{image_base} | Coverage: {coverage:.2f}%",
                        os.path.join(run_dir, "visualizations", f"{image_base}_overlay.png"),
                    )

                if generate_slice_viz:
                    viz.save_slice_visualization(
                        slice_info, results, image_base,
                        os.path.join(run_dir, "visualizations"),
                    )

                if generate_threshold_viz:
                    viz.create_threshold_visualization(
                        image_path, image_base, water_lab,
                        os.path.join(run_dir, "visualizations"),
                        device,
                        **{k: process_kwargs[k] for k in ("threshold", "slice_size", "slice_overlap")
                           if k in process_kwargs},
                    )

                if generate_erosion_viz and hierarchical:
                    _use_erosion    = process_kwargs.get("use_erosion",    cfg.USE_EROSION_MERGE)
                    _erosion_kernel = process_kwargs.get("erosion_kernel", cfg.EROSION_KERNEL_SIZE)
                    pre_erosion_np  = (~coarse_water).cpu().numpy()
                    post_erosion_np = (
                        erode_mask_gpu(~coarse_water, _erosion_kernel).cpu().numpy()
                        if _use_erosion else pre_erosion_np
                    )
                    viz.save_erosion_visualization(
                        original_image,
                        pre_erosion_np,
                        post_erosion_np,
                        f"{image_base} | Erosion Visualization",
                        os.path.join(run_dir, "visualizations", f"{image_base}_erosion.png"),
                    )

                image_id, latitude, longitude = _get_tator_metadata(image_path, tator_csv)

                all_results_dict[image_name] = {
                    "image_name":         image_name,
                    "image_id":           int(image_id)    if image_id   is not None else None,
                    "latitude":           float(latitude)  if latitude   is not None else None,
                    "longitude":          float(longitude) if longitude  is not None else None,
                    "coverage_percentage": coverage,
                }
                images_since_save += 1

                if images_since_save >= cfg.CHECKPOINT_INTERVAL:
                    _write_checkpoint(json_path, all_results_dict, command_str, run_args_dict, existing_top)
                    images_since_save = 0
                    if verbose:
                        print(f"Checkpoint saved ({cfg.CHECKPOINT_INTERVAL} images).")

            except Exception as e:
                print(f"ERROR processing {image_name}: {e}")
                with open(os.path.join(run_dir, "error_log.txt"), "a") as f:
                    f.write(f"{image_path}: {e}\n")
                continue

    finally:
        if images_since_save > 0:
            _write_checkpoint(json_path, all_results_dict, command_str, run_args_dict, existing_top)
            if verbose:
                print("Final checkpoint saved.")


def _get_tator_metadata(
    image_path: str, tator_csv: Optional[str]
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not tator_csv or not os.path.exists(tator_csv):
        return None, None, None
    try:
        import pandas as pd
        df   = pd.read_csv(tator_csv)
        name = os.path.basename(image_path)
        row  = df[df["$name"] == name]
        if row.empty:
            return None, None, None
        return (
            row["$id"].values[0]        if "$id"        in row.columns else None,
            row["latitude"].values[0]   if "latitude"   in row.columns else None,
            row["longitude"].values[0]  if "longitude"  in row.columns else None,
        )
    except Exception:
        return None, None, None


def run_analysis(
    image_paths: List[str],
    model_name: str,
    checkpoint: str,
    water_lab_opencv: Tuple[int, int, int],
    run_dir: str,
    command_str: str,
    run_args_dict: Dict,
    **kwargs,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_model(model_name, checkpoint, device)

    import kelp_coverage.models.mobile_sam  # noqa: F401

    l, a, b   = water_lab_opencv
    water_lab = convert_opencv_lab(l, a, b, device)

    process_site(
        image_paths=image_paths,
        model=model,
        water_lab=water_lab,
        device=device,
        run_dir=run_dir,
        command_str=command_str,
        run_args_dict=run_args_dict,
        **kwargs,
    )
