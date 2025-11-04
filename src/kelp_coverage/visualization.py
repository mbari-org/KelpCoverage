import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from tqdm import tqdm
from typing import List, Tuple, Dict, Any, Optional
from matplotlib.patches import Patch

from .segmentation_processors import HierarchicalProcessor
from .sahisam import SAHISAM


def _calculate_coverage(mask: np.ndarray) -> float:
    """Calculates the percentage of the mask that is True (kelp)."""
    if mask.size == 0:
        return 0.0
    total_pixels = mask.size
    kelp_pixels = np.sum(mask)
    return (kelp_pixels / total_pixels) * 100


def _get_image_metadata(
    image_path: str, tator_csv: Optional[str]
) -> Tuple[str, Optional[str], Optional[float], Optional[float]]:
    image_name = os.path.basename(image_path)
    if not tator_csv or not os.path.exists(tator_csv):
        return image_name, None, None, None
    tator_df = pd.read_csv(tator_csv)
    image_row = tator_df[tator_df["$name"] == image_name]
    if image_row.empty:
        return image_name, None, None, None
    row = image_row.iloc[0]
    return image_name, row.get("$id"), row.get("latitude"), row.get("longitude")


def _save_binary_mask(kelp_mask: np.ndarray, image_base: str, mask_dir: str, threshold: Optional[float] = None) -> None:
    os.makedirs(mask_dir, exist_ok=True)
    thresh_str = f"_thresh_{threshold:.2f}" if threshold is not None else ""
    kelp_mask_save_path = os.path.join(mask_dir, f"{image_base}_kelp_mask{thresh_str}.png")
    kelp_binary_mask_img = (kelp_mask.astype(np.uint8)) * 255
    cv2.imwrite(kelp_mask_save_path, kelp_binary_mask_img)

def _save_overlay(
    original_image_rgb: np.ndarray,
    masks_to_overlay: Dict[str, np.ndarray],
    title: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    # Ensure the visualization directory exists
    viz_dir = os.path.dirname(output_path)
    os.makedirs(viz_dir, exist_ok=True)
    
    # --- Matplotlib Figure Setup ---
    plt.figure(figsize=(12, 12))
    plt.imshow(original_image_rgb)

    # Use a constant red color in RGBA format (Red, Green, Blue, Alpha)
    red_color_rgba = (1.0, 0.0, 0.0, 0.45)  # R=1, G=0, B=0 at 45% opacity

    legend_elements = []
    for name, kelp_mask in masks_to_overlay.items():
        if kelp_mask is None:
            continue
        # Create and display the red overlay for the visualization
        overlay = np.zeros((*kelp_mask.shape, 4), dtype=np.float32)
        overlay[kelp_mask] = red_color_rgba
        plt.imshow(overlay, interpolation="nearest")
        
        legend_elements.append(Patch(facecolor=red_color_rgba, edgecolor='red', label=f"{name} Kelp"))

    # --- Finalize and Save the Overlay Visualization ---
    plt.title(title, fontsize=14)
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", fontsize="large")
    plt.axis("off")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"  - Saved overlay to: {output_path}")


def _save_confidence_heatmap(
    confidence_map_gpu: torch.Tensor,
    original_image_path: str,
    output_path: str,
    title: str,
    verbose: bool = False
):
    """Generates and saves a raw kelp confidence heatmap and an overlayed version."""
    if verbose:
        print(f"  - Generating heatmaps for '{title}'...")

    # --- Setup Paths ---
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    path_root, path_ext = os.path.splitext(output_path)
    raw_heatmap_path = f"{path_root}_raw{path_ext}"

    # --- Data Preparation ---
    # The processor provides WATER confidence, so we invert for KELP confidence visualization
    kelp_confidence_np = (1.0 - confidence_map_gpu).cpu().numpy()
    
    # --- 1. Save Raw Heatmap (Color Visualization Only) ---
    fig_raw, ax_raw = plt.subplots(figsize=(15, 15))
    im = ax_raw.imshow(kelp_confidence_np, cmap='viridis', vmin=0, vmax=1)
    ax_raw.axis('off')
    # Save just the content of the plot without extra whitespace
    plt.savefig(raw_heatmap_path, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close(fig_raw)
    if verbose:
        print(f"    - Saved raw heatmap to: {raw_heatmap_path}")

    # --- 2. Save Overlayed Heatmap ---
    original_image = cv2.imread(original_image_path)
    if original_image is None:
        print(f"    - ERROR: Could not read original image at {original_image_path}")
        return
    
    # Ensure heatmap dimensions match the original image for overlay
    if kelp_confidence_np.shape != original_image.shape[:2]:
        kelp_confidence_np = cv2.resize(kelp_confidence_np, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    fig_overlay, ax_overlay = plt.subplots(figsize=(15, 15))
    ax_overlay.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    heatmap = ax_overlay.imshow(kelp_confidence_np, cmap='viridis', alpha=0.5, vmin=0, vmax=1)
    
    cbar = fig_overlay.colorbar(heatmap, ax=ax_overlay, fraction=0.046, pad=0.04)
    cbar.set_label('Kelp Confidence', size='large')
    
    ax_overlay.set_title(title, fontsize=16)
    ax_overlay.axis('off')
    
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig_overlay)
    if verbose:
        print(f"    - Saved overlayed heatmap to: {output_path}")


def run_sahi_sam_visualization(
    image_paths: list,
    processor: Any,
    run_dir: str,
    command_str: str,
    run_args_dict: dict,
    site_name: Optional[str] = None,
    tator_csv: Optional[str] = None,
    verbose: bool = False,
    generate_overlay: bool = False,
    generate_slice_viz: bool = False,
    generate_threshold_viz: bool = False,
    generate_erosion_viz: bool = False,
    generate_component_viz: bool = False,
    slice_viz_max_size: int = 256,
    coverage_only: bool = False,
    overwrite: bool = False,
    generate_fine_heatmap: bool = False,
    generate_coarse_heatmap: bool = False,
    generate_merged_heatmap: bool = False,
) -> None:
    viz_dir = os.path.join(run_dir, "visualizations")
    mask_dir = os.path.join(run_dir, "masks")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    output_json_path = os.path.join(run_dir, "results.json")

    all_results_dict = {}

    if os.path.exists(output_json_path) and not overwrite:
        print(f"Results for this configuration already exist in {run_dir}. Use --overwrite to re-run.")
        return

    image_iterator = (
        tqdm(image_paths, desc=f"Processing Images for {site_name}")
        if not verbose
        else image_paths
    )

    for image_path in image_iterator:
        image_name = os.path.basename(image_path)
        image_base = os.path.splitext(image_name)[0]

        try:
            if verbose:
                print(f"--- Processing {image_base} ---")

            model = getattr(processor, 'fine_model', getattr(processor, 'model', None))
            original_image_rgb = model._load(image_path)
            image_lab_tensor_cpu = model._get_lab_tensor(original_image_rgb).cpu()

            results, slice_info = processor.process_image(
                image_path, full_lab_tensor_cpu=image_lab_tensor_cpu
            )
            
            reconstruction_output = processor.reconstruct_full_mask(
                results,
                slice_info,
                image_lab_tensor_cpu=image_lab_tensor_cpu,
                image_path=image_path,
                run_dir=run_dir,
                coverage_only=coverage_only,
            )

            _, image_id, latitude, longitude = _get_image_metadata(
                image_path, tator_csv
            )

            if coverage_only:
                # reconstruction_output is a dict of {threshold: coverage}
                coverage_percentages = reconstruction_output
                if verbose:
                     print(f"--- Finished {image_base} ---")
                     for thresh, cov in coverage_percentages.items():
                         print(f"  - Threshold {thresh:.2f}: Coverage {cov:.2f}%")
                
                # Store results for each threshold
                for thresh, cov in coverage_percentages.items():
                    result_key = f"{image_name}_{thresh}"
                    result_data = {
                        "image_name": f"{image_name}_thresh_{thresh:.2f}",
                        "image_id": int(image_id) if image_id is not None else None,
                        "latitude": float(latitude) if latitude is not None else None,
                        "longitude": float(longitude) if longitude is not None else None,
                        "coverage_percentage": cov,
                        "threshold": thresh
                    }
                    all_results_dict[result_key] = result_data
            else:
                # reconstruction_output is a dict of {threshold: mask}
                final_kelp_masks = reconstruction_output
                if not final_kelp_masks:
                    if verbose:
                        print(f"--- Finished {image_base} (no masks generated) ---")
                    continue
                
                # MODIFIED: Loop through each generated mask to save outputs for each threshold
                for threshold, mask in final_kelp_masks.items():
                    # Save the binary mask for the current threshold
                    _save_binary_mask(mask, image_base, mask_dir, threshold=threshold)

                    # If overlay generation is enabled, create one for the current threshold
                    if generate_overlay:
                        coverage_percentage = _calculate_coverage(mask)
                        overlay_output_path = os.path.join(viz_dir, f"{image_base}_overlay_thresh_{threshold:.2f}.png")
                        _save_overlay(
                            original_image_rgb,
                            {"Final": mask},
                            f"{image_base} | Kelp Coverage (Thresh {threshold:.2f}): {coverage_percentage:.2f}%",
                            overlay_output_path,
                            verbose=verbose,
                        )

                if isinstance(processor, HierarchicalProcessor):
                    if generate_fine_heatmap:
                        fine_map = processor.get_fine_confidence_map()
                        if fine_map is not None:
                            _save_confidence_heatmap(fine_map, image_path, os.path.join(viz_dir, f"{image_base}_heatmap_fine.png"), f"Fine Pass Kelp Confidence\n{image_base}", verbose)
                    
                    if generate_coarse_heatmap:
                        coarse_map = processor.get_coarse_confidence_map()
                        if coarse_map is not None:
                            _save_confidence_heatmap(coarse_map, image_path, os.path.join(viz_dir, f"{image_base}_heatmap_coarse.png"), f"Coarse Pass Kelp Confidence\n{image_base}", verbose)

                    if generate_merged_heatmap:
                        merged_map = processor.get_merged_confidence_map()
                        if merged_map is not None:
                            _save_confidence_heatmap(merged_map, image_path, os.path.join(viz_dir, f"{image_base}_heatmap_merged.png"), f"Merged Kelp Confidence\n{image_base}", verbose)

                # Store results for each threshold
                for thresh, mask in final_kelp_masks.items():
                    cov = _calculate_coverage(mask)
                    result_key = f"{image_name}_{thresh}"
                    result_data = {
                        "image_name": f"{image_name}_thresh_{thresh:.2f}",
                        "image_id": int(image_id) if image_id is not None else None,
                        "latitude": float(latitude) if latitude is not None else None,
                        "longitude": float(longitude) if longitude is not None else None,
                        "coverage_percentage": cov,
                        "threshold": thresh
                    }
                    all_results_dict[result_key] = result_data


        except Exception as e:
            print(f"\n--- ERROR processing {image_name}: {e} ---")
            import traceback
            traceback.print_exc()
            with open(os.path.join(run_dir, "error_log.txt"), "a") as f:
                f.write(f"Error on {image_name}: {e}\n")
            continue

    final_output = {
        "command": command_str,
        "run_args": run_args_dict,
        "results": list(all_results_dict.values()),
    }
    with open(output_json_path, "w") as f:
        json.dump(final_output, f, indent=4)

    if verbose:
        print(f"--- Analysis complete. All results saved in: {run_dir} ---")

