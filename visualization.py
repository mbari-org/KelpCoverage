import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from typing import List, Tuple, Dict, Any, Optional
import torch
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from segmentation_processors import HierarchicalProcessor, SinglePassProcessor
from sahisam import SAHISAM
from matplotlib.patches import Patch

def _calculate_coverage(mask: np.ndarray) -> float:
    if mask.size == 0:
        return 0.0
    total_pixels = mask.size
    water_pixels = np.sum(mask)
    kelp_pixels = total_pixels - water_pixels
    return (kelp_pixels / total_pixels) * 100

def _get_image_metadata(image_path: str, tator_csv: Optional[str]) -> Tuple[str, Optional[str], Optional[float], Optional[float]]:
    image_name = os.path.basename(image_path)
    if not tator_csv or not os.path.exists(tator_csv):
        return image_name, None, None, None
    tator_df = pd.read_csv(tator_csv)
    image_row = tator_df[tator_df['$name'] == image_name]
    if image_row.empty:
        return image_name, None, None, None
    row = image_row.iloc[0]
    return image_name, row.get('$id'), row.get('latitude'), row.get('longitude')

def _save_coverage_to_csv(image_path: str, coverage_percentage: float, results_dir: str, site_name: str, param_string: str, tator_csv: Optional[str]) -> None:
    image_name, image_id, latitude, longitude = _get_image_metadata(image_path, tator_csv)
    if image_id is None:
        return

    coverage_data = {
        'image_name': image_name, 'image_id': image_id, 'latitude': latitude, 
        'longitude': longitude, 'coverage_percentage': coverage_percentage,
        'param_string': param_string
    }
    site_dir = os.path.join(results_dir, site_name)
    os.makedirs(site_dir, exist_ok=True)
    site_coverage_csv = os.path.join(site_dir, "coverage_values.csv")
    try:
        if os.path.exists(site_coverage_csv):
            existing_df = pd.read_csv(site_coverage_csv)
            existing_df = existing_df[~((existing_df['image_id'] == image_id) & (existing_df['param_string'] == param_string))]
            coverage_df = pd.concat([existing_df, pd.DataFrame([coverage_data])], ignore_index=True)
        else:
            coverage_df = pd.DataFrame([coverage_data])
        coverage_df.to_csv(site_coverage_csv, index=False, float_format='%.7f')
    except Exception as e:
        print(f"Error saving to site CSV: {e}")

def _save_binary_mask(full_mask: np.ndarray, image_base: str, mask_dir: str) -> None:
    os.makedirs(mask_dir, exist_ok=True)
    kelp_mask_save_path = os.path.join(mask_dir, f"{image_base}_kelp_mask.png")
    kelp_binary_mask_img = ((~full_mask).astype(np.uint8)) * 255
    cv2.imwrite(kelp_mask_save_path, kelp_binary_mask_img)

def _save_overlay(original_image: np.ndarray, masks_to_overlay: Dict[str, np.ndarray], title: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    num_masks = len(masks_to_overlay)
    cmap = plt.cm.get_cmap('viridis', num_masks) if num_masks > 1 else plt.cm.get_cmap('ocean')
    colors = [cmap(i) for i in range(num_masks)]
    
    legend_elements = []
    for i, (name, water_mask) in enumerate(masks_to_overlay.items()):
        if water_mask is None: continue
        kelp_mask = ~water_mask
        color = colors[i]
        overlay = np.zeros((*kelp_mask.shape, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3] = np.where(kelp_mask, 0.45, 0)
        plt.imshow(overlay)
        legend_elements.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=name))
        
    plt.title(title, fontsize=14)
    if legend_elements:
        plt.legend(handles=legend_elements, loc='upper right', fontsize='large')
    plt.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def _save_slice_visualization(slice_info: Dict[str, Any], processed_results: List[Tuple[torch.Tensor, np.ndarray]], image_base: str, viz_dir: str, model: SAHISAM, max_size: int = 256) -> None:
    os.makedirs(viz_dir, exist_ok=True)
    img_list = slice_info['img_list']
    if not img_list: return
    
    num_slices = len(img_list)
    cols = len(sorted(set(pt[0] for pt in slice_info['img_starting_pts'])))
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes = axes.flatten()
    
    for i in range(num_slices):
        img = img_list[i]
        mask_tensor, points = processed_results[i]
        h_orig, w_orig, _ = img.shape
        scale = max_size / max(h_orig, w_orig)
        w_new, h_new = int(w_orig * scale), int(h_orig * scale)
        display_img = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)
        
        ax = axes[i]
        ax.imshow(display_img)
        
        if mask_tensor.numel() > 0:
            mask_cpu = mask_tensor.cpu().numpy()
            padding_scaled = int(model.padding * scale)
            
            content_h, content_w = w_new - 2*padding_scaled, h_new - 2*padding_scaled
            if content_h > 0 and content_w > 0:
                content_mask = cv2.resize(mask_cpu.astype(np.uint8), (content_w, content_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                slice_mask_full = np.zeros(display_img.shape[:2], dtype=bool)
                slice_mask_full[padding_scaled:-padding_scaled, padding_scaled:-padding_scaled] = content_mask
                
                mask_overlay_rgba = np.zeros((h_new, w_new, 4), dtype=np.uint8)
                mask_overlay_rgba[..., 2] = 255
                mask_overlay_rgba[..., 3] = np.where(~slice_mask_full, int(255 * 0.2), 0)
                ax.imshow(mask_overlay_rgba)

        if len(points) > 0:
            points_scaled = (np.array(points) * scale).astype(int)
            ax.plot(points_scaled[:, 0], points_scaled[:, 1], 'o', color='red', markersize=3)
            
        ax.axis('off')
        ax.set_title(f"Slice {i}", fontsize=10)
        
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    slice_save_path = os.path.join(viz_dir, f"{image_base}_slices_with_points.png")
    plt.savefig(slice_save_path, dpi=150, bbox_inches='tight')
    plt.close()

def _create_threshold_visualization(model: SAHISAM, image_path: str, image_base: str, viz_dir: str) -> None:
    original_image = model._load(image_path)
    slice_info = model._slice(original_image)
    img_list = slice_info['img_list']
    if not img_list: return
    
    num_slices = len(img_list)
    cols = len(sorted(set(pt[0] for pt in slice_info['img_starting_pts'])))
    rows = -(-num_slices // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()
    
    vmax = model.threshold * 2 
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap('viridis_r')
    
    for i, img in enumerate(img_list):
        ax = axes[i]
        ax.imshow(img)
        if model.water_lab_tensor is not None:
            dist_map = torch.linalg.norm(model._get_lab_tensor(img) - model.water_lab_tensor, dim=2).cpu().numpy()
            masked_array = np.ma.masked_where(dist_map > vmax, dist_map)
            ax.imshow(masked_array, alpha=0.5, cmap=cmap, norm=norm)
        ax.set_title(f"Slice {i}", fontsize=10)
        ax.axis('off')
        
    for j in range(num_slices, len(axes)):
        axes[j].axis('off')
        
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes.tolist(), label='Distance to Water LAB', shrink=0.8, aspect=20)
    plt.tight_layout()
    threshold_save_path = os.path.join(viz_dir, f"{image_base}_threshold_grid.png")
    plt.savefig(threshold_save_path, dpi=200, bbox_inches='tight')
    plt.close()

def _save_erosion_visualization(original_image: np.ndarray, pre_erosion_mask: np.ndarray, post_erosion_mask: np.ndarray, title: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    
    white_overlay = np.zeros((*pre_erosion_mask.shape, 4))
    white_overlay[..., :3] = [1, 1, 1]
    white_overlay[..., 3] = np.where(pre_erosion_mask, 0.5, 0)
    plt.imshow(white_overlay)
    
    red_overlay = np.zeros((*post_erosion_mask.shape, 4))
    red_overlay[..., 0] = 1
    red_overlay[..., 3] = np.where(post_erosion_mask, 0.6, 0)
    plt.imshow(red_overlay)
    
    legend_elements = [
        Patch(facecolor='white', alpha=0.5, label='Coarse Mask (Before Erosion)'),
        Patch(facecolor='red', alpha=0.6, label='Coarse Mask (After Erosion)')
    ]
    plt.title(title, fontsize=14)
    plt.legend(handles=legend_elements, loc='upper right', fontsize='large')
    plt.axis('off')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

def run_sahi_sam_visualization( image_path: str, processor: Any, results_dir: str = "results", site_name: Optional[str] = None, param_string: str = "", generate_overlay: bool = False, generate_slice_viz: bool = False,
                               generate_threshold_viz: bool = False, generate_erosion_viz: bool = False, tator_csv: Optional[str] = None, verbose: bool = False, slice_viz_max_size: int = 256, coverage_only: bool = False) -> None:
    image_base = os.path.splitext(os.path.basename(image_path))[0]

    if not site_name:
        site_name = "error"

    if not coverage_only:
        viz_dir = os.path.join(results_dir, site_name, "visualizations", param_string)
        mask_dir = os.path.join(results_dir, site_name, "masks", param_string)
        os.makedirs(viz_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
    
    if verbose: print(f"--- Processing {image_base} ---")
    
    if generate_threshold_viz and not coverage_only:
        model_for_viz = getattr(processor, 'fine_model', getattr(processor, 'model', None))
        if model_for_viz:
            _create_threshold_visualization(model_for_viz, image_path, image_base, viz_dir)
    
    results, slice_info = processor.process_image(image_path)
    full_mask = processor.reconstruct_full_mask(results, slice_info, image_path=image_path, coverage_only=False)
    
    if full_mask is None:
        if verbose: print(f"--- Finished {image_base} (no mask generated) ---")
        return

    coverage_percentage = _calculate_coverage(full_mask)
    if verbose: print(f"Coverage for {image_base}: {coverage_percentage:.7f}%")
    
    _save_coverage_to_csv(image_path, coverage_percentage, results_dir, site_name, param_string, tator_csv)

    if coverage_only:
        if verbose: print(f"--- Finished {image_base} (coverage only) ---")
        return

    original_image = cv2.imread(image_path)
    
    if (generate_erosion_viz and isinstance(processor, HierarchicalProcessor) and
            processor.pre_erosion_mask is not None and processor.post_erosion_mask is not None):
        erosion_viz_path = os.path.join(viz_dir, f"{image_base}_erosion_effect.png")
        _save_erosion_visualization(
            original_image=original_image,
            pre_erosion_mask=processor.pre_erosion_mask,
            post_erosion_mask=processor.post_erosion_mask,
            title=f"{image_base} | Erosion Effect (Kernel: {processor.erosion_kernel_size})",
            output_path=erosion_viz_path
        )

    if isinstance(processor, HierarchicalProcessor):
        component_masks = processor.get_component_masks()
        fine_mask, coarse_mask = component_masks.get("Fine Pass"), component_masks.get("Coarse Pass")
        if fine_mask is not None: _save_binary_mask(fine_mask, f"{image_base}_fine_pass", mask_dir)
        if coarse_mask is not None: _save_binary_mask(coarse_mask, f"{image_base}_coarse_pass", mask_dir)
        _save_binary_mask(full_mask, f"{image_base}_combined", mask_dir)
        
        if generate_overlay:
            _save_overlay(original_image, {"Fine Pass Kelp": fine_mask, "Coarse Pass Kelp": coarse_mask},
                          f"{image_base} | Fine vs. Coarse Pass", os.path.join(viz_dir, f"{image_base}_comparison_overlay.png"))
            _save_overlay(original_image, {"Final Kelp Mask": full_mask},
                          f"{image_base} | Final Mask | Coverage: {coverage_percentage:.2f}%", os.path.join(viz_dir, f"{image_base}_final_overlay.png"))
    else:
        _save_binary_mask(full_mask, image_base, mask_dir)
        if generate_overlay:
            _save_overlay(original_image, {"Kelp": full_mask},
                          f"{image_base} | Kelp Coverage: {coverage_percentage:.2f}%", os.path.join(viz_dir, f"{image_base}_overlay.png"))

    if generate_slice_viz:
        if isinstance(processor, HierarchicalProcessor):
            fine_results, fine_slice_info = processor.get_fine_pass_data()
            if fine_results and fine_slice_info:
                _save_slice_visualization(fine_slice_info, fine_results, f"{image_base}_fine", viz_dir, processor.fine_model, max_size=slice_viz_max_size)
            coarse_results, coarse_slice_info = processor.get_coarse_pass_data()
            if coarse_results and coarse_slice_info:
                _save_slice_visualization(coarse_slice_info, coarse_results, f"{image_base}_coarse", viz_dir, processor.coarse_model, max_size=slice_viz_max_size)
        else:
            model_for_viz = getattr(processor, 'model', None)
            if model_for_viz:
                _save_slice_visualization(slice_info, results, image_base, viz_dir, model_for_viz, max_size=slice_viz_max_size)
                
    if verbose: print(f"--- Finished {image_base} ---")

