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
    if mask.size == 0:
        return 0.0
    total_pixels = mask.size
    water_pixels = np.sum(mask)
    kelp_pixels = total_pixels - water_pixels
    coverage_percentage = (kelp_pixels / total_pixels) * 100
    return coverage_percentage


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


def _save_binary_mask(full_mask: np.ndarray, image_base: str, mask_dir: str) -> None:
    os.makedirs(mask_dir, exist_ok=True)
    kelp_mask_save_path = os.path.join(mask_dir, f"{image_base}_kelp_mask.png")
    kelp_binary_mask_img = ((~full_mask).astype(np.uint8)) * 255
    cv2.imwrite(kelp_mask_save_path, kelp_binary_mask_img)


def _save_overlay(
    original_image: np.ndarray,
    masks_to_overlay: Dict[str, np.ndarray],
    title: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(original_image)

    num_masks = len(masks_to_overlay)
    if num_masks == 1:
        colors = [plt.cm.get_cmap("ocean")(0.5)]
    else:
        cmap = plt.cm.get_cmap("viridis", num_masks)
        colors = [cmap(i) for i in range(num_masks)]

    legend_elements = []
    for i, (name, water_mask) in enumerate(masks_to_overlay.items()):
        if water_mask is None:
            continue

        kelp_mask = ~water_mask

        color = colors[i]
        overlay = np.zeros((*kelp_mask.shape, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3] = np.where(kelp_mask, 0.45, 0)
        plt.imshow(overlay)

        legend_elements.append(
            Patch(facecolor=color, edgecolor=color, alpha=0.5, label=f"{name} Kelp")
        )

    plt.title(title, fontsize=14)
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", fontsize="large")
    plt.axis("off")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved overlay to: {output_path}")


def _save_slice_visualization(
    slice_info: Dict[str, Any],
    processed_results: List[Tuple[torch.Tensor, np.ndarray]],
    image_base: str,
    viz_dir: str,
    model: SAHISAM,
    max_size: int = 256,
) -> None:
    os.makedirs(viz_dir, exist_ok=True)
    img_list = slice_info["img_list"]
    if not img_list:
        return

    num_slices = len(img_list)
    cols = len(sorted(set(pt[0] for pt in slice_info["img_starting_pts"])))
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
            mask_overlay_rgba = np.zeros((h_new, w_new, 4), dtype=np.uint8)
            mask_overlay_rgba[..., 2] = 255

            padding_scaled = int(model.padding * scale)
            slice_mask_full = np.zeros(display_img.shape[:2], dtype=bool)

            content_w, content_h = (
                w_new - 2 * padding_scaled,
                h_new - 2 * padding_scaled,
            )
            if padding_scaled > 0 and content_w > 0 and content_h > 0:
                content_mask = cv2.resize(
                    mask_cpu.astype(np.uint8),
                    (content_w, content_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
                slice_mask_full[
                    padding_scaled:-padding_scaled, padding_scaled:-padding_scaled
                ] = content_mask
            else:
                slice_mask_full = cv2.resize(
                    mask_cpu.astype(np.uint8),
                    (w_new, h_new),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            mask_overlay_rgba[..., 3] = np.where(slice_mask_full, int(255 * 0.2), 0)
            ax.imshow(mask_overlay_rgba)

        if len(points) > 0:
            points_scaled = (np.array(points) * scale).astype(int)
            ax.plot(
                points_scaled[:, 0], points_scaled[:, 1], "o", color="red", markersize=3
            )

        ax.axis("off")
        ax.set_title(f"Slice {i}", fontsize=10)

    for j in range(num_slices, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    slice_save_path = os.path.join(viz_dir, f"{image_base}_slices_with_points.png")
    plt.savefig(slice_save_path, dpi=150, bbox_inches="tight")
    plt.close()


def _create_threshold_visualization(
    model: SAHISAM,
    image_path: str,
    image_base: str,
    viz_dir: str,
    verbose: bool = False,
) -> None:
    if verbose:
        print("Generating threshold visualization...")
    original_image = model._load(image_path)
    slice_info = model._slice(original_image)
    img_list = slice_info["img_list"]
    if not img_list:
        return

    num_slices = len(img_list)
    cols = len(sorted(set(pt[0] for pt in slice_info["img_starting_pts"])))
    rows = num_slices // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes = axes.flatten()

    vmax = model.threshold * 2
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap("viridis_r")

    for i, img in enumerate(img_list):
        ax = axes[i]
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if model.water_lab_tensor is not None:
            dist_map = (
                torch.linalg.norm(
                    model._get_lab_tensor(img) - model.water_lab_tensor, dim=2
                )
                .cpu()
                .numpy()
            )
            masked_array = np.ma.masked_where(dist_map > vmax, dist_map)
            ax.imshow(masked_array, alpha=0.5, cmap=cmap, norm=norm)
        ax.set_title(f"Slice {i}", fontsize=10)
        ax.axis("off")

    for j in range(num_slices, len(axes)):
        axes[j].axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(
        sm, ax=axes.tolist(), label="Distance to Water LAB", shrink=0.8, aspect=20
    )
    plt.tight_layout()
    threshold_save_path = os.path.join(viz_dir, f"{image_base}_threshold_grid.png")
    plt.savefig(threshold_save_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved threshold grid visualization to: {threshold_save_path}")


def _save_erosion_visualization(
    original_image: np.ndarray,
    pre_erosion_mask: np.ndarray,
    post_erosion_mask: np.ndarray,
    title: str,
    output_path: str,
    verbose: bool = False,
) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(12, 12))
    plt.imshow(original_image)

    white_overlay = np.zeros((*pre_erosion_mask.shape, 4))
    white_overlay[..., :3] = [1, 1, 1]
    white_overlay[..., 3] = np.where(pre_erosion_mask, 0.5, 0)
    plt.imshow(white_overlay)

    red_overlay = np.zeros((*post_erosion_mask.shape, 4))
    red_overlay[..., 0] = 1
    red_overlay[..., 3] = np.where(post_erosion_mask, 0.6, 0)
    plt.imshow(red_overlay)

    legend_elements = [
        Patch(facecolor="white", alpha=0.5, label="Coarse Kelp (Before Erosion)"),
        Patch(facecolor="red", alpha=0.6, label="Coarse Kelp (After Erosion)"),
    ]
    plt.title(title, fontsize=14)
    plt.legend(handles=legend_elements, loc="upper right", fontsize="large")
    plt.axis("off")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved erosion visualization to: {output_path}")


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
) -> None:
    viz_dir = os.path.join(run_dir, "visualizations")
    mask_dir = os.path.join(run_dir, "masks")
    os.makedirs(viz_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    output_json_path = os.path.join(run_dir, "results.json")

    all_results_dict = {}
    existing_data = {}

    if os.path.exists(output_json_path):
        if verbose:
            print(f"Found existing results file. Loading data from {run_dir}")
        with open(output_json_path, "r") as f:
            try:
                existing_data = json.load(f)
                results_list = existing_data.get("results", [])
                all_results_dict = {res["image_name"]: res for res in results_list}
            except (json.JSONDecodeError, KeyError):
                print(
                    "Warning: Could not read results.json or format is incorrect. Starting fresh."
                )
                existing_data = {}

    image_iterator = (
        tqdm(image_paths, desc=f"Processing Images for {site_name}")
        if not verbose
        else image_paths
    )

    images_processed_since_save = 0

    try:
        for image_path in image_iterator:
            image_name = os.path.basename(image_path)

            if not overwrite and image_name in all_results_dict:
                if verbose:
                    print(f"Skipping already processed image: {image_name}")
                continue

            try:
                image_base = os.path.splitext(image_name)[0]
                if verbose:
                    print(f"--- Processing {image_base} ---")

                if hasattr(processor, "model"):
                    model = processor.model
                elif hasattr(processor, "fine_model"):
                    model = processor.coarse_model
                original_image = model._load(image_path)
                image_lab_tensor_cpu = model._get_lab_tensor(original_image).cpu()

                results, slice_info = processor.process_image(
                    image_path, full_lab_tensor_cpu=image_lab_tensor_cpu
                )

                if coverage_only:
                    coverage_percentage = processor.reconstruct_full_mask(
                        results,
                        slice_info,
                        image_lab_tensor_cpu=image_lab_tensor_cpu,
                        image_path=image_path,
                        run_dir=run_dir,
                        coverage_only=True,
                    )
                else:
                    full_mask = processor.reconstruct_full_mask(
                        results,
                        slice_info,
                        image_lab_tensor_cpu=image_lab_tensor_cpu,
                        image_path=image_path,
                        run_dir=run_dir,
                        coverage_only=False,
                    )

                    if full_mask is None:
                        if verbose:
                            print(f"--- Finished {image_base} (no mask generated) ---")
                        continue

                    coverage_percentage = _calculate_coverage(full_mask)
                    _save_binary_mask(full_mask, image_base, mask_dir)

                    if generate_overlay:
                        _save_overlay(
                            original_image,
                            {"Final": full_mask},
                            f"{image_base} | Kelp Coverage: {coverage_percentage:.2f}%",
                            os.path.join(viz_dir, f"{image_base}_overlay.png"),
                            verbose=verbose,
                        )

                    if generate_component_viz and isinstance(
                        processor, HierarchicalProcessor
                    ):
                        masks = processor.get_component_masks()
                        _save_overlay(
                            original_image,
                            masks,
                            f"{image_base} | Component Comparison",
                            os.path.join(
                                viz_dir, f"{image_base}_component_overlay.png"
                            ),
                            verbose=verbose,
                        )

                    if generate_erosion_viz and isinstance(
                        processor, HierarchicalProcessor
                    ):
                        _save_erosion_visualization(
                            original_image=original_image,
                            pre_erosion_mask=processor.pre_erosion_mask,
                            post_erosion_mask=processor.post_erosion_mask,
                            title=f"{image_base} | Erosion Effect (Kernel: {processor.erosion_kernel_size})",
                            output_path=os.path.join(
                                viz_dir, f"{image_base}_erosion_effect.png"
                            ),
                            verbose=verbose,
                        )

                    if generate_slice_viz:
                        if isinstance(processor, HierarchicalProcessor):
                            fine_results, fine_slice_info = (
                                processor.get_fine_pass_data()
                            )
                            if fine_results and fine_slice_info:
                                _save_slice_visualization(
                                    fine_slice_info,
                                    fine_results,
                                    f"{image_base}_fine",
                                    viz_dir,
                                    processor.fine_model,
                                    max_size=slice_viz_max_size,
                                )
                            coarse_results, coarse_slice_info = (
                                processor.get_coarse_pass_data()
                            )
                            if coarse_results and coarse_slice_info:
                                _save_slice_visualization(
                                    coarse_slice_info,
                                    coarse_results,
                                    f"{image_base}_coarse",
                                    viz_dir,
                                    processor.coarse_model,
                                    max_size=slice_viz_max_size,
                                )
                        else:
                            _save_slice_visualization(
                                slice_info,
                                results,
                                image_base,
                                viz_dir,
                                model,
                                max_size=slice_viz_max_size,
                            )

                _, image_id, latitude, longitude = _get_image_metadata(
                    image_path, tator_csv
                )

                result_data = {
                    "image_name": image_name,
                    "image_id": int(image_id) if image_id is not None else None,
                    "latitude": float(latitude) if latitude is not None else None,
                    "longitude": float(longitude) if longitude is not None else None,
                    "coverage_percentage": coverage_percentage,
                }
                all_results_dict[image_name] = result_data
                images_processed_since_save += 1
                if images_processed_since_save >= 25:
                    if verbose:
                        print(
                            f"\n--- Saving progress ({images_processed_since_save} images processed)... ---"
                        )
                    final_output = {
                        "command": existing_data.get("command", command_str),
                        "run_args": existing_data.get("run_args", run_args_dict),
                        "results": list(all_results_dict.values()),
                    }
                    with open(output_json_path, "w") as f:
                        json.dump(final_output, f, indent=4)
                    images_processed_since_save = 0

            except Exception as e:
                print(f"\n--- ERROR processing {os.path.basename(image_path)}: {e} ---")
                with open(os.path.join(run_dir, "error_log.txt"), "a") as f:
                    f.write(f"Error on {image_path}: {e}\n")
                continue
    finally:
        if images_processed_since_save > 0:
            if verbose:
                print(
                    f"\n--- Final save before exiting ({images_processed_since_save} new images)... ---"
                )
            final_output = {
                "command": existing_data.get("command", command_str),
                "run_args": existing_data.get("run_args", run_args_dict),
                "results": list(all_results_dict.values()),
            }
            with open(output_json_path, "w") as f:
                json.dump(final_output, f, indent=4)
            if verbose:
                print("Final results saved.")

    if verbose:
        print(f"--- Analysis complete. All results saved in: {run_dir} ---")
