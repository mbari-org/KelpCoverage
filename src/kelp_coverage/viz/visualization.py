import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from matplotlib.patches import Patch, Rectangle
from typing import Any, Dict, List, Optional, Tuple

from kelp_coverage import config
# FIX(baseline): use standalone core functions instead of duck-typing into model internals
from kelp_coverage.core.color import load_image, rgb_to_lab_gpu
from kelp_coverage.core.slicing import slice_image as core_slice_image

def save_binary_mask(full_mask: np.ndarray, image_base: str, mask_dir: str) -> None:
    os.makedirs(mask_dir, exist_ok=True)
    kelp_binary = (~full_mask).astype(np.uint8) * 255
    cv2.imwrite(os.path.join(mask_dir, f"{image_base}_kelp_mask.png"), kelp_binary)

def save_overlay(
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
        cmap   = plt.cm.get_cmap("viridis", num_masks)
        colors = [cmap(i) for i in range(num_masks)]

    legend_elements = []
    for i, (name, water_mask) in enumerate(masks_to_overlay.items()):
        if water_mask is None:
            continue
        kelp_mask = ~water_mask
        color     = colors[i]
        overlay   = np.zeros((*kelp_mask.shape, 4))
        overlay[..., :3] = color[:3]
        overlay[..., 3]  = np.where(kelp_mask, 0.45, 0)
        plt.imshow(overlay)
        legend_elements.append(Patch(facecolor=color, edgecolor=color, alpha=0.5, label=f"{name} Kelp"))

    plt.title(title, fontsize=14)
    if legend_elements:
        plt.legend(handles=legend_elements, loc="upper right", fontsize="large")
    plt.axis("off")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved overlay: {output_path}")

def save_slice_visualization(
    slice_info: Dict,
    results: List[Tuple[torch.Tensor, Any]],
    image_base: str,
    viz_dir: str,
    padding: int = config.PADDING,
    max_size: int = 256,
) -> None:
    os.makedirs(viz_dir, exist_ok=True)
    img_list = slice_info["img_list"]
    if not img_list:
        return

    num_slices = len(img_list)
    cols = len(set(pt[0] for pt in slice_info["img_starting_pts"]))
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i in range(num_slices):
        img = img_list[i]
        mask_tensor, points = results[i]
        h_orig, w_orig = img.shape[:2]
        scale = max_size / max(h_orig, w_orig)
        w_new, h_new  = int(w_orig * scale), int(h_orig * scale)
        display_img   = cv2.resize(img, (w_new, h_new), interpolation=cv2.INTER_AREA)

        ax = axes_flat[i]
        ax.imshow(display_img)

        if mask_tensor.numel() > 0:
            mask_np  = mask_tensor.cpu().numpy()
            pad_s    = int(padding * scale)
            overlay  = np.zeros((h_new, w_new, 4), dtype=np.uint8)
            overlay[..., 2] = 255  # blue channel

            c_w = w_new - 2 * pad_s
            c_h = h_new - 2 * pad_s
            if pad_s > 0 and c_w > 0 and c_h > 0:
                content = cv2.resize(mask_np.astype(np.uint8), (c_w, c_h), interpolation=cv2.INTER_NEAREST).astype(bool)
                full_mask = np.zeros((h_new, w_new), dtype=bool)
                full_mask[pad_s:-pad_s, pad_s:-pad_s] = content
            else:
                full_mask = cv2.resize(mask_np.astype(np.uint8), (w_new, h_new), interpolation=cv2.INTER_NEAREST).astype(bool)

            overlay[..., 3] = np.where(full_mask, int(255 * 0.2), 0)
            ax.imshow(overlay)

        if len(points) > 0:
            pts_scaled = (np.array(points) * scale).astype(int)
            ax.plot(pts_scaled[:, 0], pts_scaled[:, 1], "o", color="red", markersize=3)

        ax.axis("off")
        ax.set_title(f"Slice {i}", fontsize=10)

    for j in range(num_slices, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{image_base}_slices_with_points.png"), dpi=150, bbox_inches="tight")
    plt.close()

def create_threshold_visualization(
    image_path: str,
    image_base: str,
    water_lab: torch.Tensor,
    viz_dir: str,
    device: str,
    threshold: int           = config.THRESHOLD,
    slice_size: int          = config.SLICE_SIZE,
    slice_overlap: float     = config.SLICE_OVERLAP,
    downsample_factor: float = config.DOWNSAMPLE_FACTOR,
    clahe: bool              = config.CLAHE_ENABLED,
    verbose: bool            = False,
) -> None:
    os.makedirs(viz_dir, exist_ok=True)
    image      = load_image(image_path, downsample_factor, clahe)
    slice_info = core_slice_image(image, slice_size, slice_overlap)
    img_list   = slice_info["img_list"]
    if not img_list:
        return

    num_slices = len(img_list)
    cols = len(set(pt[0] for pt in slice_info["img_starting_pts"]))
    rows = (num_slices + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    vmax = threshold * 2
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = cm.get_cmap("viridis_r")

    for i, img in enumerate(img_list):
        ax = axes_flat[i]
        ax.imshow(img)
        lab = rgb_to_lab_gpu(img, device)
        dist_map = torch.linalg.norm(lab - water_lab, dim=2).cpu().numpy()
        masked   = np.ma.masked_where(dist_map > vmax, dist_map)
        ax.imshow(masked, alpha=0.5, cmap=cmap, norm=norm)
        ax.set_title(f"Slice {i}", fontsize=10)
        ax.axis("off")

    for j in range(num_slices, len(axes_flat)):
        axes_flat[j].axis("off")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_flat.tolist(), label="Distance to Water LAB", shrink=0.8, aspect=20)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f"{image_base}_threshold_grid.png"), dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved threshold grid: {viz_dir}/{image_base}_threshold_grid.png")

def save_erosion_visualization(
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

    white = np.zeros((*pre_erosion_mask.shape, 4))
    white[..., :3] = 1
    white[..., 3]  = np.where(pre_erosion_mask, 0.5, 0)
    plt.imshow(white)

    red = np.zeros((*post_erosion_mask.shape, 4))
    red[..., 0] = 1
    red[..., 3] = np.where(post_erosion_mask, 0.6, 0)
    plt.imshow(red)

    plt.legend(handles=[
        Patch(facecolor="white", alpha=0.5, label="Before Erosion"),
        Patch(facecolor="red",   alpha=0.6, label="After Erosion"),
    ], loc="upper right", fontsize="large")
    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    if verbose:
        print(f"Saved erosion viz: {output_path}")

def _draw_grid_overlay(
    ax,
    uniform_grids: np.ndarray,
    water_color_grids: np.ndarray,
    img_shape: Tuple[int, int],
    grid_size: int,
    alpha: float = 0.3,
) -> None:
    h, w = img_shape
    gh   = (h + grid_size - 1) // grid_size
    gw   = (w + grid_size - 1) // grid_size

    for gy in range(gh):
        for gx in range(gw):
            is_uniform = uniform_grids[gy, gx] if uniform_grids is not None else False
            is_water   = water_color_grids[gy, gx] if water_color_grids is not None else False

            if is_uniform and is_water:
                color = "green"
            elif is_uniform:
                color = "purple"
            else:
                color = "red"

            rect = Rectangle(
                (gx * grid_size, gy * grid_size),
                min(grid_size, w - gx * grid_size),
                min(grid_size, h - gy * grid_size),
                linewidth=0.5, edgecolor="white",
                facecolor=color, alpha=alpha,
            )
            ax.add_patch(rect)


def build_debug_figures(
    slice_img: np.ndarray,
    diagnostics: Dict,
    threshold: int,
    slice_index: int,
    is_shortcut: bool,
    show_stages: bool = True,
    show_heatmap: bool = True,
    water_lab: Optional[torch.Tensor] = None,
    device: str = "cpu",
    grid_size: int = config.GRID_SIZE,
) -> Dict[str, plt.Figure]:
    h, w = slice_img.shape[:2]
    uniform_grids       = diagnostics.get("uniform_grids")
    water_color_grids   = diagnostics.get("water_color_grids")
    valid_water_grids   = diagnostics.get("valid_water_grids")
    initial_candidates  = diagnostics.get("initial_candidates", np.array([]))
    grid_filtered       = diagnostics.get("grid_filtered_candidates", np.array([]))
    final_points_xy     = diagnostics.get("final_points_xy")

    points_per_grid = 10
    if (len(initial_candidates) > 0
            and valid_water_grids is not None):
        grid_iy = initial_candidates[:, 0] // grid_size
        grid_ix = initial_candidates[:, 1] // grid_size
        grid_h, grid_w_g = valid_water_grids.shape
        sampled = []
        for r in range(grid_h):
            for c in range(grid_w_g):
                if valid_water_grids[r, c]:
                    mask = (grid_iy == r) & (grid_ix == c)
                    pts  = initial_candidates[mask]
                    if len(pts) > 0:
                        n = min(len(pts), points_per_grid)
                        idx = np.random.choice(len(pts), n, replace=False)
                        sampled.append(pts[idx])
        plot_candidates = np.vstack(sampled) if sampled else np.array([])
    else:
        plot_candidates = initial_candidates

    _LEG_KW = dict(loc="upper center", bbox_to_anchor=(0.5, -0.025),
                   fancybox=True, shadow=True, ncol=3)

    figures = {}

    if show_stages:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(slice_img)
        if uniform_grids is not None:
            _draw_grid_overlay(ax, uniform_grids, water_color_grids, (h, w), grid_size, alpha=0.3)
            ax.legend(handles=[
                Patch(facecolor="green",  alpha=0.3, label="Valid Grid (Uniform & Water Color)"),
                Patch(facecolor="purple", alpha=0.3, label="Uniform Only"),
                Patch(facecolor="red",    alpha=0.3, label="Non-Uniform"),
            ], **_LEG_KW)
        ax.set_title(f"Grid Validation | Slice {slice_index}" + (" (Shortcut)" if is_shortcut else ""))
        ax.axis("off")
        ax.margins(0.01)
        figures["stage1_grid_validation"] = fig

    if show_stages:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(slice_img)
        if uniform_grids is not None:
            _draw_grid_overlay(ax, uniform_grids, water_color_grids, (h, w), grid_size, alpha=0.3)
        legend_elems = []
        if uniform_grids is not None:
            legend_elems.append(Patch(facecolor="green", alpha=0.3, label="Valid Grid"))
        if len(plot_candidates) > 0:
            ax.scatter(plot_candidates[:, 1], plot_candidates[:, 0],
                       c="gray", s=15, alpha=0.6)
            legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                label=f"Unselected Points ({len(plot_candidates)} | {len(initial_candidates)})",
                markerfacecolor="gray", markersize=10))
        if len(grid_filtered) > 0:
            ax.scatter(grid_filtered[:, 1], grid_filtered[:, 0],
                       c="orange", s=30, edgecolor="black", lw=0.5)
            legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                label=f"Grid Filtered ({len(grid_filtered)})",
                markerfacecolor="orange", markersize=10))
        ax.set_title(f"Point Selection over Grid | Slice {slice_index}")
        ax.axis("off")
        ax.margins(0.01)
        if legend_elems:
            ax.legend(handles=legend_elems, **_LEG_KW)
        figures["stage2_point_selection"] = fig

    if show_stages:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(slice_img)
        legend_elems = []
        if len(grid_filtered) > 0:
            ax.scatter(grid_filtered[:, 1], grid_filtered[:, 0],
                       c="orange", s=30, alpha=0.4)
            legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
                label=f"Grid Filtered Pool ({len(grid_filtered)})",
                markerfacecolor="orange", alpha=0.4, markersize=10))
        if final_points_xy is not None and len(final_points_xy) > 0:
            ax.scatter(final_points_xy[:, 0], final_points_xy[:, 1],
                       c="red", s=60, marker="X", edgecolor="white", lw=1)
            legend_elems.append(plt.Line2D([0], [0], marker="X", color="w",
                label=f"Final Prompts ({len(final_points_xy)})",
                markerfacecolor="red", markersize=12))
        ax.set_title(f"Final Point Selection | Slice {slice_index}")
        ax.axis("off")
        ax.margins(0.01)
        if legend_elems:
            ax.legend(handles=legend_elems, **_LEG_KW)
        figures["stage3_final_selection"] = fig

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(slice_img)
    legend_elems = []

    if show_heatmap and water_lab is not None:
        from kelp_coverage.core.color import rgb_to_lab_gpu
        lab      = rgb_to_lab_gpu(slice_img, device)
        dist_map = torch.linalg.norm(lab - water_lab, dim=2).cpu().numpy()
        vmax     = threshold * 2
        cmap     = cm.get_cmap("viridis_r")
        ax.imshow(np.ma.masked_where(dist_map > vmax, dist_map),
                  alpha=0.5, cmap=cmap,
                  norm=mcolors.Normalize(vmin=0, vmax=vmax))
        legend_elems.append(Patch(facecolor=cmap(0.5), alpha=0.5,
                                  label=f"Color Distance (<= {vmax})"))

    if uniform_grids is not None:
        _draw_grid_overlay(ax, uniform_grids, water_color_grids, (h, w), grid_size, alpha=0.25)
        legend_elems.extend([
            Patch(facecolor="green",  alpha=0.3, label="Valid Grid"),
            Patch(facecolor="purple", alpha=0.3, label="Uniform Grid"),
            Patch(facecolor="red",    alpha=0.3, label="Non-Uniform Grid"),
        ])

    if len(plot_candidates) > 0:
        ax.scatter(plot_candidates[:, 1], plot_candidates[:, 0],
                   c="yellow", s=15, alpha=0.6)
        legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
            label=f"Sampled Initial ({len(plot_candidates)} of {len(initial_candidates)})",
            markerfacecolor="yellow", markersize=10))

    if len(grid_filtered) > 0:
        ax.scatter(grid_filtered[:, 1], grid_filtered[:, 0],
                   c="orange", s=30, edgecolor="black", lw=0.5)
        legend_elems.append(plt.Line2D([0], [0], marker="o", color="w",
            label=f"Grid Filtered ({len(grid_filtered)})",
            markerfacecolor="orange", markersize=10))

    if final_points_xy is not None and len(final_points_xy) > 0:
        ax.scatter(final_points_xy[:, 0], final_points_xy[:, 1],
                   c="red", s=60, marker="X", edgecolor="white", lw=1)
        legend_elems.append(plt.Line2D([0], [0], marker="X", color="w",
            label=f"Final Prompts ({len(final_points_xy)})",
            markerfacecolor="red", markersize=12))

    ax.set_title(f"Final Composite Overlay | Slice {slice_index}" + (" (Shortcut)" if is_shortcut else ""))
    ax.axis("off")
    ax.margins(0.01)
    if legend_elems:
        ax.legend(handles=legend_elems, **_LEG_KW)
    figures["stage4_final_overlay"] = fig

    return figures


def save_debug_visualization(
    figures: Dict[str, plt.Figure],
    output_dir: str,
    image_base: str,
    slice_index: int,
    threshold: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, f"{image_base}_slice_{slice_index}_thresh{threshold}")
    for stage_name, fig in figures.items():
        fig.savefig(f"{base}_{stage_name}.png", bbox_inches="tight", dpi=200)
        plt.close(fig)
    print(f"Saved debug viz to: {output_dir}")
