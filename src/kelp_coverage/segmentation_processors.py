import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pprint
import os
import matplotlib.pyplot as plt
import time

from typing import Tuple, Dict, Any, Optional
from .sahisam import SAHISAM


class SinglePassProcessor:
    def __init__(self, model_args: Any, water_lab: Tuple[int, int, int]):
        sahisam_args = {
            "sam_checkpoint": model_args.sam_checkpoint,
            "sam_model_type": model_args.sam_model_type,
            "water_lab": water_lab,
            "use_mobile_sam": model_args.use_mobile_sam,
            "slice_size": model_args.slice_size,
            "slice_overlap": model_args.slice_overlap,
            "padding": model_args.padding,
            "clahe": model_args.clahe,
            "downsample_factor": model_args.downsample_factor,
            "num_points": model_args.num_points,
            "threshold": model_args.threshold,
            "threshold_max": model_args.threshold_max,
            "verbose": model_args.verbose,
            "final_point_strategy": model_args.final_point_strategy,
            "grid_size": model_args.grid_size,
            "uniformity_check": model_args.uniformity_check,
            "uniformity_std_threshold": model_args.uniformity_std_threshold,
            "uniform_grid_thresh": model_args.uniform_grid_thresh,
            "water_grid_thresh": model_args.water_grid_thresh,
            "fallback_brightness_threshold": model_args.fallback_brightness_threshold,
            "fallback_distance_threshold": model_args.fallback_distance_threshold,
            "gpu_batch_size": model_args.gpu_batch_size,
        }
        if model_args.verbose:
            print("-" * 10)
            print("Initializing Single-Pass Processor with arguments:")
            pprint.pprint(sahisam_args)
            print("-" * 10)
        self.model = SAHISAM(**sahisam_args)

    def process_image(
        self, image_path: str, full_lab_tensor_cpu: Optional[torch.Tensor] = None
    ) -> Tuple[Any, Any]:
        if self.model.verbose:
            print("Running single-pass (detailed point search on all slices)...")
        return self.model.process_image(
            image_path=image_path, full_lab_tensor_cpu=full_lab_tensor_cpu
        )

    def reconstruct_full_mask(
        self,
        results: Any,
        slice_info: Dict[str, Any],
        image_lab_tensor_cpu: torch.Tensor,
        image_path: str,
        run_dir: str,
        coverage_only: bool = False,
    ) -> Any:
        return self.model.reconstruct_full_mask_gpu(
            results, slice_info, coverage_only=coverage_only
        )


class HierarchicalProcessor:
    def __init__(self, model_args: Any, water_lab: Tuple[int, int, int]):
        if model_args.verbose:
            print("--- Initializing Hierarchical Processor ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        common_sahisam_args = {
            "sam_checkpoint": model_args.sam_checkpoint,
            "sam_model_type": model_args.sam_model_type,
            "water_lab": water_lab,
            "use_mobile_sam": model_args.use_mobile_sam,
            "slice_overlap": model_args.slice_overlap,
            "padding": model_args.padding,
            "clahe": model_args.clahe,
            "downsample_factor": model_args.downsample_factor,
            "num_points": model_args.num_points,
            "threshold": model_args.threshold,
            "threshold_max": model_args.threshold_max,
            "verbose": model_args.verbose,
            "final_point_strategy": model_args.final_point_strategy,
            "grid_size": model_args.grid_size,
            "uniformity_check": model_args.uniformity_check,
            "uniformity_std_threshold": model_args.uniformity_std_threshold,
            "uniform_grid_thresh": model_args.uniform_grid_thresh,
            "water_grid_thresh": model_args.water_grid_thresh,
            "fallback_brightness_threshold": model_args.fallback_brightness_threshold,
            "fallback_distance_threshold": model_args.fallback_distance_threshold,
            "device": self.device,
            "gpu_batch_size": model_args.gpu_batch_size,
        }
        fine_args = common_sahisam_args.copy()
        fine_args["slice_size"] = model_args.slice_size

        if model_args.verbose:
            print("\n--- Initializing FINE-PASS SAHISAM with arguments: ---")
            pprint.pprint(fine_args)
            print("---------------------------------------------------------")
        self.fine_model = SAHISAM(**fine_args)

        coarse_args = common_sahisam_args.copy()
        coarse_args["slice_size"] = model_args.hierarchical_slice_size

        if model_args.verbose:
            print("\n--- Initializing COARSE-PASS SAHISAM with arguments: ---")
            pprint.pprint(coarse_args)
            print("----------------------------------------------------------")
        self.coarse_model = SAHISAM(**coarse_args)

        self.use_erosion_merge = getattr(model_args, "use_erosion_merge", False)
        self.erosion_kernel_size = getattr(model_args, "erosion_kernel_size", 15)
        self.use_color_validation = getattr(model_args, "use_color_validation", True)
        self.generate_merge_viz = getattr(model_args, "generate_merge_viz", False)
        self.merge_color_threshold = getattr(model_args, "merge_color_threshold", 15)
        self.merge_lightness_threshold = getattr(
            model_args, "merge_lightness_threshold", 75.0
        )

        self.water_lab_tensor = self.fine_model.water_lab_tensor

        self.internal_fine_results: Optional[Any] = None
        self.fine_slice_info: Optional[Dict[str, Any]] = None
        self.internal_coarse_results: Optional[Any] = None
        self.coarse_slice_info: Optional[Dict[str, Any]] = None
        self.fine_pass_water_mask_gpu: Optional[torch.Tensor] = None
        self.coarse_pass_water_mask_gpu: Optional[torch.Tensor] = None
        self.pre_erosion_mask: Optional[np.ndarray] = None
        self.post_erosion_mask: Optional[np.ndarray] = None

    def _erode_gpu(
        self, kelp_mask_tensor: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        padding = kernel_size // 2
        inverted_mask = ~kelp_mask_tensor
        dilated_inverted_mask = (
            F.max_pool2d(
                inverted_mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
            .squeeze()
            .bool()
        )
        return ~dilated_inverted_mask

    def _merge_masks_gpu(
        self,
        fine_pass_water_mask: torch.Tensor,
        coarse_pass_water_mask: torch.Tensor,
        image_lab_tensor_cpu: torch.Tensor,
        image_path: str,
        run_dir: str,
    ) -> torch.Tensor:
        fine_pass_kelp_mask = ~fine_pass_water_mask
        coarse_pass_kelp_mask = ~coarse_pass_water_mask

        if self.use_erosion_merge:
            kernel_size = self.erosion_kernel_size
            if kernel_size % 2 == 0:
                kernel_size += 1
            self.pre_erosion_mask = coarse_pass_kelp_mask.cpu().numpy()
            coarse_pass_kelp_mask = self._erode_gpu(coarse_pass_kelp_mask, kernel_size)
            self.post_erosion_mask = coarse_pass_kelp_mask.cpu().numpy()

        agreed_kelp = fine_pass_kelp_mask & coarse_pass_kelp_mask
        disagreement_zone = fine_pass_kelp_mask ^ coarse_pass_kelp_mask

        validated_kelp_in_disagreement = torch.zeros_like(disagreement_zone)

        if self.use_color_validation and torch.any(disagreement_zone):
            if self.fine_model.verbose:
                print("\n--- [Debug] Symmetrical Mask Merge Analysis ---")

            torch.cuda.synchronize()
            t_color_val_start = time.time()

            disagreement_zone_cpu = disagreement_zone.cpu()
            disagreement_pixels_lab_cpu = image_lab_tensor_cpu[disagreement_zone_cpu]
            disagreement_pixels_lab = disagreement_pixels_lab_cpu.to(self.device)

            lightness_values = disagreement_pixels_lab[:, 0]
            is_not_too_bright = lightness_values < self.merge_lightness_threshold
            color_values = disagreement_pixels_lab[:, 1:]
            color_distances = torch.linalg.norm(
                color_values - self.water_lab_tensor[1:], dim=1
            )
            is_different_color = color_distances > self.merge_color_threshold
            is_validated_as_kelp_flat = is_not_too_bright & is_different_color

            validated_kelp_in_disagreement[disagreement_zone] = (
                is_validated_as_kelp_flat
            )
            torch.cuda.synchronize()
            if self.fine_model.verbose:
                print(
                    f"    - Color validation step took: {time.time() - t_color_val_start:.2f}s"
                )
                print(
                    f"    - Disagreement pixels validated as KELP: {torch.sum(validated_kelp_in_disagreement).item()}"
                )
            if self.generate_merge_viz:
                self._save_merge_visualization(
                    image_path,
                    run_dir,
                    disagreement_zone_cpu,
                    color_distances,
                    self.merge_color_threshold,
                )
        final_kelp_mask = agreed_kelp | validated_kelp_in_disagreement
        if self.fine_model.verbose:
            print(
                f"  - Total KELP pixels in final merged mask: {torch.sum(final_kelp_mask).item()}"
            )
            print("-------------------------------------\n")
        return ~final_kelp_mask

    def _save_merge_visualization(
        self, image_path, run_dir, disagreement_zone, distances_flat, threshold
    ) -> None:
        image_base = os.path.splitext(os.path.basename(image_path))[0]
        viz_dir = os.path.join(run_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        output_path = os.path.join(
            viz_dir, f"{image_base}_merge_disagreement_heatmap.png"
        )

        original_image = cv2.imread(image_path)

        if original_image.shape[:2] != disagreement_zone.shape:
            original_image = cv2.resize(
                original_image,
                (disagreement_zone.shape[1], disagreement_zone.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

        disagreement_zone_np = disagreement_zone.cpu().numpy()
        distances_np = distances_flat.cpu().numpy()

        heatmap = np.full(disagreement_zone_np.shape, np.nan, dtype=np.float32)
        heatmap[disagreement_zone_np] = distances_np

        plt.figure(figsize=(15, 15))
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        vmax = 25
        cmap = plt.get_cmap("viridis_r")

        plt.imshow(heatmap, cmap=cmap, alpha=0.6, vmin=0, vmax=vmax)

        cbar = plt.colorbar(shrink=0.7)
        cbar.set_label("LAB Distance to Water Color", size="large")
        cbar.ax.axhline(threshold, color="red", linestyle="--", linewidth=2)
        cbar.ax.text(
            1.5,
            threshold,
            " Kelp Threshold",
            color="red",
            va="center",
            ha="left",
            fontsize="medium",
        )

        plt.title(
            f"Disagreement Zone Heatmap for {image_base}\n(Pixels above red line are validated as Kelp)",
            fontsize=16,
        )
        plt.axis("off")
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        if self.fine_model.verbose:
            print(f"  - Saved merge disagreement heatmap to: {output_path}")

    def process_image(
        self, image_path: str, full_lab_tensor_cpu: Optional[torch.Tensor] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        if self.fine_model.verbose:
            print("\n--- [Hierarchical] Running passes sequentially ---")
        if self.fine_model.verbose:
            print("   > Starting FINE pass (small slices)...")
        self.internal_fine_results, self.fine_slice_info = (
            self.fine_model.process_image(
                image_path=image_path, full_lab_tensor_cpu=full_lab_tensor_cpu
            )
        )
        if self.coarse_model.verbose:
            print("   > Starting COARSE pass (large slices)...")
        self.internal_coarse_results, self.coarse_slice_info = (
            self.coarse_model.process_image(
                image_path=image_path, full_lab_tensor_cpu=full_lab_tensor_cpu
            )
        )
        if self.fine_model.verbose:
            print("--- [Hierarchical] Both passes complete ---")
        return self.internal_fine_results, self.fine_slice_info

    def reconstruct_full_mask(
        self,
        results: Any,
        slice_info: Dict[str, Any],
        image_lab_tensor_cpu: torch.Tensor,
        image_path: str,
        run_dir: str,
        coverage_only: bool = False,
    ) -> Any:
        if self.fine_model.verbose:
            print("\n--- [Hierarchical] Reconstructing and combining masks on GPU ---")
        t_start = time.time()

        torch.cuda.synchronize()
        t_fine_start = time.time()
        self.fine_pass_water_mask_gpu = self.fine_model.reconstruct_full_mask_gpu(
            masks=results,
            slice_info=slice_info,
            return_gpu_tensor=True,
            merge_logic="OR",
        )
        torch.cuda.synchronize()
        if self.fine_model.verbose:
            print(
                f"  > Fine pass reconstruction took: {time.time() - t_fine_start:.2f}s"
            )

        torch.cuda.synchronize()
        t_coarse_start = time.time()
        self.coarse_pass_water_mask_gpu = self.coarse_model.reconstruct_full_mask_gpu(
            masks=self.internal_coarse_results,
            slice_info=self.coarse_slice_info,
            return_gpu_tensor=True,
            merge_logic="AND",
        )
        torch.cuda.synchronize()
        if self.fine_model.verbose:
            print(
                f"  > Coarse pass reconstruction took: {time.time() - t_coarse_start:.2f}s"
            )

        torch.cuda.synchronize()
        t_merge_start = time.time()
        combined_water_mask_gpu = self._merge_masks_gpu(
            self.fine_pass_water_mask_gpu,
            self.coarse_pass_water_mask_gpu,
            image_lab_tensor_cpu,
            image_path,
            run_dir,
        )
        torch.cuda.synchronize()
        if self.fine_model.verbose:
            print(f"  > Final mask merge took: {time.time() - t_merge_start:.2f}s")

        if self.fine_model.verbose:
            print(
                f"--- [Hierarchical] GPU mask combination complete. Total time: {time.time() - t_start:.2f}s ---"
            )

        if coverage_only:
            total_pixels = combined_water_mask_gpu.numel()
            if total_pixels == 0:
                return 0.0
            water_pixels = torch.sum(combined_water_mask_gpu)
            kelp_pixels = total_pixels - water_pixels
            return ((kelp_pixels.float() / total_pixels) * 100.0).item()
        else:
            return combined_water_mask_gpu.cpu().numpy()

    def get_fine_pass_data(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        return self.internal_fine_results, self.fine_slice_info

    def get_coarse_pass_data(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        return self.internal_coarse_results, self.coarse_slice_info

    def get_component_masks(self) -> Dict[str, Optional[np.ndarray]]:
        fine_water_mask_np = (
            self.fine_pass_water_mask_gpu.cpu().numpy()
            if self.fine_pass_water_mask_gpu is not None
            else None
        )
        coarse_water_mask_np = (
            self.coarse_pass_water_mask_gpu.cpu().numpy()
            if self.coarse_pass_water_mask_gpu is not None
            else None
        )
        return {"Coarse Pass": coarse_water_mask_np, "Fine Pass": fine_water_mask_np}
