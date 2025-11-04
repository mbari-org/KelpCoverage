import cv2
import numpy as np
import torch
import torch.nn.functional as F
import pprint
import os
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
            "kelp_confidence_threshold": model_args.kelp_confidence_threshold,
        }
        if model_args.verbose:
            print("-" * 10)
            print("Initializing Single-Pass Processor with arguments:")
            pprint.pprint(sahisam_args)
            print("-" * 10)
        self.model = SAHISAM(**sahisam_args)
        self.use_post_processing = getattr(model_args, "use_post_processing", False)
        self.morph_kernel_size = getattr(model_args, "morph_kernel_size", 3)
        self.blur_post_merge = getattr(model_args, "blur_post_merge", False)
        self.blur_kernel_size = getattr(model_args, "blur_kernel_size", 7)
        self.blur_sigma = getattr(model_args, "blur_sigma", 1.5)
        self.kelp_confidence_thresholds = model_args.kelp_confidence_threshold


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
        final_water_confidence_map = self.model.reconstruct_full_mask_gpu(
            results,
            slice_info,
            return_gpu_tensor=True, 
            apply_blur=self.blur_post_merge,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )
        final_kelp_confidence_map = 1.0 - final_water_confidence_map

        if coverage_only:
            coverage_results = {}
            total_pixels = final_kelp_confidence_map.numel()
            if total_pixels == 0:
                return {t: 0.0 for t in self.kelp_confidence_thresholds}
            
            for thresh in self.kelp_confidence_thresholds:
                mask = final_kelp_confidence_map >= thresh
                coverage = ((mask.sum().float() / total_pixels) * 100.0).item()
                coverage_results[thresh] = coverage
            return coverage_results

        mask_results = {}
        for thresh in self.kelp_confidence_thresholds:
            final_kelp_mask = (final_kelp_confidence_map >= thresh).cpu().numpy()

            if self.use_post_processing:
                if self.model.verbose:
                    print(f"--- [Post-Processing] Applying morphological opening with kernel size {self.morph_kernel_size} for threshold {thresh}... ---")
                kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
                final_kelp_mask = cv2.morphologyEx(final_kelp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            
            mask_results[thresh] = final_kelp_mask
            
        return mask_results


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
            "kelp_confidence_threshold": model_args.kelp_confidence_threshold,
        }
        fine_args = common_sahisam_args.copy()
        fine_args["slice_size"] = model_args.slice_size
        self.fine_model = SAHISAM(**fine_args)

        coarse_args = common_sahisam_args.copy()
        coarse_args["slice_size"] = model_args.hierarchical_slice_size
        self.coarse_model = SAHISAM(**coarse_args)

        self.use_color_validation = model_args.use_color_validation
        self.merge_color_threshold = model_args.merge_color_threshold
        self.merge_lightness_threshold = model_args.merge_lightness_threshold
        
        self.merge_strategy = getattr(model_args, "merge_strategy", "kelp-based")
        self.use_post_processing = getattr(model_args, "use_post_processing", False)
        self.morph_kernel_size = getattr(model_args, "morph_kernel_size", 3)
        self.blur_pre_merge = getattr(model_args, "blur_pre_merge", False)
        self.blur_post_merge = getattr(model_args, "blur_post_merge", False)
        self.blur_kernel_size = getattr(model_args, "blur_kernel_size", 7)
        self.blur_sigma = getattr(model_args, "blur_sigma", 1.5)
        self.kelp_confidence_thresholds = model_args.kelp_confidence_threshold

        self.water_lab_tensor = self.fine_model.water_lab_tensor
        self.internal_fine_results: Optional[Any] = None
        self.fine_slice_info: Optional[Dict[str, Any]] = None
        self.internal_coarse_results: Optional[Any] = None
        self.coarse_slice_info: Optional[Dict[str, Any]] = None
        self.fine_pass_confidence_map_gpu: Optional[torch.Tensor] = None
        self.coarse_pass_confidence_map_gpu: Optional[torch.Tensor] = None
        self.merged_water_confidence_map_gpu: Optional[torch.Tensor] = None

    def _merge_masks_gpu(
        self,
        fine_pass_confidence_map: torch.Tensor,
        coarse_pass_confidence_map: torch.Tensor,
        image_lab_tensor_cpu: torch.Tensor,
    ) -> torch.Tensor:
        if self.merge_strategy == 'kelp-based':
            merged_water_confidence = torch.min(fine_pass_confidence_map, coarse_pass_confidence_map)
        
        elif self.merge_strategy == 'average':
            merged_water_confidence = (fine_pass_confidence_map + coarse_pass_confidence_map) / 2.0
        
        elif self.merge_strategy == 'water-based':
            merged_water_confidence = torch.max(fine_pass_confidence_map, coarse_pass_confidence_map)
        
        else:
            raise ValueError(f"Unknown merge strategy: {self.merge_strategy}")

        if self.use_color_validation:
            image_lab_tensor = image_lab_tensor_cpu.to(self.device)
            
            is_confidently_kelp = merged_water_confidence <= (1.0 - self.kelp_confidence_thresholds[0])
            
            kelp_pixels_lab = image_lab_tensor[is_confidently_kelp]
            
            if kelp_pixels_lab.numel() > 0:
                lightness_values = kelp_pixels_lab[:, 0]
                color_values = kelp_pixels_lab[:, 1:]
                
                is_too_bright = lightness_values >= self.merge_lightness_threshold
                color_distances = torch.linalg.norm(color_values - self.water_lab_tensor[1:], dim=1)
                is_water_colored = color_distances <= self.merge_color_threshold
                is_invalid_kelp_flat = is_too_bright | is_water_colored
                final_water_confidence = merged_water_confidence.clone()
                invalid_mask = torch.zeros_like(is_confidently_kelp, dtype=torch.bool)
                invalid_mask[is_confidently_kelp] = is_invalid_kelp_flat
                final_water_confidence[invalid_mask] = 1.0
            else:
                final_water_confidence = merged_water_confidence
        else:
            final_water_confidence = merged_water_confidence
        
        return final_water_confidence

    def process_image(
        self, image_path: str, full_lab_tensor_cpu: Optional[torch.Tensor] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        self.internal_fine_results, self.fine_slice_info = self.fine_model.process_image(image_path=image_path, full_lab_tensor_cpu=full_lab_tensor_cpu)
        self.internal_coarse_results, self.coarse_slice_info = self.coarse_model.process_image(image_path=image_path, full_lab_tensor_cpu=full_lab_tensor_cpu)
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
        self.fine_pass_confidence_map_gpu = self.fine_model.reconstruct_full_mask_gpu(
            masks=results,
            slice_info=slice_info, 
            return_gpu_tensor=True,
            apply_blur=self.blur_pre_merge,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )
        self.coarse_pass_confidence_map_gpu = self.coarse_model.reconstruct_full_mask_gpu(
            masks=self.internal_coarse_results, 
            slice_info=self.coarse_slice_info, 
            return_gpu_tensor=True,
            apply_blur=self.blur_pre_merge,
            blur_kernel_size=self.blur_kernel_size,
            blur_sigma=self.blur_sigma,
        )
        
        merged_map_gpu = self._merge_masks_gpu(
            self.fine_pass_confidence_map_gpu,
            self.coarse_pass_confidence_map_gpu,
            image_lab_tensor_cpu,
        )

        if self.blur_post_merge:
            if self.fine_model.verbose:
                print(f"--- Applying Gaussian blur to POST-MERGE confidence map (k={self.blur_kernel_size}, s={self.blur_sigma}) ---")
            merged_map_gpu = self.fine_model._apply_gaussian_blur_gpu(
                merged_map_gpu, self.blur_kernel_size, self.blur_sigma
            )
        
        self.merged_water_confidence_map_gpu = merged_map_gpu
        
        kelp_confidence = 1.0 - self.merged_water_confidence_map_gpu

        if coverage_only:
            coverage_results = {}
            total_pixels = kelp_confidence.numel()
            if total_pixels == 0:
                return {t: 0.0 for t in self.kelp_confidence_thresholds}

            for thresh in self.kelp_confidence_thresholds:
                mask = kelp_confidence >= thresh
                coverage = ((mask.sum().float() / total_pixels) * 100.0).item()
                coverage_results[thresh] = coverage
            return coverage_results
        
        mask_results = {}
        for thresh in self.kelp_confidence_thresholds:
            final_kelp_mask = (kelp_confidence >= thresh).cpu().numpy()

            if self.use_post_processing:
                if self.fine_model.verbose:
                    print(f"--- [Post-Processing] Applying morphological opening with kernel size {self.morph_kernel_size} for threshold {thresh}... ---")
                kernel = np.ones((self.morph_kernel_size, self.morph_kernel_size), np.uint8)
                final_kelp_mask = cv2.morphologyEx(final_kelp_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)
            
            mask_results[thresh] = final_kelp_mask
            
        return mask_results

    def get_fine_confidence_map(self) -> Optional[torch.Tensor]:
        return self.fine_pass_confidence_map_gpu

    def get_coarse_confidence_map(self) -> Optional[torch.Tensor]:
        return self.coarse_pass_confidence_map_gpu

    def get_merged_confidence_map(self) -> Optional[torch.Tensor]:
        return self.merged_water_confidence_map_gpu


