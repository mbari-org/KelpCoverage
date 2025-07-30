import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sahisam import SAHISAM
import pprint
from typing import Tuple, Dict, Any, Optional

class SinglePassProcessor:
    def __init__(self, model_args: Any, water_lab: Tuple[int, int, int]):
        sahisam_args = {
            "sam_checkpoint": model_args.sam_checkpoint,
            "sam_model_type": model_args.sam_model,
            "water_lab": water_lab,
            "use_mobile_sam": model_args.use_mobile_sam,
            "slice_size": model_args.slice_size,
            "padding": model_args.padding,
            "num_points": model_args.num_points,
            "threshold": model_args.threshold,
            "threshold_max": model_args.threshold_max,
            "threshold_step": model_args.threshold_step,
            "validate_points": model_args.validate_points,
            "verbose": model_args.verbose,
            "final_point_strategy": model_args.final_point_strategy,
            "grid_size": model_args.grid_size,
            "uniformity_check": model_args.uniformity_check,
            "uniformity_std_threshold": model_args.uniformity_std_threshold,
            "use_grid_uniformity": model_args.use_grid_uniformity,
            "uniform_grid_thresh": model_args.uniform_grid_thresh,
            "water_grid_thresh": model_args.water_grid_thresh
        }
        if model_args.verbose:
            print("--- Initializing Single-Pass Processor with arguments: ---")
            pprint.pprint(sahisam_args)
            print("---------------------------------------------------------")
        self.model = SAHISAM(**sahisam_args)

    def process_image(self, image_path: str) -> Tuple[Any, Any]:
        if self.model.verbose:
            print("Running single-pass (detailed point search on all slices)...")
        return self.model.process_image(image_path=image_path)

    def reconstruct_full_mask(self, results: Any, slice_info: Dict[str, Any], image_path: str, coverage_only: bool = False) -> Any:
        return self.model.reconstruct_full_mask_gpu(results, slice_info, coverage_only=coverage_only)

class HierarchicalProcessor:
    def __init__(self, model_args: Any, water_lab: Tuple[int, int, int]):
        if model_args.verbose:
            print("--- Initializing Hierarchical Processor ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        common_sahisam_args = {
            "sam_checkpoint": model_args.sam_checkpoint,
            "sam_model_type": model_args.sam_model,
            "water_lab": water_lab,
            "use_mobile_sam": model_args.use_mobile_sam,
            "padding": model_args.padding,
            "num_points": model_args.num_points,
            "threshold": model_args.threshold,
            "threshold_max": model_args.threshold_max,
            "threshold_step": model_args.threshold_step,
            "validate_points": model_args.validate_points,
            "verbose": model_args.verbose,
            "final_point_strategy": model_args.final_point_strategy,
            "grid_size": model_args.grid_size,
            "uniformity_check": model_args.uniformity_check,
            "uniformity_std_threshold": model_args.uniformity_std_threshold,
            "use_grid_uniformity": model_args.use_grid_uniformity,
            "uniform_grid_thresh": model_args.uniform_grid_thresh,
            "water_grid_thresh": model_args.water_grid_thresh,
            "device": self.device
        }
        fine_args = common_sahisam_args.copy()
        fine_args['slice_size'] = model_args.slice_size
        self.fine_model = SAHISAM(**fine_args)
        
        coarse_args = common_sahisam_args.copy()
        coarse_args['slice_size'] = model_args.hierarchical_slice_size
        self.coarse_model = SAHISAM(**coarse_args)
        
        self.use_erosion_merge = getattr(model_args, 'use_erosion_merge', False)
        self.erosion_kernel_size = getattr(model_args, 'erosion_kernel_size', 15)
        self.use_color_validation = getattr(model_args, 'use_color_validation', False)
        self.color_validation_threshold = getattr(model_args, 'color_validation_threshold', 50)
        self.water_lab_tensor = torch.tensor(water_lab, device=self.device, dtype=torch.float32)
        
        self.internal_fine_results: Optional[Any] = None
        self.fine_slice_info: Optional[Dict[str, Any]] = None
        self.internal_coarse_results: Optional[Any] = None
        self.coarse_slice_info: Optional[Dict[str, Any]] = None
        self.fine_mask_gpu: Optional[torch.Tensor] = None
        self.coarse_mask_gpu: Optional[torch.Tensor] = None
        self.pre_erosion_mask: Optional[np.ndarray] = None
        self.post_erosion_mask: Optional[np.ndarray] = None

    def _image_to_lab_gpu(self, rgb_tensor: torch.Tensor) -> torch.Tensor:
        rgb_norm = rgb_tensor.float() / 255.0
        r, g, b = rgb_norm[..., 0], rgb_norm[..., 1], rgb_norm[..., 2]

        X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456
        Y = (0.212671 * r + 0.715160 * g + 0.072169 * b)
        Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754

        T = 0.008856
        fX = torch.where(X > T, torch.pow(X, 1./3.), 7.787 * X + 16./116.)
        fY = torch.where(Y > T, torch.pow(Y, 1./3.), 7.787 * Y + 16./116.)
        fZ = torch.where(Z > T, torch.pow(Z, 1./3.), 7.787 * Z + 16./116.)

        L = torch.where(Y > T, 116. * fY - 16.0, 903.3 * Y)
        a = 500. * (fX - fY)
        b = 200. * (fY - fZ)
        
        return torch.stack([L, a, b], dim=-1)

    def _erode_gpu(self, kelp_mask_tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        padding = kernel_size // 2
        inverted_mask = ~kelp_mask_tensor
        dilated_inverted_mask = F.max_pool2d(
            inverted_mask.float().unsqueeze(0).unsqueeze(0),
            kernel_size=kernel_size,
            stride=1,
            padding=padding
        ).squeeze().bool()
        return ~dilated_inverted_mask

    def _merge_masks_gpu(self, coarse_water_mask: torch.Tensor, fine_water_mask: torch.Tensor, image_path: str) -> torch.Tensor:
        coarse_kelp_mask = ~coarse_water_mask
        fine_kelp_mask = ~fine_water_mask

        if self.use_erosion_merge:
            kernel_size = self.erosion_kernel_size
            if kernel_size % 2 == 0: kernel_size += 1
            eroded_coarse_kelp = self._erode_gpu(coarse_kelp_mask, kernel_size)
            self.pre_erosion_mask = coarse_kelp_mask.cpu().numpy()
            self.post_erosion_mask = eroded_coarse_kelp.cpu().numpy()
        else:
            eroded_coarse_kelp = torch.zeros_like(coarse_kelp_mask)

        if self.use_color_validation:
            disagreement_zone = fine_kelp_mask & coarse_water_mask
            if torch.any(disagreement_zone):
                original_image_bgr = cv2.imread(image_path)
                if original_image_bgr.shape[:2] != coarse_water_mask.shape:
                    original_image_bgr = cv2.resize(original_image_bgr, (coarse_water_mask.shape[1], coarse_water_mask.shape[0]), interpolation=cv2.INTER_AREA)
                
                image_rgb_tensor = torch.from_numpy(cv2.cvtColor(original_image_bgr, cv2.COLOR_BGR2RGB)).to(self.device)
                image_lab_tensor = self._image_to_lab_gpu(image_rgb_tensor)

                disagreement_pixels_lab = image_lab_tensor[disagreement_zone]
                distances = torch.linalg.norm(disagreement_pixels_lab - self.water_lab_tensor, dim=1)
                is_validated_as_kelp = distances > self.color_validation_threshold
                
                validated_kelp_in_disagreement = torch.zeros_like(disagreement_zone)
                validated_kelp_in_disagreement[disagreement_zone] = is_validated_as_kelp
            else:
                validated_kelp_in_disagreement = torch.zeros_like(disagreement_zone)
                
            trusted_fine_mask = (fine_kelp_mask & coarse_kelp_mask) | validated_kelp_in_disagreement
        else:
            trusted_fine_mask = fine_kelp_mask

        final_kelp_mask = trusted_fine_mask | eroded_coarse_kelp
        return ~final_kelp_mask

    def process_image(self, image_path: str) -> Tuple[Any, Dict[str, Any]]:
        if self.fine_model.verbose: print("\n--- [Hierarchical] Running passes sequentially ---")
        if self.fine_model.verbose: print("   > Starting FINE pass (small slices)...")
        self.internal_fine_results, self.fine_slice_info = self.fine_model.process_image(image_path=image_path)
        if self.coarse_model.verbose: print("   > Starting COARSE pass (large slices)...")
        self.internal_coarse_results, self.coarse_slice_info = self.coarse_model.process_image(image_path=image_path)
        if self.fine_model.verbose: print("--- [Hierarchical] Both passes complete ---")
        return self.internal_fine_results, self.fine_slice_info

    def reconstruct_full_mask(self, results: Any, slice_info: Dict[str, Any], image_path: str, coverage_only: bool = False) -> Any:
        if self.fine_model.verbose: print("\n--- [Hierarchical] Reconstructing and combining masks on GPU ---")
        
        self.fine_mask_gpu = self.fine_model.reconstruct_full_mask_gpu(results, slice_info, return_gpu_tensor=True)
        self.coarse_mask_gpu = self.coarse_model.reconstruct_full_mask_gpu(self.internal_coarse_results, self.coarse_slice_info, return_gpu_tensor=True)
        
        if self.coarse_mask_gpu is None or self.fine_mask_gpu is None:
             raise RuntimeError("Failed to generate one or both masks in hierarchical processing.")

        combined_mask_gpu = self._merge_masks_gpu(self.coarse_mask_gpu, self.fine_mask_gpu, image_path)
        
        if self.fine_model.verbose: print("--- [Hierarchical] GPU mask combination complete. ---")
        
        if coverage_only:
            return self.fine_model.reconstruct_full_mask_gpu([ (combined_mask_gpu, []) ], {'original_shape': combined_mask_gpu.shape}, coverage_only=True)
        else:
            return combined_mask_gpu.cpu().numpy()

    def get_fine_pass_data(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        return self.internal_fine_results, self.fine_slice_info

    def get_coarse_pass_data(self) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        return self.internal_coarse_results, self.coarse_slice_info

    def get_component_masks(self) -> Dict[str, Optional[np.ndarray]]:
        fine_mask_np = self.fine_mask_gpu.cpu().numpy() if self.fine_mask_gpu is not None else None
        coarse_mask_np = self.coarse_mask_gpu.cpu().numpy() if self.coarse_mask_gpu is not None else None
        return {"Fine Pass": fine_mask_np, "Coarse Pass": coarse_mask_np}

