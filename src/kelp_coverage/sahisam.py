import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from tqdm import tqdm
from typing import Tuple, List, Dict, Optional, Any
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from sahi.slicing import slice_image
from matplotlib.patches import Rectangle, Patch
from ultralytics import SAM as UltralyticsSAM


class SAHISAM:
    def __init__(
        self,
        sam_checkpoint: str,
        sam_model_type: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        slice_size: int = 1024,
        slice_overlap: float = 0.2,
        water_lab: Optional[Tuple[int, int, int]] = None,
        clahe: bool = False,
        verbose: bool = False,
        padding: int = 0,
        padding_color: Tuple[int, int, int] = (0, 255, 0),
        num_points: int = 3,
        threshold: int = 15,
        threshold_max: int = 20,
        downsample_factor: float = 1.0,
        use_mobile_sam: bool = False,
        final_point_strategy: str = "poisson_disk",
        grid_size: int = 64,
        uniformity_check: bool = True,
        uniformity_std_threshold: float = 5.0,
        uniform_grid_thresh: float = 0.98,
        water_grid_thresh: float = 0.95,
        points_per_grid: int = 10,
        fallback_brightness_threshold: float = 100.0,
        fallback_distance_threshold: float = 55.0,
        gpu_batch_size: int = 16,
    ) -> None:
        local_vars = locals()
        self.config_args = {key: local_vars[key] for key in local_vars if key != "self"}

        self.device = device
        self.water_lab = water_lab
        self.clahe = clahe
        self.verbose = verbose
        self.slice_size = slice_size
        self.slice_overlap = slice_overlap
        self.padding = padding
        self.padding_color = padding_color
        self.num_points = num_points
        self.threshold = threshold
        self.threshold_max = threshold_max
        self.downsample_factor = downsample_factor
        self.use_mobile_sam = use_mobile_sam
        self.final_point_strategy = final_point_strategy
        self.grid_size = grid_size
        self.uniformity_check = uniformity_check
        self.uniformity_std_threshold = uniformity_std_threshold
        self.uniform_grid_thresh = uniform_grid_thresh
        self.water_grid_thresh = water_grid_thresh
        self.points_per_grid = points_per_grid
        self.fallback_brightness_threshold = fallback_brightness_threshold
        self.fallback_distance_threshold = fallback_distance_threshold
        self.gpu_batch_size = gpu_batch_size

        if self.use_mobile_sam:
            if self.verbose:
                print(f"Loading MobileSAM model from: {sam_checkpoint}")
            sam_wrapper = UltralyticsSAM(sam_checkpoint)
            self.model = sam_wrapper.model.to(self.device)
        else:
            if self.verbose:
                print(
                    f"Loading standard SAM model '{sam_model_type}' from: {sam_checkpoint}"
                )
            if sam_model_type not in sam_model_registry:
                raise KeyError(
                    f"Model type '{sam_model_type}' not found in sam_model_registry."
                )
            self.model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.model.to(device=self.device)

        self.model.eval()
        self.transform = ResizeLongestSide(self.model.image_encoder.img_size)

        # convert from cv2 LAB representation to 0-100, -127 - 128
        l_opencv, a_opencv, b_opencv = self.water_lab
        true_l = l_opencv * 100.0 / 255.0
        true_a = float(a_opencv - 128)
        true_b = float(b_opencv - 128)
        self.water_lab_tensor = torch.tensor(
            [true_l, true_a, true_b], device=self.device, dtype=torch.float32
        )

        # pixel mean / std taken directly from SAM github
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L27
        self.pixel_mean = torch.tensor(
            [123.675, 116.28, 103.53], device=self.device
        ).view(-1, 1, 1)
        self.pixel_std = torch.tensor([58.395, 57.12, 57.375], device=self.device).view(
            -1, 1, 1
        )
        if self.verbose:
            print("--- slice shortcut enabled with two conditions: ---")
            print(f"    1. Uniform Grid % >= {self.uniform_grid_thresh * 100:.1f}%")
            print(f"    2. Water Color Grid % >= {self.water_grid_thresh * 100:.1f}%")

    def _load(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.downsample_factor > 1.0:
            new_width = int(img.shape[1] / self.downsample_factor)
            new_height = int(img.shape[0] / self.downsample_factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        if self.clahe:
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)
        return img

    def _slice(self, image: np.ndarray) -> Dict[str, Any]:
        content_slice_size = self.slice_size - (2 * self.padding)
        if content_slice_size <= 0:
            raise ValueError("Slice size must be greater than twice the padding.")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        sliced_image = slice_image(
            image=image_bgr,
            slice_height=content_slice_size,
            slice_width=content_slice_size,
            overlap_height_ratio=self.slice_overlap,
            overlap_width_ratio=self.slice_overlap,
        )
        padded_img_list = []
        for s in sliced_image.images:
            padded_slice = cv2.copyMakeBorder(
                s,
                self.padding,
                self.padding,
                self.padding,
                self.padding,
                cv2.BORDER_CONSTANT,
                value=self.padding_color,
            )
            padded_img_list.append(cv2.cvtColor(padded_slice, cv2.COLOR_BGR2RGB))
        return {
            "img_list": padded_img_list,
            "img_starting_pts": sliced_image.starting_pixels,
            "original_shape": image.shape,
        }

    def _get_lab_tensor(self, image: np.ndarray) -> torch.Tensor:
        # rgb to lab code taken from cv2 implementation
        # https://github.com/opencv/opencv/blob/7ab4e1bf56849e9c5584ce1400adf9705710ca32/modules/ts/misc/color.py#L191
        image_rgb_tensor = torch.from_numpy(image).to(self.device)
        rgb_normalized = image_rgb_tensor.float() / 255.0

        # Apply sRGB gamma de-correction to get linear RGB values
        gamma_mask = rgb_normalized <= 0.04045
        linear_rgb = torch.where(
            gamma_mask,
            rgb_normalized / 12.92,
            torch.pow((rgb_normalized + 0.055) / 1.055, 2.4),
        )
        r, g, b = linear_rgb[..., 0], linear_rgb[..., 1], linear_rgb[..., 2]
        X = (0.412453 * r + 0.357580 * g + 0.180423 * b) / 0.950456
        Y = 0.212671 * r + 0.715160 * g + 0.072169 * b
        Z = (0.019334 * r + 0.119193 * g + 0.950227 * b) / 1.088754
        T = 0.008856
        fX = torch.where(X > T, torch.pow(X, 1.0 / 3.0), 7.787 * X + 16.0 / 116.0)
        fY = torch.where(Y > T, torch.pow(Y, 1.0 / 3.0), 7.787 * Y + 16.0 / 116.0)
        fZ = torch.where(Z > T, torch.pow(Z, 1.0 / 3.0), 7.787 * Z + 16.0 / 116.0)
        L = torch.where(Y > T, 116.0 * fY - 16.0, 903.3 * Y)
        a = 500.0 * (fX - fY)
        b = 200.0 * (fY - fZ)
        return torch.stack([L, a, b], dim=-1)

    def _get_initial_candidates_gpu(
        self, lab_image_tensor: torch.Tensor, override_threshold: Optional[int] = None
    ) -> Tuple[torch.Tensor, Dict[str, np.ndarray]]:
        current_threshold = (
            override_threshold if override_threshold is not None else self.threshold
        )
        h, w, _ = lab_image_tensor.shape
        grid_size = self.grid_size

        pixel_color_dist = torch.linalg.norm(
            lab_image_tensor - self.water_lab_tensor, dim=2
        )
        water_color_pixels = pixel_color_dist <= current_threshold

        grid_diagnostics = {}
        if self.uniformity_check:
            l_channel = lab_image_tensor[:, :, 0].unsqueeze(0).unsqueeze(0)
            pooled_l_sq = F.avg_pool2d(
                l_channel**2, kernel_size=grid_size, stride=grid_size
            )
            pooled_l = F.avg_pool2d(l_channel, kernel_size=grid_size, stride=grid_size)
            grid_stds = torch.sqrt(
                torch.clamp(pooled_l_sq.squeeze() - pooled_l.squeeze() ** 2, min=0)
            )
            uniform_grids = grid_stds <= self.uniformity_std_threshold

            a_channel = lab_image_tensor[:, :, 1].unsqueeze(0).unsqueeze(0)
            b_channel = lab_image_tensor[:, :, 2].unsqueeze(0).unsqueeze(0)
            pooled_a = F.avg_pool2d(a_channel, kernel_size=grid_size, stride=grid_size)
            pooled_b = F.avg_pool2d(b_channel, kernel_size=grid_size, stride=grid_size)
            avg_lab_grids = (
                torch.cat([pooled_l, pooled_a, pooled_b], dim=1)
                .squeeze(0)
                .permute(1, 2, 0)
            )
            grid_color_dist = torch.linalg.norm(
                avg_lab_grids - self.water_lab_tensor, dim=2
            )
            water_color_grids = grid_color_dist <= current_threshold

            valid_water_grids = water_color_grids & uniform_grids
            pixel_validity_mask = (
                F.interpolate(
                    valid_water_grids.float().unsqueeze(0).unsqueeze(0),
                    size=(h, w),
                    mode="nearest",
                )
                .squeeze()
                .bool()
            )
            grid_diagnostics = {
                "uniform_grids": uniform_grids.cpu().numpy(),
                "water_color_grids": water_color_grids.cpu().numpy(),
                "valid_water_grids": valid_water_grids.cpu().numpy(),
            }
        else:
            pixel_validity_mask = water_color_pixels
            grid_diagnostics = {
                "uniform_grids": None,
                "water_color_grids": None,
                "valid_water_grids": None,
            }

        candidates_yx = torch.nonzero(pixel_validity_mask)
        return candidates_yx, grid_diagnostics

    def _poisson_disk_sampling(
        self, points: np.ndarray, n_samples: int, k: int = 30
    ) -> List[int]:
        if len(points) < n_samples:
            return np.arange(len(points)).tolist()
        if len(points) == 0:
            return []

        first_index = np.random.randint(0, len(points))
        samples_indices = [first_index]
        min_dist_sq = (self.slice_size**2 * 2) / (n_samples**2 * 4)
        max_attempts_per_sample = k

        while len(samples_indices) < n_samples:
            found_new_point = False
            for _ in range(max_attempts_per_sample):
                candidate_idx = np.random.randint(0, len(points))
                candidate_pt = points[candidate_idx]
                all_sample_pts = points[samples_indices]

                diff = all_sample_pts - candidate_pt
                dist_sq = np.sum(diff**2, axis=1)

                if np.all(dist_sq > min_dist_sq):
                    samples_indices.append(candidate_idx)
                    found_new_point = True
                    break
            if not found_new_point:
                break

        return samples_indices

    def _select_prompt_points_from_grid(
        self,
        lab_image_tensor: torch.Tensor,
        return_diagnostics: bool = False,
        threshold: Optional[int] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
        h, w, _ = lab_image_tensor.shape
        search_threshold = threshold if threshold is not None else self.threshold_max
        initial_candidates_yx, grid_diagnostics = self._get_initial_candidates_gpu(
            lab_image_tensor, override_threshold=search_threshold
        )
        diagnostics_data = None
        if return_diagnostics:
            if grid_diagnostics is not None:
                grid_diagnostics["final_threshold_used"] = (
                    search_threshold if initial_candidates_yx.numel() > 0 else None
                )
            diagnostics_data = grid_diagnostics
        if initial_candidates_yx.shape[0] < self.num_points:
            if self.verbose:
                print(
                    f"  > Found {initial_candidates_yx.shape[0]} points at max threshold {search_threshold}. Not enough for prompting."
                )
            if return_diagnostics:
                if diagnostics_data is None:
                    diagnostics_data = {}
                diagnostics_data.update(
                    {
                        "initial_candidates": np.array([]),
                        "grid_filtered_candidates": np.array([]),
                    }
                )
            return None, diagnostics_data

        final_candidate_points_yx = initial_candidates_yx
        if self.uniformity_check:
            grid_size = self.grid_size
            grid_indices = (initial_candidates_yx[:, 0] // grid_size) * (
                (w + grid_size - 1) // grid_size
            ) + (initial_candidates_yx[:, 1] // grid_size)
            perm = torch.randperm(initial_candidates_yx.shape[0], device=self.device)
            shuffled_candidates = initial_candidates_yx[perm]
            shuffled_indices = grid_indices[perm]
            unique_grid_ids, inverse_indices = torch.unique(
                shuffled_indices, return_inverse=True
            )
            arange_perm = torch.arange(
                inverse_indices.size(0), device=inverse_indices.device
            )
            initial_fill_value = initial_candidates_yx.shape[0] + 1
            first_indices = torch.full(
                (unique_grid_ids.size(0),),
                fill_value=initial_fill_value,
                dtype=arange_perm.dtype,
                device=arange_perm.device,
            )
            first_indices.scatter_reduce_(
                0, inverse_indices, arange_perm, reduce="amin", include_self=True
            )
            final_candidate_points_yx = shuffled_candidates[first_indices]

        final_selection_yx = None
        num_candidates = final_candidate_points_yx.shape[0]
        if num_candidates == 0:
            if return_diagnostics:
                diagnostics_data.update(
                    {
                        "initial_candidates": initial_candidates_yx.cpu().numpy(),
                        "grid_filtered_candidates": np.array([]),
                    }
                )
            return None, diagnostics_data
        num_to_sample = min(self.num_points, num_candidates)
        if self.final_point_strategy == "poisson_disk":
            candidates_np = final_candidate_points_yx.cpu().numpy()
            selected_indices = self._poisson_disk_sampling(
                candidates_np, num_to_sample, k=30
            )
            if len(selected_indices) > 0:
                final_selection_yx = final_candidate_points_yx[selected_indices]
        elif self.final_point_strategy == "center_bias":
            center_yx = torch.tensor(
                [h / 2, w / 2], device=self.device, dtype=torch.float32
            )
            distances = torch.linalg.norm(
                final_candidate_points_yx.float() - center_yx, dim=1
            )
            weights = 1.0 / (distances + 1e-6)
            if torch.sum(weights) > 0 and not torch.isinf(weights).any():
                indices = torch.multinomial(weights, num_to_sample, replacement=False)
                final_selection_yx = final_candidate_points_yx[indices]
        if final_selection_yx is None:
            perm_indices = torch.randperm(num_candidates, device=self.device)
            final_selection_yx = final_candidate_points_yx[perm_indices[:num_to_sample]]
        final_points_xy = (
            final_selection_yx[:, [1, 0]].cpu().numpy()
            if final_selection_yx is not None and final_selection_yx.shape[0] > 0
            else None
        )
        if return_diagnostics:
            diagnostics_data.update(
                {
                    "initial_candidates": initial_candidates_yx.cpu().numpy(),
                    "grid_filtered_candidates": final_candidate_points_yx.cpu().numpy(),
                }
            )
        return final_points_xy, diagnostics_data

    def _postprocess_masks_manual(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        # taken directly from SAM github
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L130
        masks = F.interpolate(
            masks,
            size=(self.model.image_encoder.img_size, self.model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(
            masks, size=original_size, mode="bilinear", align_corners=False
        )
        return masks

    def _manual_batch_predict(
        self, slices_for_batch: List[np.ndarray], batch_points: List[np.ndarray]
    ) -> List[torch.Tensor]:
        # implementation based on forward function from SAM model
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L130
        if not slices_for_batch:
            return []
        preprocessed_slices = [self.transform.apply_image(s) for s in slices_for_batch]
        batch_input_tensor = torch.stack(
            [torch.as_tensor(s, device=self.device) for s in preprocessed_slices], dim=0
        )
        processed_batch = batch_input_tensor.float().permute(0, 3, 1, 2)
        processed_batch = (processed_batch - self.pixel_mean) / self.pixel_std
        with torch.no_grad():
            batch_embeddings = self.model.image_encoder(processed_batch)
            generated_masks = []
            for i, slice_img in enumerate(slices_for_batch):
                image_embedding = batch_embeddings[i]
                points = batch_points[i]
                input_points = self.transform.apply_coords(points, slice_img.shape[:2])
                input_labels = np.ones(len(input_points), dtype=np.int32)
                input_points_torch = torch.as_tensor(
                    input_points, device=self.device
                ).unsqueeze(0)
                input_labels_torch = torch.as_tensor(
                    input_labels, device=self.device
                ).unsqueeze(0)
                sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=(input_points_torch, input_labels_torch),
                    boxes=None,
                    masks=None,
                )
                low_res_masks, _ = self.model.mask_decoder(
                    image_embeddings=image_embedding.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                input_size_tuple = self.transform.apply_image(slice_img).shape[:2]
                original_size_tuple = slice_img.shape[:2]
                final_mask_torch = self._postprocess_masks_manual(
                    low_res_masks,
                    input_size=input_size_tuple,
                    original_size=original_size_tuple,
                )
                final_mask = final_mask_torch[0, 0] > 0.0
                generated_masks.append(final_mask)
        return generated_masks

    def _check_shortcut_condition(
        self, diagnostics: Dict[str, Any]
    ) -> Tuple[bool, float, float]:
        if not self.uniformity_check:
            return False, 0.0, 0.0
        uniform_grids = diagnostics.get("uniform_grids")
        water_color_grids = diagnostics.get("water_color_grids")
        if uniform_grids is None or water_color_grids is None:
            return False, 0.0, 0.0
        total_grids = uniform_grids.size
        if total_grids == 0:
            return False, 0.0, 0.0
        uniform_pct = np.sum(uniform_grids) / total_grids
        water_pct = np.sum(water_color_grids) / total_grids
        is_shortcut = (
            uniform_pct >= self.uniform_grid_thresh
            and water_pct >= self.water_grid_thresh
        )
        return is_shortcut, uniform_pct, water_pct

    def _create_debug_visualization(
        self,
        slice_img: np.ndarray,
        current_threshold: int,
        image_path: str,
        slice_index: int,
        output_dir: str,
        diagnostics: Dict[str, Any],
        show_heatmap: bool = False,
        show_stages: bool = False,
        is_shortcut: bool = False,
    ) -> None:
        lab_tensor_for_debug = self._get_lab_tensor(slice_img)
        selected_points_xy, _ = self._select_prompt_points_from_grid(
            lab_tensor_for_debug, return_diagnostics=False, threshold=current_threshold
        )
        initial_candidates = diagnostics.get("initial_candidates", np.array([]))
        grid_filtered_candidates = diagnostics.get(
            "grid_filtered_candidates", np.array([])
        )
        uniform_grids = diagnostics.get("uniform_grids")
        water_color_grids = diagnostics.get("water_color_grids")
        valid_water_grids = diagnostics.get("valid_water_grids")

        plot_candidates = []
        if (
            len(initial_candidates) > 0
            and valid_water_grids is not None
            and self.uniformity_check
        ):
            grid_indices_y, grid_indices_x = (
                initial_candidates[:, 0] // self.grid_size,
                initial_candidates[:, 1] // self.grid_size,
            )
            grid_h, grid_w = valid_water_grids.shape
            plot_candidates_list = []
            for r in range(grid_h):
                for c in range(grid_w):
                    if valid_water_grids[r, c]:
                        in_grid_mask = (grid_indices_y == r) & (grid_indices_x == c)
                        grid_candidates = initial_candidates[in_grid_mask]
                        if len(grid_candidates) > 0:
                            num_to_sample = min(
                                len(grid_candidates), self.points_per_grid
                            )
                            sample_indices = np.random.choice(
                                grid_candidates.shape[0], num_to_sample, replace=False
                            )
                            plot_candidates_list.append(grid_candidates[sample_indices])
            if plot_candidates_list:
                plot_candidates = np.vstack(plot_candidates_list)
        elif len(initial_candidates) > 0:
            plot_candidates = initial_candidates

        image_base = os.path.splitext(os.path.basename(image_path))[0]
        base_save_path = os.path.join(
            output_dir, f"{image_base}_slice_{slice_index}_thresh{current_threshold}"
        )

        if show_stages:
            if (
                self.uniformity_check
                and uniform_grids is not None
                and water_color_grids is not None
            ):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(slice_img)
                grid_h, grid_w = uniform_grids.shape
                for r in range(grid_h):
                    for c in range(grid_w):
                        is_uniform, is_water_color = (
                            uniform_grids[r, c],
                            water_color_grids[r, c],
                        )
                        color = (
                            "green"
                            if is_uniform and is_water_color
                            else ("purple" if is_uniform else "red")
                        )
                        ax.add_patch(
                            Rectangle(
                                (c * self.grid_size, r * self.grid_size),
                                self.grid_size,
                                self.grid_size,
                                facecolor=color,
                                alpha=0.3,
                                edgecolor="white",
                                lw=0.5,
                            )
                        )
                legend_elements = [
                    Patch(
                        facecolor="green",
                        alpha=0.3,
                        label="Valid Grid (Uniform & Water Color)",
                    ),
                    Patch(facecolor="purple", alpha=0.3, label="Uniform Only"),
                    Patch(facecolor="red", alpha=0.3, label="Non-Uniform"),
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.025),
                    fancybox=True,
                    shadow=True,
                    ncol=3,
                )
                ax.set_title(
                    f"Grid Validation | Slice {slice_index}"
                    + (" (Shortcut)" if is_shortcut else ""),
                    loc="center",
                )
                ax.axis("off")
                ax.margins(0.01)
                plt.savefig(
                    f"{base_save_path}_stage1_grid_validation.png",
                    bbox_inches="tight",
                    dpi=200,
                )
                plt.close(fig)

            if (
                self.uniformity_check
                and uniform_grids is not None
                and water_color_grids is not None
            ):
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(slice_img)
                grid_h, grid_w = uniform_grids.shape
                for r in range(grid_h):
                    for c in range(grid_w):
                        is_uniform, is_water_color = (
                            uniform_grids[r, c],
                            water_color_grids[r, c],
                        )
                        color = (
                            "green"
                            if is_uniform and is_water_color
                            else ("purple" if is_uniform else "red")
                        )
                        ax.add_patch(
                            Rectangle(
                                (c * self.grid_size, r * self.grid_size),
                                self.grid_size,
                                self.grid_size,
                                facecolor=color,
                                alpha=0.3,
                                edgecolor="white",
                                lw=0.5,
                            )
                        )

                if len(plot_candidates) > 0:
                    ax.scatter(
                        plot_candidates[:, 1],
                        plot_candidates[:, 0],
                        c="gray",
                        s=15,
                        alpha=0.6,
                    )
                if len(grid_filtered_candidates) > 0:
                    ax.scatter(
                        grid_filtered_candidates[:, 1],
                        grid_filtered_candidates[:, 0],
                        c="orange",
                        s=30,
                        edgecolor="black",
                        lw=0.5,
                    )

                legend_elements = [
                    Patch(facecolor="green", alpha=0.3, label="Valid Grid"),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=f"Unselected Points ({len(plot_candidates)} | {len(initial_candidates)})",
                        markerfacecolor="gray",
                        markersize=10,
                    ),
                    plt.Line2D(
                        [0],
                        [0],
                        marker="o",
                        color="w",
                        label=f"Grid Filtered ({len(grid_filtered_candidates)})",
                        markerfacecolor="orange",
                        markersize=10,
                    ),
                ]
                ax.legend(
                    handles=legend_elements,
                    loc="upper center",
                    bbox_to_anchor=(0.5, -0.025),
                    fancybox=True,
                    shadow=True,
                    ncol=3,
                )
                ax.set_title(
                    f"Point Selection over Grid | Slice {slice_index}", loc="center"
                )
                ax.axis("off")
                ax.margins(0.01)
                plt.savefig(
                    f"{base_save_path}_stage2_point_selection.png",
                    bbox_inches="tight",
                    dpi=200,
                )
                plt.close(fig)

            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(slice_img)
            if len(grid_filtered_candidates) > 0:
                ax.scatter(
                    grid_filtered_candidates[:, 1],
                    grid_filtered_candidates[:, 0],
                    c="orange",
                    s=30,
                    alpha=0.4,
                )
            if selected_points_xy is not None and len(selected_points_xy) > 0:
                ax.scatter(
                    selected_points_xy[:, 0],
                    selected_points_xy[:, 1],
                    c="red",
                    s=60,
                    marker="X",
                    edgecolor="white",
                    lw=1,
                )

            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Grid Filtered Pool ({len(grid_filtered_candidates)})",
                    markerfacecolor="orange",
                    alpha=0.4,
                    markersize=10,
                ),
                plt.Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    label=f"Final Prompts ({len(selected_points_xy) if selected_points_xy is not None else 0})",
                    markerfacecolor="red",
                    markersize=12,
                ),
            ]
            ax.legend(
                handles=legend_elements,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.025),
                fancybox=True,
                shadow=True,
                ncol=3,
            )
            ax.set_title(f"Final Point Selection | Slice {slice_index}", loc="center")
            ax.axis("off")
            ax.margins(0.01)
            plt.savefig(
                f"{base_save_path}_stage3_final_selection.png",
                bbox_inches="tight",
                dpi=200,
            )
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(slice_img)
        legend_elements = []

        if show_heatmap:
            dist_map = (
                torch.linalg.norm(lab_tensor_for_debug - self.water_lab_tensor, dim=2)
                .cpu()
                .numpy()
            )
            vmax = current_threshold * 2
            norm = mcolors.Normalize(vmin=0, vmax=vmax)
            cmap = cm.get_cmap("viridis_r")
            ax.imshow(
                np.ma.masked_where(dist_map > vmax, dist_map),
                alpha=0.5,
                cmap=cmap,
                norm=norm,
            )
            legend_elements.append(
                Patch(
                    facecolor=cmap(0.5), alpha=0.5, label=f"Color Distance (<= {vmax})"
                )
            )

        if (
            self.uniformity_check
            and uniform_grids is not None
            and water_color_grids is not None
        ):
            grid_h, grid_w = uniform_grids.shape
            for r in range(grid_h):
                for c in range(grid_w):
                    is_uniform, is_water_color = (
                        uniform_grids[r, c],
                        water_color_grids[r, c],
                    )
                    color = (
                        "green"
                        if is_uniform and is_water_color
                        else ("purple" if is_uniform else "red")
                    )
                    ax.add_patch(
                        Rectangle(
                            (c * self.grid_size, r * self.grid_size),
                            self.grid_size,
                            self.grid_size,
                            facecolor=color,
                            alpha=0.25,
                            edgecolor="white",
                            lw=0.5,
                        )
                    )
            legend_elements.extend(
                [
                    Patch(facecolor="green", alpha=0.3, label="Valid Grid"),
                    Patch(facecolor="purple", alpha=0.3, label="Uniform Grid"),
                    Patch(facecolor="red", alpha=0.3, label="Non-Uniform Grid"),
                ]
            )

        if len(plot_candidates) > 0:
            ax.scatter(
                plot_candidates[:, 1],
                plot_candidates[:, 0],
                c="yellow",
                s=15,
                alpha=0.6,
            )
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Sampled Initial ({len(plot_candidates)} of {len(initial_candidates)})",
                    markerfacecolor="yellow",
                    markersize=10,
                )
            )

        if len(grid_filtered_candidates) > 0:
            ax.scatter(
                grid_filtered_candidates[:, 1],
                grid_filtered_candidates[:, 0],
                c="orange",
                s=30,
                edgecolor="black",
                lw=0.5,
            )
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=f"Grid Filtered ({len(grid_filtered_candidates)})",
                    markerfacecolor="orange",
                    markersize=10,
                )
            )

        if selected_points_xy is not None and len(selected_points_xy) > 0:
            ax.scatter(
                selected_points_xy[:, 0],
                selected_points_xy[:, 1],
                c="red",
                s=60,
                marker="X",
                edgecolor="white",
                lw=1,
            )
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="X",
                    color="w",
                    label=f"Final Prompts ({len(selected_points_xy)})",
                    markerfacecolor="red",
                    markersize=12,
                )
            )

        ax.set_title(
            f"Final Composite Overlay | Slice {slice_index}"
            + (" (Shortcut)" if is_shortcut else ""),
            loc="center",
        )
        ax.axis("off")
        ax.legend(
            handles=legend_elements,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.025),
            fancybox=True,
            shadow=True,
            ncol=3,
        )
        ax.margins(0.01)
        plt.savefig(
            f"{base_save_path}_stage4_final_overlay.png", bbox_inches="tight", dpi=200
        )
        plt.close(fig)

        print(f"Saved debug visualization(s) to: {os.path.dirname(base_save_path)}")

    def _generate_debug_visualization(
        self,
        slice_img: np.ndarray,
        image_path: str,
        slice_index: int,
        diagnostics: Dict[str, Any],
        output_dir: str,
        show_heatmap: bool,
        show_stages: bool,
        debug_threshold: Optional[int],
    ) -> None:
        is_shortcut, uniform_pct, water_pct = self._check_shortcut_condition(
            diagnostics
        )
        used_threshold = (
            diagnostics.get("final_threshold_used") or debug_threshold or self.threshold
        )
        if self.verbose:
            print(f"\n--- Debug Visualization for Slice {slice_index} ---")
            if self.uniformity_check:
                print(
                    f"Uniform Grid %: {uniform_pct * 100:.2f}% (Threshold: >={self.uniform_grid_thresh * 100:.1f}%)"
                )
                print(
                    f"Water Color Grid %: {water_pct * 100:.2f}% (Threshold: >={self.water_grid_thresh * 100:.1f}%)"
                )
                if is_shortcut:
                    print(
                        "\n--- Slice met shortcut conditions. Visualization shows pre-SAM state. ---"
                    )
        self._create_debug_visualization(
            slice_img,
            used_threshold,
            image_path,
            slice_index,
            output_dir,
            diagnostics,
            show_heatmap=show_heatmap,
            show_stages=show_stages,
            is_shortcut=is_shortcut,
        )

    def process_image(
        self,
        image_path: str,
        full_lab_tensor_cpu: Optional[torch.Tensor] = None,
        visualize_slice_indices: Optional[List[int]] = None,
        visualize_output_dir: Optional[str] = None,
        visualize_heatmap: bool = False,
        visualize_stages: bool = False,
        debug_threshold: Optional[int] = None,
    ) -> Tuple[List[Tuple[torch.Tensor, np.ndarray]], Dict[str, Any]]:
        original_image = self._load(image_path)
        slice_info = self._slice(original_image)
        num_slices = len(slice_info["img_list"])
        results: List[Tuple[torch.Tensor, np.ndarray]] = [
            (torch.empty(0), np.array([])) for _ in range(num_slices)
        ]

        if full_lab_tensor_cpu is None:
            if self.verbose:
                print("NOTE: Pre-computed LAB tensor not provided. Generating one now.")
            full_lab_tensor_cpu = self._get_lab_tensor(original_image).cpu()

        slices_to_process_indices = []
        batch_points = []
        slices_for_batch = []

        desc = f"Analyzing {os.path.basename(image_path)} slices"
        slice_iterator = (
            tqdm(
                enumerate(slice_info["img_list"]),
                total=num_slices,
                desc=desc,
                leave=False,
            )
            if self.verbose
            else enumerate(slice_info["img_list"])
        )

        for i, slice_img in slice_iterator:
            is_debug_slice = (
                visualize_slice_indices is not None and i in visualize_slice_indices
            )
            point_selection_threshold = (
                debug_threshold
                if is_debug_slice and debug_threshold is not None
                else None
            )

            start_x, start_y = slice_info["img_starting_pts"][i]
            h, w, _ = slice_img.shape
            content_h, content_w = h - 2 * self.padding, w - 2 * self.padding

            end_y, end_x = (
                min(start_y + content_h, full_lab_tensor_cpu.shape[0]),
                min(start_x + content_w, full_lab_tensor_cpu.shape[1]),
            )

            lab_slice_full_cpu = full_lab_tensor_cpu[start_y:end_y, start_x:end_x]
            lab_image_tensor_gpu = F.pad(
                lab_slice_full_cpu,
                (0, 0, self.padding, self.padding, self.padding, self.padding),
                "constant",
                0,
            ).to(self.device)
            input_points, diagnostics = self._select_prompt_points_from_grid(
                lab_image_tensor_gpu,
                return_diagnostics=True,
                threshold=point_selection_threshold,
            )

            if is_debug_slice and visualize_output_dir and diagnostics:
                self._generate_debug_visualization(
                    slice_img,
                    image_path,
                    i,
                    diagnostics,
                    visualize_output_dir,
                    visualize_heatmap,
                    visualize_stages,
                    debug_threshold,
                )
                continue

            is_shortcut = False
            if diagnostics:
                is_shortcut, _, _ = self._check_shortcut_condition(diagnostics)
                if is_shortcut:
                    if self.verbose:
                        print(f"Slice {i}: Uniform and water-colored. Skipping SAM.")
                    full_water_mask = torch.ones(
                        (content_h, content_w), dtype=torch.bool, device=self.device
                    )
                    results[i] = (full_water_mask, np.array([]))
                    continue

            if input_points is None or len(input_points) == 0:
                avg_brightness = lab_image_tensor_gpu[:, :, 0].mean().item()
                pixel_distances = torch.linalg.norm(
                    lab_image_tensor_gpu - self.water_lab_tensor, dim=2
                )
                avg_lab_distance = pixel_distances.mean().item()

                is_water_fallback = (
                    avg_brightness > self.fallback_brightness_threshold
                    and avg_lab_distance < self.fallback_distance_threshold
                )

                if is_water_fallback:
                    if self.verbose:
                        print(
                            f"Slice {i}: No points, but passed fallback check. Classifying as 'all water'."
                        )
                    final_mask = torch.ones(
                        (content_h, content_w), dtype=torch.bool, device=self.device
                    )
                else:
                    if self.verbose:
                        print(
                            f"Slice {i}: No points and failed fallback check. Classifying as 'all kelp'."
                        )
                    final_mask = torch.zeros(
                        (content_h, content_w), dtype=torch.bool, device=self.device
                    )

                results[i] = (final_mask, np.array([]))
                continue

            slices_to_process_indices.append(i)
            slices_for_batch.append(slice_img)
            batch_points.append(input_points)

        if slices_for_batch:
            all_generated_masks = []
            num_slices_to_process = len(slices_for_batch)
            num_batches = (
                num_slices_to_process + self.gpu_batch_size - 1
            ) // self.gpu_batch_size

            for i in range(0, num_slices_to_process, self.gpu_batch_size):
                batch_end = min(i + self.gpu_batch_size, num_slices_to_process)
                current_slice_chunk = slices_for_batch[i:batch_end]
                current_point_chunk = batch_points[i:batch_end]

                if self.verbose:
                    print(
                        f"Processing SAM batch {i // self.gpu_batch_size + 1}/{num_batches}: {len(current_slice_chunk)} slices..."
                    )

                batch_masks = self._manual_batch_predict(
                    current_slice_chunk, current_point_chunk
                )
                all_generated_masks.extend(batch_masks)

            for j, final_mask in enumerate(all_generated_masks):
                original_slice_index = slices_to_process_indices[j]
                mask_to_store = final_mask
                if self.padding > 0:
                    h_mask, w_mask = final_mask.shape
                    if h_mask > 2 * self.padding and w_mask > 2 * self.padding:
                        mask_to_store = final_mask[
                            self.padding : -self.padding, self.padding : -self.padding
                        ]

                results[original_slice_index] = (
                    mask_to_store.to(self.device),
                    batch_points[j],
                )

        return results, slice_info

    def reconstruct_full_mask_gpu(
        self,
        masks: List[Tuple[torch.Tensor, np.ndarray]],
        slice_info: Dict[str, Any],
        coverage_only: bool = False,
        return_gpu_tensor: bool = False,
        merge_logic: str = "OR",
    ) -> Any:
        H, W = slice_info["original_shape"][:2]

        has_valid_masks = any(torch.is_tensor(m[0]) and m[0].numel() > 0 for m in masks)
        if not has_valid_masks:
            full_mask_gpu = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        elif merge_logic == "OR":
            full_mask_gpu = torch.zeros((H, W), dtype=torch.bool, device=self.device)
        elif merge_logic == "AND":
            full_mask_gpu = torch.ones((H, W), dtype=torch.bool, device=self.device)
            processed_pixels_mask = torch.zeros(
                (H, W), dtype=torch.bool, device=self.device
            )
        else:
            raise ValueError("merge_logic must be either 'OR' or 'AND'")

        if has_valid_masks:
            for i, (mask_tensor, _) in enumerate(masks):
                if not torch.is_tensor(mask_tensor) or mask_tensor.numel() == 0:
                    continue

                start_x, start_y = slice_info["img_starting_pts"][i]
                h, w = mask_tensor.shape
                end_y, end_x = min(start_y + h, H), min(start_x + w, W)
                if start_y >= H or start_x >= W:
                    continue

                region_h, region_w = end_y - start_y, end_x - start_x
                current_region_slice = (slice(start_y, end_y), slice(start_x, end_x))
                slice_mask_data = mask_tensor[:region_h, :region_w] > 0.0

                if merge_logic == "OR":
                    full_mask_gpu[current_region_slice] |= slice_mask_data
                elif merge_logic == "AND":
                    overlap_region = processed_pixels_mask[current_region_slice]
                    full_mask_gpu[current_region_slice][overlap_region] &= (
                        slice_mask_data[overlap_region]
                    )
                    full_mask_gpu[current_region_slice][~overlap_region] = (
                        slice_mask_data[~overlap_region]
                    )
                    processed_pixels_mask[current_region_slice] = True

        if coverage_only:
            total_pixels = full_mask_gpu.numel()
            if total_pixels == 0:
                return 0.0
            water_pixels = torch.sum(full_mask_gpu)
            kelp_pixels = total_pixels - water_pixels
            return ((kelp_pixels.float() / total_pixels) * 100.0).item()
        elif return_gpu_tensor:
            return full_mask_gpu
        else:
            return full_mask_gpu.cpu().numpy()
