import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from segment_anything.utils.transforms import ResizeLongestSide
from ultralytics import SAM as UltralyticsSAM

from kelp_coverage import config
from kelp_coverage.models.protocol import register_model


class MobileSAMModel:
    def __init__(self, checkpoint: str, device: str) -> None:
        self.device = device
        sam_wrapper = UltralyticsSAM(checkpoint)
        self.model = sam_wrapper.model.to(device).eval()
        self.img_size = self.model.image_encoder.img_size  
        self.transform = ResizeLongestSide(self.img_size)

        # pixel mean / std taken directly from SAM github
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L27
        self.pixel_mean = torch.tensor(
            config.SAM_PIXEL_MEAN, device=device
        ).view(-1, 1, 1)
        self.pixel_std = torch.tensor(
            config.SAM_PIXEL_STD, device=device
        ).view(-1, 1, 1)

    def preprocess(
        self, slices: List[np.ndarray]
    ) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        resized      = [self.transform.apply_image(s) for s in slices]
        input_sizes  = [r.shape[:2] for r in resized]

        batch = torch.stack(
            [torch.as_tensor(r, device=self.device, dtype=torch.float32) for r in resized]
        )                                                
        batch = batch.permute(0, 3, 1, 2)
        batch = (batch - self.pixel_mean) / self.pixel_std

        return batch, input_sizes

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model.image_encoder(batch)

    def decode(
        self,
        embedding: torch.Tensor,
        points_xy: np.ndarray,
        original_size: Tuple[int, int],
        input_size: Tuple[int, int],
    ) -> torch.Tensor:
        input_points = self.transform.apply_coords(points_xy, original_size)
        input_labels = np.ones(len(input_points), dtype=np.int32)

        pts_torch    = torch.as_tensor(input_points, device=self.device, dtype=torch.float32).unsqueeze(0)
        labels_torch = torch.as_tensor(input_labels, device=self.device, dtype=torch.int64).unsqueeze(0)

        # implementation based on forward function from SAM model
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L130
        with torch.no_grad():
            sparse_emb, dense_emb = self.model.prompt_encoder(
                points=(pts_torch, labels_torch),
                boxes=None,
                masks=None,
            )
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=embedding,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=False,
            )

        mask = self._postprocess(low_res_masks, input_size, original_size)
        return mask[0, 0] > 0.0

    def _postprocess(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        # taken directly from SAM github
        # https://github.com/facebookresearch/segment-anything/blob/dca509fe793f601edb92606367a655c15ac00fdf/segment_anything/modeling/sam.py#L130
        """Two-step resize: upsample to img_size, crop, then resize to original."""
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., :input_size[0], :input_size[1]]
        masks = F.interpolate(
            masks,
            size=original_size,
            mode="bilinear",
            align_corners=False,
        )
        return masks

@register_model("mobile_sam")
def load_mobile_sam(checkpoint: str, device: str) -> MobileSAMModel:
    return MobileSAMModel(checkpoint, device)
