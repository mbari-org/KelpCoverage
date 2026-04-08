import cv2
import numpy as np
import torch
from kelp_coverage import config

def load_image(
    path: str,
    downsample_factor: float = config.DOWNSAMPLE_FACTOR,
    clahe: bool = config.CLAHE_ENABLED,
) -> np.ndarray:
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if downsample_factor > 1.0:
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (int(w / downsample_factor), int(h / downsample_factor)),
            interpolation=cv2.INTER_AREA,
        )

    if clahe:
        img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe_obj = cv2.createCLAHE(
            clipLimit=config.CLAHE_CLIP_LIMIT,
            tileGridSize=config.CLAHE_TILE_GRID_SIZE,
        )
        img_lab[:, :, 0] = clahe_obj.apply(img_lab[:, :, 0])
        img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)

    return img


def rgb_to_lab_GPU(
    image: np.ndarray,
    device: str,
) -> torch.Tensor:
    rgb = torch.from_numpy(image).to(device=device, dtype=torch.float32)/255.0
    # rgb to lab code taken from cv2 implementation
    # https://github.com/opencv/opencv/blob/7ab4e1bf56849e9c5584ce1400adf9705710ca32/modules/ts/misc/color.py#L191
    linear = torch.where(
        rgb <= config.SRGB_GAMMA_THRESHOLD,
        rgb / config.SRGB_LINEAR_SCALE,
        ((rgb + config.SRGB_GAMMA_OFFSET) / config.SRGB_GAMMA_DIVISOR) ** config.SRGB_GAMMA_EXPONENT,
    )

    R, G, B = linear[..., 0], linear[..., 1], linear[..., 2]

    X = (config.RGB_TO_XYZ[0][0] * R + config.RGB_TO_XYZ[0][1] * G + config.RGB_TO_XYZ[0][2] * B) / config.D65_X
    Y =  config.RGB_TO_XYZ[1][0] * R + config.RGB_TO_XYZ[1][1] * G + config.RGB_TO_XYZ[1][2] * B
    Z = (config.RGB_TO_XYZ[2][0] * R + config.RGB_TO_XYZ[2][1] * G + config.RGB_TO_XYZ[2][2] * B) / config.D65_Z

    def _f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(
            t > config.CIE_T,
            t ** (1.0 / 3.0),
            config.CIE_F_SLOPE * t + config.CIE_F_SHIFT,
        )

    fX, fY, fZ = _f(X), _f(Y), _f(Z)

    L = torch.where(Y > config.CIE_T, 116.0 * fY - 16.0, config.CIE_KAPPA * Y)
    a = 500.0 * (fX - fY)
    b = 200.0 * (fY - fZ)

    return torch.stack([L, a, b], dim=-1)

def convert_opencv_lab(
    l: int,
    a: int,
    b: int,
    device: str,
) -> torch.Tensor:
    true_l = l * config.OPENCV_L_SCALE
    true_a = float(a - config.OPENCV_AB_SHIFT)
    true_b = float(b - config.OPENCV_AB_SHIFT)
    return torch.tensor([true_l, true_a, true_b], device=device, dtype=torch.float32)
