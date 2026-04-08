import cv2
import numpy as np
from typing import List, Tuple
from sahi.slicing import slice_image as _sahi_slice

def slice_iamge(
    image: np.ndarray,
    slice_size: int,
    overlap: float,
    padding: int = 0,
    padding_color: Tuple[int, int, int] = (0, 0, 0),
) -> dict:
    if padding > 0 and slice_size - 2 * padding <= 0:
        raise ValueError("slice_size must be greater than twice the padding.")

    content_size = slice_size - 2 * padding
    original_shape = image.shape

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    sliced = _sahi_slice(
        image=image_bgr,
        slice_height=content_size,
        slice_width=content_size,
        overlap_height_ratio=overlap,
        overlap_width_ratio=overlap,
    )

    img_list: List[np.ndarray] = []
    img_starting_pts: List[Tuple[int, int]] = []

    for s, start in zip(sliced.images, sliced.starting_pixels):
        if padding > 0:
            s = cv2.copyMakeBorder(
                s, padding, padding, padding, padding,
                cv2.BORDER_CONSTANT, value=padding_color,
            )
        img_list.append(cv2.cvtColor(s, cv2.COLOR_BGR2RGB))
        img_starting_pts.append((start[0], start[1]))

    return {
        "img_list": img_list,
        "img_starting_pts": img_starting_pts,
        "original_shape": original_shape,
    }
