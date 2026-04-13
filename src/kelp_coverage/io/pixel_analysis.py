import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from typing import Optional, Tuple

def find_representative_lab_color(
    directory: str,
    samples_per_image: int = 50000,
    visualize: bool = False,
) -> Optional[Tuple[int, int, int]]:
    image_files = [
        f for f in os.listdir(directory)
    ]
    if not image_files:
        print(f"No image files found in directory: {directory}")
        return None

    l_vals, a_vals, b_vals = [], [], []

    for filename in tqdm(image_files, desc="Analysing pixels in LAB space"):
        image_path = os.path.join(directory, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: could not read {filename}, skipping.")
                continue
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            h, w, _ = image_lab.shape
            num_pixels = h * w
            if num_pixels == 0:
                continue
            samples    = min(samples_per_image, num_pixels)
            random_idx = random.sample(range(num_pixels), samples)
            pixels     = image_lab.reshape(-1, 3)[random_idx]
            l_vals.extend(pixels[:, 0])
            a_vals.extend(pixels[:, 1])
            b_vals.extend(pixels[:, 2])
        except Exception as e:
            print(f"Warning: error processing {filename}: {e}")
            continue

    if not l_vals:
        print("No pixels were sampled.")
        return None

    representative = (int(np.median(l_vals)), int(np.median(a_vals)), int(np.median(b_vals)))

    if visualize:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, vals, label, color, med in zip(
            axes,
            [l_vals, a_vals, b_vals],
            ["L* Channel", "a* Channel", "b* Channel"],
            ["gray", "green", "blue"],
            representative,
        ):
            ax.hist(vals, bins=32, color=color, alpha=0.7)
            ax.set_title(label)
            ax.axvline(med, color="r", linestyle="dashed", linewidth=2)
        fig.suptitle(f"Median LAB: {representative}")
        plt.show()

    return representative


def extract_location(filename: str) -> Optional[str]:
    parts = filename.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    return None
