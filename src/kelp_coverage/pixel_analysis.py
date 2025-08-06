import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from tqdm import tqdm
from typing import Optional, Tuple


def find_representative_lab_color(
    directory: str, samples_per_image: int = 50000, visualize: bool = False
) -> Optional[Tuple[int, int, int]]:
    l_vals, a_vals, b_vals = [], [], []
    image_files = [
        f
        for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print(f"No image files found in directory: {directory}")
        return None

    for filename in tqdm(image_files, desc="Analyzing pixels in LAB space"):
        image_path = os.path.join(directory, filename)
        try:
            image = cv2.imread(image_path)
            if image is None:
                continue
            image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            h, w, _ = image_lab.shape
            num_pixels = h * w
            if num_pixels == 0:
                continue

            samples = min(samples_per_image, num_pixels)
            random_idx = random.sample(range(num_pixels), samples)
            random_samples = image_lab.reshape(-1, 3)[random_idx]

            l_vals.extend(random_samples[:, 0])
            a_vals.extend(random_samples[:, 1])
            b_vals.extend(random_samples[:, 2])
        except Exception as e:
            print(f"Error processing image {filename}: {e}")
            continue

    if not l_vals:
        print("No pixels were sampled.")
        return None

    median_l = np.median(l_vals)
    median_a = np.median(a_vals)
    median_b = np.median(b_vals)
    representative_color = (int(median_l), int(median_a), int(median_b))

    if visualize:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(l_vals, bins=32, color="gray", alpha=0.7)
        plt.title("L* Channel Histogram")
        plt.axvline(median_l, color="r", linestyle="dashed", linewidth=2)
        plt.subplot(1, 3, 2)
        plt.hist(a_vals, bins=32, color="green", alpha=0.7)
        plt.title("a* Channel Histogram")
        plt.axvline(median_a, color="r", linestyle="dashed", linewidth=2)
        plt.subplot(1, 3, 3)
        plt.hist(b_vals, bins=32, color="blue", alpha=0.7)
        plt.title("b* Channel Histogram")
        plt.axvline(median_b, color="r", linestyle="dashed", linewidth=2)
        plt.suptitle(f"Median LAB: {representative_color}")
        plt.show()

    return representative_color


def extract_location(filename: str) -> Optional[str]:
    parts = filename.split("_")
    if len(parts) >= 3:
        return f"{parts[0]}_{parts[1]}"
    return None
