# KLP: Kelp Location Profiler Tool

[![PyPI version](https://img.shields.io/pypi/v/kelp-coverage?label=pypi%20package)](https://pypi.org/project/kelp-coverage/)

**Version: 1.0**

`kelp-coverage` is a command-line package designed to analyze UAV imagery of the ocean to calculate the percentage of kelp coverage. It utilizes the Segment Anything Model (SAM) in combination with Sliced Aided Hyper-Inference (SAHI) to create high-resolution segmentations of kelp from water.

## Table of Contents
1.  [Installation](#installation)
2.  [Core Concepts](#core-concepts)
3.  [Workflow Overview](#workflow-overview)
4.  [Command Reference](#command-reference)
5.  [Full Argument Reference](#full-argument-reference)
6.  [Examples](#examples)

## Installation

1.  **Install from PyPI:**
    Install the package directly from PyPI using [pip](https://test.pypi.org/project/kelp-coverage/).

    ```bash
    pip install -i [https://test.pypi.org/simple/](https://test.pypi.org/simple/) kelp-coverage
    ```
2.  **Download a SAM Checkpoint:**
    The tool defaults to using MobileSAM. Download the checkpoint file into your working directory:

    ```bash
    wget -q [https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt](https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt)
    ```
    If you wish to use the original, larger SAM models, download one of the official checkpoints from the [SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Core Concepts

* **Site:** A `site` refers to a specifc drone survey represented by a unique name (`trinity-2_20250123T185918`). The tool organizes results and images based on these site names

* **Representative Water Pixel Value:** To generate accurate coverage calculations the program first needs to have access to the "average" water pixel within a site. The `setup` command samples a 50000 pixels from each image and then takes the median of all the pixels. It then saves this pixel value (in LAB color space) and uses it as a baseline for generating seed points for segmentation.

* **Hierarchical Processing (`--hierarchical`):** For more accurate results using a hierarchical appraoch is recommended.
    1.  **Coarse Pass:** The tool first analyzes the image using large slices (e.g., 4096x4096) to generate a high-level, coarse mask.
    2.  **Fine Pass:** Next, the tool uses smaller, more detailed slices (e.g., 1024x1024) to create a fine-grained mask that better captures edges.
    3.  **Intelligent Merge:** The two masks are combined using a sophisticated merge logic. By default, this involves:
        * **Erosion (`--use-erosion-merge`):** The coarse mask is slightly eroded to remove small, noisy detections and provide a more reliable baseline.
        * **Color Validation (`--use-color-validation`):** Across the areas of disagreement, the tool analyzes pixel values in order to correct any misidentifications of kelp

## Workflow Overview

The typical workflow involves four main steps:

1.  **`setup`:** Download the image metadata CSV from the **[drone website URL]** and use it to download images from Tator. This step also calculates the representative water color for each site. This only needs to be done once per dataset.

2.  **`analyze`:** Run the main segmentation analysis on the downloaded images. This is where kelp coverage is calculated. You can run this multiple times with different parameters.

3.  **`debug-slice`:** (Optional) If you notice strange results for a particular image, use this command to inspect the segmentation process on specific slices of that image.

4.  **`heatmap`:** Once you have coverage data for a site, generate a spatial heatmap to visualize the distribution of kelp.

---

## Command Reference

### `setup`
Downloads images via Tator and computes representative water pixel values for each site.
```bash
kelp-coverage setup --tator-csv <path_to_metadata.csv> --tator-token <your_api_token> [options]
```

### `analyze`
Runs the kelp segmentation analysis. Can be run on a single site or all sites found in `images/`.
```bash
# Analyze a single site
kelp-coverage analyze --site <site_name> --tator-csv <path_to_metadata.csv> [options]

# Analyze all sites
kelp-coverage analyze --tator-csv <path_to_metadata.csv> [options]
```

#### The `analyze` Process: A Technical Look
The `analyze` command executes a sophisticated pipeline for each image:

1.  **Image Loading & Pre-processing:** The image is loaded. It can be enhanced with downsampling (`--downsample-factor`) or contrast adjustment (`--clahe`). The full image is converted to the LAB color space on the CPU.
2.  **Slicing:** The image is divided into smaller, overlapping tiles using SAHI, defined by `--slice-size` and `--slice-overlap`.
3.  **Prompt Point Selection (Per Slice):** For each slice, the tool samples `n` points (`--num-points`) that represent water to "prompt" the SAM model. This involves:
    * Transferring the LAB data for the slice to the GPU.
    * Filtering a grid based on color distance (`--threshold`) and texture uniformity (`--uniformity-std-threshold`) to find the best candidate points.
4.  **SAM Inference:** If a slice isn't obviously empty water (based on `--uniform-grid-thresh` and `--water-grid-thresh`), the prompt points are sent to SAM to generate a water mask for that slice. If no points are found, a fallback logic using brightness and color distance classifies the slice.
5.  **Mask Reconstruction & Merging:** The individual masks from all slices are stitched together. In `--hierarchical` mode, this involves the intelligent merge of the coarse and fine passes.
6.  **Output Generation:** The final kelp mask is used to calculate the coverage percentage, which is saved to a CSV file. Optional visualizations can also be generated.

### `debug-slice`
Performs a detailed debug analysis on specific image slices, generating visualizations of the point selection pipeline.
```bash
kelp-coverage debug-slice --image-path <path_to_image.JPG> --slice-index <index> --site <site_name> [options]
```

### `heatmap`
Generates a spatial heatmap from a combined coverage CSV file.
```bash
kelp-coverage heatmap --coverage-data <path_to_coverage.csv> [options]
```

---

## Full Argument Reference
## Full Argument Reference

### Core Model and Slicing Arguments
| Argument | Default | Description |
| --- | --- | --- |
| `--sam-checkpoint` | `mobile_sam.pt` | Path to the SAM model checkpoint. |
| `--use-mobile-sam` | `True` | Use the lightweight MobileSAM model. Disable with `--no-use-mobile-sam`. |
| `--sam-model-type` | `vit_h` | Model type for standard SAM (e.g., `vit_h`, `vit_l`). Ignored for MobileSAM. |
| `--slice-size` | `1024` | Size of the slices generated by SAHI. |
| `--slice-overlap` | `0.2` | Overlap ratio between adjacent slices (0.0 to 1.0). |
| `--padding` | `0` | Pixel padding to add to each slice before processing. |

### Pre-processing and Point Selection
| Argument | Default | Description |
| --- | --- | --- |
| `--clahe` | `False` | Apply Contrast Limited Adaptive Histogram Equalization (CLAHE). |
| `--downsample-factor` | `1.0` | Factor to downsample the image by (e.g., 2.0 for half size). |
| `--pixel-csv` | `pixel_values.csv` | Path to the CSV storing representative LAB pixel values. |
| `--num-points` | `3` | Number of seed points provided to SAM. |
| `--threshold` | `20` | LAB color distance threshold to identify water pixels. |
| `--threshold-max` | `20` | Maximum LAB color threshold to search up to if no points are found. |
| `--final-point-strategy` | `poisson_disk` | Algorithm for selecting final prompt points (`poisson_disk`, `center_bias`, `random`). |
| `--grid-size` | `64` | Pixel size of the grid for initial point filtering. |
| `--uniformity-check` | `True` | Enable/disable the grid uniformity check. Disable with `--no-uniformity-check`. |
| `--uniformity-std-threshold` | `4.0` | Standard deviation threshold for a grid cell to be "uniform". |

### Shortcut & Fallback Arguments
| Argument | Default | Description |
| --- | --- | --- |
| `--uniform-grid-thresh` | `0.85` | Percentage of uniform grids required to shortcut SAM. |
| `--water-grid-thresh` | `0.95` | Percentage of water-colored grids required to shortcut SAM. |
| `--fallback-brightness-threshold` | `100.0` | Brightness threshold to classify a slice as water if no points are found. |
| `--fallback-distance-threshold` | `55.0` | LAB color distance threshold to classify a slice as water if no points are found. |

### Hierarchical Mode Arguments (`analyze` only)
| Argument | Default | Description |
| --- | --- | --- |
| `--hierarchical` | `True` | Use the two-pass hierarchical method. Disable with `--no-hierarchical`. |
| `--hierarchical-slice-size` | `4096` | Slice size for the coarse pass. |
| `--use-erosion-merge` | `True` | Use erosion on the coarse mask. Disable with `--no-use-erosion-merge`. |
| `--erosion-kernel-size` | `51` | Kernel size for the erosion merge. |
| `--use-color-validation` | `True` | Use color validation to resolve mask disagreements. Disable with `--no-use-color-validation`. |
| `--merge-color-threshold`| `15` | LAB color distance threshold for validating kelp in the merge disagreement zone. |
| `--merge-lightness-threshold`| `75.0` | Lightness (L*) threshold for validating kelp in the merge disagreement zone. |

### Output and Visualization Arguments
| Argument | Default | Description |
| --- | --- | --- |
| `--verbose` / `-v` | `False` | Enable verbose output, including a list of all active flags. |
| `--generate-overlay` | `False` | Generate a transparent overlay of the kelp mask on the original image. |
| `--generate-slice-viz` | `False` | Generate a grid visualization of all processed slices. |
| `--slice-viz-max-size` | `256` | Maximum dimension for slices in the visualization grid. |
| `--generate-threshold-viz` | `False` | Generate a visualization of the color distance threshold for each slice. |
| `--generate-erosion-viz` | `False` | [HIERARCHICAL] Generate a visualization of the erosion merge effect. |
| `--generate-merge-viz` | `False` | [HIERARCHICAL] Generate a heatmap of the disagreement area during mask merging. |
| `--overwrite` | `False` | Overwrite existing results for a site/parameter combination. |
| `--coverage-only` | `False` | Only compute and save coverage values, skipping all visualization outputs. |

---
## Examples

**Standard Workflow:**

1.  **Setup:** Download metadata and images for a project.
    ```bash
    kelp-coverage setup --tator-csv all_sites_metadata.csv --tator-token <your_api_token> --images 5
    ```

2.  **Analyze a single site with recommended hierarchical settings:**
    ```bash
    kelp-coverage analyze \
      --site "hopkins-1_20250315T190000" \
      --tator-csv all_sites_metadata.csv \
      --hierarchical \
      --generate-overlay
    ```

3.  **Analyze all sites with basic settings (faster, less accurate):**
    ```bash
    kelp-coverage analyze --tator-csv all_sites_metadata.csv --coverage-only
    ```

4.  **Debug a problematic image:**
    ```bash
    kelp-coverage debug-slice \
      --site "hopkins-1_20250315T190000" \
      --image-path "images/hopkins-1_20250315T190000/DSC01234.JPG" \
      --slice-index 42 55 \
      --visualize-stages \
      --heatmap
    ```

5.  **Create a heatmap from results:**

    ```bash    
    kelp-coverage heatmap --coverage-data all_coverage.csv --show-points
    ```

