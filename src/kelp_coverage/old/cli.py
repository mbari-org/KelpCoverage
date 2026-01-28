import argparse
import os
import pandas as pd
import random
import sys
import json
import hashlib

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

from .tator_download import download_images_and_get_pixels
from .visualization import run_sahi_sam_visualization
from .segmentation_processors import SinglePassProcessor, HierarchicalProcessor
from .sahisam import SAHISAM
from .heatmap import generate_heatmap


def _generate_run_hash(args: argparse.Namespace) -> str:
    hash_key_args = {
        "slice_size": args.slice_size,
        "slice_overlap": args.slice_overlap,
        "padding": args.padding,
        "clahe": args.clahe,
        "downsample_factor": args.downsample_factor,
        "num_points": args.num_points,
        "threshold": args.threshold,
        "threshold_max": args.threshold_max,
        "final_point_strategy": args.final_point_strategy,
        "grid_size": args.grid_size,
        "uniformity_check": args.uniformity_check,
        "uniformity_std_threshold": args.uniformity_std_threshold,
        "uniform_grid_thresh": args.uniform_grid_thresh,
        "water_grid_thresh": args.water_grid_thresh,
        "fallback_brightness_threshold": args.fallback_brightness_threshold,
        "fallback_distance_threshold": args.fallback_distance_threshold,
        "hierarchical": args.hierarchical,
        "use_post_processing": getattr(args, "use_post_processing", False),
        "morph_kernel_size": getattr(args, "morph_kernel_size", 3),
        "blur_pre_merge": getattr(args, "blur_pre_merge", False),
        "blur_post_merge": getattr(args, "blur_post_merge", False),
        "blur_kernel_size": getattr(args, "blur_kernel_size", 7),
        "blur_sigma": getattr(args, "blur_sigma", 1.5),
    }
    if args.hierarchical:
        hash_key_args.update(
            {
                "hierarchical_slice_size": args.hierarchical_slice_size,
                "merge_strategy": args.merge_strategy,
                "use_color_validation": args.use_color_validation,
                "merge_color_threshold": args.merge_color_threshold,
                "merge_lightness_threshold": args.merge_lightness_threshold,
            }
        )
    
    sorted_args = dict(sorted(hash_key_args.items()))
    args_string = json.dumps(sorted_args, sort_keys=True)
    return hashlib.sha256(args_string.encode()).hexdigest()[:8]


def _ensure_directories(
    results_dir: str = "results", images_dir: str = "images"
) -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)


def _save_pixel_data(
    loc_to_pixel: Optional[Dict[str, Tuple[int, int, int]]], csv_path: str
) -> None:
    if not loc_to_pixel:
        return

    new_pixel_data = [
        {"location": loc, "L": p[0], "A": p[1], "B": p[2]}
        for loc, p in loc_to_pixel.items()
        if p
    ]
    if not new_pixel_data:
        print("No new pixel data to save.")
        return
    new_df = pd.DataFrame(new_pixel_data)

    if os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path)
            updated_locations = new_df['location'].tolist()
            existing_df = existing_df[~existing_df['location'].isin(updated_locations)]
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        except pd.errors.EmptyDataError:
            combined_df = new_df
    else:
        combined_df = new_df
        
    combined_df.to_csv(csv_path, index=False)
    print(f"Pixel values updated in {csv_path}")

def _load_pixel_data(csv_path: str) -> Dict[str, Tuple[int, int, int]]:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        row["location"]: (int(row["L"]), int(row["A"]), int(row["B"]))
        for _, row in df.iterrows()
    }


def _get_image_paths(args: argparse.Namespace, site_path: str) -> List[str]:
    if args.images:
        image_names = [img.strip() for img in args.images.split(",")]
        return [
            os.path.join(site_path, f)
            for f in image_names
            if os.path.exists(os.path.join(site_path, f))
        ]
    else:
        paths = [
            os.path.join(site_path, f)
            for f in os.listdir(site_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        paths.sort()
        if args.count != -1:
            random.shuffle(paths)
            return paths[: args.count]
        else:
            return paths


def _setup_data(args: argparse.Namespace) -> None:
    results_dir = "results"
    images_dir = "images"
    _ensure_directories(results_dir, images_dir)
    pixel_csv_path = os.path.join(results_dir, "pixel_values.csv")
    loc_to_pixel = download_images_and_get_pixels(
        file_path=args.tator_csv,
        tator_token=args.tator_token,
        images_dir=images_dir,
        images_per_location=args.images,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        visualize=args.visualize,
        verbose=args.verbose,
    )
    _save_pixel_data(loc_to_pixel, pixel_csv_path)


def _process_site(
    site: str,
    args: argparse.Namespace,
    loc_to_pixel: Dict[str, Tuple[int, int, int]],
    run_dir: str,
    command_str: str,
    run_args_dict: dict,
) -> None:
    site_path = os.path.join("images", site)
    if not os.path.exists(site_path):
        print(f"Site path not found: {site_path}, skipping.")
        return

    image_paths = _get_image_paths(args, site_path)
    if not image_paths:
        print(f"No images found for site: {site}")
        return

    water_lab = loc_to_pixel[site]

    try:
        if args.hierarchical:
            processor: Any = HierarchicalProcessor(args, water_lab)
        else:
            processor = SinglePassProcessor(args, water_lab)
    except (KeyError, NotImplementedError, RuntimeError) as e:
        print(f"Skipping site {site} due to processor initialization error: {e}")
        return

    run_sahi_sam_visualization(
        image_paths=image_paths,
        processor=processor,
        run_dir=run_dir,
        command_str=command_str,
        run_args_dict=run_args_dict,
        tator_csv=args.tator_csv,
        verbose=args.verbose,
        site_name=site,
        generate_overlay=args.generate_overlay,
        generate_slice_viz=args.generate_slice_viz,
        generate_threshold_viz=args.generate_threshold_viz,
        generate_erosion_viz=args.generate_erosion_viz,
        generate_component_viz=args.generate_component_viz,
        slice_viz_max_size=args.slice_viz_max_size,
        coverage_only=args.coverage_only,
        overwrite=args.overwrite,
        generate_fine_heatmap=args.generate_fine_heatmap,
        generate_coarse_heatmap=args.generate_coarse_heatmap,
        generate_merged_heatmap=args.generate_merged_heatmap,
    )


def _run_analysis(args: argparse.Namespace) -> None:
    results_dir = "results"
    _ensure_directories(results_dir)
    pixel_csv_path = os.path.join(results_dir, args.pixel_csv)
    loc_to_pixel = _load_pixel_data(pixel_csv_path)

    sites_to_process = [args.site] if args.site else list(loc_to_pixel.keys())

    command_str = " ".join(sys.argv)
    run_args_dict = {k: v for k, v in vars(args).items() if v is not None}
    run_hash = _generate_run_hash(args)

    if not args.verbose and not args.site:
        iterable = tqdm(sites_to_process, desc="Overall Progress")
    else:
        iterable = sites_to_process

    for site in iterable:
        if isinstance(iterable, tqdm):
            iterable.set_description(f"Processing Site: {site}")

        if site not in loc_to_pixel:
            print(f"Site {site} not found. Available: {list(loc_to_pixel.keys())}")
            continue

        run_dir = os.path.join(results_dir, site, run_hash)
        os.makedirs(run_dir, exist_ok=True)

        _process_site(site, args, loc_to_pixel, run_dir, command_str, run_args_dict)


def _run_debug(args: argparse.Namespace) -> None:
    results_dir = "results"
    debug_dir = os.path.join(results_dir, "debug")
    _ensure_directories(debug_dir)
    pixel_csv_path = os.path.join(results_dir, args.pixel_csv)
    if not os.path.exists(pixel_csv_path):
        print(f"Pixel data not found at {pixel_csv_path}. Run 'setup' command first.")
        return
    loc_to_pixel = _load_pixel_data(pixel_csv_path)
    if args.site not in loc_to_pixel:
        print(
            f"Site {args.site} not found in pixel data. Available: {list(loc_to_pixel.keys())}"
        )
        return

    water_lab = loc_to_pixel[args.site]
    model = SAHISAM(
        sam_checkpoint=args.sam_checkpoint,
        sam_model_type=args.sam_model_type,
        water_lab=water_lab,
        use_mobile_sam=args.use_mobile_sam,
        slice_size=args.slice_size,
        padding=args.padding,
        num_points=args.num_points,
        threshold=args.threshold,
        threshold_max=args.threshold_max,
        verbose=args.verbose,
        final_point_strategy=args.final_point_strategy,
        grid_size=args.grid_size,
        uniformity_check=args.uniformity_check,
        uniformity_std_threshold=args.uniformity_std_threshold,
        uniform_grid_thresh=args.uniform_grid_thresh,
        water_grid_thresh=args.water_grid_thresh,
        points_per_grid=getattr(args, 'points_per_grid', 10),
        fallback_brightness_threshold=args.fallback_brightness_threshold,
        fallback_distance_threshold=args.fallback_distance_threshold,
        kelp_confidence_threshold=args.kelp_confidence_threshold[0], 
    )
    for image_path in args.image_path:
        if not os.path.exists(image_path):
            print(f"Warning: Image path not found, skipping: {image_path}")
            continue
        print(f"\n--- Running Debug on: {os.path.basename(image_path)} ---")
        model.process_image(
            image_path=image_path,
            visualize_slice_indices=args.slice_index,
            visualize_output_dir=debug_dir,
            visualize_heatmap=args.heatmap,
            visualize_stages=args.visualize_stages,
        )
    print(f"\n--- Debug processing complete. Visualizations saved in: {debug_dir} ---")


def _run_heatmap(args: argparse.Namespace) -> None:
    print("Generating heatmap...")
    generate_heatmap(
        coverage_json=args.coverage_data,
        output_path=args.output,
        grid_cell_size=args.grid_size,
        show_grid_values=args.show_grid_values,
        show_points=args.show_points,
        show_point_labels=args.show_point_labels,
    )


def _print_active_flags(args: argparse.Namespace):
    print("--- Active Command-Line Flags ---")
    for arg, value in sorted(vars(args).items()):
        if isinstance(value, bool) and value:
            print(f"  --{arg.replace('_', '-')}")
    print("---------------------------------")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="A command-line tool for segmenting kelp in UAV imagery using Segment Anything Models.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(
        dest="command", help="Available commands", required=True
    )

    sam_model_parser = argparse.ArgumentParser(add_help=False)
    sam_model_parser.add_argument("--sam-checkpoint", type=str, default="mobile_sam.pt", help="Path to the downloaded SAM model checkpoint.")
    sam_model_parser.add_argument("--use-mobile-sam", action=argparse.BooleanOptionalAction, default=True, help="Use the lightweight MobileSAM model.")
    sam_model_parser.add_argument("--sam-model-type", type=str, default="vit_h", help="Model type for standard SAM (e.g., vit_h, vit_l, vit_b).")
    sam_model_parser.add_argument("--slice-size", type=int, default=1024, help="Size of the slices generated by SAHI.")
    sam_model_parser.add_argument("--slice-overlap", type=float, default=0.2, help="Overlap ratio between adjacent slices (0.0 to 1.0).")
    sam_model_parser.add_argument("--padding", type=int, default=0, help="Pixel padding to add to each slice before processing.")
    sam_model_parser.add_argument("--clahe", action="store_true", help="Apply CLAHE to images before processing.")
    sam_model_parser.add_argument("--downsample-factor", type=float, default=1.0, help="Factor by which to downsample the image.")
    sam_model_parser.add_argument("--pixel-csv", type=str, default="pixel_values.csv", help="Path to the CSV file storing representative LAB pixel values.")
    sam_model_parser.add_argument("--num-points", type=int, default=3, help="Number of seed points provided to SAM for segmentation.")
    sam_model_parser.add_argument("--threshold", type=int, default=20, help="Threshold for LAB color distance to identify water pixels.")
    sam_model_parser.add_argument("--threshold-max", type=int, default=20, help="Maximum LAB color threshold to search up to.")
    sam_model_parser.add_argument("--final-point-strategy", type=str, default="poisson_disk", choices=["poisson_disk", "center_bias", "random"], help="Algorithm for selecting the final prompt points.")
    sam_model_parser.add_argument("--grid-size", type=int, default=64, help="Pixel size of the grid used for initial point filtering.")
    sam_model_parser.add_argument("--uniformity-check", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable the grid uniformity check.")
    sam_model_parser.add_argument("--uniformity-std-threshold", type=float, default=4.0, help='Standard deviation threshold for a grid cell to be "uniform".')
    sam_model_parser.add_argument("--uniform-grid-thresh", type=float, default=0.98, help="Percentage of uniform grids required to shortcut SAM.")
    sam_model_parser.add_argument("--water-grid-thresh", type=float, default=0.98, help="Percentage of water-colored grids required to shortcut SAM.")
    sam_model_parser.add_argument("--fallback-brightness-threshold", type=float, default=100.0, help="[FALLBACK] Brightness threshold for water classification.")
    sam_model_parser.add_argument("--fallback-distance-threshold", type=float, default=55.0, help="[FALLBACK] LAB color distance threshold for water classification.")
    sam_model_parser.add_argument("--kelp-confidence-threshold", type=float, nargs='+', default=[0.5], help="One or more confidence thresholds for classifying a pixel as KELP (0.0 to 1.0).")

    general_parser = argparse.ArgumentParser(add_help=False)
    general_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    setup_parser = subparsers.add_parser("setup", parents=[general_parser], help="Download images and compute representative water pixel for each site.")
    setup_parser.add_argument("--tator-csv", type=str, default="tator_data.csv", help="Path to the image CSV file.")
    setup_parser.add_argument("--tator-token", required=True, type=str, help="API token for Tator.")
    setup_parser.add_argument("--images", type=int, default=-1, help="Number of images to download per site. -1 for all.")
    setup_parser.add_argument("--start-idx", type=int, help="Optional start index for images to process from CSV.")
    setup_parser.add_argument("--end-idx", type=int, help="Optional end index for images to process from CSV.")
    setup_parser.add_argument("--visualize", action="store_true", help="Display histograms of the LAB color channels.")

    analyze_parser = subparsers.add_parser("analyze", parents=[general_parser, sam_model_parser], help="Run the kelp segmentation analysis.")
    analyze_parser.add_argument("--site", type=str, help="Specify a specific site to process. Processes all sites if omitted.")
    analyze_parser.add_argument("--tator-csv", type=str, required=True, help="Required path to the CSV to link results with metadata.")
    analyze_parser.add_argument("--images", type=str, help="A comma-separated list of specific image filenames to process.")
    analyze_parser.add_argument("--count", type=int, default=-1, help="Number of images to randomly select per site. -1 for all.")
    analyze_parser.add_argument("--gpu-batch-size", type=int, default=32, help="Maximum number of slices to send to the GPU in a single batch.")
    analyze_parser.add_argument("--generate-overlay", action="store_true", help="Generate and save a transparent overlay of the kelp mask.")
    analyze_parser.add_argument("--generate-slice-viz", action="store_true", help="Generate a grid visualization of segmented slices.")
    analyze_parser.add_argument("--slice-viz-max-size", type=int, default=256, help="Maximum dimension for slices in visualization grid.")
    analyze_parser.add_argument("--generate-threshold-viz", action="store_true", help="Generate threshold visualization for each slice.")
    analyze_parser.add_argument("--hierarchical", action=argparse.BooleanOptionalAction, default=True, help="Use a two-pass hierarchical method.")
    analyze_parser.add_argument("--hierarchical-slice-size", type=int, default=4096, help="[HIERARCHICAL] Slice size for the coarse pass.")
    analyze_parser.add_argument("--generate-erosion-viz", action="store_true", help="[HIERARCHICAL] Generate a visualization of the erosion merge effect.")
    analyze_parser.add_argument("--generate-component-viz", action="store_true", help="[HIERARCHICAL] Generate a visualization of the fine and coarse masks.")
    analyze_parser.add_argument("--use-color-validation", action=argparse.BooleanOptionalAction, default=True, help="[HIERARCHICAL] Use color validation in merge.")
    analyze_parser.add_argument("--generate-merge-viz", action="store_true", help="[HIERARCHICAL] Generate a heatmap of the merge disagreement area.")
    analyze_parser.add_argument("--merge-color-threshold", type=int, default=15, help="[HIERARCHICAL] LAB color distance threshold for merge validation.")
    analyze_parser.add_argument("--merge-lightness-threshold", type=float, default=75.0, help="[HIERARCHICAL] Lightness (L*) threshold for merge validation.")
    analyze_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results for a site.")
    analyze_parser.add_argument("--coverage-only", action="store_true", help="Only compute and save coverage values.")
    analyze_parser.add_argument("--generate-fine-heatmap", action="store_true", help="[HIERARCHICAL] Generate a heatmap of the fine pass confidence.")
    analyze_parser.add_argument("--generate-coarse-heatmap", action="store_true", help="[HIERARCHICAL] Generate a heatmap of the coarse pass confidence.")
    analyze_parser.add_argument("--generate-merged-heatmap", action="store_true", help="[HIERARCHICAL] Generate a heatmap of the final merged confidence.")
    analyze_parser.add_argument("--merge-strategy", type=str, default="kelp-based", choices=["kelp-based", "average", "water-based"], help="[HIERARCHICAL] Strategy for merging fine and coarse passes. 'kelp-based' is the original behavior.")
    analyze_parser.add_argument("--use-post-processing", action=argparse.BooleanOptionalAction, default=False, help="Enable morphological opening to clean the final mask.")
    analyze_parser.add_argument("--morph-kernel-size", type=int, default=3, help="[POST-PROCESSING] Kernel size for morphological opening.")
    analyze_parser.add_argument("--blur-pre-merge", action="store_true", help="[HIERARCHICAL] Apply Gaussian blur to fine/coarse confidence maps BEFORE merging.")
    analyze_parser.add_argument("--blur-post-merge", action="store_true", help="Apply Gaussian blur to the final confidence map BEFORE thresholding.")
    analyze_parser.add_argument("--blur-kernel-size", type=int, default=7, help="Kernel size (width and height) for the Gaussian blur. Must be an odd number.")
    analyze_parser.add_argument("--blur-sigma", type=float, default=1.5, help="Standard deviation for the Gaussian blur.")

    debug_parser = subparsers.add_parser("debug-slice", parents=[general_parser, sam_model_parser], help="Detailed debug analysis on specific image slices.")
    debug_parser.add_argument("--image-path", type=str, required=True, nargs="+", help="One or more full paths to images to debug.")
    debug_parser.add_argument("--slice-index", type=int, required=True, nargs="+", help="One or more slice indices to debug for each image.")
    debug_parser.add_argument("--site", type=str, required=True, help="Site name.")
    debug_parser.add_argument("--override-threshold", type=int, help="Manually set LAB color threshold.")
    debug_parser.add_argument("--points-per-grid", type=int, default=10, help="Number of candidate points to show per valid grid cell.")
    debug_parser.add_argument("--heatmap", action="store_true", help="Generate threshold visualization of color distances.")
    debug_parser.add_argument("--visualize-stages", action="store_true", help="Visualize the point filtering pipeline step-by-step.")

    heatmap_parser = subparsers.add_parser("heatmap", parents=[general_parser], help="Generate a spatial heatmap from coverage data.")
    heatmap_parser.add_argument("--coverage-data", type=str, required=True, help="Path to the results.json file from an analysis run.")
    heatmap_parser.add_argument("--output", type=str, help='Path to save the output heatmap image.')
    heatmap_parser.add_argument("--grid-size", type=int, default=30, help="Size of heatmap grid cells.")
    heatmap_parser.add_argument("--show-grid-values", action="store_true", help="Show numerical coverage values on the grid cells.")
    heatmap_parser.add_argument("--show-points", action="store_true", help="Show the location of data points on the map.")
    heatmap_parser.add_argument("--show-point-labels", action="store_true", help="Show labels for the data points.")

    args = parser.parse_args()

    if args.verbose:
        _print_active_flags(args)

    if args.command == "setup":
        _setup_data(args)
    elif args.command == "analyze":
        _run_analysis(args)
    elif args.command == "debug-slice":
        _run_debug(args)
    elif args.command == "heatmap":
        _run_heatmap(args)


if __name__ == "__main__":
    main()

