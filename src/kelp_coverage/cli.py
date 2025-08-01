import argparse
import os
import pandas as pd
import random

from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Any

from .tator_download import download_images_and_get_pixels
from .visualization import run_sahi_sam_visualization
from .segmentation_processors import SinglePassProcessor, HierarchicalProcessor
from .sahisam import SAHISAM
from .heatmap import generate_heatmap

def _ensure_directories(results_dir: str = "results", images_dir: str = "images") -> None:
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(images_dir, exist_ok=True)

def _save_pixel_data(loc_to_pixel: Optional[Dict[str, Tuple[int, int, int]]], csv_path: str) -> None:
    if not loc_to_pixel:
        return
    pixel_data = [{'location': loc, 'L': p[0], 'A': p[1], 'B': p[2]} for loc, p in loc_to_pixel.items() if p]
    if not pixel_data:
        print("No pixel data to save.")
        return
    df = pd.DataFrame(pixel_data)
    df.to_csv(csv_path, index=False)
    print(f"Pixel values saved to {csv_path}")

def _load_pixel_data(csv_path: str) -> Dict[str, Tuple[int, int, int]]:
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {row['location']: (int(row['L']), int(row['A']), int(row['B'])) for _, row in df.iterrows()}

def _get_image_paths(args: argparse.Namespace, site_path: str) -> List[str]:
    if args.images:
        image_names = [img.strip() for img in args.images.split(',')]
        return [os.path.join(site_path, f) for f in image_names if os.path.exists(os.path.join(site_path, f))]
    else:
        paths = [os.path.join(site_path, f) for f in os.listdir(site_path) if f.lower().endswith('.jpg')]
        paths.sort()
        if args.count != -1:
            random.shuffle(paths)
            return paths[:args.count]
        else:
            return paths

def _build_param_string(args: argparse.Namespace) -> str:
    param_string = f"slice{args.slice_size}_pts{args.num_points}_{'mobile' if args.use_mobile_sam else args.sam_model_type}"
    if args.hierarchical:
        param_string += f"_hierarchical{getattr(args, 'hierarchical_slice_size', 4096)}"
        if args.use_erosion_merge and args.use_color_validation:
            param_string += f"_erosion{args.erosion_kernel_size}_color-val{args.color_validation_threshold}"
        elif args.use_color_validation:
            param_string += f"_color-val{args.color_validation_threshold}"
        elif args.use_erosion_merge:
            param_string += f"_erosion{args.erosion_kernel_size}"
    return param_string

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
        visualize=args.visualize
    )
    _save_pixel_data(loc_to_pixel, pixel_csv_path)

def _process_site(site: str, args: argparse.Namespace, loc_to_pixel: Dict[str, Tuple[int, int, int]], results_dir: str) -> None:
    site_path = os.path.join("images", site)
    if not os.path.exists(site_path):
        print(f"Site path not found: {site_path}, skipping.")
        return

    image_paths = _get_image_paths(args, site_path)
    if not image_paths:
        print(f"No images found for site: {site}")
        return

    water_lab = loc_to_pixel[site]
    param_string = _build_param_string(args)

    site_coverage_csv = os.path.join(results_dir, site, "coverage_values.csv")
    processed_entries = set()
    if os.path.exists(site_coverage_csv) and not args.overwrite:
        try:
            existing_df = pd.read_csv(site_coverage_csv)
            processed_entries = set(zip(existing_df['image_name'], existing_df['param_string']))
        except (pd.errors.EmptyDataError, KeyError):
            processed_entries = set()

    try:
        if args.hierarchical:
            processor: Any = HierarchicalProcessor(args, water_lab)
        else:
            processor = SinglePassProcessor(args, water_lab)
    except (KeyError, NotImplementedError, RuntimeError) as e:
        print(f"Skipping site {site} due to processor initialization error: {e}")
        return

    image_iterator = tqdm(image_paths, desc=f"Processing site {site}", leave=True) if not args.verbose and args.site else image_paths

    for image_path in image_iterator:
        image_name = os.path.basename(image_path)
        if (image_name, param_string) in processed_entries:
            if isinstance(image_iterator, tqdm):
                image_iterator.set_postfix_str(f"Skipping {image_name[:15]}...")
            elif args.verbose:
                print(f"Skipping {image_name}, already processed. Use --overwrite to re-run.")
            continue
        try:
            run_sahi_sam_visualization(
                image_path=image_path,
                processor=processor,
                results_dir=results_dir,
                site_name=site,
                param_string=param_string,
                generate_overlay=args.generate_overlay,
                generate_slice_viz=args.generate_slice_viz,
                generate_threshold_viz=args.generate_threshold_viz,
                generate_erosion_viz=getattr(args, 'generate_erosion_viz', False),
                tator_csv=args.tator_csv,
                verbose=args.verbose,
                slice_viz_max_size=args.slice_viz_max_size,
                coverage_only=args.coverage_only
            )
        except Exception as e:
            print(f"\n--- ERROR processing {image_name}: {e} ---")
            with open("error_log.txt", "a") as f:
                f.write(f"Error processing {image_path}: {e}\n")
            continue

def _run_analysis(args: argparse.Namespace) -> None:
    results_dir = "results"
    _ensure_directories(results_dir)
    pixel_csv_path = os.path.join(results_dir, args.pixel_csv)
    if not os.path.exists(pixel_csv_path):
        print(f"Pixel data not found at {pixel_csv_path}. Run 'setup' command first.")
        return
    loc_to_pixel = _load_pixel_data(pixel_csv_path)
    sites_to_process = [args.site] if args.site else list(loc_to_pixel.keys())
    
    site_iterator = tqdm(sites_to_process, desc="Overall Progress") if not args.verbose and not args.site else sites_to_process
    
    for site in site_iterator:
        if site not in loc_to_pixel:
            print(f"Site {site} not found. Available: {list(loc_to_pixel.keys())}")
            continue
        if isinstance(site_iterator, tqdm):
            site_iterator.set_description(f"Processing Site: {site}")
        _process_site(site, args, loc_to_pixel, results_dir)

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
        print(f"Site {args.site} not found in pixel data. Available: {list(loc_to_pixel.keys())}")
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
        points_per_grid=args.points_per_grid
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
        coverage_csv=args.coverage_data, 
        output_path=args.output,
        grid_cell_size=args.grid_size,
        show_grid_values=args.show_grid_values,
        show_points=args.show_points,
        show_point_labels=args.show_point_labels
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description='A command-line tool for segmenting kelp in UAV imagery using Segment Anything Models.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands', required=True)

    base_parser = argparse.ArgumentParser(add_help=False)
    
    # Core Model and Slicing Arguments
    base_parser.add_argument('--sam-checkpoint', type=str, default='mobile_sam.pt',  help='Path to the downloaded SAM model checkpoint. Defaults to MobileSAM model')
    base_parser.add_argument('--use-mobile-sam', action=argparse.BooleanOptionalAction, default=True, help='Use the lightweight MobileSAM model (default). Use --no-use-mobile-sam for the original SAM model.')
    base_parser.add_argument('--sam-model-type', type=str, default='vit_h', help='Model type for standard SAM (e.g., vit_h, vit_l, vit_b). Ignored if using MobileSAM.')
    base_parser.add_argument('--slice-size', type=int, default=1024, help='Size of the slices generated by SAHI.')
    base_parser.add_argument('--slice-overlap', type=float, default=0.2, help='Overlap ratio between adjacent slices (0.0 to 1.0).')
    base_parser.add_argument('--padding', type=int, default=0, help='Pixel padding to add to each slice before processing.')
    
    # Pre-processing Arguments
    base_parser.add_argument('--clahe', action='store_true', help='Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to images before processing.')
    base_parser.add_argument('--downsample-factor', type=float, default=1.0, help='Factor by which to downsample the image (e.g., 2.0 for half size).')
    
    # Point Selection and Filtering Arguments
    base_parser.add_argument('--pixel-csv', type=str, default='pixel_values.csv', help='Path to the CSV file storing representative LAB pixel values for each site.')
    base_parser.add_argument('--num-points', type=int, default=3, help='Number of seed points provided to SAM for segmentation.')
    base_parser.add_argument('--threshold', type=int, default=25, help='Threshold for LAB color distance to identify water pixels.')
    base_parser.add_argument('--threshold-max', type=int, default=30, help='Maximum LAB color threshold to search up to if no points are found.')
    base_parser.add_argument('--final-point-strategy', type=str, default='poisson_disk', choices=['poisson_disk', 'center_bias', 'random'], help='Algorithm for selecting the final prompt points.')
    base_parser.add_argument('--grid-size', type=int, default=64, help='Pixel size of the grid used for initial point filtering.')
    base_parser.add_argument('--uniformity-check', action=argparse.BooleanOptionalAction, default=True, help='Enable/disable the grid uniformity check during point selection.')
    base_parser.add_argument('--uniformity-std-threshold', type=float, default=4.0, help='Standard deviation threshold for a grid cell to be considered "uniform".')
    
    # Shortcut Arguments
    base_parser.add_argument('--uniform-grid-thresh', type=float, default=0.98, help='Percentage of uniform grids required to shortcut SAM.')
    base_parser.add_argument('--water-grid-thresh', type=float, default=0.98, help='Percentage of water-colored grids required to shortcut SAM.')

    base_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')

    # --- Setup Parser ---
    setup_parser = subparsers.add_parser('setup', parents=[base_parser], help='Download images and compute representative water pixel for each site.')
    setup_parser.add_argument('--tator-csv', type=str, default='tator_data.csv', help='Path to the image CSV file.')
    setup_parser.add_argument('--tator-token', type=str, help='API token for Tator.')
    setup_parser.add_argument('--images', type=int, default=-1, help='Number of images to download per site. -1 for all images (default).')
    setup_parser.add_argument('--visualize', action='store_true', help='Display histograms of the LAB color channels for each site.')
    setup_parser.add_argument('--start-idx', type=int, help='Optional start index for images to process from CSV.')
    setup_parser.add_argument('--end-idx', type=int, help='Optional end index for images to process from CSV.')

    # --- Analyze Parser ---
    analyze_parser = subparsers.add_parser('analyze', parents=[base_parser], help='Run the kelp segmentation analysis on a directory of images.')
    analyze_parser.add_argument('--site', type=str, help='Specify specific site to process. Processes all sites if omitted.')
    analyze_parser.add_argument('--tator-csv', type=str, required=True, help='Required path to the CSV to link results with metadata.')
    analyze_parser.add_argument('--images', type=str, help='A comma-separated list of specific image filenames to process.')
    analyze_parser.add_argument('--count', type=int, default=-1, help='Number of images to randomly select and process from each site. -1 for all images(default).')
    analyze_parser.add_argument('--generate-overlay', action='store_true', help='Generate and save a transparent overlay of the kelp mask on the original image.')
    analyze_parser.add_argument('--generate-slice-viz', action='store_true', help='Generate a grid visualization of all slices containing seed points and segmentations.')
    analyze_parser.add_argument('--slice-viz-max-size', type=int, default=256, help='Maximum dimension for slices in visualization grid.')
    analyze_parser.add_argument('--generate-threshold-viz', action='store_true', help='Generate threshold visualization for each slice.')
    analyze_parser.add_argument('--hierarchical', action='store_true', help='Use a two-pass hierarchical method (coarse and fine) for segmentation.')
    analyze_parser.add_argument('--hierarchical-slice-size', type=int, default=4096, help='[HIERARCHICAL] Slice size for the coarse pass.')
    analyze_parser.add_argument('--use-erosion-merge', action='store_true', help='[HIERARCHICAL] Toggle erosion on the coarse mask.')
    analyze_parser.add_argument('--erosion-kernel-size', type=int, default=15, help='[HIERARCHICAL] Kernel size for the erosion merge.')
    analyze_parser.add_argument('--generate-erosion-viz', action='store_true', help='[HIERARCHICAL] Generate a visualization of the erosion merge effect.')
    analyze_parser.add_argument('--use-color-validation', action='store_true', help='[HIERARCHICAL] Toggle color validation to resolve mask disagreements.')
    analyze_parser.add_argument('--color-validation-threshold', type=int, default=50, help='[HIERARCHICAL] LAB distance threshold for color validation.')
    analyze_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing results for a site.')
    analyze_parser.add_argument('--coverage-only', action='store_true', help='Only compute and save coverage values.')

    # --- Debug Parser ---
    debug_parser = subparsers.add_parser('debug-slice', parents=[base_parser], help='Detailed debug analysis on specific image slices.')
    debug_parser.add_argument('--image-path', type=str, required=True, nargs='+', help='One or more full paths to images to debug.')
    debug_parser.add_argument('--slice-index', type=int, required=True, nargs='+', help='One or more slice indices to debug for each image.')
    debug_parser.add_argument('--site', type=str, required=True, help='Site name.')
    debug_parser.add_argument('--override-threshold', type=int, help='Manually set LAB color threshold.')
    debug_parser.add_argument('--heatmap', action='store_true', help='Generate threshold visualization of color distances.')
    debug_parser.add_argument('--visualize-stages', action='store_true', help='Visualize the point filtering pipeline step-by-step.')
    debug_parser.add_argument('--points-per-grid', type=int, default=10, help='Number of candidate points to show per valid grid cell in debug mode.')

    # --- Heatmap Parser ---
    heatmap_parser = subparsers.add_parser('heatmap', help='Generate a spatial heatmap from coverage data.')
    heatmap_parser.add_argument('--coverage-data', type=str, required=True, help='Path to the combined coverage CSV file.')
    heatmap_parser.add_argument('--output', type=str, help='Path to save the output heatmap image. Defaults to "results/heatmap/".')
    heatmap_parser.add_argument('--grid-size', type=int, default=30, help='Size of heatmap grid cells.')
    heatmap_parser.add_argument('--show-grid-values', action='store_true', help='Show numerical coverage values on the grid cells.')
    heatmap_parser.add_argument('--show-points', action='store_true', help='Show the location of data points on the map.')
    heatmap_parser.add_argument('--show-point-labels', action='store_true', help='Show labels for the data points.')

    args = parser.parse_args()

    if args.command == 'setup':
        _setup_data(args)
    elif args.command == 'analyze':
        _run_analysis(args)
    elif args.command == 'debug-slice':
        _run_debug(args)
    elif args.command == 'heatmap':
        _run_heatmap(args)

if __name__ == "__main__":
    main()
