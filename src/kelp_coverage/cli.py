import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple
from kelp_coverage import config as cf

def _generate_run_hash(args: argparse.Namespace) -> str:
    key_args = {
        "slice_size":                   args.slice_size,
        "slice_overlap":                args.slice_overlap,
        "padding":                      args.padding,
        "clahe":                        args.clahe,
        "downsample_factor":            args.downsample_factor,
        "num_points":                   args.num_points,
        "threshold":                    args.threshold,
        "grid_size":                    args.grid_size,
        "uniformity_std_threshold":     args.uniformity_std_threshold,
        "uniform_grid_thresh":          args.uniform_grid_thresh,
        "water_grid_thresh":            args.water_grid_thresh,
        "fallback_brightness_threshold": args.fallback_brightness_threshold,
        "fallback_distance_threshold":  args.fallback_distance_threshold,
        "hierarchical":                 args.hierarchical,
    }
    if args.hierarchical:
        key_args.update({
            "hierarchical_slice_size":  args.hierarchical_slice_size,
            "use_erosion_merge":        args.use_erosion_merge,
            "erosion_kernel_size":      args.erosion_kernel_size,
            "use_color_validation":     args.use_color_validation,
            "merge_color_threshold":    args.merge_color_threshold,
            "merge_lightness_threshold": args.merge_lightness_threshold,
        })
    args_str = json.dumps(dict(sorted(key_args.items())), sort_keys=True)
    return hashlib.sha256(args_str.encode()).hexdigest()[:8]


def _load_pixel_data(csv_path: str) -> Dict[str, Tuple[int, int, int]]:
    import pandas as pd
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    return {
        row["location"]: (int(row["L"]), int(row["A"]), int(row["B"]))
        for _, row in df.iterrows()
    }


def _save_pixel_data(loc_to_pixel: Optional[Dict], csv_path: str) -> None:
    import pandas as pd
    if not loc_to_pixel:
        return
    rows = [{"location": loc, "L": p[0], "A": p[1], "B": p[2]}
            for loc, p in loc_to_pixel.items() if p]
    if not rows:
        return
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Pixel values saved to {csv_path}")


def _get_image_paths(args: argparse.Namespace, site_path: str) -> List[str]:
    if hasattr(args, "images") and args.images:
        names = [n.strip() for n in args.images.split(",")]
        return [os.path.join(site_path, n) for n in names
                if os.path.exists(os.path.join(site_path, n))]
    paths = sorted(
        os.path.join(site_path, f) for f in os.listdir(site_path)
        if f.lower().endswith(".jpg")
    )
    if hasattr(args, "count") and args.count != -1:
        random.shuffle(paths)
        return paths[:args.count]
    return paths


def _cmd_setup(args: argparse.Namespace) -> None:
    from kelp_coverage.io.tator_download import download_images_and_get_pixels
    os.makedirs("results", exist_ok=True)
    os.makedirs("images",  exist_ok=True)
    loc_to_pixel = download_images_and_get_pixels(
        file_path=args.tator_csv,
        tator_token=args.tator_token,
        images_dir="images",
        images_per_location=args.images,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        visualize=args.visualize,
    )
    _save_pixel_data(loc_to_pixel, os.path.join("results", "pixel_values.csv"))


def _cmd_analyze(args: argparse.Namespace) -> None:
    from tqdm import tqdm
    from kelp_coverage.pipeline import run_analysis
    os.makedirs("results", exist_ok=True)
    pixel_csv    = os.path.join("results", args.pixel_csv)
    loc_to_pixel = _load_pixel_data(pixel_csv)
    sites        = [args.site] if args.site else list(loc_to_pixel.keys())
    command_str  = " ".join(sys.argv)
    run_hash     = _generate_run_hash(args)

    iterator = tqdm(sites, desc="Overall Progress") if not args.verbose else sites

    for site in iterator:
        if site not in loc_to_pixel:
            print(f"Site '{site}' not in pixel CSV. Available: {list(loc_to_pixel.keys())}")
            continue

        site_path = os.path.join("images", site)
        if not os.path.exists(site_path):
            print(f"Site path not found: {site_path}, skipping.")
            continue

        image_paths = _get_image_paths(args, site_path)
        if not image_paths:
            print(f"No images found for site: {site}")
            continue

        run_dir = os.path.join("results", site, run_hash)
        os.makedirs(run_dir, exist_ok=True)

        run_args_dict = vars(args).copy()
        try:
            run_args_dict["site_name"] = os.path.basename(image_paths[0]).split("_")[2]
        except IndexError:
            run_args_dict["site_name"] = None

        run_analysis(
            image_paths=image_paths,
            model_name=cfg.DEFAULT_MODEL_NAME,
            checkpoint=args.sam_checkpoint,
            water_lab_opencv=loc_to_pixel[site],
            run_dir=run_dir,
            command_str=command_str,
            run_args_dict=run_args_dict,
            site_name=site,
            tator_csv=args.tator_csv,
            verbose=args.verbose,
            overwrite=args.overwrite,
            generate_overlay=args.generate_overlay,
            generate_slice_viz=args.generate_slice_viz,
            generate_threshold_viz=args.generate_threshold_viz,
            generate_erosion_viz=args.generate_erosion_viz,
            hierarchical=args.hierarchical,
            hierarchical_slice_size=args.hierarchical_slice_size,
            slice_size=args.slice_size,
            slice_overlap=args.slice_overlap,
            padding=args.padding,
            num_points=args.num_points,
            threshold=args.threshold,
            grid_size=args.grid_size,
            std_threshold=args.uniformity_std_threshold,
            uniform_thresh=args.uniform_grid_thresh,
            water_thresh=args.water_grid_thresh,
            fallback_brightness=args.fallback_brightness_threshold,
            fallback_distance=args.fallback_distance_threshold,
            gpu_batch_size=args.gpu_batch_size,
            downsample_factor=args.downsample_factor,
            clahe=args.clahe,
            use_erosion=args.use_erosion_merge,
            erosion_kernel=args.erosion_kernel_size,
            use_color_validation=args.use_color_validation,
            color_threshold=args.merge_color_threshold,
            lightness_threshold=args.merge_lightness_threshold,
        )


def _cmd_debug(args: argparse.Namespace) -> None:
    import torch
    import torch.nn.functional as F
    from kelp_coverage.models.protocol import load_model
    import kelp_coverage.models.mobile_sam  # noqa: F401 — registers "mobile_sam"
    from kelp_coverage.core.color import load_image, rgb_to_lab_gpu, convert_opencv_lab
    from kelp_coverage.core.candidates import select_prompt_points, check_shortcut_condition
    from kelp_coverage.core.slicing import slice_image
    from kelp_coverage.viz.visualization import build_debug_figures, save_debug_visualization

    pixel_csv    = os.path.join("results", args.pixel_csv)
    loc_to_pixel = _load_pixel_data(pixel_csv)
    if args.site not in loc_to_pixel:
        print(f"Site '{args.site}' not in pixel CSV.")
        return

    device    = "cuda" if torch.cuda.is_available() else "cpu"
    model     = load_model(cfg.DEFAULT_MODEL_NAME, args.sam_checkpoint, device)
    water_lab = convert_opencv_lab(*loc_to_pixel[args.site], device=device)
    debug_dir = os.path.join("results", "debug")
    os.makedirs(debug_dir, exist_ok=True)

    for image_path in args.image_path:
        if not os.path.exists(image_path):
            print(f"Not found: {image_path}")
            continue
        image      = load_image(image_path, args.downsample_factor, args.clahe)
        lab_cpu    = rgb_to_lab_gpu(image, device).cpu()
        slice_info = slice_image(image, args.slice_size, args.slice_overlap, args.padding)
        padding    = args.padding

        for i, slice_img in enumerate(slice_info["img_list"]):
            if i not in args.slice_index:
                continue

            start_x, start_y = slice_info["img_starting_pts"][i]
            h, w, _   = slice_img.shape
            content_h = h - 2 * padding
            content_w = w - 2 * padding
            end_y     = min(start_y + content_h, lab_cpu.shape[0])
            end_x     = min(start_x + content_w, lab_cpu.shape[1])
            lab_slice = lab_cpu[start_y:end_y, start_x:end_x]
            if padding > 0:
                lab_slice = F.pad(lab_slice,
                                  (0, 0, padding, padding, padding, padding),
                                  "constant", 0)
            lab_gpu = lab_slice.to(device)

            threshold = args.override_threshold or args.threshold
            points_xy, diagnostics = select_prompt_points(
                lab_gpu, water_lab,
                threshold=threshold,
                num_points=args.num_points,
                grid_size=args.grid_size,
                std_threshold=args.uniformity_std_threshold,
                slice_size=args.slice_size,
                return_diagnostics=True,
            )
            if points_xy is not None:
                diagnostics["final_points_xy"] = points_xy
            is_shortcut, _, _ = check_shortcut_condition(diagnostics)
            figs = build_debug_figures(
                slice_img, diagnostics, threshold, i,
                is_shortcut=is_shortcut,
                show_stages=args.visualize_stages,
                show_heatmap=args.heatmap,
                water_lab=water_lab,
                device=device,
                grid_size=args.grid_size,
            )
            save_debug_visualization(
                figs, debug_dir,
                os.path.splitext(os.path.basename(image_path))[0],
                i, threshold,
            )


def _cmd_heatmap(args: argparse.Namespace) -> None:
    from kelp_coverage.io.heatmap import generate_heatmap
    generate_heatmap(
        coverage_json=args.coverage_data,
        output_path=args.output,
        grid_cell_size=args.grid_size,
        show_grid_values=args.show_grid_values,
        show_points=args.show_points,
        show_point_labels=args.show_point_labels,
    )


def _build_base_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--sam-checkpoint",              type=str,   default="mobile_sam.pt")
    p.add_argument("--slice-size",                  type=int,   default=cfg.SLICE_SIZE)
    p.add_argument("--slice-overlap",               type=float, default=cfg.SLICE_OVERLAP)
    p.add_argument("--padding",                     type=int,   default=cfg.PADDING)
    p.add_argument("--clahe",                       action="store_true", default=cfg.CLAHE_ENABLED)
    p.add_argument("--downsample-factor",           type=float, default=cfg.DOWNSAMPLE_FACTOR)
    p.add_argument("--pixel-csv",                   type=str,   default="pixel_values.csv")
    p.add_argument("--num-points",                  type=int,   default=cfg.NUM_POINTS)
    p.add_argument("--threshold",                   type=int,   default=cfg.THRESHOLD)
    p.add_argument("--grid-size",                   type=int,   default=cfg.GRID_SIZE)
    p.add_argument("--uniformity-std-threshold",    type=float, default=cfg.UNIFORMITY_STD_THRESHOLD)
    p.add_argument("--uniform-grid-thresh",         type=float, default=cfg.UNIFORM_GRID_THRESH)
    p.add_argument("--water-grid-thresh",           type=float, default=cfg.WATER_GRID_THRESH)
    p.add_argument("--fallback-brightness-threshold", type=float, default=cfg.FALLBACK_BRIGHTNESS_THRESHOLD)
    p.add_argument("--fallback-distance-threshold", type=float, default=cfg.FALLBACK_DISTANCE_THRESHOLD)
    p.add_argument("--verbose", "-v",               action="store_true")
    return p


def main() -> None:
    from kelp_coverage.splash import play as _splash
    _splash(force='--splash' in sys.argv)

    parser = argparse.ArgumentParser(
        description="Kelp coverage segmentation using Segment Anything Models."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    base = _build_base_parser()

    sp = sub.add_parser("setup", help="Download images and compute water LAB color per site.")
    sp.add_argument("--tator-csv",   type=str, default="tator_data.csv")
    sp.add_argument("--tator-token", type=str, required=True)
    sp.add_argument("--images",      type=int, default=-1)
    sp.add_argument("--start-idx",   type=int)
    sp.add_argument("--end-idx",     type=int)
    sp.add_argument("--visualize",   action="store_true")

    ap = sub.add_parser("analyze", parents=[base], help="Run kelp segmentation analysis.")
    ap.add_argument("--site",         type=str)
    ap.add_argument("--tator-csv",    type=str, required=True)
    ap.add_argument("--images",       type=str, help="Comma-separated image filenames.")
    ap.add_argument("--count",        type=int, default=-1)
    ap.add_argument("--gpu-batch-size", type=int, default=cfg.GPU_BATCH_SIZE)
    ap.add_argument("--overwrite",    action="store_true")
    ap.add_argument("--coverage-only", action="store_true")
    ap.add_argument("--generate-overlay",        action="store_true")
    ap.add_argument("--generate-slice-viz",      action="store_true")
    ap.add_argument("--slice-viz-max-size",      type=int, default=256)
    ap.add_argument("--generate-threshold-viz",  action="store_true")
    ap.add_argument("--generate-erosion-viz",    action="store_true")
    ap.add_argument("--generate-component-viz",  action="store_true")
    ap.add_argument("--hierarchical",            action=argparse.BooleanOptionalAction, default=cfg.HIERARCHICAL)
    ap.add_argument("--hierarchical-slice-size", type=int,   default=cfg.HIERARCHICAL_SLICE_SIZE)
    ap.add_argument("--use-erosion-merge",       action=argparse.BooleanOptionalAction, default=cfg.USE_EROSION_MERGE)
    ap.add_argument("--erosion-kernel-size",     type=int,   default=cfg.EROSION_KERNEL_SIZE)
    ap.add_argument("--use-color-validation",    action=argparse.BooleanOptionalAction, default=cfg.USE_COLOR_VALIDATION)
    ap.add_argument("--merge-color-threshold",   type=int,   default=cfg.MERGE_COLOR_THRESHOLD)
    ap.add_argument("--merge-lightness-threshold", type=float, default=cfg.MERGE_LIGHTNESS_THRESHOLD)

    dp = sub.add_parser("debug-slice", parents=[base], help="Per-slice debug visualizations.")
    dp.add_argument("--image-path",        type=str, required=True, nargs="+")
    dp.add_argument("--slice-index",       type=int, required=True, nargs="+")
    dp.add_argument("--site",              type=str, required=True)
    dp.add_argument("--override-threshold", type=int)
    dp.add_argument("--heatmap",           action="store_true")
    dp.add_argument("--visualize-stages",  action="store_true")

    hp = sub.add_parser("heatmap", help="Generate geospatial coverage heatmap.")
    hp.add_argument("--coverage-data",    type=str, required=True)
    hp.add_argument("--output",           type=str)
    hp.add_argument("--grid-size",        type=int, default=30)
    hp.add_argument("--show-grid-values", action="store_true")
    hp.add_argument("--show-points",      action="store_true")
    hp.add_argument("--show-point-labels", action="store_true")
    hp.add_argument("--verbose", "-v",    action="store_true")

    args = parser.parse_args()

    if args.command == "setup":
        _cmd_setup(args)
    elif args.command == "analyze":
        _cmd_analyze(args)
    elif args.command == "debug-slice":
        _cmd_debug(args)
    elif args.command == "heatmap":
        _cmd_heatmap(args)


if __name__ == "__main__":
    main()
