from tator.openapi import tator_openapi
import tator
import os
import pandas as pd
import urllib3
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .pixel_analysis import find_representative_lab_color, extract_location

# hacky way to get rid of error msg for now
urllib3.disable_warnings()


def download_images_and_get_pixels(
    file_path: str,
    tator_token: str,
    images_dir: str = "images",
    images_per_location: int = -1,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    visualize: bool = False,
    verbose: bool = False,
) -> Dict[str, Optional[Tuple[int, int, int]]]:
    v_print = print if verbose else lambda *a, **k: None

    df = pd.read_csv(file_path)
    df["location"] = df["$name"].apply(extract_location)
    filtered_df = df.dropna(subset=["location"])
    grouped_df = filtered_df.groupby("location")

    host = "https://drone.mbari.org"
    config = tator_openapi.Configuration()
    config.host = host
    config.verify_ssl = False
    if tator_token:
        config.api_key["Authorization"] = tator_token
        config.api_key_prefix["Authorization"] = "Token"
    else:
        print("Warning: No Tator token provided.")
        return None
    api = tator_openapi.TatorApi(tator_openapi.ApiClient(config))

    loc_to_pixel: Dict[str, Optional[Tuple[int, int, int]]] = {}

    location_iterable = grouped_df
    if not verbose:
        location_iterable = tqdm(grouped_df, desc="Processing locations")

    for location, group_df in location_iterable:
        if not verbose and isinstance(location_iterable, tqdm):
            location_iterable.set_description(f"Processing {location}")
        
        v_print(f"Processing location: {location}")
        group_df = group_df.sort_values(by="$id").reset_index(drop=True)
        s_idx = start_idx if start_idx is not None else 0
        e_idx = end_idx if end_idx is not None else len(group_df)
        subset_df = group_df.iloc[s_idx:e_idx]

        if images_per_location == -1 or images_per_location >= len(subset_df):
            images_to_download = subset_df
        else:
            images_to_download = subset_df.sample(n=images_per_location, replace=False)

        location_path = os.path.join(images_dir, str(location))
        os.makedirs(location_path, exist_ok=True)
        v_print(f"Checking for existing images in {location_path}")

        try:
            existing_files = set(os.listdir(location_path))
            if existing_files:
                v_print(f"Found {len(existing_files)} existing files.")
        except OSError as e:
            print(f"Error reading directory {location_path}: {e}") 
            existing_files = set()

        image_iterable = images_to_download.iterrows()
        if not verbose:
            image_iterable = tqdm(
                images_to_download.iterrows(),
                desc=f"Checking images",
                total=len(images_to_download),
                leave=False
            )

        for _, row in image_iterable:
            media_name = row["$name"]
            if not verbose and isinstance(image_iterable, tqdm):
                image_iterable.set_description(f"Checking {media_name[:20]}...")

            if media_name in existing_files:
                v_print(f"  Skipping {media_name}, file already exists.")
                continue

            media_id_to_download = row["$id"]
            out_path = os.path.join(location_path, media_name)

            v_print(f"  Downloading {media_name} (ID: {media_id_to_download})")
            try:
                media = api.get_media(media_id_to_download)
                
                if verbose:
                    for progress in tator.util.download_media(api, media, out_path):
                        if progress % 50 == 0:
                            v_print(f"  Progress at {progress}%")
                else:
                    with tqdm(total=100, desc=f"Downloading {media_name[:20]}...", leave=False, unit='%', unit_scale=True) as pbar:
                        last_progress = 0
                        for progress in tator.util.download_media(api, media, out_path):
                            pbar.update(progress - last_progress)
                            last_progress = progress

                v_print(f"  Successfully downloaded {media_name}")
                existing_files.add(media_name)
            except Exception as e:
                print(f"  ERROR downloading {media_name}: {e}") 

        v_print(f"Finished downloading for: {location}")
        loc_to_pixel[location] = find_representative_lab_color(
            location_path, visualize=visualize
        )
        v_print(f"Representative pixel value: {loc_to_pixel[location]}")

    v_print("Finished processing all locations.")
    return loc_to_pixel
