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

    pbar = None
    if not verbose:
        total_image_count = 0
        df_for_count = filtered_df.groupby("location")
        for location, group_df in df_for_count:
            group_df_sorted = group_df.sort_values(by="$id").reset_index(drop=True)
            s_idx = start_idx if start_idx is not None else 0
            e_idx = end_idx if end_idx is not None else len(group_df_sorted)
            subset_df = group_df_sorted.iloc[s_idx:e_idx]

            if images_per_location == -1 or images_per_location >= len(subset_df):
                num_images = len(subset_df)
            else:
                num_images = images_per_location
            total_image_count += num_images
        
        pbar = tqdm(total=total_image_count, desc="Processing all images", unit="img")

    for location, group_df in grouped_df:
        if verbose:
            print(f"Processing location: {location}")
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
        if verbose:
            print(f"Checking for existing images in {location_path}")

        try:
            existing_files = set(os.listdir(location_path))
            if existing_files and verbose:
                print(f"Found {len(existing_files)} existing files.")
        except OSError as e:
            print(f"Error reading directory {location_path}: {e}")
            existing_files = set()

        for _, row in images_to_download.iterrows():
            media_name = row["$name"]

            if media_name in existing_files:
                if verbose:
                    print(f"  Skipping {media_name}, file already exists.")
                if pbar:
                    pbar.update(1)
                continue

            media_id_to_download = row["$id"]
            out_path = os.path.join(location_path, media_name)

            if verbose:
                print(f"  Downloading {media_name} (ID: {media_id_to_download})")
            try:
                media = api.get_media(media_id_to_download)
                
                if verbose:
                    for progress in tator.util.download_media(api, media, out_path):
                        if progress % 50 == 0:
                            print(f"  Progress at {progress}%")
                else:
                    for _ in tator.util.download_media(api, media, out_path):
                        pass
                
                if verbose:
                    print(f"  Successfully downloaded {media_name}")
                existing_files.add(media_name)
            except Exception as e:
                print(f"  ERROR downloading {media_name}: {e}")

            if pbar:
                pbar.update(1)

        if verbose:
            print(f"Finished downloading for: {location}")
        loc_to_pixel[location] = find_representative_lab_color(
            location_path, visualize=visualize
        )
        if verbose:
            print(f"Representative pixel value: {loc_to_pixel[location]}")

    if pbar:
        pbar.close()
    
    if verbose:
        print("Finished processing all locations.")
    return loc_to_pixel
