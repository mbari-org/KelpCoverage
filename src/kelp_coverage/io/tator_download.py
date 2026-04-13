import os
import pandas as pd
import urllib3
import tator
from tator.openapi import tator_openapi
from typing import Dict, Optional, Tuple

from kelp_coverage.io.pixel_analysis import find_representative_lab_color, extract_location

# hacky way to ignore warning
urllib3.disable_warnings()


def download_images_and_get_pixels(
    file_path: str,
    tator_token: str,
    images_dir: str = "images",
    images_per_location: int = -1,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    visualize: bool = False,
) -> Optional[Dict[str, Optional[Tuple[int, int, int]]]]:
    df = pd.read_csv(file_path)
    df["location"] = df["$name"].apply(extract_location)
    filtered_df    = df.dropna(subset=["location"])

    cfg = tator_openapi.Configuration()
    cfg.host       = "https://drone.mbari.org"
    cfg.verify_ssl = False  # MBARI internal server
    if not tator_token:
        print("Warning: No Tator token provided.")
        return None
    cfg.api_key["Authorization"]        = tator_token
    cfg.api_key_prefix["Authorization"] = "Token"
    api = tator_openapi.TatorApi(tator_openapi.ApiClient(cfg))

    loc_to_pixel: Dict[str, Optional[Tuple[int, int, int]]] = {}

    for location, group_df in filtered_df.groupby("location"):
        print(f"Processing location: {location}")
        group_df = group_df.sort_values(by="$id").reset_index(drop=True)

        s = start_idx if start_idx is not None else 0
        e = end_idx   if end_idx   is not None else len(group_df)
        subset_df = group_df.iloc[s:e]

        if images_per_location == -1 or images_per_location >= len(subset_df):
            to_download = subset_df
        else:
            to_download = subset_df.sample(n=images_per_location, replace=False)

        location_path = os.path.join(images_dir, str(location))
        os.makedirs(location_path, exist_ok=True)

        try:
            existing = set(os.listdir(location_path))
        except OSError as e:
            print(f"Error reading {location_path}: {e}")
            existing = set()

        for _, row in to_download.iterrows():
            name = row["$name"]
            if name in existing:
                print(f"  Skipping {name}, already exists.")
                continue
            out_path = os.path.join(location_path, name)
            print(f"  Downloading {name} (ID: {row['$id']})")
            try:
                media = api.get_media(row["$id"])
                for progress in tator.util.download_media(api, media, out_path):
                    if progress % 50 == 0:
                        print(f"  Progress: {progress}%")
                existing.add(name)
            except Exception as ex:
                print(f"  ERROR downloading {name}: {ex}")

        loc_to_pixel[location] = find_representative_lab_color(location_path, visualize=visualize)
        print(f"  Representative pixel: {loc_to_pixel[location]}")

    return loc_to_pixel
