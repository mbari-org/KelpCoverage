from tator.openapi import tator_openapi
import tator
import os
import pandas as pd
import urllib3
from typing import Dict, Tuple, Optional

from pixel_analysis import find_representative_lab_color, extract_location

# hacky way to get rid of error msg for now
urllib3.disable_warnings()

def download_images_and_get_pixels(
    file_path: str,
    images_dir: str = "images",
    images_per_location: int = -1,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    visualize: bool = False
) -> Dict[str, Optional[Tuple[int, int, int]]]:

    df = pd.read_csv(file_path)
    df['location'] = df['$name'].apply(extract_location)
    filtered_df = df.dropna(subset=['location'])
    grouped_df = filtered_df.groupby('location')
    host = "https://drone.mbari.org"
    token = ""
    config = tator_openapi.Configuration()
    config.host = host
    config.verify_ssl = False
    if token:
        config.api_key['Authorization'] = token
        config.api_key_prefix['Authorization'] = 'Token'
    api = tator_openapi.TatorApi(tator_openapi.ApiClient(config))
    loc_to_pixel: Dict[str, Optional[Tuple[int, int, int]]] = {}

    for location, group_df in grouped_df:
        print(f"Processing location: {location}")
        group_df = group_df.sort_values(by='$id').reset_index(drop=True)
        
        s_idx = start_idx if start_idx is not None else 0
        e_idx = end_idx if end_idx is not None else len(group_df)
        subset_df = group_df.iloc[s_idx:e_idx]
        
        if images_per_location == -1 or images_per_location >= len(subset_df):
            images_to_download = subset_df
        else:
            images_to_download = subset_df.sample(n=images_per_location, replace=False)
            
        location_path = os.path.join(images_dir, str(location))
        os.makedirs(location_path, exist_ok=True)
        print(f"Downloading to {location_path}")
        
        for _, row in images_to_download.iterrows():
            media_id_to_download = row['$id']
            media_name = row['$name']
            out_path = os.path.join(location_path, media_name)
            
            if os.path.exists(out_path):
                print(f"  Skipping {media_name}, already exists.")
                continue
                
            print(f"  Downloading {media_name} (ID: {media_id_to_download})")
            media = api.get_media(media_id_to_download)
            for progress in tator.util.download_media(api, media, out_path):
                if progress % 50 == 0:
                      print(f"  Progress at {progress}%")
            print(f"  Successfully downloaded {media_name}")
            
        print(f"Finished downloading for: {location}")
        loc_to_pixel[location] = find_representative_lab_color(location_path, visualize=visualize)
        print(f"Representative pixel value: {loc_to_pixel[location]}")

    print("Finished processing all locations.")
    return loc_to_pixel

