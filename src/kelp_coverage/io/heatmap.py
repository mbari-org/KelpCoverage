import json
import os

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from typing import Optional


def _create_analysis_grid(gdf: gpd.GeoDataFrame, cell_size: int = 50) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gdf.total_bounds
    cells = [
        Polygon([(x, y), (x + cell_size, y), (x + cell_size, y + cell_size), (x, y + cell_size)])
        for x in np.arange(minx, maxx, cell_size)
        for y in np.arange(miny, maxy, cell_size)
    ]
    return gpd.GeoDataFrame({"geometry": cells}, crs="EPSG:3857")


def generate_heatmap(
    coverage_json: str,
    output_path: Optional[str] = None,
    grid_cell_size: int = 30,
    figsize: tuple = (20, 20),
    show_grid_values: bool = True,
    show_points: bool = True,
    show_point_labels: bool = True,
    map_buffer_percentage: float = 0.1,
    colorbar_fontsize: int = 30,
    title_fontsize: int = 50,
) -> None:
    with open(coverage_json, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data["results"])
    if df.empty:
        print(f"No results in {coverage_json}. Skipping heatmap.")
        return

    site_prefix = df["image_name"].iloc[0].split("_")[1]

    df["longitude"] = df["longitude"].abs() * -1

    df["num_id"] = (
        df["image_name"].str.split("_").str[-1].str.replace(r"DSC|\.JPG", "", regex=True)
    )

    gdf_pts = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.longitude, df.latitude),
        crs="EPSG:4326",
    )
    gdf_merc = gdf_pts.to_crs(epsg=3857)
    gdf_merc["geometry"] = gdf_merc.geometry.buffer(18)

    grid = _create_analysis_grid(gdf_merc, cell_size=grid_cell_size)
    grid = grid.reset_index().rename(columns={"index": "grid_id"})

    intersection = gpd.overlay(grid, gdf_merc, how="intersection")
    intersection["area"]         = intersection.geometry.area
    intersection["weighted_cov"] = intersection["coverage_percentage"] * intersection["area"]

    grouped      = intersection.groupby("grid_id").agg(
        wc_sum=("weighted_cov", "sum"), area_sum=("area", "sum")
    )
    weighted_mean = (grouped["wc_sum"] / grouped["area_sum"]).rename("coverage_percentage")
    grid_final    = grid.join(weighted_mean, on="grid_id")
    grid_to_plot  = grid_final.dropna(subset=["coverage_percentage"])

    if grid_to_plot.empty:
        print(f"No data to plot for '{site_prefix}'. Skipping.")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    grid_to_plot.plot(
        column="coverage_percentage", cmap="viridis_r", ax=ax,
        legend=False, edgecolor="black", linewidth=0,
    )

    cbar = fig.colorbar(ax.collections[0], ax=ax, orientation="horizontal", pad=0.01, shrink=0.9)
    cbar.set_label("Area-Weighted Kelp Coverage %", size=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    gdf_proj = gdf_pts.to_crs(grid.crs)

    if show_points:
        gdf_proj.plot(ax=ax, marker="o", color="red", markersize=20)

    if show_grid_values:
        for _, row in grid_to_plot.iterrows():
            c = row.geometry.centroid
            ax.text(c.x, c.y, f"{row['coverage_percentage']:.2f}",
                    ha="center", va="center", color="white", fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"))

    if show_point_labels:
        for _, row in gdf_proj.iterrows():
            plt.annotate(text=row["num_id"],
                         xy=(row.geometry.x, row.geometry.y),
                         xytext=(12, 12), textcoords="offset points",
                         fontsize=10, color="white",
                         bbox=dict(facecolor="red", alpha=0.5, edgecolor="none"))

    minx, miny, maxx, maxy = grid_to_plot.total_bounds
    xb = (maxx - minx) * map_buffer_percentage
    yb = (maxy - miny) * map_buffer_percentage
    ax.set_xlim(minx - xb, maxx + xb)
    ax.set_ylim(miny - yb, maxy + yb)

    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
    ax.set_axis_off()
    ax.set_title(f"{site_prefix} Heatmap", fontsize=title_fontsize)

    if not output_path:
        default_dir = os.path.join("results", "heatmap")
        os.makedirs(default_dir, exist_ok=True)
        output_path = os.path.join(default_dir, f"{site_prefix}_heatmap.png")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Heatmap saved to {output_path}")
    plt.close(fig)
