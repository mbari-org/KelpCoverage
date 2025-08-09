import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import os
import json

from shapely.geometry import Polygon
import numpy as np
from typing import Optional


def _create_analysis_grid(
    gdf: gpd.GeoDataFrame, cell_size: int = 50
) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = gdf.total_bounds
    grid = []
    for x in np.arange(minx, maxx, cell_size):
        for y in np.arange(miny, maxy, cell_size):
            grid.append(
                Polygon(
                    [
                        (x, y),
                        (x + cell_size, y),
                        (x + cell_size, y + cell_size),
                        (x, y + cell_size),
                    ]
                )
            )
    return gpd.GeoDataFrame({"geometry": grid}, crs="EPSG:3857")


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
):
    with open(coverage_json, "r") as f:
        data = json.load(f)

    df_coverage = pd.DataFrame(data["results"])

    if df_coverage.empty:
        print(f"No results found in {coverage_json}. Skipping heatmap generation.")
        return

    site_prefix = df_coverage["image_name"].iloc[0].split("_")[1]
    df_coverage["longitude"] = df_coverage["longitude"].abs() * -1
    df_coverage["num_id"] = (
        df_coverage["image_name"]
        .str.split("_")
        .str[-1]
        .str.replace(r"DSC|\.JPG", "", regex=True)
    )

    gdf_pts = gpd.GeoDataFrame(
        df_coverage,
        geometry=gpd.points_from_xy(df_coverage.longitude, df_coverage.latitude),
        crs="EPSG:4326",
    )

    gdf_pts_mercator = gdf_pts.to_crs(epsg=3857)
    gdf_pts_mercator["geometry"] = gdf_pts_mercator.geometry.buffer(18)

    the_grid = _create_analysis_grid(gdf_pts_mercator, cell_size=grid_cell_size)
    the_grid = the_grid.reset_index().rename(columns={"index": "grid_id"})

    intersection = gpd.overlay(the_grid, gdf_pts_mercator, how="intersection")
    intersection["area"] = intersection.geometry.area
    intersection["weighted_cov"] = (
        intersection["coverage_percentage"] * intersection["area"]
    )

    grouped = intersection.groupby("grid_id").agg(
        weighted_cov_sum=("weighted_cov", "sum"), area_sum=("area", "sum")
    )
    weighted_mean = (grouped["weighted_cov_sum"] / grouped["area_sum"]).rename(
        "coverage_percentage"
    )
    grid_final = the_grid.join(weighted_mean, on="grid_id")

    grid_to_plot = grid_final.dropna(subset=["coverage_percentage"])
    if grid_to_plot.empty:
        print(
            f"No data to plot for site prefix '{site_prefix}'. Skipping heatmap generation."
        )
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    grid_to_plot.plot(
        column="coverage_percentage",
        cmap="viridis_r",
        ax=ax,
        legend=False,
        edgecolor="black",
        linewidth=0,
    )

    mappable = ax.collections[0]
    cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.01, shrink=0.9)
    cbar.set_label("Area-Weighted Kelp Coverage %", size=colorbar_fontsize)
    cbar.ax.tick_params(labelsize=colorbar_fontsize)

    gdf_coverage_proj = gdf_pts.to_crs(the_grid.crs)

    if show_points:
        gdf_coverage_proj.plot(ax=ax, marker="o", color="red", markersize=20)

    if show_grid_values:
        for _, row in grid_to_plot.iterrows():
            value_text = f"{row['coverage_percentage']:.2f}"
            centroid = row.geometry.centroid
            ax.text(
                centroid.x,
                centroid.y,
                value_text,
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                bbox=dict(facecolor="black", alpha=0.4, edgecolor="none"),
            )

    if show_point_labels:
        for _, row in gdf_coverage_proj.iterrows():
            plt.annotate(
                text=row["num_id"],
                xy=(row.geometry.x, row.geometry.y),
                xytext=(12, 12),
                textcoords="offset points",
                fontsize=10,
                color="white",
                bbox=dict(facecolor="red", alpha=0.5, edgecolor="none"),
            )

    minx, miny, maxx, maxy = grid_to_plot.total_bounds
    x_buffer = (maxx - minx) * map_buffer_percentage
    y_buffer = (maxy - miny) * map_buffer_percentage
    ax.set_xlim(minx - x_buffer, maxx + x_buffer)
    ax.set_ylim(miny - y_buffer, maxy + y_buffer)

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
