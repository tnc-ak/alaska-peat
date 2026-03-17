#!/usr/bin/env python3
"""Compute and plot yearly burned area on peat soils from fire polygons.

Reads the layer `AK_fire_location_polygons_AKAlbersNAD83` from file geodatabase
and intersects fire polygons with peat raster values (1 or 2) in an Albers raster.

Outputs:
 - CSV: outputs/fire_area_on_peat_by_year.csv
 - PNG plot: outputs/fire_area_on_peat_trend.png

Usage:
 python src/fire_peat_trends.py \
   --gdb data/fire/AlaskaFireHistory_Polygons.gdb \
   --layer AK_fire_location_polygons_AKAlbersNAD83 \
   --peat-raster data/fire/AlbersPeatMap.tif

"""
import argparse
import os

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import rasterio
from rasterstats import zonal_stats


def compute_fire_areas(fire_gdf, peat_raster_path, peat_values=(1, 2)):
    """Return a DataFrame indexed by FIREYEAR with burned areas (ha).
    
    Columns:
    - total_area_ha: total burned area
    - peat_area_ha: burned area on peat
    - peat_proportion: fraction of burned area on peat

    fire_gdf must contain a column named 'FIREYEAR' and geometries in the same CRS
    as the peat raster (the script does not reproject automatically).
    """
    # get raster transform to compute pixel area
    with rasterio.open(peat_raster_path) as src:
        transform = src.transform
        # pixel width and height (may be negative for y)
        px_w = transform.a
        px_h = -transform.e
        pixel_area_m2 = abs(px_w * px_h)

    # Use rasterstats to get counts of peat categories per polygon
    # nodata=0 so non-peat is 0 or other values; categorical returns counts per value
    stats = zonal_stats(
        fire_gdf.geometry,
        peat_raster_path,
        stats=None,
        categorical=True,
        nodata=0,
        all_touched=False,
    )

    # Build DataFrame with peat pixel counts per feature
    peat_counts = []
    for s in stats:
        if not s:
            peat_counts.append(0)
            continue
        cnt = 0
        for v in peat_values:
            cnt += s.get(v, 0)
        peat_counts.append(cnt)

    # attach to fire_gdf and compute areas
    df = fire_gdf.copy()
    df["peat_pixel_count"] = peat_counts
    df["peat_fire_area_ha"] = df["peat_pixel_count"] * (pixel_area_m2 / 10000.0)
    
    # compute total area in hectares from geometry
    df["total_fire_area_ha"] = df.geometry.area / 10000.0  # convert m² to ha

    # Clean FIREYEAR
    # ensure FIREYEAR numeric, drop invalid
    if "FIREYEAR" not in df.columns:
        raise KeyError("Input fire layer has no 'FIREYEAR' field")

    df["year"] = pd.to_numeric(df["FIREYEAR"], errors="coerce").astype(pd.Int64Dtype())
    df = df[df["year"].notna()]
    
    # Group by year and compute metrics
    summary = df.groupby("year").agg({
        "total_fire_area_ha": "sum",
        "peat_fire_area_ha": "sum"
    })
    summary["peat_proportion"] = summary["peat_fire_area_ha"] / summary["total_fire_area_ha"]
    
    return summary.sort_index()


def plot_trends(df, out_png_base, show=False):
    """Plot yearly burned area trends and save to PNG files.
    
    Creates three plots:
    - total and peat burned areas
    - proportion of burned area on peat
    - just peat burned area with trend (original plot)
    """
    # 1. Combined total and peat area plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df[["total_fire_area_ha", "peat_fire_area_ha"]].plot(ax=ax, marker="o", linestyle="-")
    ax.set_xlabel("Year")
    ax.set_ylabel("Burned area (ha)")
    ax.set_title("Yearly burned area: total vs peat")
    ax.grid(True, linestyle=":")
    ax.legend(["Total burned area", "Burned peat area"])
    
    fig.tight_layout()
    fig.savefig(f"{out_png_base}_areas.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    
    # 2. Proportion plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df["peat_proportion"].plot(ax=ax, marker="o", linestyle="-", color="darkgreen")
    ax.set_xlabel("Year")
    ax.set_ylabel("Proportion")
    ax.set_title("Yearly proportion of burned area on peat")
    ax.grid(True, linestyle=":")
    
    fig.tight_layout()
    fig.savefig(f"{out_png_base}_proportion.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    
    # 3. Original peat area plot with trend
    fig, ax = plt.subplots(figsize=(12, 6))
    df["peat_fire_area_ha"].plot(ax=ax, marker="o", linestyle="-")
    ax.set_xlabel("Year")
    ax.set_ylabel("Burned area on peat (ha)")
    ax.set_title("Yearly burned area on peat soils")
    ax.grid(True, linestyle=":")

    # add simple linear trend
    try:
        import numpy as np

        x = df.index.astype(int).to_numpy()
        y = df["peat_area_ha"].to_numpy()
        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            ax.plot(x, m * x + b, color="red", linestyle="--", 
                   label=f"trend: {m:.1f} ha/yr")
            ax.legend()
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(f"{out_png_base}.png", dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute fire area on peat soils by year")
    parser.add_argument("--gdb", default="data/fire/AlaskaFireHistory_Polygons.gdb", help="Path to file geodatabase")
    parser.add_argument("--layer", default="AK_fire_location_polygons_AKAlbersNAD83", help="Layer name inside the geodatabase")
    parser.add_argument("--peat-raster", default="data/fire/AlbersPeatMap.tif", help="Peat raster where values 1 or 2 indicate wet/moist peat")
    parser.add_argument("--out-csv", default="outputs/fire_perimeters/fire_area_on_peat_by_year.csv", help="Output CSV path")
    parser.add_argument("--out-png", default="outputs/exploratory/fire_perimeters/fire_area_on_peat_trend.png", help="Output plot PNG path")
    parser.add_argument("--show", action="store_true", help="Show the plot interactively")

    args = parser.parse_args()

    # read fire layer
    if not os.path.exists(args.gdb):
        raise FileNotFoundError(f"GDB not found: {args.gdb}")

    print(f"Reading fire layer '{args.layer}' from {args.gdb}...")
    fire_gdf = gpd.read_file(args.gdb, layer=args.layer)
    print(f"Loaded {len(fire_gdf)} features")

    if not os.path.exists(args.peat_raster):
        raise FileNotFoundError(f"Peat raster not found: {args.peat_raster}")

    # NOTE: ensure CRS alignment — expect both in Albers NAD83 (projected meters)
    # If CRSs differ, try to reproject the GeoDataFrame to raster CRS
    with rasterio.open(args.peat_raster) as src:
        raster_crs = src.crs

    if fire_gdf.crs is None:
        raise ValueError("Fire layer has no CRS; cannot align with peat raster")
    if raster_crs is None:
        raise ValueError("Peat raster has no CRS; cannot align with fire polygons")

    if fire_gdf.crs != raster_crs:
        print("Reprojecting fire geometries to peat raster CRS...")
        fire_gdf = fire_gdf.to_crs(raster_crs)

    # compute
    summary = compute_fire_areas(fire_gdf, args.peat_raster)

    # Save CSVs
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    
    # Full data with all metrics
    summary.reset_index().to_csv(args.out_csv, index=False)
    print(f"Wrote full data CSV to {args.out_csv}")

    # plot all versions
    out_png_base = args.out_png.replace(".png", "")
    plot_trends(summary, out_png_base, show=args.show)
    print(f"Wrote plot variations to {out_png_base}_*.png")


if __name__ == "__main__":
    main()
