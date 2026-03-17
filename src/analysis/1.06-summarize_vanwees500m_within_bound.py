"""
1.06 - Summarize vanWees 500 m emissions & burned area within Alaska.

Reads the per-year Alaska tiles extracted by 1.05 and aggregates:
  - C_AG_TOT  (aboveground carbon emissions, g C m⁻² month⁻¹)
  - C_BG_TOT  (belowground carbon emissions, g C m⁻² month⁻¹)
  - BA_TOT    (burned-area fraction of grid cell month⁻¹)

For each tile the Alaska boundary is rasterised onto the sinusoidal grid
to produce a pixel mask.  Emissions are multiplied by pixel area (m²) to
give total grams of C; burned-area fractions are likewise multiplied by
pixel area to give m².

Results are written as a wide CSV with columns:
    year | C_AG_TOT | C_BG_TOT | BA_TOT_M2

Usage:
    python src/analysis/1.06-summarize_vanwees_500m_within_bound.py
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import netCDF4
import geopandas as gpd
from rasterio.features import geometry_mask


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TILE_DIR = os.path.join("data", "fire", "vanWees", "500m_alaska")
BOUNDARY_PATH = os.path.join("data", "alaska_buffered.shp")
OUTPUT_FILE = os.path.join("outputs", "vanwees_500m_emissions_ba_in_alaska.csv")

EMISSION_VARS = ["C_AG_TOT", "C_BG_TOT"]
BA_VARS = ["BA_TOT"]

TILE_PATTERN = re.compile(r"_(h\d{2}v\d{2})_")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _read_tile_meta(nc_path):
    """Return (proj4, GeoTransform, ny, nx) from a 500 m tile."""
    nc = netCDF4.Dataset(nc_path, "r")
    grid = nc.groups["MOD_Grid"]
    crs_var = grid.variables["_HDFEOS_CRS"]
    proj4 = str(crs_var.proj4).strip()
    geo = crs_var.GeoTransform  # [x_origin, dx, 0, y_origin, 0, dy]
    ny = len(grid.dimensions["YDim"])
    nx = len(grid.dimensions["XDim"])
    nc.close()
    return proj4, geo, ny, nx


def _build_mask(nc_path, boundary_gdf):
    """
    Build a boolean pixel mask (True = inside Alaska) for a 500 m tile.

    The Alaska boundary is reprojected into the tile's sinusoidal CRS and
    then rasterised at the tile's resolution.
    """
    proj4, geo, ny, nx = _read_tile_meta(nc_path)

    # GeoTransform: [x_origin, pixel_width, 0, y_origin, 0, pixel_height]
    x_origin = float(geo[0])
    dx = float(geo[1])
    y_origin = float(geo[3])
    dy = float(geo[5])  # negative

    # Affine transform for rasterio (same convention as GDAL GeoTransform)
    from rasterio.transform import Affine

    transform = Affine(dx, 0.0, x_origin, 0.0, dy, y_origin)

    # Reproject boundary into the tile's sinusoidal CRS
    boundary_sinu = boundary_gdf.to_crs(proj4)

    mask = geometry_mask(
        boundary_sinu.geometry,
        transform=transform,
        invert=True,  # True where geometry is
        out_shape=(ny, nx),
    )
    return mask, transform, dx, dy


def summarize_vanwees_500m(boundary_path, output_file):
    """
    Summarize vanWees 500 m carbon emissions and burned area within the
    Alaska boundary across all extracted tiles and years.
    """
    boundary_gdf = gpd.read_file(boundary_path)
    print(f"Loaded boundary: {boundary_path}")

    # Discover year directories
    year_dirs = sorted(glob.glob(os.path.join(TILE_DIR, "[0-9][0-9][0-9][0-9]")))
    if not year_dirs:
        raise FileNotFoundError(
            f"No year directories found in {TILE_DIR}. "
            "Run 1.05-extract_alaska_tiles_from_zips.py first."
        )
    print(f"Found {len(year_dirs)} year directories in {TILE_DIR}")

    # ----- Pre-compute per-tile masks (same grid every year) --------
    # Use the first year's tiles as reference
    ref_tiles = sorted(glob.glob(os.path.join(year_dirs[0], "*.nc")))
    tile_ids = []
    tile_masks = {}  # tile_id -> (mask, pixel_area_m2)

    print("\nPre-computing Alaska masks for each tile …")
    for nc_path in ref_tiles:
        fname = os.path.basename(nc_path)
        m = TILE_PATTERN.search(fname)
        if not m:
            continue
        tile_id = m.group(1)
        tile_ids.append(tile_id)

        mask, transform, dx, dy = _build_mask(nc_path, boundary_gdf)
        pixel_area_m2 = abs(dx * dy)  # m²
        n_inside = int(mask.sum())
        tile_masks[tile_id] = (mask, pixel_area_m2)
        print(
            f"  {tile_id}: {n_inside:,} pixels inside Alaska "
            f"(pixel area = {pixel_area_m2:,.1f} m²)"
        )

    # ----- Process each year ----------------------------------------
    results = []

    for year_dir in year_dirs:
        year_label = os.path.basename(year_dir)
        nc_files = sorted(glob.glob(os.path.join(year_dir, "*.nc")))
        print(f"\nProcessing {year_label} ({len(nc_files)} tiles) …")

        for nc_path in nc_files:
            fname = os.path.basename(nc_path)
            m = TILE_PATTERN.search(fname)
            if not m:
                continue
            tile_id = m.group(1)

            if tile_id not in tile_masks:
                print(f"  Warning: no mask for {tile_id}, skipping.")
                continue

            mask, pixel_area_m2 = tile_masks[tile_id]

            nc = netCDF4.Dataset(nc_path, "r")
            grid = nc.groups["MOD_Grid"]
            time_var = grid.variables["time"]
            time_units = time_var.units
            time_cal = getattr(time_var, "calendar", "standard")
            dates = netCDF4.num2date(time_var[:], time_units, time_cal)
            n_time = len(dates)

            em_grp = grid.groups["emissions"]
            ba_grp = grid.groups["burned_area"]

            for t in range(n_time):
                year = dates[t].year
                month = dates[t].month

                # --- emissions (g C m⁻² month⁻¹ → total g C) ---
                for var_name in EMISSION_VARS:
                    arr = em_grp.variables[var_name][t, :, :]  # (YDim, XDim)
                    arr = np.where(mask, arr, 0.0)
                    total_gc = float(np.nansum(arr) * pixel_area_m2)
                    results.append(
                        {
                            "year": year,
                            "month": month,
                            "variable": var_name,
                            "value": total_gc,
                        }
                    )

                # --- burned area (fraction → m²) ---
                for var_name in BA_VARS:
                    arr = ba_grp.variables[var_name][t, :, :]
                    arr = np.where(mask, arr, 0.0)
                    total_m2 = float(np.nansum(arr) * pixel_area_m2)
                    results.append(
                        {
                            "year": year,
                            "month": month,
                            "variable": f"{var_name}_M2",
                            "value": total_m2,
                        }
                    )

            nc.close()

    # ----- Aggregate and save ---------------------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.pivot_table(
        index="year",
        columns="variable",
        values="value",
        aggfunc="sum",
    ).reset_index()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(results_df.to_string(index=False))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summarize_vanwees_500m(BOUNDARY_PATH, OUTPUT_FILE)
