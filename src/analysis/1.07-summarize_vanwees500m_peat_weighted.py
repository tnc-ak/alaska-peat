"""
1.07 - Summarize vanWees 500 m emissions & burned area on peatlands in Alaska.

Same structure as 1.06, but every vanWees pixel is weighted by its
peat-cover fraction before aggregation.

The peat fraction is derived from the Lara peat map
(data/fire/AlbersPeatMap.tif — 20 m, EPSG:3338) in two steps:

  1. **Pre-aggregate** the 20 m binary peat raster (values 1|2 → 1, else 0)
     to 500 m resolution in its native EPSG:3338 projection by block-by-block
     averaging.  Because the average of binary values *is* the proportion,
     this preserves exact peat fractions from the full 20 m data — no
     downsampling artefacts.  The result is a small temporary GeoTIFF.

  2. **Reproject** the 500 m peat-fraction raster onto each vanWees
     sinusoidal tile grid using ``rasterio.warp.reproject`` with bilinear
     resampling (source and target are both ~500 m, so this is purely a CRS
     conversion).  This avoids the PROJ crash that occurs when
     ``transform_bounds`` is called on sinusoidal tiles that span near-polar
     extents.

Peat is defined as pixel values 1 or 2 in the source raster; 0 = non-peat,
NaN = no data.

Results are written as a wide CSV with columns:
    year | C_AG_TOT | C_BG_TOT | BA_TOT_M2

Usage:
    python src/analysis/1.07-summarize_vanwees_500m_peat_weighted.py
"""

import os
import re
import glob
import tempfile

import numpy as np
import pandas as pd
import netCDF4
import rasterio
import rasterio.windows
from rasterio.transform import Affine
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask
import geopandas as gpd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TILE_DIR = os.path.join("data", "fire", "vanWees", "500m_alaska")
BOUNDARY_PATH = os.path.join("data", "alaska_buffered.shp")
PEAT_PATH = os.path.join("data", "peat", "AlbersPeatMap.tif")
OUTPUT_FILE = os.path.join("outputs", "vanwees_500m_peat_emissions_ba_in_alaska.csv")

EMISSION_VARS = ["C_AG_TOT", "C_BG_TOT"]
BA_VARS = ["BA_TOT"]

TILE_PATTERN = re.compile(r"_(h\d{2}v\d{2})_")

# Target resolution (m) for the pre-aggregated peat fraction raster.
# Must be a multiple of the 20 m source resolution.
PEAT_AGG_RES = 500


# ---------------------------------------------------------------------------
# Helper: read tile metadata
# ---------------------------------------------------------------------------


def _read_tile_meta(nc_path):
    """Return (proj4, GeoTransform array, ny, nx) from a 500 m tile."""
    nc = netCDF4.Dataset(nc_path, "r")
    grid = nc.groups["MOD_Grid"]
    crs_var = grid.variables["_HDFEOS_CRS"]
    proj4 = str(crs_var.proj4).strip()
    geo = crs_var.GeoTransform  # [x_origin, dx, 0, y_origin, 0, dy]
    ny = len(grid.dimensions["YDim"])
    nx = len(grid.dimensions["XDim"])
    nc.close()
    return proj4, geo, ny, nx


def _tile_transform(nc_path):
    """Return (Affine, dx, dy, ny, nx, proj4) for a vanWees tile."""
    proj4, geo, ny, nx = _read_tile_meta(nc_path)
    dx = float(geo[1])
    dy = float(geo[5])  # negative
    transform = Affine(dx, 0.0, float(geo[0]), 0.0, dy, float(geo[3]))
    return transform, dx, dy, ny, nx, proj4


# ---------------------------------------------------------------------------
# Helper: Alaska mask
# ---------------------------------------------------------------------------


def _build_alaska_mask(nc_path, boundary_gdf):
    """Boolean mask (True = inside Alaska) for one tile."""
    transform, dx, dy, ny, nx, proj4 = _tile_transform(nc_path)
    boundary_sinu = boundary_gdf.to_crs(proj4)
    mask = geometry_mask(
        boundary_sinu.geometry,
        transform=transform,
        invert=True,
        out_shape=(ny, nx),
    )
    return mask, transform, dx, dy, ny, nx, proj4


# ---------------------------------------------------------------------------
# Helper: pre-aggregate 20 m peat to 500 m peat fraction
# ---------------------------------------------------------------------------


def _preagg_peat_raster(peat_path, target_res=PEAT_AGG_RES):
    """
    Read the 20 m peat raster block-by-block, reclassify to binary
    (peat = 1, non-peat = 0), and average into a coarser grid whose pixel
    size is *target_res* metres.

    Returns the path to a temporary GeoTIFF (float32, values in [0, 1])
    in the source CRS (EPSG:3338).
    """
    with rasterio.open(peat_path) as src:
        src_res = src.res[0]  # 20 m
        factor = int(target_res / src_res)
        if factor < 1:
            raise ValueError(
                f"target_res ({target_res}) must be >= source res ({src_res})"
            )

        out_height = src.height // factor
        out_width = src.width // factor
        out_transform = Affine(
            src.transform.a * factor,
            src.transform.b,
            src.transform.c,
            src.transform.d,
            src.transform.e * factor,
            src.transform.f,
        )

        tmp_path = os.path.join(tempfile.gettempdir(), "peat_frac_500m.tif")
        with rasterio.open(
            tmp_path,
            "w",
            driver="GTiff",
            height=out_height,
            width=out_width,
            count=1,
            dtype="float32",
            crs=src.crs,
            transform=out_transform,
            nodata=0.0,
            compress="lzw",
        ) as dst:
            # Process in blocks of BLOCK_SIZE output rows
            BLOCK_SIZE = 200
            for out_row in range(0, out_height, BLOCK_SIZE):
                n_out = min(BLOCK_SIZE, out_height - out_row)
                src_row = out_row * factor
                n_src = min(n_out * factor, src.height - src_row)
                if n_src <= 0:
                    continue

                window = rasterio.windows.Window(0, src_row, src.width, n_src)
                data = src.read(1, window=window).astype(np.float32)

                # Binary: peat (1 or 2) → 1.0, else 0.0
                binary = np.where((data >= 1) & (data <= 2), 1.0, 0.0)

                # Average into coarser grid
                usable_rows = (n_src // factor) * factor
                usable_cols = (src.width // factor) * factor
                block = binary[:usable_rows, :usable_cols]
                block = block.reshape(
                    usable_rows // factor,
                    factor,
                    usable_cols // factor,
                    factor,
                )
                avg = block.mean(axis=(1, 3)).astype(np.float32)

                out_win = rasterio.windows.Window(0, out_row, out_width, avg.shape[0])
                dst.write(avg, 1, window=out_win)

    return tmp_path


# ---------------------------------------------------------------------------
# Helper: reproject peat fraction onto a vanWees sinusoidal tile
# ---------------------------------------------------------------------------


def _reproject_peat_to_tile(peat_500m_path, transform, ny, nx, proj4):
    """
    Reproject the pre-aggregated 500 m peat-fraction raster (EPSG:3338) onto
    one vanWees sinusoidal tile grid.  Returns a float32 array (ny, nx) with
    values in [0, 1].
    """
    dst_array = np.zeros((1, ny, nx), dtype=np.float32)
    dst_crs = CRS.from_proj4(proj4)

    with rasterio.open(peat_500m_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.bilinear,
            src_nodata=0.0,
            dst_nodata=0.0,
        )

    result = dst_array[0]
    np.clip(result, 0.0, 1.0, out=result)
    return result


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------


def summarize_vanwees_500m_peat(boundary_path, peat_path, output_file):
    """
    Summarize vanWees 500 m carbon emissions and burned area on peatlands
    within Alaska, weighting each pixel by its peat-cover fraction.
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

    # ------------------------------------------------------------------
    # Step 1: Pre-aggregate peat raster (one-time)
    # ------------------------------------------------------------------
    print(
        f"\nStep 1: Pre-aggregating {PEAT_AGG_RES} m peat fraction from 20 m raster …"
    )
    peat_500m_path = _preagg_peat_raster(peat_path, target_res=PEAT_AGG_RES)
    print(f"  Temp peat-fraction raster: {peat_500m_path}")

    # ------------------------------------------------------------------
    # Step 2: Pre-compute per-tile Alaska mask + peat fraction
    # ------------------------------------------------------------------
    ref_tiles = sorted(glob.glob(os.path.join(year_dirs[0], "*.nc")))
    tile_data = {}  # tile_id -> (weight, pixel_area_m2)

    print("\nStep 2: Pre-computing Alaska masks and peat fractions …")
    for nc_path in ref_tiles:
        fname = os.path.basename(nc_path)
        m = TILE_PATTERN.search(fname)
        if not m:
            continue
        tile_id = m.group(1)

        mask, transform, dx, dy, ny, nx, proj4 = _build_alaska_mask(
            nc_path, boundary_gdf
        )
        pixel_area_m2 = abs(dx * dy)

        peat_frac = _reproject_peat_to_tile(peat_500m_path, transform, ny, nx, proj4)

        # Combined weight: Alaska mask × peat fraction
        weight = np.where(mask, peat_frac, 0.0).astype(np.float32)
        n_ak = int(mask.sum())
        n_peat = int((weight > 0).sum())
        mean_frac = float(weight[mask].mean()) if n_ak > 0 else 0.0

        tile_data[tile_id] = (weight, pixel_area_m2)
        print(
            f"  {tile_id}: {n_ak:>8,} AK pixels, "
            f"{n_peat:>8,} with peat, "
            f"mean peat frac = {mean_frac:.4f}"
        )

    # Clean up temp raster
    try:
        os.remove(peat_500m_path)
    except OSError:
        pass

    # ------------------------------------------------------------------
    # Step 3: Process each year
    # ------------------------------------------------------------------
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

            if tile_id not in tile_data:
                continue

            weight, pixel_area_m2 = tile_data[tile_id]

            # Skip tiles with zero peat overlap
            if weight.max() == 0:
                continue

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

                # --- emissions (g C m⁻² month⁻¹) × peat_frac × pixel_area → g C
                for var_name in EMISSION_VARS:
                    arr = em_grp.variables[var_name][t, :, :]
                    total_gc = float(np.nansum(arr * weight) * pixel_area_m2)
                    results.append(
                        {
                            "year": year,
                            "month": month,
                            "variable": var_name,
                            "value": total_gc,
                        }
                    )

                # --- burned area (fraction) × peat_frac × pixel_area → m²
                for var_name in BA_VARS:
                    arr = ba_grp.variables[var_name][t, :, :]
                    total_m2 = float(np.nansum(arr * weight) * pixel_area_m2)
                    results.append(
                        {
                            "year": year,
                            "month": month,
                            "variable": f"{var_name}_M2",
                            "value": total_m2,
                        }
                    )

            nc.close()

    # ------------------------------------------------------------------
    # Step 4: Aggregate and save
    # ------------------------------------------------------------------
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
    summarize_vanwees_500m_peat(BOUNDARY_PATH, PEAT_PATH, OUTPUT_FILE)
