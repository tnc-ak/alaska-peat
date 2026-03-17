"""
1.05 - Extract Alaska-overlapping tiles from vanWees 500m zip archives.

This script:
1. Takes a directory of .zip files, each containing .nc tiles for one year.
2. Determines which tiles (identified by h##v## in filenames) overlap with the
   Alaska boundary shapefile (data/alaska_buffered.shp).
3. Extracts only the overlapping tiles from each zip into per-year folders.

Tile identification is done once using the first zip file and reused for all
subsequent years, since tile numbering is consistent across years.

Usage:
    python src/analysis/1.05-extract_alaska_tiles_from_zips.py
"""

import os
import re
import zipfile
import glob
import shutil

import netCDF4
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import Polygon


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZIP_DIR = os.path.join("data", "fire", "vanWees", "500m")
BOUNDARY_PATH = os.path.join("data", "alaska_buffered.shp")
OUTPUT_DIR = os.path.join("data", "fire", "vanWees", "500m_alaska")
PLOT_DIR = os.path.join("outputs", "exploratory", "tile_footprints")

TILE_PATTERN = re.compile(r"_(h\d{2}v\d{2})_")

# Alaska Albers Equal Area Conic – minimises distortion for statewide mapping
ALASKA_ALBERS = "EPSG:3338"


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def parse_tile_id(filename):
    """Extract the tile id (e.g. 'h06v03') from a filename."""
    match = TILE_PATTERN.search(filename)
    if match:
        return match.group(1)
    return None


def get_nc_bounding_box(nc_path, target_crs=ALASKA_ALBERS, n_edge_pts=64):
    """
    Open a .nc file and return its footprint as a Shapely polygon in
    *target_crs* (default: Alaska Albers EPSG:3338).

    Reads the MOD_Grid group's ``_HDFEOS_CRS`` variable which stores
    ``UpperLeftPointMtrs`` and ``LowerRightMtrs`` (in metres, MODIS
    sinusoidal projection) as well as a ``proj4`` string.

    Because a simple rectangular box in sinusoidal coordinates becomes
    heavily distorted when reprojected (especially at high latitudes),
    the four edges are densified with *n_edge_pts* intermediate vertices
    before reprojection.
    """
    nc = netCDF4.Dataset(nc_path, "r")
    try:
        grid = nc.groups["MOD_Grid"]
        crs_var = grid.variables["_HDFEOS_CRS"]

        # Upper-left and lower-right corners in metres (sinusoidal)
        ul = crs_var.UpperLeftPointMtrs  # [x_min, y_max]
        lr = crs_var.LowerRightMtrs  # [x_max, y_min]
        proj4 = str(crs_var.proj4).strip()

        x_min, y_max = float(ul[0]), float(ul[1])
        x_max, y_min = float(lr[0]), float(lr[1])
    finally:
        nc.close()

    # Densify each edge so the reprojection accurately curves the outline
    bottom = np.column_stack(
        [np.linspace(x_min, x_max, n_edge_pts), np.full(n_edge_pts, y_min)]
    )
    right = np.column_stack(
        [np.full(n_edge_pts, x_max), np.linspace(y_min, y_max, n_edge_pts)]
    )
    top = np.column_stack(
        [np.linspace(x_max, x_min, n_edge_pts), np.full(n_edge_pts, y_max)]
    )
    left = np.column_stack(
        [np.full(n_edge_pts, x_min), np.linspace(y_max, y_min, n_edge_pts)]
    )
    coords = np.vstack([bottom, right, top, left])

    tile_geom = Polygon(coords)
    tile_gdf = gpd.GeoDataFrame(geometry=[tile_geom], crs=proj4)
    tile_gdf = tile_gdf.to_crs(target_crs)
    return tile_gdf.geometry.values[0]


def find_alaska_tiles(nc_dir, boundary_gdf):
    """
    Given a directory of .nc tile files and a boundary GeoDataFrame,
    return the set of tile ids (e.g. {'h06v03', 'h07v03'}) whose footprints
    intersect the boundary.

    All geometry operations are performed in Alaska Albers (EPSG:3338) to
    minimise distortion.
    """
    boundary_union = boundary_gdf.to_crs(ALASKA_ALBERS).union_all()

    nc_files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"No .nc files found in {nc_dir}")

    tile_boxes = {}  # tile_id -> shapely geometry (EPSG:3338)
    for nc_file in nc_files:
        tile_id = parse_tile_id(os.path.basename(nc_file))
        if tile_id is None:
            continue
        try:
            bbox = get_nc_bounding_box(nc_file, target_crs=ALASKA_ALBERS)
            tile_boxes[tile_id] = bbox
        except Exception as e:
            print(f"  Warning: could not read bbox for {nc_file}: {e}")

    intersecting = set()
    for tile_id, geom in tile_boxes.items():
        if geom.intersects(boundary_union):
            intersecting.add(tile_id)

    return intersecting, tile_boxes


def plot_tile_footprints(tile_boxes, alaska_tiles, boundary_gdf, save_path=None):
    """
    Plot all tile footprints, highlighting those that overlap Alaska.

    All geometries are plotted in Alaska Albers (EPSG:3338).

    Parameters
    ----------
    tile_boxes : dict
        {tile_id: shapely geometry in EPSG:3338}
    alaska_tiles : set
        Tile ids that intersect the Alaska boundary.
    boundary_gdf : GeoDataFrame
        Alaska boundary (will be reprojected to EPSG:3338 for plotting).
    save_path : str, optional
        If provided, save figure to this path.
    """
    boundary_ak = boundary_gdf.to_crs(ALASKA_ALBERS)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Only plot tiles that intersect Alaska (non-intersecting ones are far
    # away and would dominate the extent of the plot)
    for tile_id, geom in tile_boxes.items():
        if tile_id not in alaska_tiles:
            continue
        gpd.GeoSeries([geom], crs=ALASKA_ALBERS).plot(
            ax=ax,
            facecolor="tab:blue",
            edgecolor="grey",
            alpha=0.35,
            linewidth=0.5,
        )
        centroid = geom.centroid
        ax.text(
            centroid.x,
            centroid.y,
            tile_id,
            fontsize=6,
            ha="center",
            va="center",
            color="black",
            fontweight="bold",
        )

    # Plot Alaska boundary
    boundary_ak.boundary.plot(ax=ax, color="red", linewidth=1.5)

    ax.set_title("MODIS sinusoidal tile footprints overlapping Alaska (EPSG:3338)")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    legend_elements = [
        Patch(
            facecolor="tab:blue",
            alpha=0.35,
            edgecolor="grey",
            label="Intersecting tiles",
        ),
        plt.Line2D([0], [0], color="red", linewidth=1.5, label="Alaska boundary"),
    ]
    ax.legend(handles=legend_elements, loc="lower left")

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Tile footprint plot saved to: {save_path}")
    plt.close(fig)


def extract_tiles_from_zip(zip_path, tile_ids, output_dir):
    """
    Extract only the .nc files matching the given tile_ids from a zip archive
    into output_dir.

    Parameters
    ----------
    zip_path : str
        Path to the zip file.
    tile_ids : set
        Set of tile ids (e.g. {'h06v03'}) to extract.
    output_dir : str
        Directory to extract matching files into.

    Returns
    -------
    list of str
        Paths of the extracted files.
    """
    os.makedirs(output_dir, exist_ok=True)
    extracted = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            # Skip directories and macOS resource forks
            if member.endswith("/") or "__MACOSX" in member:
                continue
            if not member.lower().endswith(".nc"):
                continue

            basename = os.path.basename(member)
            tile_id = parse_tile_id(basename)
            if tile_id and tile_id in tile_ids:
                # Extract to flat output_dir (no nested subdirectories)
                target_path = os.path.join(output_dir, basename)
                with zf.open(member) as src, open(target_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target_path)

    return extracted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("1.05 - Extract Alaska-overlapping tiles from vanWees 500m zips")
    print("=" * 70)

    # Load Alaska boundary
    boundary_gdf = gpd.read_file(BOUNDARY_PATH)
    print(f"Loaded boundary: {BOUNDARY_PATH}")

    # Discover zip files
    zip_files = sorted(glob.glob(os.path.join(ZIP_DIR, "*.zip")))
    if not zip_files:
        raise FileNotFoundError(f"No .zip files found in {ZIP_DIR}")
    print(f"Found {len(zip_files)} zip files in {ZIP_DIR}")

    # ------------------------------------------------------------------
    # Step 1: Determine Alaska tiles using the first zip file
    # ------------------------------------------------------------------
    first_zip = zip_files[0]
    first_year = re.search(r"(\d{4})", os.path.basename(first_zip))
    first_year = first_year.group(1) if first_year else "unknown"
    print(f"\nStep 1: Identifying Alaska tiles from {os.path.basename(first_zip)} ...")

    # Extract first zip to a temporary directory to inspect tiles
    tmp_dir = os.path.join(OUTPUT_DIR, "_tmp_tile_detection")
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(first_zip, "r") as zf:
        nc_members = [
            m
            for m in zf.namelist()
            if m.lower().endswith(".nc") and "__MACOSX" not in m
        ]
        for member in nc_members:
            basename = os.path.basename(member)
            target = os.path.join(tmp_dir, basename)
            if not os.path.exists(target):
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

    alaska_tiles, tile_boxes = find_alaska_tiles(tmp_dir, boundary_gdf)
    print(f"  Total tiles in zip: {len(tile_boxes)}")
    print(f"  Tiles intersecting Alaska: {len(alaska_tiles)}")
    print(f"  Tile ids: {sorted(alaska_tiles)}")

    # Clean up temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Step 2: Plot tile footprints (diagnostic)
    # ------------------------------------------------------------------
    print("\nStep 2: Plotting tile footprints ...")
    plot_path = os.path.join(PLOT_DIR, "tile_footprints_vs_alaska.png")
    plot_tile_footprints(tile_boxes, alaska_tiles, boundary_gdf, save_path=plot_path)

    # ------------------------------------------------------------------
    # Step 3: Extract Alaska tiles from all zips
    # ------------------------------------------------------------------
    print("\nStep 3: Extracting Alaska tiles from all zip files ...")
    for zip_path in zip_files:
        zip_name = os.path.basename(zip_path)
        year_match = re.search(r"(\d{4})", zip_name)
        year_label = year_match.group(1) if year_match else "unknown"
        year_out_dir = os.path.join(OUTPUT_DIR, year_label)

        # Skip if already extracted
        if os.path.isdir(year_out_dir):
            existing_nc = glob.glob(os.path.join(year_out_dir, "*.nc"))
            if len(existing_nc) == len(alaska_tiles):
                print(
                    f"  {zip_name}: already extracted ({len(existing_nc)} files), skipping."
                )
                continue

        extracted = extract_tiles_from_zip(zip_path, alaska_tiles, year_out_dir)
        print(
            f"  {zip_name}: extracted {len(extracted)} Alaska tiles -> {year_out_dir}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
