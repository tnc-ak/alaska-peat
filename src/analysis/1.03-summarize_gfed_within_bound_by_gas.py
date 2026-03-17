
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray as rxr
from rasterio.features import geometry_mask

def summarize_gfed_within_boundary(boundary_path, output_file):
    """
    Summarize GFED emissions within a given boundary.
    Args:
        boundary_path (str): Path to the boundary shapefile.
        output_file (str): Path to save the output CSV file.
    """
    gfed_dir = os.path.join("data", "fire", "GFED5.1", "GFED5.1_monthly")
    # output_dir = os.path.join("data", "GFED5.1", "processed")
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, output_file)
    emission_vars = ["CO2", "CH4", "N2O", "CO", "C", "DM"]
    # emission_vars = ["burned_area", "carbon_emissions"]
    boundary_gdf = gpd.read_file(boundary_path)
    boundary_geom = boundary_gdf.geometry.values[0]
    gfed_files = glob.glob(os.path.join(gfed_dir, "GFED5.1_monthly_*.nc"))
    gfed_files.sort()
    
    if not gfed_files:
        raise FileNotFoundError(f"No GFED files found in {gfed_dir}. Please check the directory path.")
    
    print(f"Found {len(gfed_files)} GFED files to process")
    results = []
    for gfed_file in gfed_files:
        print(f"Processing {os.path.basename(gfed_file)}")
        ds = xr.open_dataset(gfed_file)
        ds = ds.rename({"lon": "x", "lat": "y"})
        ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        ds.rio.write_crs("EPSG:4326", inplace=True)
        mask = geometry_mask(
            [boundary_geom],
            transform=ds.rio.transform(),
            invert=True,
            out_shape=(ds.sizes["y"], ds.sizes["x"]),
        )
        for month in range(1, 13):
            year = int(ds["time.year"].values[month - 1])
            for var in emission_vars:
                arr = ds[var].isel(time=month - 1)
                arr = arr
                arr = arr.where(mask)
                total = float(arr.sum().values)
                results.append(
                    {
                        "year": year,
                        "variable": var,
                        "value": total,
                    }
                )
        ds.close()
    
    if not results:
        raise ValueError("No emission data was collected. Check if the boundary intersects the data.")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.pivot_table(
        index="year", columns="variable", values="value", aggfunc="sum"
    ).reset_index()
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Example usage
    boundary_path = os.path.join("data", "alaska_buffered.shp")
    output_file = os.path.join("outputs", "gfed_emissions_in_alaska_by_gas_gram.csv")
    summarize_gfed_within_boundary(boundary_path, output_file)
