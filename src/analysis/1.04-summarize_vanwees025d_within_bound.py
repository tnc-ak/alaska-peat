import os
import glob
import pandas as pd
import xarray as xr
import geopandas as gpd
from rasterio.features import geometry_mask


def summarize_vanwees_within_boundary(boundary_path, output_file):
    """
    Summarize vanWees carbon emissions and burned area within a given boundary.

    Extracts variables:
    - C_AG_TOT and C_BG_TOT from emissions group (above-ground and below-ground carbon)
    - BA_TOT from burned_area group (total burned area)
    - grid_cell_area from MOD_CMG025 group (grid cell area)

    Args:
        boundary_path (str): Path to the boundary shapefile.
        output_file (str): Path to save the output CSV file.
    """
    vanwees_dir = os.path.join("data", "fire", "vanWees")
    boundary_gdf = gpd.read_file(boundary_path)
    boundary_geom = boundary_gdf.geometry.values[0]

    # Find all .nc files in vanWees directory
    vanwees_files = glob.glob(os.path.join(vanwees_dir, "*.nc"))
    vanwees_files.sort()

    if not vanwees_files:
        raise FileNotFoundError(
            f"No NetCDF files found in {vanwees_dir}. Please check the directory path."
        )

    print(f"Found {len(vanwees_files)} vanWees files to process")

    results = []
    for nc_file in vanwees_files:
        print(f"Processing {os.path.basename(nc_file)}")

        ds_data = xr.open_dataset(
            nc_file, group="MOD_CMG025/emissions", decode_times=True
        )
        ds_coord = xr.open_dataset(nc_file, group="MOD_CMG025", decode_times=True)
        ds_ba = xr.open_dataset(
            nc_file, group="MOD_CMG025/burned_area", decode_times=True
        )

        # Assign coordinates and spatial dimensions for data variables
        ds_data = ds_data.assign_coords(
            lat=ds_coord["lat"],
            lon=ds_coord["lon"],
            time=ds_coord["time"],
        )

        ds_data["lat"] = ds_data["lat"].assign_attrs(
            standard_name="latitude", units="degrees_north"
        )
        ds_data["lon"] = ds_data["lon"].assign_attrs(
            standard_name="longitude", units="degrees_east"
        )

        ds_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        ds_data.rio.write_crs("EPSG:4326", inplace=True)

        # Assign coordinates and spatial dimensions for burned area
        ds_ba = ds_ba.assign_coords(
            lat=ds_coord["lat"],
            lon=ds_coord["lon"],
            time=ds_coord["time"],
        )

        ds_ba["lat"] = ds_ba["lat"].assign_attrs(
            standard_name="latitude", units="degrees_north"
        )
        ds_ba["lon"] = ds_ba["lon"].assign_attrs(
            standard_name="longitude", units="degrees_east"
        )

        ds_ba.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        ds_ba.rio.write_crs("EPSG:4326", inplace=True)

        mask = geometry_mask(
            [boundary_geom],
            transform=ds_ba.rio.transform(),
            invert=True,
            out_shape=(ds_ba.sizes["lat"], ds_ba.sizes["lon"]),
        )

        grid_area = ds_coord.grid_cell_area

        # Extract variables from emissions group, weighted by grid_cell_area
        for month in range(1, 13):
            for var_name in ["C_AG_TOT", "C_BG_TOT"]:
                arr = ds_data[var_name].isel(time=month - 1)
                # Weight by grid cell area if available
                arr_weighted = arr.where(mask) * grid_area.where(mask)
                total_val = float(arr_weighted.sum().values)
                results.append(
                    {
                        "year": int(ds_data["time.year"].values[month - 1]),
                        "month": int(ds_data["time.month"].values[month - 1]),
                        "variable": var_name,
                        "value": total_val,
                    }
                )
            # Extract burned area from burned_area group
            ba_arr = ds_ba["BA_TOT"].isel(time=month - 1)
            ba_arr = ba_arr.where(mask) * grid_area.where(mask)
            total_ba = float(ba_arr.sum().values)
            results.append(
                {
                    "year": int(ds_ba["time.year"].values[month - 1]),
                    "month": int(ds_ba["time.month"].values[month - 1]),
                    "variable": "BA_TOT_M2",
                    "value": total_ba,
                }
            )

        ds_data.close()
        ds_coord.close()
        ds_ba.close()

    # Create DataFrame and save
    results_df = pd.DataFrame(results)
    results_df = results_df.pivot_table(
        index="year", columns="variable", values="value", aggfunc="sum"
    ).reset_index()
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    # Example usage
    boundary_path = os.path.join("data", "alaska_buffered.shp")
    output_file = os.path.join("outputs", "vanwees_emissions_ba_in_alaska.csv")
    summarize_vanwees_within_boundary(boundary_path, output_file)
