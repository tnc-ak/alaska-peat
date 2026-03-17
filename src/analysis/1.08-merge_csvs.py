"""Merge GFED/vanWees output CSVs and peat CSVs into one combined CSV.

Reads:
 - outputs/gfed_emissions_and_ba_in_alaska.csv
 - outputs/gfed_emissions_in_alaska_by_gas_gram.csv
 - outputs/vanwees_emissions_ba_in_alaska.csv
 - outputs/vanwees_500m_emissions_ba_in_alaska.csv
 - outputs/vanwees_500m_peat_emissions_ba_in_alaska.csv
 - data/peat/LaraPeat_MCD64A1_Alaska_Annual_Burned_Area_2001_2024.csv
 - data/peat/Total_MCD64A1_Alaska_Annual_Burned_Area_2001_2024.csv
 - data/fire/WFEIS_alaska__MTBS__1984-2024__yearly.csv
 - data/fire/WFEIS_alaska__WFIGS_archive__1950-2026__yearly.csv

Writes:
 - outputs/merged_emissions_and_ba.csv (default)

Usage:
    python src/analysis/1.04-merge_csvs.py --out outputs/merged_emissions_and_ba.csv
"""

import os
from pathlib import Path
import pandas as pd


def main(output_path: str) -> None:
    root = Path(__file__).resolve().parents[2]

    # Input files (relative to repo root)
    gfed_total_fp = root / "outputs" / "gfed_emissions_and_ba_in_alaska.csv"
    gfed_by_gas_fp = root / "outputs" / "gfed_emissions_in_alaska_by_gas_gram.csv"
    vanwees_fp = root / "outputs" / "vanwees_emissions_ba_in_alaska.csv"
    vanwees_500m_fp = root / "outputs" / "vanwees_500m_emissions_ba_in_alaska.csv"
    vanwees_500m_peat_fp = (
        root / "outputs" / "vanwees_500m_peat_emissions_ba_in_alaska.csv"
    )
    modis_lara_fp = (
        root
        / "data"
        / "peat"
        / "LaraPeat_MCD64A1_Alaska_Annual_Burned_Area_2001_2024.csv"
    )
    modis_total_fp = (
        root / "data" / "peat" / "Total_MCD64A1_Alaska_Annual_Burned_Area_2001_2024.csv"
    )
    fire_inventory_fp = (
        root / "data" / "fire" / "alaska_dec_fire_areas_by_year_2005_2020.csv"
    )
    fire_perimeters_fp = (
        root
        / "outputs"
        / "exploratory"
        / "fire_perimeters"
        / "fire_area_on_peat_by_year.csv"
    )
    wfeis_fp = (
        root / "data" / "fire" / "WFEIS_alaska__MTBS__1984-2024__yearly.csv"
    )
    wfigs_archive_fp = (
        root / "data" / "fire" / "WFEIS_alaska__WFIGS_archive__1950-2026__yearly.csv"
    )

    gfed_ecosystem = pd.read_csv(gfed_total_fp)
    gfed_monthly = pd.read_csv(gfed_by_gas_fp)
    vanwees = pd.read_csv(vanwees_fp)
    vanwees_500m = pd.read_csv(vanwees_500m_fp)
    vanwees_500m_peat = pd.read_csv(vanwees_500m_peat_fp)
    modis_lara = pd.read_csv(modis_lara_fp)
    modis_total = pd.read_csv(modis_total_fp)
    fire_inventory = pd.read_csv(fire_inventory_fp)
    fire_perimeters = pd.read_csv(fire_perimeters_fp)
    wfeis = pd.read_csv(wfeis_fp)
    wfigs_archive = pd.read_csv(wfigs_archive_fp)

    # Convert to hectares
    gfed_ecosystem["burned_area_ha"] = gfed_ecosystem["burned_area"] / 10000.0
    gfed_ecosystem = gfed_ecosystem.drop(columns=["burned_area"])

    # VanWees: 'BA_TOT_M2' is in m2
    vanwees["burned_area_ha"] = vanwees["BA_TOT_M2"] / 10000.0
    vanwees = vanwees.drop(columns=["BA_TOT_M2"])

    # VanWees 500 m (all land)
    vanwees_500m["burned_area_ha"] = vanwees_500m["BA_TOT_M2"] / 10000.0
    vanwees_500m = vanwees_500m.drop(columns=["BA_TOT_M2"])

    # VanWees 500 m (peat-weighted)
    vanwees_500m_peat["burned_area_ha"] = vanwees_500m_peat["BA_TOT_M2"] / 10000.0
    vanwees_500m_peat = vanwees_500m_peat.drop(columns=["BA_TOT_M2"])

    # Fire inventory: convert acres to hectares (1 acre = 0.40468564224 ha)
    fire_inventory["burned_area_ha"] = (
        fire_inventory["total_burned_area_acres"] * 0.40468564224
    )
    fire_inventory = fire_inventory.drop(
        columns=["total_burned_area_acres", "Wildfire_Acres", "WFU", "Prescribed_Acres"]
    )

    # Fire perimeters: rename columns and drop peat_proportion
    fire_perimeters = fire_perimeters.rename(
        columns={
            "total_fire_area_ha": "burned_area_ha",
            "peat_fire_area_ha": "burned_area_ha_peat",
        }
    )
    fire_perimeters = fire_perimeters.drop(columns=["peat_proportion"])

    # WFEIS: filter for Alaska, convert units, select columns, rename Year to year
    # wfeis = wfeis[wfeis["aoi"] == "Alaska"].copy()
    wfeis["burned_area_ha"] = wfeis["area_km2"] * 100  # km2 to ha
    wfeis["emissions_co2_g"] = wfeis["consume_output__co2_mg"] / 1000  # mg to g
    wfeis["emissions_ch4_g"] = wfeis["consume_output__ch4_mg"] / 1000
    wfeis["emissions_co_g"] = wfeis["consume_output__co_mg"] / 1000
    wfeis["emissions_c_g"] = wfeis["consume_output__carbon_mg"] / 1000
    wfeis = wfeis[
        [
            "year",
            "burned_area_ha",
            "emissions_co2_g",
            "emissions_ch4_g",
            "emissions_co_g",
            "emissions_c_g",
        ]
    ]

    # WFIGS Archive: same processing as WFEIS
    wfigs_archive["burned_area_ha"] = wfigs_archive["area_km2"] * 100  # km2 to ha
    wfigs_archive["emissions_co2_g"] = wfigs_archive["consume_output__co2_mg"] / 1000  # mg to g
    wfigs_archive["emissions_ch4_g"] = wfigs_archive["consume_output__ch4_mg"] / 1000
    wfigs_archive["emissions_co_g"] = wfigs_archive["consume_output__co_mg"] / 1000
    wfigs_archive["emissions_c_g"] = wfigs_archive["consume_output__carbon_mg"] / 1000
    wfigs_archive = wfigs_archive[
        [
            "year",
            "burned_area_ha",
            "emissions_co2_g",
            "emissions_ch4_g",
            "emissions_co_g",
            "emissions_c_g",
        ]
    ]

    # Prefix columns (keep 'year' as the merge key)
    def _suffix_cols(df, suffix):
        """Return a copy of df with all columns (except year) suffixed."""
        return df.rename(
            columns={c: f"{c}{suffix}" for c in df.columns if c not in ["year"]}
        )

    gfed_ecosystem = _suffix_cols(gfed_ecosystem, "_GFED5.1_ecosystem")
    gfed_monthly = _suffix_cols(gfed_monthly, "_GFED5.1_monthly")
    vanwees = _suffix_cols(vanwees, "_VanWees")
    vanwees_500m = _suffix_cols(vanwees_500m, "_VanWees500m")
    vanwees_500m_peat = _suffix_cols(vanwees_500m_peat, "_VanWees500m_peat")
    modis_lara = _suffix_cols(modis_lara, "_MODIS_Lara")
    modis_total = _suffix_cols(modis_total, "_MODIS_Total")
    fire_inventory = _suffix_cols(fire_inventory, "_ADEC")
    fire_perimeters = _suffix_cols(fire_perimeters, "_Perimeters")
    wfeis = _suffix_cols(wfeis, "_WFEIS")
    wfigs_archive = _suffix_cols(wfigs_archive, "_WFIGS_Archive")

    # Start from gfed_ecosystem and merge others
    merged = pd.merge(gfed_ecosystem, gfed_monthly, on="year", how="outer")
    merged = pd.merge(merged, vanwees, on="year", how="outer")
    merged = pd.merge(merged, vanwees_500m, on="year", how="outer")
    merged = pd.merge(merged, vanwees_500m_peat, on="year", how="outer")
    merged = pd.merge(merged, modis_lara, on="year", how="outer")
    merged = pd.merge(merged, modis_total, on="year", how="outer")
    merged = pd.merge(merged, fire_inventory, on="year", how="outer")
    merged = pd.merge(merged, fire_perimeters, on="year", how="outer")
    merged = pd.merge(merged, wfeis, on="year", how="outer")
    merged = pd.merge(merged, wfigs_archive, on="year", how="outer")

    # Sort and write
    merged = merged.sort_values("year").reset_index(drop=True)

    out_fp = Path(output_path)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_fp, index=False)


if __name__ == "__main__":
    default_out = os.path.join("outputs", "merged_emissions_and_ba.csv")
    main(default_out)
