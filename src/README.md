# Fire on peat trend analysis

This small script computes the yearly burned area on peat (wet or moist) from the
Alaska Fire History polygons by intersecting fire polygons with the peat raster
(`data/fire/AlbersPeatMap.tif`) where peat values are 1 or 2.

Files added:
- `src/fire_peat_trends.py` — main script
- `environment.yml` — Conda environment specification

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate alaska-peat
```

## Usage (from repository root):

```bash
python src/fire_peat_trends.py \
  --gdb data/fire/AlaskaFireHistory_Polygons.gdb \
  --layer AK_fire_location_polygons_AKAlbersNAD83 \
  --peat-raster data/fire/AlbersPeatMap.tif
```

Outputs are written to `outputs/` by default:

CSVs:
- `outputs/fire_area_on_peat_by_year.csv` — table with yearly metrics:
  - total burned area (ha)
  - burned area on peat (ha)
  - proportion of burned area on peat

Plots:
- `outputs/fire_area_on_peat_trend_areas.png` — combined plot showing total and peat burned areas
- `outputs/fire_area_on_peat_trend_proportion.png` — proportion of burned area on peat by year
- `outputs/fire_area_on_peat_trend.png` — burned area on peat with trend line (original plot)

Notes:
- The script expects both the fire layer and peat raster to be in the same projected
  CRS (Albers NAD83 meters is expected). If they differ, the script will reproject
  the fire layer into the raster CRS automatically.
- Install dependencies in a suitable Python environment. See `src/requirements.txt`.
