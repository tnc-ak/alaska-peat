from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union


# --------------------------
# USER PARAMETERS
# --------------------------
FEATURESERVER_URL = "https://arcgis.dnr.alaska.gov/arcgis/rest/services/Mapper/Ownership_Layers/FeatureServer"
BOUNDARY_PATH = Path("data/alaska-boundary/alaska_buffered.shp")
TARGET_CRS = "EPSG:3338"

# Choose where municipal lands should map. Options: "State" or "Private"
MUNICIPAL_POLICY = "Private"

# Keep uncertain areas separate while validating. Set False to fold into State.
KEEP_UNCERTAIN_CLASS = True

# Optional precedence removes overlaps between broad classes for clean cartography.
APPLY_PRECEDENCE = True
PRECEDENCE = ["Tribal", "Federal", "State", "Private", "Uncertain"]

# Fill any remaining unassigned boundary area as Federal in final output.
ASSIGN_REMAINDER_TO_FEDERAL = True

OUTDIR = Path("outputs/exploratory/ownership")
OUTDIR.mkdir(parents=True, exist_ok=True)


# Polygon layers relevant to ownership simplification.
LAYER_MAP = {
    3: "Federal Action Poly",
    4: "Land Disposal Conveyed Poly",
    5: "Management Agreement Poly",
    6: "Mental Health Trust Land Poly",
    7: "Municipal Entitlement Poly",
    8: "Municipal Tideland Poly",
    9: "State Interest Native Allotment Poly",
    10: "ANILCA Topfiled All Poly",
    11: "State Selected Land All Poly",
    12: "Other State Acquired All Poly",
    13: "State TA Patented All Poly",
    49: "Survey Boundary Poly",
}


def layer_crosswalk(municipal_policy: str) -> Dict[str, str]:
    if municipal_policy not in {"State", "Private"}:
        raise ValueError("MUNICIPAL_POLICY must be 'State' or 'Private'.")

    return {
        "Federal Action Poly": "Federal",
        "Land Disposal Conveyed Poly": "Private",
        "Survey Boundary Poly": "Private",
        "Management Agreement Poly": "State",
        "Mental Health Trust Land Poly": "State",
        "Municipal Entitlement Poly": municipal_policy,
        "Municipal Tideland Poly": municipal_policy,
        "State Interest Native Allotment Poly": "Tribal",
        "ANILCA Topfiled All Poly": "State",
        "State Selected Land All Poly": "State",
        "Other State Acquired All Poly": "State",
        "State TA Patented All Poly": "State",
    }


def chunked(items: List[int], size: int) -> Iterable[List[int]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def query_json(url: str, data: Dict[str, str]) -> Dict:
    response = requests.post(url, data=data, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if "error" in payload:
        raise RuntimeError(f"ArcGIS query error: {payload['error']}")
    return payload


def fetch_object_ids(layer_id: int) -> List[int]:
    query_url = f"{FEATURESERVER_URL}/{layer_id}/query"
    payload = query_json(
        query_url,
        {
            "f": "json",
            "where": "1=1",
            "returnIdsOnly": "true",
        },
    )
    return sorted(payload.get("objectIds") or [])


def fetch_layer_geojson(layer_id: int, layer_name: str) -> gpd.GeoDataFrame:
    query_url = f"{FEATURESERVER_URL}/{layer_id}/query"
    object_ids = fetch_object_ids(layer_id)

    if not object_ids:
        print(f"Layer {layer_id} ({layer_name}): no features")
        return gpd.GeoDataFrame(
            columns=["source_layer", "geometry"], geometry="geometry", crs="EPSG:4326"
        )

    gdfs = []
    for oid_chunk in chunked(object_ids, 500):
        payload = query_json(
            query_url,
            {
                "f": "geojson",
                "objectIds": ",".join(str(v) for v in oid_chunk),
                "outFields": "*",
                "returnGeometry": "true",
            },
        )
        features = payload.get("features", [])
        if not features:
            continue
        gdf = gpd.GeoDataFrame.from_features(features)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        gdf["source_layer"] = layer_name
        gdfs.append(gdf)

    if not gdfs:
        return gpd.GeoDataFrame(
            columns=["source_layer", "geometry"], geometry="geometry", crs="EPSG:4326"
        )

    layer_gdf = pd.concat(gdfs, ignore_index=True)
    layer_gdf = gpd.GeoDataFrame(layer_gdf, geometry="geometry", crs=gdfs[0].crs)
    print(f"Layer {layer_id} ({layer_name}): {len(layer_gdf):,} features")
    return layer_gdf


def class_from_party_name(text: str) -> str | None:
    if not text:
        return None

    t = text.upper()
    tribal_terms = [
        "TRIBE",
        "TRIBAL",
        "NATIVE VILLAGE",
        "NATIVE CORPORATION",
        "VILLAGE OF",
    ]
    federal_terms = [
        "UNITED STATES",
        "U.S.",
        "USDA",
        "BUREAU OF LAND MANAGEMENT",
        "NATIONAL PARK SERVICE",
        "FISH AND WILDLIFE",
        "FOREST SERVICE",
    ]
    state_terms = [
        "STATE OF ALASKA",
        "MENTAL HEALTH TRUST",
        "UNIVERSITY OF ALASKA",
        "MUNICIPALITY OF",
        "CITY OF",
        "BOROUGH",
    ]

    if any(term in t for term in tribal_terms):
        return "Tribal"
    if any(term in t for term in federal_terms):
        return "Federal"
    if any(term in t for term in state_terms):
        return "State"
    return None


def assign_broad_class(row: pd.Series, crosswalk: Dict[str, str]) -> str:
    layer_name = row["source_layer"]
    broad = crosswalk.get(layer_name, "Uncertain")

    if layer_name == "State TA Patented All Poly":
        status = str(row.get("LNDSTTSDSC", "") or "").strip().upper()
        if status == "LM TO BE REVOKED (TR)":
            return "Uncertain"

    if layer_name == "Land Disposal Conveyed Poly":
        # Use party-name hints when present, otherwise default to Private.
        name_parts = [
            str(row.get("CSTMRNM", "") or ""),
            str(row.get("CSTMRLSTNM", "") or ""),
            str(row.get("MI_LABEL", "") or ""),
        ]
        guessed = class_from_party_name(" ".join(name_parts))
        if guessed is not None:
            return guessed

    return broad


def polygon_parts(geom) -> List[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, MultiPolygon):
        return [g for g in geom.geoms if not g.is_empty]
    if isinstance(geom, GeometryCollection):
        parts: List[Polygon] = []
        for g in geom.geoms:
            parts.extend(polygon_parts(g))
        return parts
    return []


def apply_precedence(gdf: gpd.GeoDataFrame, precedence: List[str]) -> gpd.GeoDataFrame:
    import shapely

    records = []
    consumed = None
    grid_size = 0.01

    for ownership_class in precedence:
        subset = gdf[gdf["ownership_class"] == ownership_class]
        if subset.empty:
            continue

        geom = unary_union(subset.geometry).buffer(0)
        if consumed is not None and not consumed.is_empty:
            geom = shapely.difference(geom, consumed, grid_size=grid_size)

        parts = polygon_parts(geom)
        for part in parts:
            records.append({"ownership_class": ownership_class, "geometry": part})

        if consumed is None:
            consumed = unary_union(parts) if parts else GeometryCollection()
        elif parts:
            consumed = unary_union([consumed, unary_union(parts)])

    out = gpd.GeoDataFrame(records, geometry="geometry", crs=gdf.crs)
    return out


def append_residual_federal(
    boundary: gpd.GeoDataFrame,
    gdf: gpd.GeoDataFrame,
    crs: str,
) -> gpd.GeoDataFrame:
    import shapely

    boundary_geom = unary_union(boundary.geometry).buffer(0)
    covered_geom = unary_union(
        [g for g in gdf.geometry if g is not None and not g.is_empty]
    ).buffer(0)
    # Snap to 1 cm grid to resolve non-noded intersection topology errors.
    grid_size = 0.01
    residual = shapely.difference(boundary_geom, covered_geom, grid_size=grid_size)
    parts = polygon_parts(residual)

    if not parts:
        return gdf

    fill = gpd.GeoDataFrame(
        {
            "ownership_class": ["Federal"] * len(parts),
            "geometry": parts,
        },
        geometry="geometry",
        crs=crs,
    )
    return gpd.GeoDataFrame(
        pd.concat([gdf, fill], ignore_index=True), geometry="geometry", crs=crs
    )


def main() -> None:
    print(f"Loading Alaska boundary from file: {BOUNDARY_PATH}...")
    boundary = gpd.read_file(BOUNDARY_PATH)
    if boundary.empty:
        raise RuntimeError(f"Boundary file is empty: {BOUNDARY_PATH}")
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")
    boundary = boundary.to_crs(TARGET_CRS)
    boundary = boundary[["geometry"]].dissolve().reset_index(drop=True)
    boundary["geometry"] = boundary.geometry.buffer(0)

    print("Downloading ownership polygons from DNR service...")
    layers = []
    for layer_id, layer_name in LAYER_MAP.items():
        layer_gdf = fetch_layer_geojson(layer_id, layer_name)
        if layer_gdf.empty:
            continue
        layers.append(layer_gdf)

    if not layers:
        raise RuntimeError("No features were downloaded from ownership layers.")

    ownership = gpd.GeoDataFrame(
        pd.concat(layers, ignore_index=True), geometry="geometry", crs=layers[0].crs
    )

    if ownership.crs is None:
        ownership = ownership.set_crs("EPSG:4326")
    ownership = ownership.to_crs(TARGET_CRS)

    # Repair and clip to study area.
    ownership = ownership[
        ownership.geometry.notna() & ~ownership.geometry.is_empty
    ].copy()
    ownership["geometry"] = ownership.geometry.buffer(0)
    ownership = gpd.clip(ownership, boundary)
    ownership = ownership[
        ownership.geometry.notna() & ~ownership.geometry.is_empty
    ].copy()

    crosswalk = layer_crosswalk(MUNICIPAL_POLICY)
    ownership["ownership_class"] = ownership.apply(
        assign_broad_class, axis=1, crosswalk=crosswalk
    )

    if not KEEP_UNCERTAIN_CLASS:
        ownership.loc[
            ownership["ownership_class"] == "Uncertain", "ownership_class"
        ] = "State"

    ownership["area_km2"] = ownership.geometry.area / 1_000_000

    # QA outputs before overlap handling.
    qa_by_source = (
        ownership.groupby(["source_layer", "ownership_class"], dropna=False)["area_km2"]
        .sum()
        .reset_index()
        .sort_values(["ownership_class", "area_km2"], ascending=[True, False])
    )
    qa_by_class = (
        ownership.groupby("ownership_class", dropna=False)["area_km2"]
        .sum()
        .reset_index()
        .sort_values("area_km2", ascending=False)
    )

    qa_by_source.to_csv(OUTDIR / "ownership_area_by_source_layer.csv", index=False)
    qa_by_class.to_csv(OUTDIR / "ownership_area_by_class_raw.csv", index=False)

    uncertain = ownership[ownership["ownership_class"] == "Uncertain"].copy()
    if not uncertain.empty:
        uncertain.head(500).to_csv(
            OUTDIR / "ownership_uncertain_samples.csv", index=False
        )

    ownership.to_file(
        OUTDIR / "alaska_ownership_simplified_raw.gpkg",
        layer="ownership_raw",
        driver="GPKG",
    )

    if APPLY_PRECEDENCE:
        print("Applying overlap precedence and dissolving classes...")
        dissolved = (
            ownership[["ownership_class", "geometry"]]
            .dissolve(by="ownership_class")
            .reset_index()
        )
        final = apply_precedence(dissolved, PRECEDENCE)
    else:
        final = (
            ownership[["ownership_class", "geometry"]]
            .dissolve(by="ownership_class")
            .reset_index()
        )

    if ASSIGN_REMAINDER_TO_FEDERAL:
        final = append_residual_federal(boundary, final, TARGET_CRS)

    final = final[final.geometry.notna() & ~final.geometry.is_empty].copy()
    final["area_km2"] = final.geometry.area / 1_000_000

    final_by_class = (
        final.groupby("ownership_class", dropna=False)["area_km2"]
        .sum()
        .reset_index()
        .sort_values("area_km2", ascending=False)
    )
    final_by_class.to_csv(OUTDIR / "ownership_area_by_class_final.csv", index=False)

    final.to_file(
        OUTDIR / "alaska_ownership_simplified_final.gpkg",
        layer="ownership_final",
        driver="GPKG",
    )

    print("Done.")
    print(f"Raw ownership polygons: {len(ownership):,}")
    print(f"Final simplified polygons: {len(final):,}")
    print(f"Outputs written to: {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
