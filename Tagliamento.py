from __future__ import annotations

from pathlib import Path
import warnings
import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------
# USER SETTINGS (EDIT THIS)
# -----------------------------
BASE = Path(r"C:\Users\saeed\OneDrive\TagRiv260205")  # <<< CHANGE
RAW = BASE / "raw"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

# Canonical CRS (meters)
CRS_CANON = "EPSG:6708"

# Inputs
MUNI_FP = RAW / "COMUNI_TAGLIAMENTO.shp"

HAZ_FPS = {
    "LPH": RAW / "LPH_Mosaicatura_ISPRA_2020_pericolosita_idraulica_bassa.shp",
    "MPH": RAW / "MPH_Mosaicatura_ISPRA_2020_pericolosita_idraulica_media.shp",
    "HPH": RAW / "HPH_Mosaicatura_ISPRA_2020_pericolosita_idraulica_elevata.shp",
}

LU_FPS = {
    1950: RAW / "MOLAND_LANDUSE_1950.shp",
    1970: RAW / "MOLAND_LANDUSE_1970.shp",
    1980: RAW / "MOLAND_LANDUSE_1980.shp",
    2000: RAW / "MOLAND_LANDUSE_2000.shp",
    # 2020: RAW / "MOLAND_LANDUSE_2020.shp",  # add if you have it
}

# Fields
MUNI_ID = "PRO_COM"
MUNI_NAME = "COMUNE"
LU_LEGENDA = "LEGENDA"

OUT_LONG = OUT / "tagliamento_long.csv"
OUT_WIDE = OUT / "tagliamento_wide.csv"

OUT_GPKG = OUT / "tagliamento_intermediate.gpkg"
SAVE_GPKG = True  # turn off when not needed because saving a geo package can fill up the space on your computer.
if SAVE_GPKG and OUT_GPKG.exists():
    OUT_GPKG.unlink()
# -----------------------------
# Helpers
# -----------------------------
def read_vector(fp: Path) -> gpd.GeoDataFrame:
    """Read with pyogrio if available (fast), else fiona fallback."""
    try:
        return gpd.read_file(fp, engine="pyogrio")
    except Exception:
        return gpd.read_file(fp)

def fix_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid and drop empties."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda g: make_valid(g) if g is not None else None)
    gdf = gdf[~gdf["geometry"].is_empty & gdf["geometry"].notna()].copy()
    return gdf

def to_canon(gdf: gpd.GeoDataFrame, crs_canon: str) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS. Define it in GIS first, then rerun.")
    if str(gdf.crs).upper() != crs_canon.upper():
        return gdf.to_crs(crs_canon)
    return gdf

def classify_macro(legend_series: pd.Series) -> pd.Series:
    """6 macro classes based on your ILIKE filters, case-insensitive substring matching."""
    s = legend_series.fillna("").astype(str).str.lower()

    # Order matters: green_urban and water_body first
    is_green_urban = s.str.contains("verdi urbane", regex=False)

    is_water = (
        s.str.contains("bacini", regex=False)
        | s.str.contains("canali", regex=False)
        | s.str.contains("fiumi", regex=False)
        | s.str.contains("saline", regex=False)
        | s.str.contains("oceani", regex=False)
    )

    is_ag = (
        s.str.contains("agro-industriali", regex=False)
        | s.str.contains("colturali", regex=False)
        | s.str.contains("frutteti", regex=False)
        | s.str.contains("agrarie", regex=False)
        | s.str.contains("irrigue", regex=False)
        | s.str.contains("vigneti", regex=False)
    )

    is_nat = (
        s.str.contains("boschi", regex=False)
        | s.str.contains("lagune", regex=False)
        | s.str.contains("paludi", regex=False)
        | s.str.contains("brughiere", regex=False)
        | s.str.contains("pascolo", regex=False)
        | s.str.contains("prati", regex=False)
        | s.str.contains("vegetazione", regex=False)
        | s.str.contains("rocce", regex=False)
        | s.str.contains("spiagge", regex=False)
    )

    is_ind = (
        s.str.contains("cantieri", regex=False)
        | s.str.contains("accesso", regex=False)
        | s.str.contains("discariche", regex=False)
        | s.str.contains("industriali", regex=False)
        | s.str.contains("militari", regex=False)
        | s.str.contains("estrattive", regex=False)
        | s.str.contains("portuali", regex=False)
    )

    is_res = (
        s.str.contains("cimiteri", regex=False)
        | s.str.contains("archeologici", regex=False)
        | s.str.contains("civili", regex=False)
        | s.str.contains("commerciali", regex=False)
        | s.str.contains("strade", regex=False)
        | s.str.contains("parcheggi", regex=False)
        | s.str.contains("luoghi", regex=False)
        | s.str.contains("servizi", regex=False)
        | s.str.contains("ferrovie", regex=False)
        | s.str.contains("tessuto", regex=False)
        | s.str.contains("sportive", regex=False)
        | s.str.contains("tecnologiche", regex=False)
    )

    out = pd.Series("other", index=s.index, dtype="object")
    out[is_res] = "residential_services"
    out[is_ind] = "non_residential_industry"
    out[is_nat] = "natural_green"
    out[is_ag] = "agriculture"
    out[is_water] = "water_body"
    out[is_green_urban] = "green_urban"
    return out

def area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    # In EPSG:6708 (UTM), geometry.area is m²
    return gdf.geometry.area / 1_000_000.0

def save_layer(gdf, gpkg_path, layer_name):
    if gdf.empty:
        return
    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")

# -----------------------------
# Main pipeline
# -----------------------------
def main() -> None:
    print("Reading municipalities…")
    muni = read_vector(MUNI_FP)
    muni = muni[[MUNI_ID, MUNI_NAME, "geometry"]].copy()
    muni = fix_geoms(muni)
    muni = to_canon(muni, CRS_CANON)

    # Dissolve municipalities into one mask for clipping
    muni_union = muni.geometry.union_all() if hasattr(muni.geometry, "union_all") else muni.geometry.unary_union

    print("Reading hazards…")
    hazards = {}
    for hz, fp in HAZ_FPS.items():
        g = read_vector(fp)
        g = g[[c for c in g.columns if c != "geometry"] + ["geometry"]].copy()
        g = fix_geoms(g)
        g = to_canon(g, CRS_CANON)
        # Clip to study area mask (faster later)
        g = gpd.clip(g, muni_union)
        hazards[hz] = g
        if SAVE_GPKG:
            save_layer(g, OUT_GPKG, layer_name=f"{hz}_clip")

    long_rows = []

    for year, fp in LU_FPS.items():
        print(f"Processing land use {year}…")
        lu = read_vector(fp)
        # keep only LEGENDA + geometry (plus anything else you want)
        if LU_LEGENDA not in lu.columns:
            raise ValueError(f"{fp.name} is missing field {LU_LEGENDA}")

        lu = lu[[LU_LEGENDA, "geometry"]].copy()
        lu = fix_geoms(lu)
        lu = to_canon(lu, CRS_CANON)
        lu = gpd.clip(lu, muni_union)

        # Macro class
        lu["macro_class"] = classify_macro(lu[LU_LEGENDA])

        # LU × Municipality (assigns PRO_COM/COMUNE to every piece)
        # This is the key step that avoids broken spatial joins later.
        lu_m = gpd.overlay(
            lu,
            muni[[MUNI_ID, MUNI_NAME, "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )
        if SAVE_GPKG:
            save_layer(
                lu_m,
                OUT_GPKG,
                layer_name=f"lu_{year}_x_muni"
            )
        lu_m["year"] = int(year)
        lu_m["hazard"] = "TOTAL"
        lu_m["area_km2"] = area_km2(lu_m)

        long_rows.append(
            lu_m[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]]
        )

        # Exposure: (LU×MUNI) × HAZARD
        for hz, hz_gdf in hazards.items():
            print(f"  - intersecting with {hz}…")
            exp = gpd.overlay(lu_m, hz_gdf[["geometry"]], how="intersection", keep_geom_type=True)
            exp["hazard"] = hz
            exp["area_km2"] = area_km2(exp)
            long_rows.append(
                exp[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]]
            )
            if SAVE_GPKG:
                save_layer(
                    exp,
                    OUT_GPKG,
                    layer_name=f"lu_{year}_{hz}_exp"
                )

    print("Aggregating to long table…")
    df_long = pd.concat(long_rows, ignore_index=True)
    df_long["area_km2"] = pd.to_numeric(df_long["area_km2"], errors="coerce").fillna(0.0)

    df_long = (
        df_long.groupby([MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class"], as_index=False)
        .agg(area_km2=("area_km2", "sum"))
    )

    df_long.to_csv(OUT_LONG, index=False)
    print(f"Wrote: {OUT_LONG}")

    print("Pivoting to wide table…")
    wide = df_long.pivot_table(
        index=[MUNI_ID, MUNI_NAME],
        columns=["year", "hazard", "macro_class"],
        values="area_km2",
        aggfunc="sum",
        fill_value=0.0,
    )
    wide.columns = [f"{y}_{hz}_{mc}_km2" for (y, hz, mc) in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(OUT_WIDE, index=False)
    print(f"Wrote: {OUT_WIDE}")

    print("DONE.")


if __name__ == "__main__":
    main()
