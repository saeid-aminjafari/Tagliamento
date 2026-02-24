from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd
import geopandas as gpd
from shapely.validation import make_valid

warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# USER SETTINGS (EDIT THIS)
# =============================================================================
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

# Land use inputs (1950–2000: MOLAND shapefiles)
LU_FPS = {
    1950: RAW / "MOLAND_LANDUSE_1950.shp",
    1970: RAW / "MOLAND_LANDUSE_1970.shp",
    1980: RAW / "MOLAND_LANDUSE_1980.shp",
    2000: RAW / "MOLAND_LANDUSE_2000.shp",
}

# 2020 inputs (GPKG with 2 layers)
LU2020_GPKG = RAW / "2020" / "mosaicatura_prgc.gpkg"
LU2020_LAYER_1 = "mosaicatura_zonizzazione"     # zoning layer with zona_omoge codes
LU2020_LAYER_2 = "mosaicatura_servizi_area"     # everything -> services_infrastructure
LU2020_CODECOL = "zona_omoge"                   # code field in LAYER_1

# Fields
MUNI_ID = "PRO_COM"
MUNI_NAME = "COMUNE"
LU_LEGENDA = "LEGENDA"

# Outputs
OUT_LONG = OUT / "tagliamento_long.csv"
OUT_WIDE = OUT / "tagliamento_wide.csv"
OUT_QC_UNCL = OUT / "qc_unclassified_labels.csv"
OUT_QC_UNCL_2020 = OUT / "qc_2020_unclassified_codes.csv"

OUT_GPKG = OUT / "tagliamento_intermediate.gpkg"
SAVE_GPKG = True  # turn off if disk space is a concern
if SAVE_GPKG and OUT_GPKG.exists():
    OUT_GPKG.unlink()


# =============================================================================
# HELPERS
# =============================================================================
def read_vector(fp: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read vector with pyogrio if available (fast), else fiona fallback."""
    try:
        return gpd.read_file(fp, layer=layer, engine="pyogrio")
    except Exception:
        return gpd.read_file(fp, layer=layer)


def fix_geoms(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Make geometries valid and drop empties."""
    gdf = gdf.copy()
    gdf["geometry"] = gdf["geometry"].apply(lambda g: make_valid(g) if g is not None else None)
    gdf = gdf[gdf["geometry"].notna() & ~gdf["geometry"].is_empty].copy()
    return gdf


def to_canon(gdf: gpd.GeoDataFrame, crs_canon: str) -> gpd.GeoDataFrame:
    """Reproject to canonical CRS; fail fast if CRS is missing."""
    if gdf.crs is None:
        raise ValueError("Input layer has no CRS. Define it in GIS first, then rerun.")
    if str(gdf.crs).upper() != crs_canon.upper():
        return gdf.to_crs(crs_canon)
    return gdf


def union_mask(gdf: gpd.GeoDataFrame):
    """Return geometry union for clipping, compatible with shapely/geopandas versions."""
    if hasattr(gdf.geometry, "union_all"):
        return gdf.geometry.union_all()
    return gdf.geometry.unary_union


def classify_macro(legend_series: pd.Series) -> pd.Series:
    """
    7 macro classes + 'unclassified' based on case-insensitive substring matching of LEGENDA.
    Priority overrides (applied last):
      - water_body
      - green_urban
    """
    s = legend_series.fillna("").astype(str).str.lower()

    # Highest-priority categories
    is_green_urban = s.str.contains("verdi urbane", regex=False)

    is_water = (
        s.str.contains("bacini", regex=False)
        | s.str.contains("canali", regex=False)
        | s.str.contains("fiumi", regex=False)
        | s.str.contains("saline", regex=False)
        | s.str.contains("oceani", regex=False)
    )

    # Broad classes
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

    is_ser = (
        s.str.contains("cimiteri", regex=False)
        | s.str.contains("archeologici", regex=False)
        | s.str.contains("civili", regex=False)
        | s.str.contains("commerciali", regex=False)
        | s.str.contains("strade", regex=False)
        | s.str.contains("parcheggi", regex=False)
        | s.str.contains("luoghi", regex=False)
        | s.str.contains("servizi", regex=False)
        | s.str.contains("ferrovie", regex=False)
        | s.str.contains("sportive", regex=False)
        | s.str.contains("ospedali", regex=False)
        | s.str.contains("tecnologiche", regex=False)
    )

    is_res = s.str.contains("tessuto residenziale", regex=False)

    out = pd.Series("unclassified", index=s.index, dtype="object")

    out[is_ag] = "agriculture"
    out[is_nat] = "natural_green"
    out[is_ind] = "non_residential_industry"
    out[is_ser] = "services_infrastructure"
    out[is_res] = "residential"

    # Priority overrides last
    out[is_water] = "water_body"
    out[is_green_urban] = "green_urban"
    return out


# -----------------------------
# 2020: zona_omoge -> macro class (CODE-ONLY mapping)
# -----------------------------
AG_2020 = {
    "E", "ET", "E2", "EP", "RZ", "EH", "E SST", "V/E", "CAVALLI",
}

NAT_2020 = {
    "F", "PARCO", "RG", "RO", "RI", "RN", "RP", "PRPG", "PT_E", "PT_AM", "PC_RP", "PC_RG",
    "FLUVIALE", "AMBIENTE", "AMBIENTALE", "A. R. I. A. n° 16", "ARIA", "ARIA N° 7",
    "AMBITO FLUVIALE", "MEDUNA", "Q", "SIC", "FALESIE", "RUPI", "RUPI BOSCATE",
    "AREE BOSCATE", "GHIAIONI", "VAF", "PUSTOT", "VINCOLO", "IG", "AF",
}

RES_2020 = {
    "A", "A0", "A6", "B", "B0", "B.0", "BG", "C", "CR", "R", "CASE", "ARCH", "BZ", "RU",
    "PDL APPROVATI",
}

IND_2020 = {
    "D", "DF", "DH", "D/H", "D2/H2", "D3H3", "HD", "H2D2", "CH",
    "AI", "AD", "DS", "CAVA", "CARBURANTI", "SDI", "I", "H",
    "TV", "TELECOM", "ZM", "Z.T.", "M", "MIL", "MILITARE", "DM", "DEMANIO MILITARE",
    "DEMANIALE", "AREE DI SERVITÙ MILITARI", "BENI DI PERTINENZA DELL?AMMINISTRAZIONE MILITARE",
    "AM",
}

SER_2020 = {
    "P", "G", "O", "G1", "GH", "GOLF", "L", "L1", "N", "SCALO", "FERROVIA", "FERROVIARIA",
    "FERR", "VIAB", "SVIAB", "FS", "ACT", "AS", "A/SAAD", "HOTEL", "ST", "PRPC",
    "AMBITI TERRITORIALI SPECIALI", "AMBITO INTERVENTO ATTUATIVO", "AMBITO UNITARIO", "AMBITO",
    "S", "ZVP", "ALTRI ELEMENTI", "RFI", "RISPOSTO", "RISPETTO", "AT", "TV", "VINCOLO",
}

GREEN_2020 = {
    "V", "VP", "VP ", "VP.", "VP,", "VP;", "VP:",  # tolerate weird formatting if it exists
    "VP", "VP",  # harmless duplicates
    "VP", "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    # actual codes
    "V", "VP", "VP", "VP",
    "VP", "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    "VP",
    # keep the real list:
    "V", "VP", "Vp", "VPR", "VR", "VERDE", "VERDE PRIVATO", "VERDE_ARCH", "V.P.", "BO", "RV", "VS",
    "RIPRISTINO AMBIENTALE", "RIPRISTINO", "MITIGAZIONE", "MP",
}


def norm_code(x) -> str:
    """Normalize zona_omoge codes for robust matching (code-only, no descriptions)."""
    s = "" if x is None else str(x)
    s = s.strip().upper()
    # normalize common variants
    s = s.replace("N°", "N°")  # keep but normalized already by upper()
    s = " ".join(s.split())    # collapse multiple spaces
    return s


def classify_macro_2020_codeonly(code_series: pd.Series) -> pd.Series:
    codes = code_series.apply(norm_code)

    out = pd.Series("unclassified", index=codes.index, dtype="object")

    out[codes.isin(AG_2020)] = "agriculture"
    out[codes.isin(NAT_2020)] = "natural_green"
    out[codes.isin(RES_2020)] = "residential"
    out[codes.isin(IND_2020)] = "non_residential_industry"
    out[codes.isin(SER_2020)] = "services_infrastructure"
    out[codes.isin(GREEN_2020)] = "green_urban"

    return out


def area_km2(gdf: gpd.GeoDataFrame) -> pd.Series:
    """In EPSG:6708 (meters), geometry.area is m²."""
    return gdf.geometry.area / 1_000_000.0


def save_layer(gdf: gpd.GeoDataFrame, gpkg_path: Path, layer_name: str) -> None:
    if gdf is None or gdf.empty:
        return
    gdf.to_file(gpkg_path, layer=layer_name, driver="GPKG")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    print("Reading municipalities…")
    muni = read_vector(MUNI_FP)

    missing = [c for c in (MUNI_ID, MUNI_NAME) if c not in muni.columns]
    if missing:
        raise ValueError(f"{MUNI_FP.name} missing fields: {missing}")

    muni = muni[[MUNI_ID, MUNI_NAME, "geometry"]].copy()
    muni = fix_geoms(muni)
    muni = to_canon(muni, CRS_CANON)

    # Dissolve municipalities into one mask for clipping
    muni_union = union_mask(muni)

    print("Reading hazards…")
    hazards: dict[str, gpd.GeoDataFrame] = {}
    for hz, fp in HAZ_FPS.items():
        if not fp.exists():
            raise FileNotFoundError(fp)

        g = read_vector(fp)
        g = fix_geoms(g)
        g = to_canon(g, CRS_CANON)
        g = gpd.clip(g, muni_union)
        hazards[hz] = g

        if SAVE_GPKG:
            save_layer(g, OUT_GPKG, layer_name=f"{hz}_clip")

    long_rows: list[pd.DataFrame] = []

    # -------------------------------------------------------------------------
    # 1950–2000 (MOLAND shapefiles)
    # -------------------------------------------------------------------------
    for year, fp in LU_FPS.items():
        if not fp.exists():
            raise FileNotFoundError(fp)

        print(f"Processing land use {year}…")
        lu = read_vector(fp)

        if LU_LEGENDA not in lu.columns:
            raise ValueError(f"{fp.name} is missing field '{LU_LEGENDA}'")

        lu = lu[[LU_LEGENDA, "geometry"]].copy()
        lu = fix_geoms(lu)
        lu = to_canon(lu, CRS_CANON)
        lu = gpd.clip(lu, muni_union)

        lu["macro_class"] = classify_macro(lu[LU_LEGENDA])

        uncl = lu[lu["macro_class"] == "unclassified"]
        if not uncl.empty:
            print(f"  Unclassified examples in {year} (top 20):")
            print(uncl[LU_LEGENDA].value_counts().head(20).to_string())

        lu_m = gpd.overlay(
            lu,
            muni[[MUNI_ID, MUNI_NAME, "geometry"]],
            how="intersection",
            keep_geom_type=True,
        )

        lu_m["year"] = int(year)
        lu_m["hazard"] = "TOTAL"
        lu_m["area_km2"] = area_km2(lu_m)

        long_rows.append(lu_m[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

        if SAVE_GPKG:
            save_layer(lu_m, OUT_GPKG, layer_name=f"lu_{year}_x_muni")

        for hz, hz_gdf in hazards.items():
            print(f"  - intersecting with {hz}…")
            exp = gpd.overlay(lu_m, hz_gdf[["geometry"]], how="intersection", keep_geom_type=True)
            exp["hazard"] = hz
            exp["area_km2"] = area_km2(exp)
            long_rows.append(exp[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

            if SAVE_GPKG:
                save_layer(exp, OUT_GPKG, layer_name=f"lu_{year}_{hz}_exp")

    # -------------------------------------------------------------------------
    # 2020 (GPKG: two layers)
    # -------------------------------------------------------------------------
    if not LU2020_GPKG.exists():
        raise FileNotFoundError(LU2020_GPKG)

    print("Processing land use 2020 (GPKG)…")

    # LAYER_1: zoning -> macro by zona_omoge
    z = read_vector(LU2020_GPKG, layer=LU2020_LAYER_1)
    if LU2020_CODECOL not in z.columns:
        raise ValueError(f"2020 layer '{LU2020_LAYER_1}' missing column '{LU2020_CODECOL}'")

    z = z[[LU2020_CODECOL, "geometry"]].copy()
    z = fix_geoms(z)
    z = to_canon(z, CRS_CANON)
    z = gpd.clip(z, muni_union)

    z["macro_class"] = classify_macro_2020_codeonly(z[LU2020_CODECOL])

    uncl2020 = z[z["macro_class"] == "unclassified"].copy()
    if not uncl2020.empty:
        qc_2020 = (
            uncl2020.assign(area_km2=area_km2(uncl2020), code=uncl2020[LU2020_CODECOL].apply(norm_code))
            .groupby("code", as_index=False)
            .agg(n=("code", "size"), area_km2=("area_km2", "sum"))
            .sort_values("area_km2", ascending=False)
        )
        qc_2020.to_csv(OUT_QC_UNCL_2020, index=False)
        print(f"  2020 unclassified codes QC written: {OUT_QC_UNCL_2020}")
        print("  Top 20 unclassified 2020 codes:")
        print(qc_2020.head(20).to_string(index=False))

    z_m = gpd.overlay(
        z,
        muni[[MUNI_ID, MUNI_NAME, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    z_m["year"] = 2020
    z_m["hazard"] = "TOTAL"
    z_m["area_km2"] = area_km2(z_m)

    long_rows.append(z_m[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

    if SAVE_GPKG:
        save_layer(z_m, OUT_GPKG, layer_name="lu_2020_zonizzazione_x_muni")

    for hz, hz_gdf in hazards.items():
        print(f"  - intersecting 2020 zoning with {hz}…")
        exp = gpd.overlay(z_m, hz_gdf[["geometry"]], how="intersection", keep_geom_type=True)
        exp["hazard"] = hz
        exp["area_km2"] = area_km2(exp)
        long_rows.append(exp[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

        if SAVE_GPKG:
            save_layer(exp, OUT_GPKG, layer_name=f"lu_2020_zonizzazione_{hz}_exp")

    # LAYER_2: servizi_area -> all services_infrastructure
    s = read_vector(LU2020_GPKG, layer=LU2020_LAYER_2)
    s = s[["geometry"]].copy()
    s = fix_geoms(s)
    s = to_canon(s, CRS_CANON)
    s = gpd.clip(s, muni_union)
    s["macro_class"] = "services_infrastructure"

    s_m = gpd.overlay(
        s,
        muni[[MUNI_ID, MUNI_NAME, "geometry"]],
        how="intersection",
        keep_geom_type=True,
    )
    s_m["year"] = 2020
    s_m["hazard"] = "TOTAL"
    s_m["area_km2"] = area_km2(s_m)

    long_rows.append(s_m[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

    if SAVE_GPKG:
        save_layer(s_m, OUT_GPKG, layer_name="lu_2020_servizi_x_muni")

    for hz, hz_gdf in hazards.items():
        print(f"  - intersecting 2020 servizi with {hz}…")
        exp = gpd.overlay(s_m, hz_gdf[["geometry"]], how="intersection", keep_geom_type=True)
        exp["hazard"] = hz
        exp["area_km2"] = area_km2(exp)
        long_rows.append(exp[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]])

        if SAVE_GPKG:
            save_layer(exp, OUT_GPKG, layer_name=f"lu_2020_servizi_{hz}_exp")

    # -------------------------------------------------------------------------
    # Final aggregation + outputs (all years including 2020)
    # -------------------------------------------------------------------------
    if not long_rows:
        raise RuntimeError("No rows produced. Check that your inputs are not empty after clipping.")

    print("Aggregating to long table…")
    df_long = pd.concat(long_rows, ignore_index=True)
    df_long["area_km2"] = pd.to_numeric(df_long["area_km2"], errors="coerce").fillna(0.0)

    df_long = (
        df_long.groupby([MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class"], as_index=False)
        .agg(area_km2=("area_km2", "sum"))
    )

    df_long.to_csv(OUT_LONG, index=False)
    print(f"Wrote: {OUT_LONG}")

    # -------------------------------------------------------------------------
    # QC: unclassified labels for MOLAND years only (1950–2000)
    # -------------------------------------------------------------------------
    print("Running classification QC (unclassified labels) for MOLAND years…")

    qc_rows: list[pd.DataFrame] = []
    for year, fp in LU_FPS.items():
        lu = read_vector(fp)
        lu = lu[[LU_LEGENDA, "geometry"]].copy()
        lu = fix_geoms(lu)
        lu = to_canon(lu, CRS_CANON)
        lu = gpd.clip(lu, muni_union)

        lu["macro_class"] = classify_macro(lu[LU_LEGENDA])
        lu["area_km2"] = area_km2(lu)
        lu["year"] = int(year)
        qc_rows.append(lu[[LU_LEGENDA, "macro_class", "year", "area_km2"]])

    qc_df = pd.concat(qc_rows, ignore_index=True)

    qc_uncl = (
        qc_df[qc_df["macro_class"] == "unclassified"]
        .groupby(["year", LU_LEGENDA], as_index=False)
        .agg(area_km2=("area_km2", "sum"))
    )

    qc_tot = qc_df.groupby("year", as_index=False).agg(total_km2=("area_km2", "sum"))
    qc_uncl = qc_uncl.merge(qc_tot, on="year", how="left")
    qc_uncl["pct_total"] = 100.0 * qc_uncl["area_km2"] / qc_uncl["total_km2"]
    qc_uncl = qc_uncl.sort_values(["year", "area_km2"], ascending=[True, False])

    qc_uncl.to_csv(OUT_QC_UNCL, index=False)
    print(f"QC file written: {OUT_QC_UNCL}")

    # -------------------------------------------------------------------------
    # Wide table
    # -------------------------------------------------------------------------
    print("\nPivoting to wide table…")
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