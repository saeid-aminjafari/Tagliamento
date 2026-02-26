"""
Tagliamento land-use & flood exposure — figure script (tidied, same features)

What’s cleaned up (without changing what the script does):
- Centralized constants / labels
- Removed duplicated helper functions (_get_muni_area_km2 was defined twice)
- Consistent key typing (municipality ID as str everywhere)
- Small structure improvements + clearer function boundaries
- Kept plotting logic, ordering, colormaps, legends, filenames, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon

CRS_CANON = "EPSG:6708"

# Optional for maps
try:
    import geopandas as gpd
except Exception:
    gpd = None


# -----------------------------
# USER SETTINGS (EDIT THIS)
# -----------------------------
BASE = Path(r"C:\Users\saeed\OneDrive\TagRiv260205")
OUT = BASE / "outputs"
FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

LONG_CSV = OUT / "tagliamento_long.csv"

# If you want maps (Fig 4 / panel maps / scatters), set this to the municipality shapefile
MUNI_SHP = BASE / "raw" / "COMUNI_TAGLIAMENTO.shp"  # adjust if needed

MUNI_ID = "PRO_COM"
MUNI_NAME = "COMUNE"

YEAR0 = 1950
YEAR1 = 2020

SNAPSHOT_YEAR = 2020  # (kept; used by your workflow conceptually)

MACRO_ORDER = [
    "agriculture",
    "natural_green",
    "non_residential_industry",
    "residential",
    "services_infrastructure",
    "green_urban",
    "water_body",
    "unclassified",
]

HAZ_ORDER = ["HPH", "MPH", "LPH"]  # display order

MACRO_LABELS: Dict[str, str] = {
    "agriculture": "Agriculture",
    "natural_green": "Natural Green",
    "non_residential_industry": "Non-residential Industry",
    "residential": "Residential",
    "services_infrastructure": "Services/Infrastructure",
    "green_urban": "Urban Green",
    "water_body": "Water Bodies",
    "unclassified": "Unclassified",
}

# Semantic colors (edit as you like)
MACRO_COLORS = {
    "water_body": "#4E79A7",                # blue
    "natural_green": "#59A14F",             # green
    "green_urban": "#8CD17D",               # light green
    "agriculture": "#F28E2B",               # orange/yellow (fields)
    "residential": "#E15759",      # red-ish (built-up)
    "non_residential_industry": "#9C755F",  # brown/gray (industrial)
    "services_infrastructure": "#EDC948",   # yellow (infrastructure)
    "unclassified": "#BAB0AC",                     # gray
}

# ============================================================
# HELPERS
# ============================================================
def require_geopandas() -> None:
    if gpd is None:
        raise ImportError("GeoPandas is required for map-based figures / municipality-normalization.")


def make_muni_id_table(muni_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Stable municipality labels 1..N ordered North -> South (by latitude).
    Uses representative_point() (always inside polygon) and sorts by Y descending.
    Tie-breaker: X ascending, then PRO_COM to keep deterministic.
    """
    require_geopandas()

    m = muni_gdf[[MUNI_ID, MUNI_NAME, "geometry"]].copy()
    m[MUNI_ID] = m[MUNI_ID].astype(str)
    m = m.drop_duplicates(subset=[MUNI_ID]).copy()

    # Ensure a projected CRS for robust representative points (and consistent units)
    if m.crs is None:
        raise ValueError("Municipality layer has no CRS. Define it in QGIS, then rerun.")
    m = m.to_crs(CRS_CANON)

    rp = m.representative_point()
    m["_x"] = rp.x
    m["_y"] = rp.y

    # North -> South: higher Y first
    m = m.sort_values(by=["_y", "_x", MUNI_ID], ascending=[False, True, True]).reset_index(drop=True)

    out = m[[MUNI_ID, MUNI_NAME]].copy()
    out["pt_id"] = np.arange(1, len(out) + 1)

    return out

# Build ID table (North -> South)
muni_for_ids = gpd.read_file(MUNI_SHP)
id_table = make_muni_id_table(muni_for_ids)

# Ensure consistent types
id_table[MUNI_ID] = id_table[MUNI_ID].astype(str)
id_table["pt_id"] = id_table["pt_id"].astype(int)

# Save permanently
ID_CSV = OUT / "Municipality_ID_table_pt_id.csv"
id_table.to_csv(ID_CSV, index=False)

print("Saved ID table to:", ID_CSV)


def add_muni_ids_to_axes(
    ax,
    muni_gdf: "gpd.GeoDataFrame",
    id_table: pd.DataFrame,
    font_size: int = 7,
) -> None:
    """Adds pt_id labels at municipality representative points (in map CRS)."""
    m = muni_gdf.copy()
    m[MUNI_ID] = m[MUNI_ID].astype(str)
    m = m.merge(id_table[[MUNI_ID, "pt_id"]], on=MUNI_ID, how="left")

    pts = m.representative_point()
    for (x, y, pid) in zip(pts.x, pts.y, m["pt_id"]):
        if pd.notna(pid):
            ax.text(x, y, str(int(pid)), fontsize=font_size, ha="center", va="center")


def get_muni_area_km2(muni_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Return DataFrame with municipality area in km² computed from geometry (recommended).
    We do NOT trust Area_kmq because it is often rounded / inconsistent.
    """
    muni = muni_gdf.copy()
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    if muni.crs is None:
        raise ValueError("Municipality layer has no CRS. Define it in QGIS, then rerun.")

    muni = muni.to_crs(CRS_CANON)

    a = muni[[MUNI_ID]].copy()
    a["muni_area_km2"] = muni.geometry.area / 1_000_000.0

    a = a.dropna(subset=["muni_area_km2"]).drop_duplicates(subset=[MUNI_ID])
    return a


def compute_delta_pct_by_muni(
    df_in: pd.DataFrame,
    year0: int,
    year1: int,
    macro: str,
    muni_area: pd.DataFrame,
    zero_tol: float = 0.01,
) -> pd.DataFrame:
    """
    Municipality delta as % of municipality area for one macro class, TOTAL only.
    Δ% = 100 * (A_y1 - A_y0) / A_muni
    """
    d = df_in[(df_in["hazard"] == "TOTAL") & (df_in["macro_class"] == macro)].copy()

    y0 = (
        d[d["year"] == year0]
        .groupby([MUNI_ID], as_index=False, observed=False)["area_km2"]
        .sum()
        .rename(columns={"area_km2": "a0"})
    )
    y1 = (
        d[d["year"] == year1]
        .groupby([MUNI_ID], as_index=False, observed=False)["area_km2"]
        .sum()
        .rename(columns={"area_km2": "a1"})
    )

    out = y0.merge(y1, on=MUNI_ID, how="outer").fillna(0.0)
    out = out.merge(muni_area, on=MUNI_ID, how="left")

    out["muni_area_km2"] = pd.to_numeric(out["muni_area_km2"], errors="coerce")
    out = out.dropna(subset=["muni_area_km2"]).copy()

    out["delta_pct"] = 100.0 * (out["a1"] - out["a0"]) / out["muni_area_km2"]

    if zero_tol > 0:
        out.loc[out["delta_pct"].abs() < zero_tol, "delta_pct"] = 0.0

    return out[[MUNI_ID, "delta_pct"]]


def macro_share_pct_by_year(
    df_in: pd.DataFrame,
    year: int,
    macro: str,
    muni_area: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each municipality: macro area in that year (TOTAL) as % of municipality area.
    Returns: PRO_COM, share_pct
    """
    d = df_in[(df_in["hazard"] == "TOTAL") & (df_in["macro_class"] == macro) & (df_in["year"] == year)].copy()
    g = d.groupby([MUNI_ID], as_index=False, observed=False)["area_km2"].sum()
    g[MUNI_ID] = g[MUNI_ID].astype(str)

    out = g.merge(muni_area, on=MUNI_ID, how="left")
    out = out.dropna(subset=["muni_area_km2"]).copy()
    out["share_pct"] = 100.0 * out["area_km2"] / out["muni_area_km2"]
    return out[[MUNI_ID, "share_pct"]]
# -----------------------------
# LOAD + BASIC CLEANUP
# -----------------------------
df = pd.read_csv(LONG_CSV)

# -----------------------------
# ADD not_mapped_2020 relative to MOLAND footprint (Option B)
# -----------------------------
NOT_MAPPED_LABEL = "not_mapped_2020"

if NOT_MAPPED_LABEL not in MACRO_ORDER:
    MACRO_ORDER = MACRO_ORDER + [NOT_MAPPED_LABEL]
MACRO_LABELS[NOT_MAPPED_LABEL] = "Not mapped (2020)"
MACRO_COLORS[NOT_MAPPED_LABEL] = "#D0D0D0"

# ensure types (do this BEFORE any grouping)
df["year"] = df["year"].astype(int)
df["hazard"] = df["hazard"].astype(str)
df["macro_class"] = df["macro_class"].astype(str)
df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce").fillna(0.0)
df[MUNI_ID] = df[MUNI_ID].astype(str)

# Load municipality names ONLY for attaching labels to synthetic rows
require_geopandas()
muni = gpd.read_file(MUNI_SHP)
muni[MUNI_ID] = muni[MUNI_ID].astype(str)
names = muni[[MUNI_ID, MUNI_NAME]].drop_duplicates(subset=[MUNI_ID])

ref_years = [1950, 1970, 1980, 2000]

# 1) reference footprint per municipality = mean mapped area over ref years
ref_footprint = (
    df[(df["hazard"] == "TOTAL") & (df["year"].isin(ref_years))]
    .groupby([MUNI_ID, "year"], as_index=False)["area_km2"].sum()
    .groupby(MUNI_ID, as_index=False)["area_km2"].mean()
    .rename(columns={"area_km2": "ref_km2"})
)

# 2) mapped 2020 per municipality
mapped_2020 = (
    df[(df["hazard"] == "TOTAL") & (df["year"] == 2020)]
    .groupby(MUNI_ID, as_index=False)["area_km2"].sum()
    .rename(columns={"area_km2": "mapped_km2"})
)

# 3) missing in 2020 relative to reference footprint (NOT municipality geometry)
miss = ref_footprint.merge(mapped_2020, on=MUNI_ID, how="left").fillna({"mapped_km2": 0.0})
miss["missing_km2"] = miss["ref_km2"] - miss["mapped_km2"]

# clamp tiny negatives
eps = 1e-3
miss.loc[miss["missing_km2"].between(-eps, 0), "missing_km2"] = 0.0

# hard fail only if seriously negative
if (miss["missing_km2"] < -eps).any():
    bad = miss.loc[miss["missing_km2"] < -eps, [MUNI_ID, "ref_km2", "mapped_km2", "missing_km2"]]
    raise ValueError(f"Some municipalities have mapped_2020 > ref footprint.\n{bad.head(10)}")

# add synthetic rows (TOTAL only)
add = miss.loc[miss["missing_km2"] > eps, [MUNI_ID, "missing_km2"]].copy()
if not add.empty:
    add = add.rename(columns={"missing_km2": "area_km2"})
    add["year"] = 2020
    add["hazard"] = "TOTAL"
    add["macro_class"] = NOT_MAPPED_LABEL
    add = add.merge(names, on=MUNI_ID, how="left")  # attach COMUNE (optional)
    df = pd.concat([df, add[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]]], ignore_index=True)

# apply categorical ordering AFTER adding synthetic rows
df["macro_class"] = pd.Categorical(df["macro_class"], categories=MACRO_ORDER, ordered=True)

# quick print check
tot_ref = (
    df[(df["hazard"] == "TOTAL") & (df["year"].isin(ref_years))]
    .groupby("year")["area_km2"].sum()
)
tot_2020 = (
    df[(df["hazard"] == "TOTAL") & (df["year"] == 2020)]
    .groupby("year")["area_km2"].sum()
)
print("TOTAL area check (km2):")
print(pd.concat([tot_ref, tot_2020]).to_string())

def _prep_pct_muni_area(df_in: pd.DataFrame, macros: List[str], muni_shp: Path) -> pd.DataFrame:
    require_geopandas()

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    df_key = df_in.copy()
    df_key[MUNI_ID] = df_key[MUNI_ID].astype(str)

    muni_area = get_muni_area_km2(muni)

    d = df_key[(df_key["hazard"] == "TOTAL") & (df_key["macro_class"].isin(macros))].copy()
    g = d.groupby(["year", "macro_class", MUNI_ID], as_index=False, observed=False)["area_km2"].sum()

    g = g.merge(muni_area, on=MUNI_ID, how="left").dropna(subset=["muni_area_km2"]).copy()
    g["pct_muni_area"] = (g["area_km2"] / g["muni_area_km2"]) * 100.0
    return g


def _friedman_by_macro(g: pd.DataFrame, macros: List[str]) -> pd.DataFrame:
    """
    Friedman test across years (repeated measures), per macro.
    Uses complete cases (municipalities present in ALL years for that macro).
    """
    out = []
    years = sorted(g["year"].unique())

    for m in macros:
        gm = g[g["macro_class"] == m].copy()
        wide = gm.pivot_table(index=MUNI_ID, columns="year", values="pct_muni_area", aggfunc="sum").dropna(axis=0)

        if wide.shape[0] < 5:  # too few municipalities for a meaningful test
            out.append({"macro_class": m, "n_muni": wide.shape[0], "friedman_p": np.nan})
            continue

        args = [wide[y].values for y in years]
        stat, p = friedmanchisquare(*args)
        out.append({"macro_class": m, "n_muni": wide.shape[0], "friedman_p": float(p)})

    res = pd.DataFrame(out)
    mask = res["friedman_p"].notna()
    res["friedman_p_fdr"] = np.nan
    if mask.any():
        res.loc[mask, "friedman_p_fdr"] = multipletests(res.loc[mask, "friedman_p"], method="fdr_bh")[1]
    return res

def _prep_pct_muni_area_total(df_in: pd.DataFrame, macros: List[str], muni_shp: Path) -> pd.DataFrame:
    """
    Build a long table with pct_muni_area (% of municipal area) for TOTAL land use.
    Output columns: year, macro_class, PRO_COM, pct_muni_area
    """
    require_geopandas()

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)
    muni_area = get_muni_area_km2(muni)

    d = df_in[(df_in["hazard"] == "TOTAL") & (df_in["macro_class"].isin(macros))].copy()
    d[MUNI_ID] = d[MUNI_ID].astype(str)

    g = d.groupby(["year", "macro_class", MUNI_ID], as_index=False, observed=False)["area_km2"].sum()
    g = g.merge(muni_area, on=MUNI_ID, how="left").dropna(subset=["muni_area_km2"]).copy()
    g["pct_muni_area"] = 100.0 * g["area_km2"] / g["muni_area_km2"]
    return g[["year", "macro_class", MUNI_ID, "pct_muni_area"]]

def pairwise_wilcoxon_consecutive_plus_endpoints_table_macros(
    df_in: pd.DataFrame,
    macros: List[str],
    muni_shp: Path,
    years: Optional[List[int]] = None,
    add_endpoint_pair: bool = True,
    p_adjust_within_macro: str = "holm",
    p_adjust_global: str = "fdr_bh",
    min_n: int = 8,
    out_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Pairwise Wilcoxon signed-rank tests on % municipality area (TOTAL), paired by municipality:
      - consecutive epoch pairs: (y1->y2), (y2->y3), ...
      - optionally also endpoint pair: (first->last), e.g., 1950->2020

    Output columns:
      macro_class, year_a, year_b, n_muni, median_a, median_b, median_diff,
      p_raw, p_adj_within_macro, p_adj_global
    """
    g = _prep_pct_muni_area_total(df_in, macros, muni_shp)

    if years is None:
        years = sorted(g["year"].unique().tolist())
    else:
        years = sorted(list(years))

    if len(years) < 2:
        return pd.DataFrame()

    # Build requested pairs
    pairs = list(zip(years[:-1], years[1:]))  # consecutive
    if add_endpoint_pair:
        endpoint = (years[0], years[-1])
        if endpoint not in pairs:
            pairs.append(endpoint)

    rows = []
    for m in macros:
        gm = g[g["macro_class"] == m].copy()
        if gm.empty:
            continue

        wide = gm.pivot_table(index=MUNI_ID, columns="year", values="pct_muni_area", aggfunc="sum")

        for ya, yb in pairs:
            if ya not in wide.columns or yb not in wide.columns:
                rows.append({
                    "macro_class": m, "year_a": ya, "year_b": yb, "n_muni": 0,
                    "median_a": np.nan, "median_b": np.nan, "median_diff": np.nan,
                    "p_raw": np.nan
                })
                continue

            pair = wide[[ya, yb]].dropna(axis=0)
            n = int(pair.shape[0])

            if n < min_n:
                rows.append({
                    "macro_class": m, "year_a": ya, "year_b": yb, "n_muni": n,
                    "median_a": float(np.nanmedian(pair[ya].values)) if n > 0 else np.nan,
                    "median_b": float(np.nanmedian(pair[yb].values)) if n > 0 else np.nan,
                    "median_diff": float(np.nanmedian((pair[yb] - pair[ya]).values)) if n > 0 else np.nan,
                    "p_raw": np.nan
                })
                continue

            diffs = (pair[yb] - pair[ya]).values

            try:
                stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided", method="auto")
            except TypeError:
                stat, p = wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")

            rows.append({
                "macro_class": m, "year_a": ya, "year_b": yb, "n_muni": n,
                "median_a": float(np.median(pair[ya].values)),
                "median_b": float(np.median(pair[yb].values)),
                "median_diff": float(np.median(diffs)),
                "p_raw": float(p)
            })

    res = pd.DataFrame(rows)

    # Within-macro correction across requested pairs (consecutive + endpoint)
    res["p_adj_within_macro"] = np.nan
    for m in res["macro_class"].dropna().unique():
        mask = (res["macro_class"] == m) & res["p_raw"].notna()
        if mask.sum() >= 1:
            res.loc[mask, "p_adj_within_macro"] = multipletests(
                res.loc[mask, "p_raw"].values, method=p_adjust_within_macro
            )[1]

    # Global correction across all macro×pair tests (optional sensitivity)
    res["p_adj_global"] = np.nan
    mask_all = res["p_raw"].notna()
    if mask_all.sum() >= 1:
        res.loc[mask_all, "p_adj_global"] = multipletests(
            res.loc[mask_all, "p_raw"].values, method=p_adjust_global
        )[1]

    # Order rows nicely: by macro then by (year_a, year_b)
    res = res.sort_values(["macro_class", "year_a", "year_b"]).reset_index(drop=True)

    if out_csv is not None:
        res.to_csv(out_csv, index=False)

    return res

# -----------------------------
# LOAD + BASIC CLEANUP
# -----------------------------
df = pd.read_csv(LONG_CSV)

# -----------------------------
# ADD not_mapped_2020 relative to MOLAND footprint (Option B)
# -----------------------------
NOT_MAPPED_LABEL = "not_mapped_2020"

if NOT_MAPPED_LABEL not in MACRO_ORDER:
    MACRO_ORDER = MACRO_ORDER + [NOT_MAPPED_LABEL]
MACRO_LABELS[NOT_MAPPED_LABEL] = "Not mapped (2020)"
MACRO_COLORS[NOT_MAPPED_LABEL] = "#D0D0D0"

# ensure types (do this BEFORE any grouping)
df["year"] = df["year"].astype(int)
df["hazard"] = df["hazard"].astype(str)
df["macro_class"] = df["macro_class"].astype(str)
df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce").fillna(0.0)
df[MUNI_ID] = df[MUNI_ID].astype(str)

# Load municipality names ONLY for attaching labels to synthetic rows
require_geopandas()
muni = gpd.read_file(MUNI_SHP)
muni[MUNI_ID] = muni[MUNI_ID].astype(str)
names = muni[[MUNI_ID, MUNI_NAME]].drop_duplicates(subset=[MUNI_ID])

ref_years = [1950, 1970, 1980, 2000]

# 1) reference footprint per municipality = mean mapped area over ref years
ref_footprint = (
    df[(df["hazard"] == "TOTAL") & (df["year"].isin(ref_years))]
    .groupby([MUNI_ID, "year"], as_index=False)["area_km2"].sum()
    .groupby(MUNI_ID, as_index=False)["area_km2"].mean()
    .rename(columns={"area_km2": "ref_km2"})
)

# 2) mapped 2020 per municipality
mapped_2020 = (
    df[(df["hazard"] == "TOTAL") & (df["year"] == 2020)]
    .groupby(MUNI_ID, as_index=False)["area_km2"].sum()
    .rename(columns={"area_km2": "mapped_km2"})
)

# 3) missing in 2020 relative to reference footprint (NOT municipality geometry)
miss = ref_footprint.merge(mapped_2020, on=MUNI_ID, how="left").fillna({"mapped_km2": 0.0})
miss["missing_km2"] = miss["ref_km2"] - miss["mapped_km2"]

# clamp tiny negatives
eps = 1e-3
miss.loc[miss["missing_km2"].between(-eps, 0), "missing_km2"] = 0.0

# hard fail only if seriously negative
if (miss["missing_km2"] < -eps).any():
    bad = miss.loc[miss["missing_km2"] < -eps, [MUNI_ID, "ref_km2", "mapped_km2", "missing_km2"]]
    raise ValueError(f"Some municipalities have mapped_2020 > ref footprint.\n{bad.head(10)}")

# add synthetic rows (TOTAL only)
add = miss.loc[miss["missing_km2"] > eps, [MUNI_ID, "missing_km2"]].copy()
if not add.empty:
    add = add.rename(columns={"missing_km2": "area_km2"})
    add["year"] = 2020
    add["hazard"] = "TOTAL"
    add["macro_class"] = NOT_MAPPED_LABEL
    add = add.merge(names, on=MUNI_ID, how="left")  # attach COMUNE (optional)
    df = pd.concat([df, add[[MUNI_ID, MUNI_NAME, "year", "hazard", "macro_class", "area_km2"]]], ignore_index=True)

# apply categorical ordering AFTER adding synthetic rows
df["macro_class"] = pd.Categorical(df["macro_class"], categories=MACRO_ORDER, ordered=True)

# quick print check
tot_ref = (
    df[(df["hazard"] == "TOTAL") & (df["year"].isin(ref_years))]
    .groupby("year")["area_km2"].sum()
)
tot_2020 = (
    df[(df["hazard"] == "TOTAL") & (df["year"] == 2020)]
    .groupby("year")["area_km2"].sum()
)
print("TOTAL area check (km2):")
print(pd.concat([tot_ref, tot_2020]).to_string())


df["year"] = df["year"].astype(int)
df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce").fillna(0.0)

# Keep consistent category ordering
df["macro_class"] = pd.Categorical(df["macro_class"], categories=MACRO_ORDER, ordered=True)
df["hazard"] = df["hazard"].astype(str)

# Keep muni IDs consistent everywhere
if MUNI_ID in df.columns:
    df[MUNI_ID] = df[MUNI_ID].astype(str)

# --- total area per year across ALL macro classes (TOTAL only) ---
tot_all = (
    df[df["hazard"] == "TOTAL"]
    .groupby(["year"], as_index=False)["area_km2"]
    .sum()
    .rename(columns={"area_km2": "sum_all_macros_km2"})
)
macros6 = [
    "agriculture",
    "natural_green",
    "non_residential_industry",
    "residential",
    "services_infrastructure",
    "green_urban",
]

# ============================================================
# FIG 2 — Study-area land-use composition over time (TOTAL only)
# ============================================================
def fig2_study_area_composition(df_in: pd.DataFrame, outpath: Path) -> None:
    d = df_in[df_in["hazard"] == "TOTAL"].copy()


    p = d.groupby(["year", "macro_class"], as_index=False, observed=False)["area_km2"].sum()
    wide = p.pivot(index="year", columns="macro_class", values="area_km2").fillna(0.0)

    # ✅ only keep macros6 (in that order)
    # Keep macros6 + the 2020 missing bucket (if present)
    keep = macros6 + [NOT_MAPPED_LABEL]
    wide = wide[[c for c in keep if c in wide.columns]]

    colors = [MACRO_COLORS.get(c, "#999999") for c in wide.columns]
    ax = wide.plot(kind="bar", stacked=True, figsize=(10, 6), color=colors)

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Area (km²)", fontsize=12)

    # legend labels consistent with other figures
    handles, labels = ax.get_legend_handles_labels()
    labels_pretty = [MACRO_LABELS.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, labels_pretty, bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


fig2_study_area_composition(df, FIGDIR / "Fig2_Land_use_composition_over_time.png")


# ============================================================
# FIG 3 and Table 3 — Boxplots across ALL municipalities (TOTAL), NORMALIZED
# ============================================================
def fig3_boxplot_two_panels_pct(
    df_in: pd.DataFrame,
    macros_high: List[str],
    macros_low: List[str],
    muni_shp: Path,
    outpath: Optional[Path] = None,
    add_sig_summary: bool = True,
    # per-panel legend placement
    legend_high_loc: str = "upper left",
    legend_high_anchor: Tuple[float, float] = (0.02, 0.98),
    legend_low_loc: str = "upper left",
    legend_low_anchor: Tuple[float, float] = (0.02, 0.98),
    # per-panel significance placement
    sig_high_anchor: Tuple[float, float] = (0.60, 0.98),
    sig_low_anchor: Tuple[float, float] = (0.60, 0.98),
    bottom_ylim: Optional[Tuple[float, float]] = (0.0, 15.0),
    top_ylim: Optional[Tuple[float, float]] = (0.0, 15.0),
) -> None:
    """
    Two-panel boxplots with a COMMON x-position grid so year labels align perfectly
    even when panels have different numbers of boxes per year.
    Legends + significance are split by panel (top: high macros; bottom: low macros).
    """
    macros_all = macros_high + macros_low
    g = _prep_pct_muni_area(df_in, macros_all, muni_shp)
    years = sorted(g["year"].unique())

    present = set(g["macro_class"].astype(str).unique().tolist())
    macros_high = [m for m in macros_high if m in present]
    macros_low  = [m for m in macros_low  if m in present]

    macro_colors = {m: MACRO_COLORS.get(m, "#999999") for m in (macros_high + macros_low)}

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 9),
        sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]}
    )

    # ---- COMMON GRID PARAMETERS ----
    K = max(len(macros_high), len(macros_low))
    group_gap = 1.5
    box_gap = 0.9
    start_pos = 1.0

    year_slot_positions: Dict[int, List[float]] = {}
    pos = start_pos
    for y in years:
        slots = []
        for _ in range(K):
            slots.append(pos)
            pos += box_gap
        year_slot_positions[y] = slots
        pos += group_gap

    year_centers = [float(np.mean(year_slot_positions[y])) for y in years]

    def _draw_panel_on_grid(ax, macros_panel: List[str], title: str):
        data, positions, colors = [], [], []
        for y in years:
            slots = year_slot_positions[y]
            for i, m in enumerate(macros_panel):
                vals = g[(g["year"] == y) & (g["macro_class"] == m)]["pct_muni_area"].values
                data.append(vals)
                positions.append(slots[i])
                colors.append(macro_colors[m])

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=True,
        )

        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.8)

        for median in bp["medians"]:
            median.set_color("black")
            median.set_linewidth(1.5)

        ax.set_title(title, loc="left", fontsize=12)
        ax.grid(axis="y", alpha=0.2)

    _draw_panel_on_grid(ax1, macros_high, "(a) High-share macro-classes")
    _draw_panel_on_grid(ax2, macros_low,  "(b) Low-share macro-classes")

    # X-axis: one tick per year at shared centers
    ax2.set_xticks(year_centers)
    ax2.set_xticklabels([str(y) for y in years], fontsize=12)

    ax1.set_ylabel("Share of municipality area (%)", fontsize=14)
    ax2.set_ylabel("Share of municipality area (%)", fontsize=14)
    ax2.set_xlabel("Year", fontsize=14)

    if bottom_ylim is not None:
        ax2.set_ylim(bottom_ylim[0], bottom_ylim[1])
    if top_ylim is not None:
        ax1.set_ylim(top_ylim[0], top_ylim[1])

    # ---- legends split by panel ----
    if macros_high:
        handles_high = [
            Patch(facecolor=macro_colors[m], edgecolor="black", label=MACRO_LABELS.get(m, m))
            for m in macros_high
        ]
        ax1.legend(
            handles=handles_high,
            loc=legend_high_loc,
            bbox_to_anchor=legend_high_anchor,
            borderaxespad=0.0,
            frameon=True,
            fontsize=12
        )

    if macros_low:
        handles_low = [
            Patch(facecolor=macro_colors[m], edgecolor="black", label=MACRO_LABELS.get(m, m))
            for m in macros_low
        ]
        ax2.legend(
            handles=handles_low,
            loc=legend_low_loc,
            bbox_to_anchor=legend_low_anchor,
            borderaxespad=0.0,
            frameon=True,
            fontsize=12
        )

    # ---- significance split by panel ----
    if add_sig_summary:
        stats = _friedman_by_macro(g, macros_high + macros_low)

        def stars(p):
            if pd.isna(p): return "n/a"
            if p < 0.001: return "***"
            if p < 0.01:  return "**"
            if p < 0.05:  return "*"
            return "ns"

        def _panel_sig_text(macros_list: List[str]) -> str:
            lines = ["Change significance (Friedman, FDR):\n"]
            for m in macros_list:
                row = stats.loc[stats["macro_class"] == m]
                p = row["friedman_p_fdr"].iloc[0] if len(row) else np.nan
                lines.append(f"{MACRO_LABELS.get(m, m)}: {stars(p)}")
            return "\n".join(lines)

        if macros_high:
            ax1.text(
                sig_high_anchor[0], sig_high_anchor[1],
                _panel_sig_text(macros_high),
                transform=ax1.transAxes,
                ha="left", va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none")
            )

        if macros_low:
            ax2.text(
                sig_low_anchor[0], sig_low_anchor[1],
                _panel_sig_text(macros_low),
                transform=ax2.transAxes,
                ha="left", va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="none")
            )

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()


fig3_boxplot_two_panels_pct(
    df,
    macros_high=["agriculture", "natural_green"],
    macros_low=["non_residential_industry", "residential", "services_infrastructure", "green_urban"],
    muni_shp=MUNI_SHP,
    outpath=FIGDIR / "Fig3_Distribution_across_municipalities_macroclass_year_normalized_two_panels.png",
    add_sig_summary=True,
    bottom_ylim=(0, 14),
    top_ylim=(0, 110),
    legend_high_anchor=(0.02, 0.98),
    sig_high_anchor=(0.55, 0.98),
    legend_low_anchor=(0.02, 0.98),
    sig_low_anchor=(0.55, 0.98),
)

PAIRWISE_CSV = FIGDIR / "TableS1_Wilcoxon_consecutive_plus_1950_2020_TOTAL_pct_muni_area.csv"

tbl_pw = pairwise_wilcoxon_consecutive_plus_endpoints_table_macros(
    df_in=df,
    macros=macros6,
    muni_shp=MUNI_SHP,
    years=[1950, 1970, 1980, 2000, 2020],
    add_endpoint_pair=True,
    min_n=8,
    out_csv=PAIRWISE_CSV,
)

print("Saved Wilcoxon table to:", PAIRWISE_CSV)
print(tbl_pw.to_string(index=False))

# ============================================================
# FIG 4 — Multi-panel municipality delta maps (PERCENT)
# 6 rows (macro classes) × 4 columns (time intervals)
# ============================================================
def fig4_delta_grid_pct(
    df_in: pd.DataFrame,
    muni_shp: Path,
    macros: List[str],
    intervals: List[Tuple[int, int]],
    outpath: Path,
) -> None:
    if gpd is None:
        print("GeoPandas not available -> skipping maps.")
        return

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    df2 = df_in.copy()
    df2[MUNI_ID] = df2[MUNI_ID].astype(str)

    muni_area = get_muni_area_km2(muni)

    delta_layers = {}
    all_vals = []

    for macro in macros:
        for (y0, y1) in intervals:
            delta = compute_delta_pct_by_muni(df2, y0, y1, macro, muni_area)
            m = muni.merge(delta, on=MUNI_ID, how="left")
            m["delta_pct"] = pd.to_numeric(m["delta_pct"], errors="coerce").fillna(0.0)
            delta_layers[(macro, y0, y1)] = m
            all_vals.append(m["delta_pct"].values)

    all_vals = np.concatenate(all_vals) if len(all_vals) else np.array([0.0])
    vmax = float(np.nanmax(np.abs(all_vals))) if np.isfinite(all_vals).any() else 1.0
    if vmax == 0:
        vmax = 1.0
    vmin = -vmax

    nrows, ncols = len(macros), len(intervals)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))

    # Remove left whitespace and tighten grid
    plt.subplots_adjust(left=0.08, right=0.92, top=0.96, bottom=0.04, wspace=0.02, hspace=0.02)


    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # diverging colormap for +/- changes
    plt.set_cmap("RdBu_r")

    for i, macro in enumerate(macros):
        for j, (y0, y1) in enumerate(intervals):
            ax = axes[i, j]
            layer = delta_layers[(macro, y0, y1)]

            nz = layer[layer["delta_pct"] != 0.0]
            z = layer[layer["delta_pct"] == 0.0]

            if len(nz) > 0:
                nz.plot(
                    column="delta_pct",
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    legend=False,
                    linewidth=0.2,
                    edgecolor="black",
                )

            if len(z) > 0:
                z.boundary.plot(ax=ax, linewidth=0.4, color="black")

            ax.set_axis_off()

            if i == 0:
                ax.set_title(f"{y0}–{y1}", fontsize=20)

            # Row labels (macro labels) — use text, NOT set_ylabel (axis is off)
            if j == 0:
                ax.text(
                -0.05, 0.5,  # slightly outside the left edge of the first column
                MACRO_LABELS.get(macro, macro),
                transform=ax.transAxes,
                va="center", ha="right",
                rotation=90,
                fontsize=18
                )

    # --------------------------------------------------
    # Colorbar on RIGHT side of grid
    # --------------------------------------------------
    # Reserve a bit more room on the right for the colorbar
    plt.subplots_adjust(left=0.08, right=0.88, top=0.96, bottom=0.04, wspace=0.02, hspace=0.02)

    cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])  # [left, bottom, width, height]

    sm = plt.cm.ScalarMappable(
    cmap=plt.get_cmap(),
    norm=plt.Normalize(vmin=vmin, vmax=vmax)
    )
    sm._A = []

    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Δ share of municipality area (%)", fontsize=20)
    cbar.ax.tick_params(labelsize=20)


    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


intervals_4 = [(1950, 1970), (1970, 1980), (1980, 2000), (2000, 2020)] #, (1950, 2020)

fig4_delta_grid_pct(
    df,
    MUNI_SHP,
    macros=macros6,
    intervals=intervals_4,
    outpath=FIGDIR / "Fig4_Municipality_level_landuse_change_share.png",
)

# ============================================================
# Fig 5, COMPACT PANEL — Baseline(1950), Baseline(2000), and Δ(1950→2000)
# for macros (rows) × 3 cols (1950 share, 2000 share, delta)
# ============================================================
def fig_panel_baseline_baseline_delta_pct(
    df_in: pd.DataFrame,
    muni_shp: Path,
    macros: List[str],
    year0: int,
    year1: int,
    outpath: Path,
    zero_tol: float = 0.01,
) -> None:
    if gpd is None:
        print("GeoPandas not available -> skipping panel maps.")
        return

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    muni_ids = make_muni_id_table(muni)  # kept (even if you comment labels out)
    muni_area = get_muni_area_km2(muni)

    layers = {}  # (macro, panel) -> GeoDataFrame with "val"
    base_vals_all = []
    delta_vals_all = []

    for macro in macros:
        b0 = macro_share_pct_by_year(df_in, year0, macro, muni_area).rename(columns={"share_pct": "val"})
        b1 = macro_share_pct_by_year(df_in, year1, macro, muni_area).rename(columns={"share_pct": "val"})

        tmp0 = b0.rename(columns={"val": "b0"})
        tmp1 = b1.rename(columns={"val": "b1"})
        dd = tmp0.merge(tmp1, on=MUNI_ID, how="outer").fillna(0.0)
        dd["val"] = dd["b1"] - dd["b0"]
        dd = dd[[MUNI_ID, "val"]]

        m0 = muni.merge(b0[[MUNI_ID, "val"]], on=MUNI_ID, how="left")
        m1 = muni.merge(b1[[MUNI_ID, "val"]], on=MUNI_ID, how="left")
        md = muni.merge(dd[[MUNI_ID, "val"]], on=MUNI_ID, how="left")

        for mm in (m0, m1, md):
            mm["val"] = pd.to_numeric(mm["val"], errors="coerce").fillna(0.0)
            if zero_tol > 0:
                mm.loc[mm["val"].abs() < zero_tol, "val"] = 0.0

        layers[(macro, "base0")] = m0
        layers[(macro, "base1")] = m1
        layers[(macro, "delta")] = md

        base_vals_all.append(m0["val"].values)
        base_vals_all.append(m1["val"].values)
        delta_vals_all.append(md["val"].values)

    base_vals_all = np.concatenate(base_vals_all) if base_vals_all else np.array([0.0])
    delta_vals_all = np.concatenate(delta_vals_all) if delta_vals_all else np.array([0.0])

    base_vmax = float(np.nanmax(base_vals_all)) if np.isfinite(base_vals_all).any() else 1.0
    if base_vmax == 0:
        base_vmax = 1.0
    base_vmin = 0.0

    delta_vmax = float(np.nanmax(np.abs(delta_vals_all))) if np.isfinite(delta_vals_all).any() else 1.0
    if delta_vmax == 0:
        delta_vmax = 1.0
    delta_vmin = -delta_vmax

    nrows = len(macros)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))
    if nrows == 1:
        axes = np.array([axes])

    cmap_base = "viridis"
    cmap_delta = "RdBu_r"

    col_titles = [f"{year0} (%)", f"{year1} (%)", f"Δ {year0}→{year1}"]

    for i, macro in enumerate(macros):
        for j, panel in enumerate(["base0", "base1", "delta"]):
            ax = axes[i, j]
            layer = layers[(macro, panel)].copy()

            nz = layer[layer["val"] != 0.0]
            z = layer[layer["val"] == 0.0]

            if panel == "delta":
                if len(nz) > 0:
                    nz.plot(
                        column="val",
                        ax=ax,
                        vmin=delta_vmin,
                        vmax=delta_vmax,
                        cmap=cmap_delta,
                        legend=False,
                        linewidth=0.2,
                        edgecolor="black",
                    )
            else:
                if len(nz) > 0:
                    nz.plot(
                        column="val",
                        ax=ax,
                        vmin=base_vmin,
                        vmax=base_vmax,
                        cmap=cmap_base,
                        legend=False,
                        linewidth=0.2,
                        edgecolor="black",
                    )

            if len(z) > 0:
                z.boundary.plot(ax=ax, linewidth=0.4, color="black")

            # If you ever want the numeric IDs on panels, just uncomment:
            # add_muni_ids_to_axes(ax=ax, muni_gdf=layer, id_table=muni_ids, font_size=7)

            ax.set_axis_off()

            if i == 0:
                ax.set_title(col_titles[j], fontsize=20)

            if j == 0:
                ax.text(
                    0.01,
                    0.5,
                    MACRO_LABELS.get(macro, macro),
                    transform=ax.transAxes,
                    va="center",
                    ha="left",
                    rotation=90,
                    fontsize=18,
                )

    # ---- Colorbars aligned with SECOND ROW (same as your original intent) ----
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])

    sm_base = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_base), norm=plt.Normalize(vmin=base_vmin, vmax=base_vmax))
    sm_base._A = []
    sm_delta = plt.cm.ScalarMappable(cmap=plt.get_cmap(cmap_delta), norm=plt.Normalize(vmin=delta_vmin, vmax=delta_vmax))
    sm_delta._A = []

    row = 1 if nrows > 1 else 0

    bbox_col2 = axes[row, 1].get_position()
    bbox_col3 = axes[row, 2].get_position()

    y0 = bbox_col2.y0
    h = bbox_col2.height

    gap_left = bbox_col2.x1
    gap_right = bbox_col3.x0
    gap = gap_right - gap_left

    cbar_w = min(0.018, gap * 0.6) if gap > 0 else 0.018
    cbar_x = gap_left + (gap - cbar_w) / 2.0 if gap > 0 else (bbox_col2.x1 + 0.01)

    cax1 = fig.add_axes([cbar_x, y0, cbar_w, h])
    cbar1 = fig.colorbar(sm_base, cax=cax1)
    cbar1.set_label("Share of municipality area (%)", fontsize=20)
    cbar1.ax.tick_params(labelsize=20)

    cax2 = fig.add_axes([bbox_col3.x1 + 0.015, y0, 0.018, h])
    cbar2 = fig.colorbar(sm_delta, cax=cax2)
    cbar2.set_label("Δ share (%)", fontsize=20)
    cbar2.ax.tick_params(labelsize=20)

    plt.savefig(outpath, dpi=300)
    plt.close()


fig_panel_baseline_baseline_delta_pct(
    df_in=df,
    muni_shp=MUNI_SHP,
    macros=["agriculture", "natural_green", "residential", "services_infrastructure"],
    year0=1950,
    year1=2020,
    outpath=FIGDIR / "Fig5_Baseline and long-term change municipality-normalized.png",
    zero_tol=0.01,
)


# ============================================================
# FIG 6 — Scatter: baseline share (%) vs Δ share (%) by municipality
# ============================================================

def fig6_scatter_three_vertical_panels(
    df_in: pd.DataFrame,
    muni_shp: Path,
    macros: List[str],
    year0: int,
    year1: int,
    outpath: Path,
    highlight_ids: List[int] = [41],   # e.g., [15]
    label_top_n: int = 8,              # label top-N |Δ| per panel
    label_extreme_baseline_n: int = 0, # optionally label top-N baseline too
    font_size: int = 12,
    point_size: int = 35,
    highlight_size: int = 90,
):

    if highlight_ids is None:
        highlight_ids = [41]

    if gpd is None:
        print("GeoPandas not available -> skipping scatter.")
        return

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    muni_area = get_muni_area_km2(muni)
    muni_ids = make_muni_id_table(muni)

    fig, axes = plt.subplots(len(macros), 1, figsize=(8.5, 10.5))
    if len(macros) == 1:
        axes = [axes]

    for i, macro in enumerate(macros):
        ax = axes[i]

        base = macro_share_pct_by_year(df_in, year0, macro, muni_area).rename(columns={"share_pct": "baseline_pct"})
        end  = macro_share_pct_by_year(df_in, year1, macro, muni_area).rename(columns={"share_pct": "end_pct"})

        x = base.merge(end, on=MUNI_ID, how="outer").fillna(0.0)
        x["delta_pct"] = x["end_pct"] - x["baseline_pct"]

        x = muni_ids.merge(x, on=MUNI_ID, how="left").fillna(
            {"baseline_pct": 0.0, "end_pct": 0.0, "delta_pct": 0.0}
        )

        # ---- choose which points to label ----
        to_label = set()

        # always label explicit highlight IDs if present
        for hid in highlight_ids:
            if hid in set(x["pt_id"].astype(int).tolist()):
                to_label.add(int(hid))

        # label top-N by |delta|
        if label_top_n and label_top_n > 0:
            top_delta = x.assign(absd=x["delta_pct"].abs()).nlargest(label_top_n, "absd")
            to_label.update(top_delta["pt_id"].astype(int).tolist())

        # optionally label top-N baseline (extreme right side)
        if label_extreme_baseline_n and label_extreme_baseline_n > 0:
            top_base = x.nlargest(label_extreme_baseline_n, "baseline_pct")
            to_label.update(top_base["pt_id"].astype(int).tolist())

        # ---- plot base points ----
        ax.scatter(x["baseline_pct"], x["delta_pct"], s=point_size, alpha=0.9)

        # ---- highlight specific municipalities ----
        mask_h = x["pt_id"].astype(int).isin(highlight_ids)
        if mask_h.any():
            ax.scatter(
                x.loc[mask_h, "baseline_pct"],
                x.loc[mask_h, "delta_pct"],
                s=highlight_size,
                edgecolor="black",
                linewidth=1.2,
                zorder=5
            )

        # reference lines
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

        # panel-specific limits (keep RS readable)
        x_margin = 0.06 * (x["baseline_pct"].max() - x["baseline_pct"].min() + 1e-6)
        y_margin = 0.08 * (x["delta_pct"].max() - x["delta_pct"].min() + 1e-6)

        ax.set_xlim(x["baseline_pct"].min() - x_margin, x["baseline_pct"].max() + x_margin)
        ax.set_ylim(x["delta_pct"].min() - y_margin, x["delta_pct"].max() + y_margin)

        # titles / labels
        ax.set_title(MACRO_LABELS.get(macro, macro), fontsize=14)
        ax.set_ylabel("Δ share (%)", fontsize=14)
        ax.tick_params(labelsize=12)

        # annotate only selected
        for _, r in x.iterrows():
            pid = int(r["pt_id"])
            if pid not in to_label:
                continue
            ax.annotate(
                str(pid),
                (r["baseline_pct"], r["delta_pct"]),
                fontsize=font_size,
                xytext=(4, 4),
                textcoords="offset points",
                zorder=6
            )

        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.4)

        # optional: small note if you want
        # ax.text(0.01, 0.95, f"labels: top {label_top_n} |Δ| + highlights", transform=ax.transAxes,
        #         ha="left", va="top", fontsize=10)

    axes[-1].set_xlabel(f"Baseline in {year0} (% of municipality area)", fontsize=14)

    plt.subplots_adjust(hspace=0.32)
    plt.savefig(outpath, dpi=300)
    plt.close()

fig6_scatter_three_vertical_panels(
    df,
    MUNI_SHP,
    macros=["agriculture", "natural_green", "residential", "services_infrastructure"],
    year0=1950,
    year1=2020,
    outpath=FIGDIR / "Fig6_Three_vertical_scatter_panels.png",
    highlight_ids=[41],        # highlight Lignano (pt_id 15)
    label_top_n=6,             # label top 6 |Δ| in each panel
    label_extreme_baseline_n=0 # optionally label 2 highest-baseline points
)


#Extras, Separated pannels
def fig_scatter_baseline_vs_delta_with_ids(
    df_in: pd.DataFrame,
    muni_shp: Path,
    macro: str,
    year0: int,
    year1: int,
    outpath: Path,
    annotate_all: bool = True,
    font_size: int = 12,
) -> pd.DataFrame:
    """
    Scatter: x = baseline share (%) in year0, y = Δ share (%) year0→year1 (pp),
    per municipality. Each point gets a numeric ID (pt_id).
    Returns a table mapping pt_id -> municipality + values.
    """
    if gpd is None:
        print("GeoPandas not available -> skipping scatter.")
        return pd.DataFrame()

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    muni_area = get_muni_area_km2(muni)
    muni_ids = make_muni_id_table(muni)

    base = macro_share_pct_by_year(df_in, year0, macro, muni_area).rename(columns={"share_pct": "baseline_pct"})
    end = macro_share_pct_by_year(df_in, year1, macro, muni_area).rename(columns={"share_pct": "end_pct"})

    x = base.merge(end, on=MUNI_ID, how="outer").fillna(0.0)
    x["delta_pct"] = x["end_pct"] - x["baseline_pct"]

    x = muni_ids.merge(x, on=MUNI_ID, how="left").fillna(
        {"baseline_pct": 0.0, "end_pct": 0.0, "delta_pct": 0.0}
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x["baseline_pct"], x["delta_pct"])
    ax.axhline(0, linewidth=0.8)

    ax.set_xlabel(f"Baseline in {year0} (% of municipality area)")
    ax.set_ylabel(f"Δ share {year0}→{year1} (%)")

    if annotate_all:
        for _, r in x.iterrows():
            ax.annotate(
                str(int(r["pt_id"])),
                (r["baseline_pct"], r["delta_pct"]),
                fontsize=font_size,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    return x[["pt_id", MUNI_ID, MUNI_NAME, "baseline_pct", "delta_pct"]].sort_values("pt_id")


tbl_ag = fig_scatter_baseline_vs_delta_with_ids(
    df,
    MUNI_SHP,
    "agriculture",
    1950,
    2020,
    FIGDIR / "Fig6_1_Agriculture_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True,
)
tbl_ag.to_csv(FIGDIR / "Scatter_ID_table_agriculture.csv", index=False)

tbl_rs = fig_scatter_baseline_vs_delta_with_ids(
    df,
    MUNI_SHP,
    "residential",
    1950,
    2020,
    FIGDIR / "Fig6_2_Residential_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True,
)
tbl_rs.to_csv(FIGDIR / "Scatter_ID_table_residential.csv", index=False)

tbl_ng = fig_scatter_baseline_vs_delta_with_ids(
    df,
    MUNI_SHP,
    "natural_green",
    1950,
    2020,
    FIGDIR / "Fig6_3_Natural_Green_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True,
)
tbl_ng.to_csv(FIGDIR / "Scatter_ID_table_natural_green.csv", index=False)


# ============================================================
# FIG 7 — Exposure snapshot (two panels: 1950 vs 2000)
# ============================================================
def exposure_table(df_in: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df_in[(df_in["year"] == year) & (df_in["hazard"].isin(HAZ_ORDER))].copy()

    p = d.groupby(["macro_class", "hazard"], as_index=False, observed=False)["area_km2"].sum()

    wide = p.pivot(index="macro_class", columns="hazard", values="area_km2").fillna(0.0)
    wide = wide.reindex([c for c in macros6 if c in wide.index])
    wide = wide[[h for h in HAZ_ORDER if h in wide.columns]]
    return wide


def fig7_exposure_two_panels(df_in: pd.DataFrame, year_top: int, year_bottom: int, outpath: Path) -> None:
    wide_top = exposure_table(df_in, year_top)
    wide_bot = exposure_table(df_in, year_bottom)

    ymax = max(wide_top.to_numpy().max(), wide_bot.to_numpy().max())
    if ymax == 0:
        ymax = 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    wide_top.plot(kind="bar", ax=ax1)
    ax1.set_title(f"Exposed land-use area by hazard class (year={year_top})", fontsize=14)
    ax1.set_ylabel("Exposed area (km²)", fontsize=14)
    ax1.set_ylim(0, ymax * 1.05)
    ax1.legend(title="Hazard", loc="upper right", frameon=True, fontsize=12, title_fontsize=12)
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)
    ax1.tick_params(axis="y", labelsize=12)

    wide_bot.plot(kind="bar", ax=ax2, legend=False)
    ax2.set_title(f"Exposed land-use area by hazard class (year={year_bottom})", fontsize=14)
    ax2.set_ylabel("Exposed area (km²)", fontsize=14)
    ax2.set_ylim(0, ymax * 1.05)
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    current_labels = [tick.get_text() for tick in ax2.get_xticklabels()]
    pretty_labels = [MACRO_LABELS.get(lbl, lbl) for lbl in current_labels]
    ax2.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=14)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


fig7_exposure_two_panels(
    df,
    year_top=1950,
    year_bottom=2020,
    outpath=FIGDIR / "Fig7_exposure_two_panels_1950_2020.png",
)

# ============================================================
# FIG 8
# ============================================================
def fig_exposure_boxplot_18boxes_yearpaired_hatched(
    df_in: pd.DataFrame,
    macros: List[str],
    muni_shp: Path,
    outpath: Path,
    years: List[int] = [1950, 2020],
    normalize_by_muni_area: bool = True,
    showfliers: bool = False,
):

    if gpd is None:
        raise ImportError("GeoPandas required.")

    # --- municipality area (only needed if normalize=True) ---
    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)
    muni_area = get_muni_area_km2(muni)

    # --- hazard colors ---
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hazard_colors = {"HPH": default_colors[0], "MPH": default_colors[1], "LPH": default_colors[2]}

    # --- year hatch styles (1950 no hatch, 2000 hatched) ---
    year_hatch = {years[0]: "", years[1]: "///"}  # adjust hatch if you want

    # --- filter data once ---
    d = df_in[
        (df_in["year"].isin(years)) &
        (df_in["macro_class"].isin(macros)) &
        (df_in["hazard"].isin(HAZ_ORDER))
    ].copy()

    if d.empty:
        print("No data found for requested settings.")
        return

    d[MUNI_ID] = d[MUNI_ID].astype(str)

    # one value per municipality per (year, macro, hazard)
    g = (
        d.groupby([MUNI_ID, "year", "macro_class", "hazard"], as_index=False, observed=False)["area_km2"]
         .sum()
         .rename(columns={"area_km2": "val"})
    )

    if normalize_by_muni_area:
        g = g.merge(muni_area, on=MUNI_ID, how="left").dropna(subset=["muni_area_km2"])
        g["val"] = 100.0 * g["val"] / g["muni_area_km2"]
        ylab = "Exposed area (% of municipality area)"
    else:
        ylab = "Exposed area (km²)"

    # --- build 18 boxes: macro -> hazard -> year(1950,2000) ---
    data, positions = [], []
    box_hazard, box_year = [], []   # to style boxes later

    tick_pos, tick_lab = [], []

    box_gap = 0.35
    pair_gap = 0.55
    macro_gap = 1.1
    pos = 1.0

    for macro in macros:
        macro_start = pos

        for hz in HAZ_ORDER:
            # year pair: 1950 then 2000
            for yr in years:
                vals = g[(g["macro_class"] == macro) & (g["hazard"] == hz) & (g["year"] == yr)]["val"].values
                data.append(vals)
                positions.append(pos)
                box_hazard.append(hz)
                box_year.append(yr)
                pos += box_gap

            # tick at center of hazard pair
            pair_center = (positions[-2] + positions[-1]) / 2
            tick_pos.append(pair_center)
            tick_lab.append(hz)

            pos += pair_gap

        macro_end = pos
        pos += macro_gap

    # --- plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=showfliers,
    )

    # style boxes: hazard color + year hatch
    for patch, hz, yr in zip(bp["boxes"], box_hazard, box_year):
        patch.set_facecolor(hazard_colors[hz])
        patch.set_alpha(0.85)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)
        patch.set_hatch(year_hatch.get(yr, ""))

    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.2)

    ax.set_ylabel(ylab, fontsize=14)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_ylabel(f"{ylab} (symlog scale)", fontsize=12)
    ax.tick_params(axis="y", labelsize=12)

    # x-axis: hazard labels repeated per macro
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lab, fontsize=12)

    # macro labels above each macro block (3 hazard ticks per macro)
    n_hz = len(HAZ_ORDER)
    for i, macro in enumerate(macros):
        start = i * n_hz
        end = start + n_hz
        mc = np.mean(tick_pos[start:end])
        ax.text(
            mc, 1.02,
            MACRO_LABELS.get(macro, macro),
            transform=ax.get_xaxis_transform(),
            ha="center", va="bottom",
            fontsize=13
        )

    ax.grid(True, axis="y", linestyle="--", linewidth=0.3, alpha=0.4)

    # --- legends INSIDE upper-right ---
    hazard_handles = [
    Patch(facecolor=hazard_colors[h], edgecolor="black", label=h)
    for h in HAZ_ORDER
    ]
    year_handles = [
    Patch(facecolor="white", edgecolor="black", hatch=year_hatch[years[0]], label=str(years[0])),
    Patch(facecolor="white", edgecolor="black", hatch=year_hatch[years[1]], label=str(years[1])),
    ]

    leg1 = ax.legend(
        handles=hazard_handles,
        title="Hazard",
        loc="best",
        #bbox_to_anchor=(1, 0.5),   # inside
        frameon=True,
        fontsize=12,
        title_fontsize=12
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=year_handles,
        title="Year",
        loc="upper right",
        bbox_to_anchor=(0.85, 1),   # inside, slightly lower
        frameon=True,
        fontsize=12,
        title_fontsize=12
    )


    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

fig_exposure_boxplot_18boxes_yearpaired_hatched(
    df,
    macros=["agriculture", "natural_green", "residential", "services_infrastructure"],
    muni_shp=MUNI_SHP,
    outpath=FIGDIR / "Fig8_Boxplot_Exposure_18boxes_yearpaired_hatched.png",
    years=[1950, 2020],
    normalize_by_muni_area=False,   # set True if you prefer %
    showfliers=True,
)



# ============================================================
# FIG 9 — Temporal evolution of exposure (ONE plot, 9 lines)
# ============================================================
def fig9_temporal_exposure_9_lines_pct(df_in: pd.DataFrame, macros: List[str], outpath: Path) -> None:
    d = df_in[(df_in["hazard"].isin(HAZ_ORDER)) & (df_in["macro_class"].isin(macros))].copy()
    if d.empty:
        print("No exposure data found for requested macros/hazards.")
        return

    p = d.groupby(["year", "macro_class", "hazard"], as_index=False, observed=False)["area_km2"].sum()

    denom = (
        df_in[df_in["hazard"] == "TOTAL"]
        .groupby(["year"], as_index=False, observed=False)["area_km2"]
        .sum()
        .rename(columns={"area_km2": "study_area_km2"})
    )

    p = p.merge(denom, on="year", how="left")
    p["pct_study_area"] = np.where(p["study_area_km2"] > 0, 100.0 * p["area_km2"] / p["study_area_km2"], 0.0)

    macro_labels_local = {
        "agriculture": MACRO_LABELS["agriculture"],
        "residential": MACRO_LABELS["residential"],
        "natural_green": MACRO_LABELS["natural_green"],
    }

    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hazard_color = {"HPH": default_colors[0], "MPH": default_colors[1], "LPH": default_colors[2]}

    macro_style = {
        "residential": "-",
        "agriculture": "--",
        "natural_green": ":",
    }

    years = sorted(p["year"].unique())

    fig, ax = plt.subplots(figsize=(11, 7))

    for macro in macros:
        for hz in HAZ_ORDER:
            sub = p[(p["macro_class"] == macro) & (p["hazard"] == hz)][["year", "pct_study_area"]].copy()
            if sub.empty:
                continue

            sub = sub.set_index("year").reindex(years)
            sub["pct_study_area"] = sub["pct_study_area"].fillna(0.0)
            sub = sub.reset_index()

            ax.plot(
                sub["year"],
                sub["pct_study_area"],
                color=hazard_color.get(hz),
                linestyle=macro_style.get(macro, "-"),
                linewidth=2,
                label=f"{macro_labels_local.get(macro, macro)} — {hz}",
            )

    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Exposed area (% of study area)", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


fig9_temporal_exposure_9_lines_pct(
    df,
    macros=["agriculture", "residential", "natural_green"],
    outpath=FIGDIR / "Fig9_Temporal_evolution_exposed_area_hazard_class_pct.png",
)


# ============================================================
# OPTIONAL — Representative municipalities (kept as-is, but note:
# your original code references compute_delta_by_muni which isn't
# defined in the snippet you pasted. So this will still be skipped.
# ============================================================
def pick_representative_munis(df_in: pd.DataFrame, year0: int, year1: int, n: int = 4) -> list:
    delta = compute_delta_by_muni(df_in, year0, year1, "residential")  # noqa: F821
    top = delta.sort_values("delta_km2", ascending=False).head(n)
    return top[MUNI_ID].tolist()


def fig_muni_composition_timeseries(df_in: pd.DataFrame, muni_id: int, outpath: Path) -> None:
    d = df_in[(df_in["hazard"] == "TOTAL") & (df_in[MUNI_ID] == str(muni_id))].copy()
    if d.empty:
        return

    p = d.groupby(["year", "macro_class"], as_index=False, observed=False)["area_km2"].sum()
    wide = p.pivot(index="year", columns="macro_class", values="area_km2").fillna(0.0)
    wide = wide[[c for c in MACRO_ORDER if c in wide.columns]]

    name = d[MUNI_NAME].iloc[0] if (MUNI_NAME in d.columns and len(d[MUNI_NAME]) > 0) else str(muni_id)

    ax = wide.plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title(f"Land-use composition over time (TOTAL) — {name} ({muni_id})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area (km²)")
    ax.legend(title="Macro class", bbox_to_anchor=(1.02, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


try:
    reps = pick_representative_munis(df, YEAR0, YEAR1, n=4)
    for mid in reps:
        fig_muni_composition_timeseries(df, mid, FIGDIR / f"Fig3_muni_composition_{mid}.png")
except Exception as e:
    print("Representative municipality plots skipped:", e)

print("All figures written to:", FIGDIR)
