from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from Tagliamento import CRS_CANON
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

# If you want maps (Fig 4), set this to the municipality shapefile
MUNI_SHP = BASE / "raw" / "COMUNI_TAGLIAMENTO.shp"  # adjust if needed

MUNI_ID = "PRO_COM"
MUNI_NAME = "COMUNE"

# Choose years of interest
YEAR0 = 1950
YEAR1 = 2000

# Exposure snapshot year for Fig 5
SNAPSHOT_YEAR = 2000

# Macro classes (ensure these match your output)
MACRO_ORDER = [
    "agriculture",
    "natural_green",
    "non_residential_industry",
    "residential_services",
    "green_urban",
    "water_body",
    "other",
]

HAZ_ORDER = ["HPH", "MPH", "LPH"]  # display order (you can change)


# -----------------------------
# LOAD + BASIC CLEANUP
# -----------------------------
df = pd.read_csv(LONG_CSV)

# Make sure types are clean
df["year"] = df["year"].astype(int)
df["area_km2"] = pd.to_numeric(df["area_km2"], errors="coerce").fillna(0.0)

# Keep consistent category ordering
df["macro_class"] = pd.Categorical(df["macro_class"], categories=MACRO_ORDER, ordered=True)
df["hazard"] = df["hazard"].astype(str)


def make_muni_id_table(muni_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """
    Create a stable label for each municipality to use across plots/maps.
    label = short integer 1..N (sorted by PRO_COM)
    """
    t = muni_gdf[[MUNI_ID, MUNI_NAME]].copy()
    t[MUNI_ID] = t[MUNI_ID].astype(str)
    t = t.drop_duplicates(subset=[MUNI_ID]).sort_values(MUNI_ID).reset_index(drop=True)
    t["pt_id"] = np.arange(1, len(t) + 1)
    return t

def add_muni_ids_to_axes(ax, muni_gdf: "gpd.GeoDataFrame", id_table: pd.DataFrame, font_size: int = 7):
    """
    Adds pt_id labels at municipality centroids (in map CRS).
    """
    m = muni_gdf.copy()
    m[MUNI_ID] = m[MUNI_ID].astype(str)
    m = m.merge(id_table[[MUNI_ID, "pt_id"]], on=MUNI_ID, how="left")

    # centroid labels (use representative_point to avoid labels outside polygons)
    pts = m.representative_point()
    for (x, y, pid) in zip(pts.x, pts.y, m["pt_id"]):
        if pd.notna(pid):
            ax.text(x, y, str(int(pid)), fontsize=font_size, ha="center", va="center")

# ============================================================
# FIG 3 — Study-area land-use composition over time (TOTAL only)
# ============================================================
def fig3_study_area_composition(df: pd.DataFrame, outpath: Path) -> None:
    d = df[df["hazard"] == "TOTAL"].copy()

    # Sum across all municipalities -> study area total
    p = (d.groupby(["year", "macro_class"], as_index=False, observed=False)["area_km2"].sum())

    # Pivot to stacked bars
    wide = p.pivot(index="year", columns="macro_class", values="area_km2").fillna(0.0)
    wide = wide[[c for c in MACRO_ORDER if c in wide.columns]]

    ax = wide.plot(kind="bar", stacked=True, figsize=(10, 6))
    #ax.set_title("Land-use composition over time (Study area, TOTAL)")
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Area (km²)", fontsize=12)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


fig3_study_area_composition(df, FIGDIR / "Fig3_Land_use_composition_over_time.png")

# ============================================================
# FIG 3b — Boxplots across ALL municipalities (TOTAL), NORMALIZED
# Each box = (% of municipality area) for a macro class in a year.
# - No x-labels (clean)
# - Macro classes color-coded (Matplotlib default cycle; consistent with Fig 3 bars)
# Requires: geopandas installed + municipality shapefile path (MUNI_SHP)
# ============================================================

def fig3b_boxplot_all_macros_one_fig_pct(df, macros, muni_shp, outpath=None):
    """
    All macro classes in ONE figure.
    For each year: multiple boxplots (one per macro),
    where values are normalized as (% of municipality area).

    df must contain: year, hazard, macro_class, PRO_COM, area_km2
    muni_shp must contain: PRO_COM and either Area_kmq or a valid geometry.
    """
    if gpd is None:
        raise ImportError("GeoPandas is required for municipality-area normalization (maps/geometry read).")

    # --- read municipality areas ---
    muni = gpd.read_file(muni_shp)

    # Ensure key types match between muni and df
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)
    df_key = df.copy()
    df_key[MUNI_ID] = df_key[MUNI_ID].astype(str)

    # Prefer attribute Area_kmq if present; otherwise compute from geometry (projected CRS)
    if "Area_kmq" in muni.columns:
        muni_area = muni[[MUNI_ID, "Area_kmq"]].rename(columns={"Area_kmq": "muni_area_km2"}).copy()
        muni_area["muni_area_km2"] = pd.to_numeric(muni_area["muni_area_km2"], errors="coerce")
    else:
        # Compute area from geometry; make sure CRS is meters
        if muni.crs is None:
            raise ValueError("Municipality layer has no CRS. Define it in QGIS, then rerun.")
        # Use your canonical CRS if you want: muni = muni.to_crs("EPSG:6708")
        muni = muni.to_crs(CRS_CANON)
        muni_area = muni[[MUNI_ID]].copy()
        muni_area["muni_area_km2"] = muni.geometry.area / 1_000_000.0

    muni_area = muni_area.dropna(subset=["muni_area_km2"]).drop_duplicates(subset=[MUNI_ID])

    # --- filter to TOTAL + macros ---
    d = df_key[(df_key["hazard"] == "TOTAL") & (df_key["macro_class"].isin(macros))].copy()

    # one value per municipality per year per macro
    g = d.groupby(["year", "macro_class", MUNI_ID], as_index=False, observed=False)["area_km2"].sum()

    # join municipality area and normalize
    g = g.merge(muni_area, on=MUNI_ID, how="left")
    g = g.dropna(subset=["muni_area_km2"]).copy()
    g["pct_muni_area"] = (g["area_km2"] / g["muni_area_km2"]) * 100.0

    years = sorted(g["year"].unique())
    macros = [m for m in macros if m in g["macro_class"].unique()]

    # --- consistent colors (same order as stacked bars) ---
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    macro_colors = {m: default_colors[i % len(default_colors)] for i, m in enumerate(macros)}

    data, positions, box_colors = [], [], []
    group_gap = 1.5
    box_gap = 0.9
    pos = 1.0

    for y in years:
        for m in macros:
            vals = g[(g["year"] == y) & (g["macro_class"] == m)]["pct_muni_area"].values
            data.append(vals)
            positions.append(pos)
            box_colors.append(macro_colors[m])
            pos += box_gap
        pos += group_gap

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        showfliers=True
    )

    # color boxes
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Set median lines to black and thicker
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

        # ------------------------------------------------------------
    # X-axis labels: one label per year (centered under 6 boxes)
    # ------------------------------------------------------------
    n_macros = len(macros)

    year_centers = []
    for i, y in enumerate(years):
        start = i * n_macros
        end = start + n_macros
        center = np.mean(positions[start:end])
        year_centers.append(center)

    ax.set_xticks(year_centers)
    ax.set_xticklabels([str(y) for y in years], fontsize=12)

    # clean axes
    #ax.set_title("Distribution across municipalities: TOTAL area by macro class and year (normalized)")
    ax.set_ylabel("Share of municipality area (%)")
    

    # Pretty legend names
    macro_labels = {
    "agriculture": "Agriculture",
    "natural_green": "Natural Green",
    "non_residential_industry": "Non-residential Industry",
    "residential_services": "Residential/Services",
    "green_urban": "Urban/Green",
    "water_body": "Water Bodies",
    }

    # legend instead
    legend_handles = [
    Patch(facecolor=macro_colors[m], edgecolor="black",
          label=macro_labels.get(m, m))
    for m in macros
    ]

    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1))

    # optional: set a sensible y-limit (comment out if you dislike)
    # ax.set_ylim(bottom=0)

    plt.tight_layout()
    if outpath is not None:
        plt.savefig(outpath, dpi=300)
        plt.close()


# -----------------------------
# CALL (put near bottom where you save figures)
# -----------------------------
macros6 = [
    "agriculture",
    "natural_green",
    "non_residential_industry",
    "residential_services",
    "green_urban",
    "water_body",
]

fig3b_boxplot_all_macros_one_fig_pct(
    df,
    macros6,
    muni_shp=MUNI_SHP,
    outpath=FIGDIR / "Fig3b_Distribution_across_municipalities_macroclass_year_normalized.png"
)

# ============================================================
# FIG 4 — Multi-panel municipality delta maps (PERCENT)
# 6 rows (macro classes) × 4 columns (time intervals)
# - Values: Δ (% of municipality area)
# - Municipalities with delta == 0: borders only (no fill)
# ============================================================

def _get_muni_area_km2(muni_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """Return DataFrame with municipality area in km² (from Area_kmq if available, else computed)."""
    muni = muni_gdf.copy()
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    if "Area_kmq" in muni.columns:
        a = muni[[MUNI_ID, "Area_kmq"]].rename(columns={"Area_kmq": "muni_area_km2"}).copy()
        a["muni_area_km2"] = pd.to_numeric(a["muni_area_km2"], errors="coerce")
    else:
        if muni.crs is None:
            raise ValueError("Municipality layer has no CRS. Define it in QGIS, then rerun.")
        muni = muni.to_crs(CRS_CANON)
        a = muni[[MUNI_ID]].copy()
        a["muni_area_km2"] = muni.geometry.area / 1_000_000.0

    a = a.dropna(subset=["muni_area_km2"]).drop_duplicates(subset=[MUNI_ID])
    return a


def compute_delta_pct_by_muni(df: pd.DataFrame, year0: int, year1: int, macro: str, muni_area: pd.DataFrame) -> pd.DataFrame:
    """
    Return municipality table with delta as % of municipality area for one macro class, TOTAL only.
    Δ% = 100 * (A_y1 - A_y0) / A_muni
    """
    d = df[(df["hazard"] == "TOTAL") & (df["macro_class"] == macro)].copy()

    y0 = d[d["year"] == year0].groupby([MUNI_ID], as_index=False)["area_km2"].sum().rename(columns={"area_km2": "a0"})
    y1 = d[d["year"] == year1].groupby([MUNI_ID], as_index=False)["area_km2"].sum().rename(columns={"area_km2": "a1"})

    out = y0.merge(y1, on=MUNI_ID, how="outer").fillna(0.0)
    out = out.merge(muni_area, on=MUNI_ID, how="left")

    out["muni_area_km2"] = pd.to_numeric(out["muni_area_km2"], errors="coerce")
    out = out.dropna(subset=["muni_area_km2"]).copy()

    out["delta_pct"] = 100.0 * (out["a1"] - out["a0"]) / out["muni_area_km2"]

    # treat tiny numerical noise as zero
    out.loc[out["delta_pct"].abs() < 0.01, "delta_pct"] = 0.0

    return out[[MUNI_ID, "delta_pct"]]


def fig4_delta_grid_pct(df: pd.DataFrame, muni_shp: Path, macros: list, intervals: list, outpath: Path) -> None:
    if gpd is None:
        print("GeoPandas not available -> skipping maps.")
        return

    muni = gpd.read_file(muni_shp)

    # Join key type consistency
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)
    df2 = df.copy()
    df2[MUNI_ID] = df2[MUNI_ID].astype(str)

    # Municipality area table for normalization
    muni_area = _get_muni_area_km2(muni)

    # Build all deltas first so we can set one shared color scale
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
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    macro_labels = {
    "agriculture": "Agriculture",
    "natural_green": "Natural Green",
    "non_residential_industry": "Non-residential Industry",
    "residential_services": "Residential/Services",
    "green_urban": "Urban/Green",
    "water_body": "Water Bodies",}

    # (Recommended) diverging colormap for +/- changes
    plt.set_cmap("RdBu_r")

    for i, macro in enumerate(macros):
        for j, (y0, y1) in enumerate(intervals):
            ax = axes[i, j]
            layer = delta_layers[(macro, y0, y1)]

            # Split into non-zero and zero for noise reduction
            nz = layer[layer["delta_pct"] != 0.0]
            z  = layer[layer["delta_pct"] == 0.0]

            # Plot non-zero filled
            if len(nz) > 0:
                nz.plot(
                    column="delta_pct",
                    ax=ax,
                    vmin=vmin,
                    vmax=vmax,
                    legend=False,
                    linewidth=0.2,
                    edgecolor="black"
                )

            # Plot zeros as borders only (no fill)
            if len(z) > 0:
                z.boundary.plot(ax=ax, linewidth=0.4, color="black")

            ax.set_axis_off()

            if i == 0:
                ax.set_title(f"{y0}–{y1}", fontsize=14)

            if j == 0:
                ax.text(
                    0.01, 0.5, macro_labels.get(macro, macro),
                    transform=ax.transAxes,
                    va="center", ha="left",
                    rotation=90,
                    fontsize=14
                )

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.get_cmap(), norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("Δ share of municipality area (%)", fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    #fig.suptitle("Municipality-level land-use change (TOTAL): Δ share by macro class and interval", y=0.995)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# --- CALL: Figure 4 grid (6×4) ---
intervals_4 = [(1950, 1970), (1970, 1980), (1980, 2000), (1950, 2000)]
macros6 = [
    "agriculture",
    "natural_green",
    "non_residential_industry",
    "residential_services",
    "green_urban",
    "water_body",
]

fig4_delta_grid_pct(
    df,
    MUNI_SHP,
    macros=macros6,
    intervals=intervals_4,
    outpath=FIGDIR / "Fig4_Municipality_level_landuse_change_share.png"
)

# ============================================================
# EXTRA FIG — Scatter: baseline share (%) vs Δ share (%) by municipality
# for agriculture + residential_services
# ============================================================

def _get_muni_area_km2(muni_gdf: "gpd.GeoDataFrame") -> pd.DataFrame:
    """Return DataFrame with municipality area in km² (from Area_kmq if available, else computed)."""
    muni = muni_gdf.copy()
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    if "Area_kmq" in muni.columns:
        a = muni[[MUNI_ID, "Area_kmq"]].rename(columns={"Area_kmq": "muni_area_km2"}).copy()
        a["muni_area_km2"] = pd.to_numeric(a["muni_area_km2"], errors="coerce")
    else:
        if muni.crs is None:
            raise ValueError("Municipality layer has no CRS. Define it in QGIS, then rerun.")
        muni = muni.to_crs(CRS_CANON)
        a = muni[[MUNI_ID]].copy()
        a["muni_area_km2"] = muni.geometry.area / 1_000_000.0

    a = a.dropna(subset=["muni_area_km2"]).drop_duplicates(subset=[MUNI_ID])
    return a


def _macro_share_pct_by_year(df: pd.DataFrame, year: int, macro: str, muni_area: pd.DataFrame) -> pd.DataFrame:
    """
    For each municipality: macro area in that year (TOTAL) as % of municipality area.
    Returns: PRO_COM, share_pct
    """
    d = df[(df["hazard"] == "TOTAL") & (df["macro_class"] == macro) & (df["year"] == year)].copy()
    g = d.groupby([MUNI_ID], as_index=False)["area_km2"].sum()
    g[MUNI_ID] = g[MUNI_ID].astype(str)

    out = g.merge(muni_area, on=MUNI_ID, how="left")
    out = out.dropna(subset=["muni_area_km2"]).copy()
    out["share_pct"] = 100.0 * out["area_km2"] / out["muni_area_km2"]
    return out[[MUNI_ID, "share_pct"]]


def fig_scatter_baseline_vs_delta_with_ids(
    df: pd.DataFrame,
    muni_shp: Path,
    macro: str,
    year0: int,
    year1: int,
    outpath: Path,
    annotate_all: bool = True,
    font_size: int = 12
) -> pd.DataFrame:
    """
    Scatter: x = baseline share (%) in year0, y = Δ share (%) year0→year1 (pp),
    per municipality. Each point gets a numeric ID (pt_id) that can also be plotted on maps.
    Returns a table mapping pt_id -> municipality.
    """
    if gpd is None:
        print("GeoPandas not available -> skipping scatter.")
        return pd.DataFrame()

    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)

    muni_area = _get_muni_area_km2(muni)
    muni_ids = make_muni_id_table(muni)  # pt_id mapping

    # baseline + end shares (per municipality)
    base = _macro_share_pct_by_year(df, year0, macro, muni_area).rename(columns={"share_pct": "baseline_pct"})
    end  = _macro_share_pct_by_year(df, year1, macro, muni_area).rename(columns={"share_pct": "end_pct"})
    x = base.merge(end, on=MUNI_ID, how="outer").fillna(0.0)
    x["delta_pct"] = x["end_pct"] - x["baseline_pct"]

    # attach pt_id and names
    x = muni_ids.merge(x, on=MUNI_ID, how="left").fillna({"baseline_pct": 0.0, "end_pct": 0.0, "delta_pct": 0.0})

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x["baseline_pct"], x["delta_pct"])

    ax.axhline(0, linewidth=0.8)
    #ax.set_title(f"{macro}: baseline share vs change (municipality-normalized)")
    ax.set_xlabel(f"Baseline in {year0} (% of municipality area)")
    ax.set_ylabel(f"Δ share {year0}→{year1} (%)")

    if annotate_all:
        for _, r in x.iterrows():
            ax.annotate(
                str(int(r["pt_id"])),
                (r["baseline_pct"], r["delta_pct"]),
                fontsize=font_size,
                xytext=(3, 3),
                textcoords="offset points"
            )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    # return mapping table (useful for caption/supp table)
    return x[[ "pt_id", MUNI_ID, MUNI_NAME, "baseline_pct", "delta_pct" ]].sort_values("pt_id")


# --- CALLS ---
tbl_ag = fig_scatter_baseline_vs_delta_with_ids(
    df, MUNI_SHP, "agriculture", 1950, 2000,
    FIGDIR / "Fig_Agriculture_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True
)
tbl_ag.to_csv(FIGDIR / "Scatter_ID_table_agriculture.csv", index=False)

tbl_rs = fig_scatter_baseline_vs_delta_with_ids(
    df, MUNI_SHP, "residential_services", 1950, 2000,
    FIGDIR / "Fig_Residential_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True
)
tbl_rs.to_csv(FIGDIR / "Scatter_ID_table_residential_services.csv", index=False)

tbl_ag = fig_scatter_baseline_vs_delta_with_ids(
    df, MUNI_SHP, "natural_green", 1950, 2000,
    FIGDIR / "Fig_Natural_Green_baseline_share_vs_change_municipality_normalized.png",
    annotate_all=True
)
tbl_ag.to_csv(FIGDIR / "Scatter_ID_table_natural_green.csv", index=False)

# ============================================================
# COMPACT PANEL FIGURE — Baseline(1950), Baseline(2000), and Δ(1950→2000)
# for two macro classes: agriculture + residential_services
# Layout: 2 rows (macros) × 3 cols (1950 share, 2000 share, delta)
# Units: % of municipality area (per-municipality normalization)
# Zeros: borders only (no fill) to reduce noise (for ALL panels)
# ============================================================

def fig_panel_baseline_baseline_delta_pct(
    df: pd.DataFrame,
    muni_shp: Path,
    macros: list,
    year0: int,
    year1: int,
    outpath: Path,
    zero_tol: float = 0.01,   # set to e.g. 0.01 to treat |value|<0.01% as zero
) -> None:
    """
    2×3 (or len(macros)×3) panel:
      col1: baseline share (%) in year0
      col2: baseline share (%) in year1
      col3: delta share (%) year0->year1 (percentage points)
    """
    if gpd is None:
        print("GeoPandas not available -> skipping panel maps.")
        return

    # Read municipalities
    muni = gpd.read_file(muni_shp)
    muni[MUNI_ID] = muni[MUNI_ID].astype(str)
    # CREATE THE ID TABLE ONCE
    muni_ids = make_muni_id_table(muni)

    # Municipality areas (km²)
    muni_area = _get_muni_area_km2(muni)

    # Precompute layers to set shared color scales
    layers = {}   # (macro, panel) -> GeoDataFrame with value column
    base_vals_all = []
    delta_vals_all = []

    for macro in macros:
        # baseline shares
        b0 = _macro_share_pct_by_year(df, year0, macro, muni_area).rename(columns={"share_pct": "val"})
        b1 = _macro_share_pct_by_year(df, year1, macro, muni_area).rename(columns={"share_pct": "val"})
        # delta
        tmp0 = b0.rename(columns={"val": "b0"})
        tmp1 = b1.rename(columns={"val": "b1"})
        dd = tmp0.merge(tmp1, on=MUNI_ID, how="outer").fillna(0.0)
        dd["val"] = dd["b1"] - dd["b0"]
        dd = dd[[MUNI_ID, "val"]]

        # join to geometry
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

    # Shared scales:
    # - Baselines: 0 .. max
    base_vmax = float(np.nanmax(base_vals_all)) if np.isfinite(base_vals_all).any() else 1.0
    if base_vmax == 0:
        base_vmax = 1.0
    base_vmin = 0.0

    # - Delta: symmetric about zero
    delta_vmax = float(np.nanmax(np.abs(delta_vals_all))) if np.isfinite(delta_vals_all).any() else 1.0
    if delta_vmax == 0:
        delta_vmax = 1.0
    delta_vmin = -delta_vmax

    # Figure layout
    nrows = len(macros)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 6 * nrows))

    if nrows == 1:
        axes = np.array([axes])

    # Choose colormaps:
    # Baselines: sequential
    cmap_base = "viridis"
    # Delta: diverging
    cmap_delta = "RdBu_r"

    col_titles = [
        f"{year0} (%)",
        f"{year1} (%)",
        f"Δ {year0}→{year1}",
    ]

    for i, macro in enumerate(macros):
        for j, panel in enumerate(["base0", "base1", "delta"]):
            ax = axes[i, j]
            layer = layers[(macro, panel)].copy()

            # Split zero vs non-zero for noise reduction
            nz = layer[layer["val"] != 0.0]
            z = layer[layer["val"] == 0.0]

            if panel == "delta":
                # Diverging for delta
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
                # Sequential for baseline
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

            # Zero polygons as borders only
            if len(z) > 0:
                z.boundary.plot(ax=ax, linewidth=0.4, color="black")

            '''add_muni_ids_to_axes(
                            ax=ax,
                            muni_gdf=layer,   # same geometry you just plotted
                            id_table=muni_ids,
                            font_size=7
            )'''

            ax.set_axis_off()

            # Top titles
            if i == 0:
                ax.set_title(col_titles[j], fontsize=14)

            # Row label on left
            if j == 0:
                ax.text(
                    0.01, 0.5, ["Agriculture", "Residential/Services", "Natural Green"][i],
                    transform=ax.transAxes,
                    va="center", ha="left",
                    rotation=90,
                    fontsize=14,
                )

        # --- leave room on the right for TWO vertical colorbars ---
    # IMPORTANT: do this BEFORE creating colorbar axes
    plt.tight_layout(rect=[0, 0, 0.88, 0.98])  # right margin reserved for cbar

    # Two colorbars: one for baselines (cols 0–1), one for delta (col 2)
    sm_base = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_base),
        norm=plt.Normalize(vmin=base_vmin, vmax=base_vmax)
    )
    sm_base._A = []

    sm_delta = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_delta),
        norm=plt.Normalize(vmin=delta_vmin, vmax=delta_vmax)
    )
    sm_delta._A = []

        # ------------------------------------------------------------
    # Leave right margin for 2 colorbars
    # ------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 0.86, 0.98])

    # Create scalar mappables
    sm_base = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_base),
        norm=plt.Normalize(vmin=base_vmin, vmax=base_vmax)
    )
    sm_base._A = []

    sm_delta = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_delta),
        norm=plt.Normalize(vmin=delta_vmin, vmax=delta_vmax)
    )
    sm_delta._A = []

        # ------------------------------------------------------------
    # Layout: reserve a LITTLE right margin only for the 2nd cbar
    # (cbar1 will be placed between col2 and col3)
    # ------------------------------------------------------------
    plt.tight_layout(rect=[0, 0, 0.90, 0.98])

    # Scalar mappables
    sm_base = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_base),
        norm=plt.Normalize(vmin=base_vmin, vmax=base_vmax)
    )
    sm_base._A = []

    sm_delta = plt.cm.ScalarMappable(
        cmap=plt.get_cmap(cmap_delta),
        norm=plt.Normalize(vmin=delta_vmin, vmax=delta_vmax)
    )
    sm_delta._A = []

    # ------------------------------------------------------------
    # Align colorbars with SECOND ROW
    # ------------------------------------------------------------
    row = 1  # second row (0-based)

    # Reference axes bboxes for the second row
    bbox_col2 = axes[row, 1].get_position()  # 2000 share column
    bbox_col3 = axes[row, 2].get_position()  # delta column

    # Vertical placement (match row height)
    y0 = bbox_col2.y0
    h  = bbox_col2.height

    # ---- cbar1: BETWEEN col2 and col3 ----
    gap_left  = bbox_col2.x1
    gap_right = bbox_col3.x0
    gap = gap_right - gap_left

    # choose a reasonable width inside the gap
    cbar_w = min(0.018, gap * 0.6)  # do not exceed gap too much
    cbar_x = gap_left + (gap - cbar_w) / 2.0

    cax1 = fig.add_axes([cbar_x, y0, cbar_w, h])

    cbar1 = fig.colorbar(sm_base, cax=cax1)
    cbar1.set_label("Share of municipality area (%)", fontsize=14)
    cbar1.ax.tick_params(labelsize=14)

    # ---- cbar2: to the RIGHT of col3 (delta) ----
    bbox_col3 = axes[row, 2].get_position()
    cax2 = fig.add_axes([bbox_col3.x1 + 0.015, y0, 0.018, h])

    cbar2 = fig.colorbar(sm_delta, cax=cax2)
    cbar2.set_label("Δ share (%)", fontsize=14)
    cbar2.ax.tick_params(labelsize=14)


    #fig.suptitle("Baseline and long-term change (municipality-normalized)", y=0.995)
    plt.savefig(outpath, dpi=300)
    plt.close()


# --- CALL: compact 2×3 panel for your two key macro classes ---
fig_panel_baseline_baseline_delta_pct(
    df=df,
    muni_shp=MUNI_SHP,
    macros=["agriculture", "residential_services", "natural_green"],
    year0=1950,
    year1=2000,
    outpath=FIGDIR / "Fig_Baseline and long-term change municipality-normalized.png",
    zero_tol=0.01  # set to 0.01 if you want to treat tiny values as zero
)


# ============================================================
# FIG 5 — Exposure snapshot (two panels: 1950 vs 2000)
# ============================================================

def _exposure_table(df: pd.DataFrame, year: int) -> pd.DataFrame:
    d = df[(df["year"] == year) & (df["hazard"].isin(HAZ_ORDER))].copy()

    p = d.groupby(["macro_class", "hazard"], as_index=False, observed=False)["area_km2"].sum()

    wide = p.pivot(index="macro_class", columns="hazard", values="area_km2").fillna(0.0)
    wide = wide.reindex([c for c in MACRO_ORDER if c in wide.index])
    wide = wide[[h for h in HAZ_ORDER if h in wide.columns]]
    return wide


def fig5_exposure_two_panels(df: pd.DataFrame, year_top: int, year_bottom: int, outpath: Path) -> None:
    wide_top = _exposure_table(df, year_top)
    wide_bot = _exposure_table(df, year_bottom)

    # Shared y-limit for comparability
    ymax = max(wide_top.to_numpy().max(), wide_bot.to_numpy().max())
    if ymax == 0:
        ymax = 1.0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    macro_labels = {
    "agriculture": "Agriculture",
    "natural_green": "Natural Green",
    "non_residential_industry": "Non-residential Industry",
    "residential_services": "Residential/Services",
    "green_urban": "Urban/Green",
    "water_body": "Water Bodies",
    }

    # --- Top panel ---
    wide_top.plot(kind="bar", ax=ax1)
    ax1.set_title(f"Exposed land-use area by hazard class (year={year_top})", fontsize=14)
    ax1.set_ylabel("Exposed area (km²)", fontsize=14)
    ax1.set_ylim(0, ymax * 1.05)

    # Legend inside top panel (top right)
    ax1.legend(title="Hazard", loc="upper right", frameon=True, fontsize=12, title_fontsize=12)

    # Remove x ticks/labels on top panel
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", bottom=False, labelbottom=False)

    # --- Bottom panel ---
    wide_bot.plot(kind="bar", ax=ax2, legend=False)
    ax2.set_title(f"Exposed land-use area by hazard class (year={year_bottom})", fontsize=14)
    ax2.set_ylabel("Exposed area (km²)", fontsize=14)
    ax2.set_ylim(0, ymax * 1.05)

    # X labels only on bottom, rotated 45
    ax2.set_xlabel("")  # no axis title, just tick labels
    ax2.tick_params(axis="x", labelsize=12)
    # Replace raw macro names with pretty labels
    current_labels = [tick.get_text() for tick in ax2.get_xticklabels()]
    pretty_labels = [macro_labels.get(lbl, lbl) for lbl in current_labels]

    ax2.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=14)


    ax1.tick_params(axis="y", labelsize=12)
    ax2.tick_params(axis="y", labelsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# --- CALL ---
fig5_exposure_two_panels(
    df,
    year_top=1950,
    year_bottom=2000,
    outpath=FIGDIR / "Fig5_exposure_two_panels_1950_2000.png"
)


# ============================================================
# FIG 6 — Temporal evolution of exposure (ONE plot, 9 lines)
# 3 macro classes × 3 hazard classes
# - y-axis: % of study-area area (not km²)
# - hazard: color (3 colors)
# - macro: line style (3 line styles)
# ============================================================

def fig6_temporal_exposure_9_lines_pct(df: pd.DataFrame, macros: list, outpath: Path) -> None:
    d = df[(df["hazard"].isin(HAZ_ORDER)) & (df["macro_class"].isin(macros))].copy()
    if d.empty:
        print("No exposure data found for requested macros/hazards.")
        return

    # km² totals per year × macro × hazard
    p = (
        d.groupby(["year", "macro_class", "hazard"], as_index=False, observed=False)["area_km2"]
        .sum()
    )

    # Denominator: study-area total (TOTAL) per year (km²)
    denom = (
        df[df["hazard"] == "TOTAL"]
        .groupby(["year"], as_index=False, observed=False)["area_km2"]
        .sum()
        .rename(columns={"area_km2": "study_area_km2"})
    )

    p = p.merge(denom, on="year", how="left")
    p["pct_study_area"] = np.where(
        p["study_area_km2"] > 0,
        100.0 * p["area_km2"] / p["study_area_km2"],
        0.0
    )

    macro_labels = {
        "agriculture": "Agriculture",
        "non_residential_industry": "Non-residential Industry",
        "residential_services": "Residential/Services",
    }

    # Hazard -> color (matplotlib default cycle first 3 colors)
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    hazard_color = {
        "HPH": default_colors[0],
        "MPH": default_colors[1],
        "LPH": default_colors[2],
    }

    # Macro -> line style
    macro_style = {
        "residential_services": "-",
        "agriculture": "--",
        "non_residential_industry": ":",
    }

    years = sorted(p["year"].unique())

    fig, ax = plt.subplots(figsize=(11, 7))

    for macro in macros:
        for hz in HAZ_ORDER:
            sub = p[(p["macro_class"] == macro) & (p["hazard"] == hz)][["year", "pct_study_area"]].copy()
            if sub.empty:
                continue

            # ensure all years appear (fill missing with 0)
            sub = sub.set_index("year").reindex(years)
            sub["pct_study_area"] = sub["pct_study_area"].fillna(0.0)
            sub = sub.reset_index()

            ax.plot(
                sub["year"],
                sub["pct_study_area"],
                color=hazard_color.get(hz),
                linestyle=macro_style.get(macro, "-"),
                linewidth=2,
                label=f"{macro_labels.get(macro, macro)} — {hz}",
            )

    ax.set_xlabel("Year", fontsize=14)
    ax.set_ylabel("Exposed area (% of study area)", fontsize=14)
    ax.tick_params(axis="both", labelsize=14)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# --- CALL ---
fig6_temporal_exposure_9_lines_pct(
    df,
    macros=["residential_services", "agriculture", "non_residential_industry"],
    outpath=FIGDIR / "Fig6_Temporal_evolution_exposed_area_hazard_class_pct.png"
)



# ============================================================
# OPTIONAL — Pick 4 “representative municipalities” automatically
#           (largest residential_services increase YEAR0→YEAR1)
#           and plot their TOTAL composition over time.
# ============================================================
def pick_representative_munis(df: pd.DataFrame, year0: int, year1: int, n: int = 4) -> list:
    delta = compute_delta_by_muni(df, year0, year1, "residential_services")
    top = delta.sort_values("delta_km2", ascending=False).head(n)
    return top[MUNI_ID].tolist()

def fig_muni_composition_timeseries(df: pd.DataFrame, muni_id: int, outpath: Path) -> None:
    d = df[(df["hazard"] == "TOTAL") & (df[MUNI_ID] == muni_id)].copy()
    if d.empty:
        return
    p = d.groupby(["year", "macro_class"], as_index=False, observed=False)["area_km2"].sum()
    wide = p.pivot(index="year", columns="macro_class", values="area_km2").fillna(0.0)
    wide = wide[[c for c in MACRO_ORDER if c in wide.columns]]

    name = d[MUNI_NAME].iloc[0] if MUNI_NAME in d.columns and len(d[MUNI_NAME]) > 0 else str(muni_id)

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


