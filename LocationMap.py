import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches

BASE = Path(r"C:\Users\saeed\OneDrive\TagRiv260205")
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
RAW = BASE / "raw"
RAW.mkdir(exist_ok=True)

FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

out_png = FIGDIR / "Fig1_Tagliamento_overview_map.png"

MUNI_SHP  = RAW / "COMUNI_TAGLIAMENTO.shp"
RIVER_SHP = RAW / "TAGLIAMENTO_RIVER.shp"
ITALY_SHP = RAW / "it.shp"


def make_muni_id_table(muni_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Stable municipality ID mapping (consistent with your other figures):
    - sort by PRO_COM (as string)
    - assign pt_id = 1..N
    """
    t = muni_gdf[["PRO_COM", "COMUNE"]].copy()
    t["PRO_COM"] = t["PRO_COM"].astype(str)
    t = t.drop_duplicates(subset=["PRO_COM"]).sort_values("PRO_COM").reset_index(drop=True)
    t["pt_id"] = range(1, len(t) + 1)
    return t


def fig_tagliamento_overview(id_table: pd.DataFrame, outpath: Path) -> None:
    # -------------------------
    # READ DATA
    # -------------------------
    muni = gpd.read_file(MUNI_SHP)
    river = gpd.read_file(RIVER_SHP)
    italy = gpd.read_file(ITALY_SHP)

    # Reproject everything to WGS84 (degrees)
    muni = muni.to_crs("EPSG:4326")
    river = river.to_crs("EPSG:4326")
    italy = italy.to_crs("EPSG:4326")

    # Attach municipality IDs
    muni["PRO_COM"] = muni["PRO_COM"].astype(str)
    id_table = id_table.copy()
    id_table["PRO_COM"] = id_table["PRO_COM"].astype(str)
    muni = muni.merge(id_table[["PRO_COM", "pt_id"]], on="PRO_COM", how="left")

    # -------------------------
    # FIGURE
    # -------------------------
    fig, ax = plt.subplots(figsize=(10, 10))

    # Municipality borders only
    muni.boundary.plot(ax=ax, linewidth=0.8, color="black")

    # River
    river.plot(ax=ax, color="royalblue", linewidth=2)

    # Larger municipality ID numbers
    pts = muni.representative_point()
    for x, y, pid in zip(pts.x, pts.y, muni["pt_id"]):
        if pd.notna(pid):
            ax.text(
                x, y,
                str(int(pid)),
                fontsize=11,   # increased size
                ha="center",
                va="center",
                weight="bold"
            )

    # -------------------------
    # LEGEND
    # -------------------------
    river_patch = mpatches.Patch(color="royalblue", label="Tagliamento River")
    ax.legend(handles=[river_patch], loc="upper right")

    # -------------------------
    # NORTH ARROW
    # -------------------------
    ax.annotate(
        "N",
        xy=(0.95, 0.18),
        xytext=(0.95, 0.08),
        arrowprops=dict(facecolor="black", width=3, headwidth=10),
        ha="center",
        va="center",
        fontsize=14,
        xycoords="axes fraction",
    )

    # -------------------------
    # COORDINATES IN DEGREES
    # -------------------------
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.tick_params(labelsize=9)

    # Optional grid
    ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

    # -------------------------
    # INSET MAP (Italy) — LOWER LEFT
    # -------------------------
    axins = inset_axes(ax, width="30%", height="30%", loc="lower left")

    italy.boundary.plot(ax=axins, linewidth=0.5, color="gray")
    muni.boundary.plot(ax=axins, linewidth=1.2, color="red")

    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Italy", fontsize=8)

    # -------------------------
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



# -------------------------
# BUILD ID TABLE (consistent with your other figures)
# -------------------------
muni_for_ids = gpd.read_file(MUNI_SHP)
id_table = make_muni_id_table(muni_for_ids)

# (Optional) save mapping for reference
id_table.to_csv(OUT / "Municipality_ID_table_pt_id.csv", index=False)

# -------------------------
# CALL
# -------------------------
fig_tagliamento_overview(
    id_table=id_table,
    outpath=out_png,
)

print("Saved:", out_png)
