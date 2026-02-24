import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

BASE = Path(r"C:\Users\saeed\OneDrive\TagRiv260205")
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
RAW = BASE / "raw"
RAW.mkdir(exist_ok=True)
ID_CSV = OUT / "Municipality_ID_table_pt_id.csv"

FIGDIR = OUT / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

out_png = FIGDIR / "Fig1_Tagliamento_overview_map.png"

MUNI_SHP  = RAW / "COMUNI_TAGLIAMENTO.shp"
RIVER_SHP = RAW / "TAGLIAMENTO_RIVER.shp"
ITALY_SHP = RAW / "it.shp"


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
    fig, ax = plt.subplots(figsize=(10, 12))

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
                fontsize=12,   # increased size
                ha="center",
                va="center",
                weight="bold"
            )

    # -------------------------
    # LEGEND
    # -------------------------
    river_handle = mpatches.Patch(color="royalblue", label="Tagliamento River")
    muni_handle = mpatches.Patch(
    facecolor="none",
    edgecolor="black",
    linewidth=1,
    label="Municipality Boundaries"
    )

    ax.legend(
    handles=[river_handle, muni_handle],
    loc="upper right",
    fontsize=12
    )

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
    ax.set_xlabel("Longitude (°)", fontsize=14)
    ax.set_ylabel("Latitude (°)", fontsize=14, rotation=90)
       
    ax.tick_params(labelsize=14)
    ax.tick_params(axis='y', labelrotation=90)

    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.3))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.3))

    # Optional grid
    ax.grid(True, linestyle="--", linewidth=0.2, alpha=0.3)

    # -------------------------
    # INSET MAP (Italy) — LOWER LEFT
    # -------------------------
    axins = inset_axes(ax, width="37%", height="37%", loc="lower left")

    italy.boundary.plot(ax=axins, linewidth=0.5, color="gray")
    muni.boundary.plot(ax=axins, linewidth=1.8, color="red")

    axins.set_xticks([])
    axins.set_yticks([])
    axins.set_title("Italy", fontsize=12)

    # -------------------------
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()



# -------------------------
# LOAD ID TABLE (use same IDs everywhere)
# -------------------------
id_table = pd.read_csv(ID_CSV, dtype={"PRO_COM": str})

# -------------------------
# CALL
# -------------------------
fig_tagliamento_overview(
    id_table=id_table,
    outpath=out_png,
)

print("Saved:", out_png)
