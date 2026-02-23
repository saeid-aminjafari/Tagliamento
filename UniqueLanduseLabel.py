from pathlib import Path
import geopandas as gpd
import pandas as pd

BASE = Path(r"C:\Users\saeed\OneDrive\TagRiv260205")
RAW = BASE / "raw" / "2020"
OUT = BASE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

GPKG = RAW / "mosaicatura_prgc.gpkg"   # <<< change
LAYER = "mosaicatura_zonizzazione"           # <<< change (or set None and list layers below)

CODE = "zona_omoge"
DESC = "zona_desc"

gdf = gpd.read_file(GPKG, layer=LAYER)

# Keep needed columns
gdf = gdf[[CODE, DESC, "geometry"]].copy()

# Reproject for correct area if needed
# (keep your canonical CRS; this assumes EPSG:6708 is meters)
if gdf.crs is None:
    raise ValueError("Layer has no CRS.")
if str(gdf.crs).upper() != "EPSG:6708":
    gdf = gdf.to_crs("EPSG:6708")

gdf["area_m2"] = gdf.geometry.area

# Clean description slightly so duplicates collapse
gdf[DESC] = gdf[DESC].fillna("").astype(str).str.strip()

# Aggregate by code
agg = (
    gdf.groupby(CODE, dropna=False, as_index=False)
       .agg(
           n=("geometry", "size"),
           area_m2=("area_m2", "sum"),
       )
)

# Pick the most common description for each code (mode)
desc_mode = (
    gdf.groupby([CODE, DESC], dropna=False)
       .size()
       .reset_index(name="desc_count")
       .sort_values([CODE, "desc_count"], ascending=[True, False])
       .drop_duplicates(subset=[CODE])
       [[CODE, DESC, "desc_count"]]
)

out = agg.merge(desc_mode, on=CODE, how="left")
out = out.sort_values("area_m2", ascending=False)

out_csv = OUT / "unique_by_zona_omoge.csv"
out.to_csv(out_csv, index=False)

print("Wrote:", out_csv)
print("Unique zona_omoge:", len(out))
print(out.head(50).to_string(index=False))

# Copy/paste friendly (first N rows)
N = 200
print("\n--- COPY/PASTE BELOW ---")
print(out.head(N).to_csv(index=False))