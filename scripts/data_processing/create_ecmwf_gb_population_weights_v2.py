from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import box, mapping
from shapely.ops import unary_union
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt

# =========================
# Paths
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

ECMWF_FILE = PROJECT_DIR / "data" / "raw" / "ecmwf_s5_weather_winter2010_NDJFM_uk.nc"
POP_FILE = PROJECT_DIR / "data" / "population" / "worldpop_uk_2020_population.tif"

OUT_DIR = PROJECT_DIR / "data" / "masks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_WEIGHTS_NC = OUT_DIR / "ecmwf_gb_population_weights_v2.nc"
OUT_WEIGHTS_CSV = OUT_DIR / "ecmwf_gb_population_weights_v2.csv"
OUT_FIG = PROJECT_DIR / "outputs" / "figures" / "ecmwf_gb_population_weights_v2.png"

# Optional: compare with v1 if it exists
V1_CSV = OUT_DIR / "ecmwf_gb_population_weights.csv"

# =========================
# Load ECMWF grid
# =========================
print("Opening ECMWF file:")
print(ECMWF_FILE)

ds = xr.open_dataset(ECMWF_FILE)
lats = ds["latitude"].values
lons = ds["longitude"].values

print("\nECMWF grid:")
print("latitudes:", lats)
print("longitudes:", lons)

# ECMWF grid here is 1 degree.
lat_step = abs(float(np.diff(np.sort(lats))[0]))
lon_step = abs(float(np.diff(np.sort(lons))[0]))
half_lat = lat_step / 2
half_lon = lon_step / 2

print("\nGrid-cell size:")
print("lat step:", lat_step)
print("lon step:", lon_step)

# =========================
# Load UK geometry and create approximate GB geometry
# =========================
# Natural Earth UK polygon includes Northern Ireland.
# For this first implementation, we remove Northern Ireland using an approximate bounding box.
# This can be refined later if needed.
# =========================

countries_shp = shpreader.natural_earth(
    resolution="10m",
    category="cultural",
    name="admin_0_countries"
)

uk_geometries = []

for record in shpreader.Reader(countries_shp).records():
    attrs = record.attributes
    name_long = attrs.get("NAME_LONG", "")
    name = attrs.get("NAME", "")

    if name_long == "United Kingdom" or name == "United Kingdom":
        uk_geometries.append(record.geometry)

if len(uk_geometries) == 0:
    raise ValueError("Could not find United Kingdom geometry in Natural Earth data.")

uk_geom = unary_union(uk_geometries)

# Approximate Northern Ireland bounding box
# lon/lat roughly covering Northern Ireland
northern_ireland_bbox = box(-8.3, 54.0, -5.0, 55.4)

# Great Britain geometry = UK minus Northern Ireland area
gb_geom = uk_geom.difference(northern_ireland_bbox)

# =========================
# Aggregate WorldPop raster to ECMWF grid cells
# =========================
# V2 improvement:
# We do NOT require the grid-point centre to be inside GB.
# Instead, for every ECMWF 1-degree cell, we check whether the cell polygon intersects GB.
# If it intersects GB, we sum population over the GB-overlap part.
# =========================

population_grid = np.zeros((len(lats), len(lons)), dtype=float)
overlap_mask_grid = np.zeros((len(lats), len(lons)), dtype=int)

print("\nOpening population raster:")
print(POP_FILE)

with rasterio.open(POP_FILE) as src:
    print("Population CRS:", src.crs)
    print("Population bounds:", src.bounds)
    print("Population nodata:", src.nodata)

    if src.crs is None or src.crs.to_string() != "EPSG:4326":
        raise ValueError(
            "Population raster is not EPSG:4326. Reprojection would be needed."
        )

    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):

            # Create ECMWF grid-cell polygon around grid point
            cell_poly = box(
                float(lon - half_lon),
                float(lat - half_lat),
                float(lon + half_lon),
                float(lat + half_lat),
            )

            # Intersect grid cell with GB geometry
            cell_gb_poly = cell_poly.intersection(gb_geom)

            if cell_gb_poly.is_empty:
                continue

            overlap_mask_grid[i, j] = 1

            try:
                out_image, out_transform = rio_mask(
                    src,
                    [mapping(cell_gb_poly)],
                    crop=True,
                    filled=False
                )

                arr = out_image[0]

                # Mask invalid and negative NoData values
                arr = np.ma.masked_invalid(arr)
                arr = np.ma.masked_less(arr, 0)

                pop_sum = float(arr.filled(0).sum())

            except ValueError:
                pop_sum = 0.0

            population_grid[i, j] = pop_sum

# =========================
# Convert population to weights
# =========================
total_population = population_grid.sum()

print("\nTotal GB population represented on ECMWF grid, v2:")
print(total_population)

if total_population <= 0:
    raise ValueError("Total population is zero. Something went wrong.")

weights_grid = population_grid / total_population

print("\nCheck sum of weights:")
print(weights_grid.sum())

print("\nNumber of ECMWF grid cells overlapping GB:")
print(overlap_mask_grid.sum())

print("\nNumber of grid cells with positive population:")
print((weights_grid > 0).sum())

# =========================
# Save outputs
# =========================
pop_da = xr.DataArray(
    population_grid,
    coords={"latitude": lats, "longitude": lons},
    dims=("latitude", "longitude"),
    name="gb_population_v2"
)

weights_da = xr.DataArray(
    weights_grid,
    coords={"latitude": lats, "longitude": lons},
    dims=("latitude", "longitude"),
    name="gb_population_weight_v2"
)

weights_da.attrs["description"] = (
    "Approximate GB population weights on the ECMWF 1-degree grid, "
    "constructed from WorldPop 2020 UK population count raster. "
    "V2 uses grid-cell polygon intersection with GB geometry, rather than "
    "only grid-point centre inclusion."
)

weights_da.attrs["source_population_file"] = str(POP_FILE)
weights_da.attrs["note"] = (
    "Northern Ireland is approximately excluded using a bounding box. "
    "This is a refined implementation compared with the centre-point GB mask version."
)

weights_da.to_netcdf(OUT_WEIGHTS_NC)

rows = []
for i, lat in enumerate(lats):
    for j, lon in enumerate(lons):
        rows.append({
            "latitude": lat,
            "longitude": lon,
            "cell_overlaps_GB": int(overlap_mask_grid[i, j]),
            "population": population_grid[i, j],
            "weight": weights_grid[i, j],
        })

weights_df = pd.DataFrame(rows)
weights_df.to_csv(OUT_WEIGHTS_CSV, index=False)

print("\nSaved population weights NetCDF:")
print(OUT_WEIGHTS_NC)

print("\nSaved population weights CSV:")
print(OUT_WEIGHTS_CSV)

print("\nTop 10 grid cells by population weight:")
print(
    weights_df.sort_values("weight", ascending=False)
    .head(10)
    .to_string(index=False)
)

# Compare with v1 if available
if V1_CSV.exists():
    v1 = pd.read_csv(V1_CSV)
    v1_total = v1["population"].sum()
    print("\nComparison with v1:")
    print("v1 total population:", v1_total)
    print("v2 total population:", total_population)
    print("increase from v1:", total_population - v1_total)

# =========================
# Plot weights
# =========================
plt.figure(figsize=(7, 6))
plt.pcolormesh(lons, lats, weights_grid, shading="auto")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("GB population weights on ECMWF grid, v2")
plt.colorbar(label="Population weight")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print("\nSaved population weights figure:")
print(OUT_FIG)