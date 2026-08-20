from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 02_process_ecmwf_daily_weather.py
#
# Purpose:
# Process ECMWF SEAS5 raw 6-hourly grid data into daily
# population-weighted weather series for winter 1982–2016.
#
# Inputs:
# - ECMWF raw NetCDF files:
#   ecmwf_s5_weather_winter1982_NDJFM_uk.nc
#   ...
#   ecmwf_s5_weather_winter2016_NDJFM_uk.nc
#
# Variables:
# - t2m: 2m temperature, usually Kelvin
# - u10: 10m u wind component
# - v10: 10m v wind component
#
# Outputs:
# - data_processed/daily/ecmwf_daily_weather_winterYYYY.csv
# - data_processed/daily/ecmwf_daily_weather_1982_2016.csv
# - outputs/tables/ecmwf_daily_processing_summary.csv
#
# Important:
# This script needs a population weights CSV with latitude,
# longitude and weight/population columns.
# It will try to auto-find such a file in week3_ecmwf_pilot.
# ============================================================


# =========================
# Project paths
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]
DISSERTATION_DIR = PROJECT_DIR

RAW_DIR = PROJECT_DIR / "data" / "raw"

OUTPUT_DAILY_DIR = PROJECT_DIR / "outputs" / "daily"
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

OUTPUT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# File settings
# =========================
FILE_PATTERN = "ecmwf_s5_weather_winter*_NDJFM_uk.nc"
EXPECTED_WINTER_YEARS = list(range(1982, 2017))


# =========================
# Population weights settings
# =========================
# If you know the exact population weights file, put it here, e.g.
# WEIGHTS_FILE = DISSERTATION_DIR / "week3_ecmwf_pilot" / "outputs" / "tables" / "ecmwf_population_weights.csv"
#
# If not sure, leave as None. The script will auto-search.
WEIGHTS_FILE = PROJECT_DIR / "data" / "processed" / "population_weights_used_by_02.csv"

# Do NOT set this to True unless you only want a temporary unweighted test.
USE_EQUAL_WEIGHTS_IF_NO_POP_WEIGHTS = False


def extract_winter_year(filename: str) -> int:
    """Extract winter year from filename."""
    match = re.search(r"winter(\d{4})", filename, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot extract winter year from filename: {filename}")
    return int(match.group(1))


def find_population_weights_file() -> Path | None:
    """
    Try to find a population weights CSV in week3 folder.

    The file should contain latitude/longitude and either:
    - weight
    - pop_weight
    - population_weight
    - normalized_weight / normalised_weight
    - population
    """
    search_root = PROJECT_DIR / "data" / "processed"

    candidates = []
    for csv_file in search_root.rglob("*.csv"):
        name_lower = csv_file.name.lower()

        # Avoid obvious non-weight files
        if "inventory" in name_lower or "summary" in name_lower:
            continue

        try:
            header = pd.read_csv(csv_file, nrows=5)
        except Exception:
            continue

        cols = [c.lower().strip() for c in header.columns]

        has_lat = any(c in cols for c in ["latitude", "lat"])
        has_lon = any(c in cols for c in ["longitude", "lon", "long"])
        has_weight = any(
            c in cols
            for c in [
                "weight",
                "weights",
                "pop_weight",
                "population_weight",
                "normalised_weight",
                "normalized_weight",
                "population",
                "pop",
            ]
        )

        if has_lat and has_lon and has_weight:
            score = 0
            if "pop" in name_lower:
                score += 2
            if "weight" in name_lower:
                score += 2
            if "ecmwf" in name_lower:
                score += 1
            candidates.append((score, csv_file))

    if not candidates:
        return None

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    print("\nPossible population weights files found:")
    for score, path in candidates[:10]:
        print(f"  score={score}: {path}")

    selected = candidates[0][1]
    print(f"\nUsing population weights file:\n{selected}")
    return selected


def standardise_weights_dataframe(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Standardise weight dataframe column names."""
    original_cols = weights_df.columns.tolist()
    lower_map = {c.lower().strip(): c for c in original_cols}

    def find_col(possible_names):
        for name in possible_names:
            if name in lower_map:
                return lower_map[name]
        return None

    lat_col = find_col(["latitude", "lat"])
    lon_col = find_col(["longitude", "lon", "long"])
    weight_col = find_col(
        [
            "weight",
            "weights",
            "pop_weight",
            "population_weight",
            "normalised_weight",
            "normalized_weight",
            "population",
            "pop",
        ]
    )

    if lat_col is None or lon_col is None or weight_col is None:
        raise ValueError(
            "Population weights file must contain latitude, longitude and weight/population columns.\n"
            f"Columns found: {original_cols}"
        )

    out = weights_df[[lat_col, lon_col, weight_col]].copy()
    out.columns = ["latitude", "longitude", "weight"]

    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out["weight"] = pd.to_numeric(out["weight"], errors="coerce")

    out = out.dropna(subset=["latitude", "longitude", "weight"])
    out = out[out["weight"] >= 0].copy()

    if out.empty:
        raise ValueError("Population weights dataframe is empty after cleaning.")

    total_weight = out["weight"].sum()
    if total_weight <= 0:
        raise ValueError("Population weights sum to zero.")

    out["weight"] = out["weight"] / total_weight

    return out


def make_weight_array_for_grid(ds: xr.Dataset, weights_df: pd.DataFrame) -> xr.DataArray:
    """
    Create an xarray DataArray of weights aligned with ECMWF latitude/longitude grid.
    Matching is done using rounded lat/lon values to avoid tiny floating-point issues.
    """
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    weights_map = {
        (round(float(row["latitude"]), 6), round(float(row["longitude"]), 6)): float(row["weight"])
        for _, row in weights_df.iterrows()
    }

    arr = np.zeros((len(lats), len(lons)), dtype=float)

    matched = 0
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            key = (round(float(lat), 6), round(float(lon), 6))
            if key in weights_map:
                arr[i, j] = weights_map[key]
                matched += 1

    weight_sum = arr.sum()

    if weight_sum <= 0:
        raise ValueError(
            "No population weights matched the ECMWF latitude/longitude grid.\n"
            "This usually means the weights file is on a different grid or has different lat/lon values."
        )

    arr = arr / weight_sum

    print(f"Matched population-weighted grid cells: {matched} / {arr.size}")
    print(f"Normalised matched weight sum: {arr.sum():.6f}")

    weights_da = xr.DataArray(
        arr,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name="population_weight",
    )

    return weights_da


def make_equal_weight_array_for_grid(ds: xr.Dataset) -> xr.DataArray:
    """
    Temporary equal-grid-cell weights.
    This is NOT population weighting.
    Only use for debugging if population weights are not available.
    """
    lats = ds["latitude"].values
    lons = ds["longitude"].values

    arr = np.ones((len(lats), len(lons)), dtype=float)
    arr = arr / arr.sum()

    return xr.DataArray(
        arr,
        coords={"latitude": lats, "longitude": lons},
        dims=("latitude", "longitude"),
        name="equal_grid_weight",
    )


def prepare_time_dimension(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert ECMWF forecast_period / valid_time structure into a normal time dimension.
    Expected current structure:
    dims: number, forecast_reference_time, forecast_period, latitude, longitude
    coord: valid_time
    """
    # Remove singleton forecast_reference_time dimension
    if "forecast_reference_time" in ds.dims and ds.sizes["forecast_reference_time"] == 1:
        ds = ds.isel(forecast_reference_time=0, drop=True)

    # Convert forecast_period dimension to valid_time dimension
    if "valid_time" in ds.coords:
        if "forecast_period" in ds.dims:
            ds = ds.swap_dims({"forecast_period": "valid_time"})
        ds = ds.rename({"valid_time": "time"})

    elif "time" not in ds.coords and "time" not in ds.dims:
        raise ValueError("No valid_time or time coordinate found in dataset.")

    return ds


def process_one_winter(file_path: Path, weights_df: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Process one ECMWF winter file into daily population-weighted weather."""
    winter_year = extract_winter_year(file_path.name)
    print(f"\nProcessing winter {winter_year}: {file_path.name}")

    ds = xr.open_dataset(file_path)
    ds = prepare_time_dimension(ds)

    required_vars = ["t2m", "u10", "v10"]
    missing_vars = [v for v in required_vars if v not in ds.data_vars]
    if missing_vars:
        raise ValueError(f"Missing variables in {file_path.name}: {missing_vars}")

    # Restrict to Nov-Mar winter window.
    # winter 1982 = Nov 1981 to Mar 1982
    start = pd.Timestamp(f"{winter_year - 1}-11-01")
    end_exclusive = pd.Timestamp(f"{winter_year}-04-01")

    ds = ds.where((ds["time"] >= start) & (ds["time"] < end_exclusive), drop=True)

    # Build weights on this grid
    if weights_df is not None:
        weights_da = make_weight_array_for_grid(ds, weights_df)
        weight_type = "population_weighted"
    else:
        if USE_EQUAL_WEIGHTS_IF_NO_POP_WEIGHTS:
            weights_da = make_equal_weight_array_for_grid(ds)
            weight_type = "equal_grid_weighted_DEBUG_ONLY"
        else:
            raise FileNotFoundError(
                "No population weights file found.\n"
                "Please provide a population weights CSV with latitude, longitude and weight/population columns."
            )

    # Temperature: usually Kelvin. Convert to Celsius if values look like Kelvin.
    t2m = ds["t2m"]
    t2m_mean_value = float(t2m.mean().values)

    if t2m_mean_value > 100:
        t2m_c = t2m - 273.15
        temp_unit = "Kelvin_to_Celsius"
    else:
        t2m_c = t2m
        temp_unit = "Already_Celsius"

    # Wind speed from u/v components
    wind10m = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)

    # Population-weighted 6-hourly mean over lat/lon
    pw_t2m_c = t2m_c.weighted(weights_da).mean(dim=("latitude", "longitude"))
    pw_wind10m = wind10m.weighted(weights_da).mean(dim=("latitude", "longitude"))

    out_6h = xr.Dataset(
        {
            "daily_mean_t2m_c": pw_t2m_c,
            "daily_mean_wind10m": pw_wind10m,
        }
    )

    # Daily mean by ensemble member
    daily = out_6h.resample(time="1D").mean()

    # Convert to dataframe
    df = daily.to_dataframe().reset_index()

    # Standardise columns
    if "number" in df.columns:
        df = df.rename(columns={"number": "member"})
    elif "member" not in df.columns:
        df["member"] = 0

    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df["winter_year"] = winter_year

    # Reorder columns
    df = df[
        [
            "winter_year",
            "member",
            "date",
            "daily_mean_t2m_c",
            "daily_mean_wind10m",
        ]
    ].copy()

    # Remove any fully missing rows
    df = df.dropna(subset=["daily_mean_t2m_c", "daily_mean_wind10m"], how="all")

    # Save one-year file
    output_file = OUTPUT_DAILY_DIR / f"ecmwf_daily_weather_winter{winter_year}.csv"
    df.to_csv(output_file, index=False)

    # Summary
    summary = {
        "winter_year": winter_year,
        "file_name": file_path.name,
        "weight_type": weight_type,
        "temperature_conversion": temp_unit,
        "n_rows": len(df),
        "n_members": df["member"].nunique(),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
        "n_days_per_member_min": int(df.groupby("member")["date"].nunique().min()),
        "n_days_per_member_max": int(df.groupby("member")["date"].nunique().max()),
        "daily_output_file": str(output_file),
    }

    ds.close()

    print(
        f"Saved {output_file.name} | rows={summary['n_rows']} | "
        f"members={summary['n_members']} | dates={summary['date_min']} to {summary['date_max']}"
    )

    return df, summary


def main():
    print("\n==============================")
    print("02 Process ECMWF daily weather")
    print("==============================")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"RAW_DIR:     {RAW_DIR}")
    print(f"OUTPUT_DAILY_DIR: {OUTPUT_DAILY_DIR}")

    if not RAW_DIR.exists():
        raise FileNotFoundError(f"RAW_DIR does not exist: {RAW_DIR}")

    files = sorted(RAW_DIR.glob(FILE_PATTERN))
    print(f"\nFound {len(files)} formal ECMWF weather files.")

    if len(files) == 0:
        raise FileNotFoundError(f"No files found with pattern: {FILE_PATTERN}")

    # Check years
    found_years = sorted(extract_winter_year(f.name) for f in files)
    missing_years = [y for y in EXPECTED_WINTER_YEARS if y not in found_years]

    print(f"Found winter years: {found_years}")
    print(f"Missing winter years: {missing_years if missing_years else 'None'}")

    # Load population weights
    weights_path = WEIGHTS_FILE
    if weights_path is None:
        weights_path = find_population_weights_file()

    weights_df = None

    if weights_path is not None and Path(weights_path).exists():
        raw_weights_df = pd.read_csv(weights_path)
        weights_df = standardise_weights_dataframe(raw_weights_df)

        weights_used_file = OUTPUT_TABLE_DIR / "population_weights_used_by_02.csv"
        weights_df.to_csv(weights_used_file, index=False)

        print(f"\nPopulation weights loaded from:\n{weights_path}")
        print(f"Cleaned population weights saved to:\n{weights_used_file}")
        print(f"Number of weight rows: {len(weights_df)}")
        print(f"Weight sum after normalisation: {weights_df['weight'].sum():.6f}")

    else:
        print("\nNo population weights file was found automatically.")

        # Create a grid template from the first ECMWF file
        first_ds = xr.open_dataset(files[0])
        first_ds = prepare_time_dimension(first_ds)

        lats = first_ds["latitude"].values
        lons = first_ds["longitude"].values

        template_rows = []
        for lat in lats:
            for lon in lons:
                template_rows.append(
                    {
                        "latitude": float(lat),
                        "longitude": float(lon),
                        "weight": 0.0,
                    }
                )

        template = pd.DataFrame(template_rows)
        template_file = OUTPUT_TABLE_DIR / "ecmwf_grid_population_weight_template.csv"
        template.to_csv(template_file, index=False)

        first_ds.close()

        if not USE_EQUAL_WEIGHTS_IF_NO_POP_WEIGHTS:
            raise FileNotFoundError(
                "\nNo population weights file found.\n\n"
                "I created a grid template here:\n"
                f"{template_file}\n\n"
                "Please either:\n"
                "1. Put the correct population weights into this template, or\n"
                "2. Set WEIGHTS_FILE in this script to your existing population weights CSV.\n"
            )

    # Process all winters
    all_daily = []
    summaries = []

    for file_path in files:
        winter_year = extract_winter_year(file_path.name)

        if winter_year not in EXPECTED_WINTER_YEARS:
            print(f"Skipping unexpected winter year: {winter_year}")
            continue

        try:
            df_one, summary_one = process_one_winter(file_path, weights_df)
            all_daily.append(df_one)
            summaries.append(summary_one)

        except Exception as e:
            print(f"ERROR processing {file_path.name}: {e}")
            summaries.append(
                {
                    "winter_year": winter_year,
                    "file_name": file_path.name,
                    "error": str(e),
                }
            )

    # Save processing summary
    summary_df = pd.DataFrame(summaries).sort_values("winter_year")
    summary_file = OUTPUT_TABLE_DIR / "ecmwf_daily_processing_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    print("\n==============================")
    print("Processing summary saved")
    print("==============================")
    print(summary_file)

    # Save combined daily file
    if len(all_daily) == 0:
        raise RuntimeError("No winters were processed successfully.")

    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined.sort_values(["winter_year", "member", "date"]).reset_index(drop=True)

    combined_file = OUTPUT_DAILY_DIR / "ecmwf_daily_weather_1982_2016.csv"
    combined.to_csv(combined_file, index=False)

    print("\n==============================")
    print("Combined daily file saved")
    print("==============================")
    print(combined_file)

    print("\nCombined daily dataset summary:")
    print(f"Rows: {len(combined)}")
    print(f"Winter years: {combined['winter_year'].min()}–{combined['winter_year'].max()}")
    print(f"Members per winter: {combined.groupby('winter_year')['member'].nunique().unique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    print("\nDone.")


if __name__ == "__main__":
    main()