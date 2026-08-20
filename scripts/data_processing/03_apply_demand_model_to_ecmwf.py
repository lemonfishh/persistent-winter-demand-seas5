from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 03_apply_demand_model_to_ecmwf.py
#
# Purpose:
# Apply Aninda's deterministic daily mean demand model to the
# ECMWF SEAS5 population-weighted daily weather dataset.
#
# Input:
# data_processed/daily/ecmwf_daily_weather_1982_2016.csv
#
# Output:
# data_processed/daily/ecmwf_daily_demand_1982_2016.csv
# outputs/tables/ecmwf_demand_processing_summary.csv
#
# Model:
# estimated_daily_mean_demand_MW
# = ALPHA
# + LAMBDA_HDD * HDD
# + LAMBDA_HDD_7DAY * HDD_7day_avg
# + GAMMA_WS * daily_mean_wind10m
# + DOW_effect
# + BETA1_2018 * DSN
# + BETA2_2018 * DSN^2
# + PSI1 * TARGET_YEAR_EFFECT
#
# Residuals are NOT added here.
# This is the deterministic weather-driven demand component.
# ============================================================


# =========================
# Project paths
# =========================
PROJECT_DIR = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_DIR / "outputs" / "daily" / "ecmwf_daily_weather_1982_2016.csv"

OUTPUT_DAILY_DIR = PROJECT_DIR / "outputs" / "daily"
OUTPUT_TABLE_DIR = PROJECT_DIR / "outputs" / "tables"

OUTPUT_DAILY_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DAILY_DIR / "ecmwf_daily_demand_1982_2016.csv"
SUMMARY_FILE = OUTPUT_TABLE_DIR / "ecmwf_demand_processing_summary.csv"


# =========================
# Demand model parameters
# =========================

# HDD base temperature
HDD_BASE_TEMP = 15.5

# Regression coefficients from Aninda's Daily Demand Model
ALPHA = 30983.77
LAMBDA_HDD = 315.31
LAMBDA_HDD_7DAY = 301.14
GAMMA_WS = 161.41

PSI1 = -399.75
TARGET_YEAR_EFFECT = 10  # transformation equation uses psi_1 * 10

BETA1_2018 = 26.20
BETA2_2018 = -0.25

# Day-of-week effects.
# Saturday is reference, so Saturday effect = 0.
# Mapping follows previous week3 implementation.
DOW_EFFECT = {
    "Sunday": -852.39,
    "Monday": 3876.48,
    "Tuesday": 4369.05,
    "Wednesday": 4389.15,
    "Thursday": 4263.68,
    "Friday": 3629.33,
    "Saturday": 0.0,
}


def main():
    print("\n==============================")
    print("03 Apply demand model to ECMWF")
    print("==============================")
    print(f"INPUT_FILE: {INPUT_FILE}")
    print(f"OUTPUT_FILE: {OUTPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    # =========================
    # Load ECMWF daily weather
    # =========================
    df = pd.read_csv(INPUT_FILE)
    df["date"] = pd.to_datetime(df["date"])

    print("\nLoaded ECMWF daily weather:")
    print(df.head().to_string(index=False))
    print("\nShape:", df.shape)
    print("Columns:", df.columns.tolist())

    required_cols = [
        "winter_year",
        "member",
        "date",
        "daily_mean_t2m_c",
        "daily_mean_wind10m",
    ]

    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Sort before rolling calculations
    df = df.sort_values(["winter_year", "member", "date"]).reset_index(drop=True)

    # =========================
    # Construct demand model inputs
    # =========================

    # Heating Degree Days
    df["HDD"] = HDD_BASE_TEMP - df["daily_mean_t2m_c"]
    df["HDD"] = df["HDD"].clip(lower=0)

    # 7-day average HDD, computed separately for each winter/member
    # Since ECMWF files start at 1 Nov, the first 6 days of each winter/member
    # will have missing HDD_7day_avg and are dropped later.
    df["HDD_7day_avg"] = (
        df.groupby(["winter_year", "member"])["HDD"]
        .transform(lambda x: x.rolling(window=7, min_periods=7).mean())
    )

    # Day of week
    df["day_name"] = df["date"].dt.day_name()
    df["DOW_effect"] = df["day_name"].map(DOW_EFFECT)

    # Days since start of November
    # Example: winter 1982 starts on 1981-11-01, so DSN = 0 on 1981-11-01.
    winter_start_dates = pd.to_datetime((df["winter_year"] - 1).astype(str) + "-11-01")
    df["DSN"] = (df["date"] - winter_start_dates).dt.days

    # =========================
    # Deterministic demand model
    # =========================
    df["estimated_daily_mean_demand_MW"] = (
        ALPHA
        + LAMBDA_HDD * df["HDD"]
        + LAMBDA_HDD_7DAY * df["HDD_7day_avg"]
        + GAMMA_WS * df["daily_mean_wind10m"]
        + df["DOW_effect"]
        + BETA1_2018 * df["DSN"]
        + BETA2_2018 * (df["DSN"] ** 2)
        + PSI1 * TARGET_YEAR_EFFECT
    )

    # Drop invalid rows
    # Main reason: first 6 days for each member/winter have no 7-day HDD average.
    before_drop = len(df)

    demand_df = df.dropna(
        subset=[
            "HDD",
            "HDD_7day_avg",
            "daily_mean_wind10m",
            "DOW_effect",
            "DSN",
            "estimated_daily_mean_demand_MW",
        ]
    ).copy()

    after_drop = len(demand_df)

    # =========================
    # Save output
    # =========================
    demand_df.to_csv(OUTPUT_FILE, index=False)

    # =========================
    # Summary table
    # =========================
    summary = (
        demand_df.groupby("winter_year")
        .agg(
            n_rows=("date", "count"),
            n_members=("member", "nunique"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            mean_temperature_c=("daily_mean_t2m_c", "mean"),
            mean_wind10m=("daily_mean_wind10m", "mean"),
            mean_HDD=("HDD", "mean"),
            mean_HDD_7day_avg=("HDD_7day_avg", "mean"),
            mean_estimated_demand_MW=("estimated_daily_mean_demand_MW", "mean"),
            max_estimated_demand_MW=("estimated_daily_mean_demand_MW", "max"),
            min_estimated_demand_MW=("estimated_daily_mean_demand_MW", "min"),
        )
        .reset_index()
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    # =========================
    # Print checks
    # =========================
    print("\n==============================")
    print("Demand model applied")
    print("==============================")
    print(f"Rows before drop: {before_drop}")
    print(f"Rows after drop:  {after_drop}")
    print(f"Rows dropped:     {before_drop - after_drop}")

    print("\nMissing values after drop:")
    print(demand_df.isna().sum())

    print("\nDemand summary:")
    print(demand_df["estimated_daily_mean_demand_MW"].describe())

    print("\nHDD summary:")
    print(demand_df["HDD"].describe())

    print("\nHDD 7-day average summary:")
    print(demand_df["HDD_7day_avg"].describe())

    print("\nDate range:")
    print(demand_df["date"].min(), "to", demand_df["date"].max())

    print("\nWinter years:")
    print(demand_df["winter_year"].min(), "to", demand_df["winter_year"].max())

    print("\nMembers per winter:")
    print(demand_df.groupby("winter_year")["member"].nunique().unique())

    print("\nTop 10 demand days:")
    top10 = demand_df.sort_values("estimated_daily_mean_demand_MW", ascending=False).head(10)
    print(
        top10[
            [
                "winter_year",
                "member",
                "date",
                "daily_mean_t2m_c",
                "daily_mean_wind10m",
                "HDD",
                "HDD_7day_avg",
                "day_name",
                "estimated_daily_mean_demand_MW",
            ]
        ].to_string(index=False)
    )

    print("\n==============================")
    print("Output files saved")
    print("==============================")
    print(f"Daily demand file: {OUTPUT_FILE}")
    print(f"Summary file:      {SUMMARY_FILE}")

    print("\nDone.")


if __name__ == "__main__":
    main()