from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# Input / output paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

input_path = (
    PROJECT_DIR
    / "outputs"
    / "daily"
    / "ecmwf_daily_demand_1982_2016.csv"
)

severity_output_dir = PROJECT_DIR / "outputs" / "severity"
severity_output_dir.mkdir(parents=True, exist_ok=True)

daily_output_dir = PROJECT_DIR / "outputs" / "daily"
daily_output_dir.mkdir(parents=True, exist_ok=True)

severity_output_path = severity_output_dir / "ecmwf_severity_summary_Nov08_1982_2016.csv"
daily_output_path = daily_output_dir / "ecmwf_daily_demand_Nov08_1982_2016.csv"

# ============================================================
# Settings
# ============================================================

DEMAND_COL = "estimated_daily_mean_demand_MW"

SEVERITY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]

# Important:
# This is NOT the winter start date.
#
# The winter/model season remains NDJFM:
#   Nov 1 to Mar 31
#
# The DSN definition also remains unchanged:
#   Nov 1 = DSN 0
#
# The daily demand values in the input file have already been calculated
# using the original winter/model-season convention.
#
# The following dates only define the analysis window used when calculating
# demand-severity metrics. We use Nov 8 as a conservative start date to avoid
# early-season incomplete input issues related to the HDD7 term and the first
# incomplete ECMWF forecast day.
ANALYSIS_START_MONTH_DAY = "11-08"
ANALYSIS_END_MONTH_DAY = "03-31"


# ============================================================
# Helper functions
# ============================================================

def month_day_string(date_series):
    return date_series.dt.strftime("%m-%d")


def compute_analysis_day(date_series):
    """
    Compute analysis day within the Nov 8-Mar 31 analysis window.

    Nov 8 = 0.
    This does NOT redefine DSN.
    DSN remains based on Nov 1 = 0.
    """
    dates = pd.to_datetime(date_series).reset_index(drop=True)

    winter_start_year = np.where(
        dates.dt.month >= 11,
        dates.dt.year,
        dates.dt.year - 1
    )

    analysis_start_dates = pd.to_datetime(
        pd.Series(winter_start_year).astype(str) + "-11-08"
    )

    analysis_day = (dates - analysis_start_dates).dt.days

    return analysis_day.to_numpy()


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])

print("=" * 80)
print("Making ECMWF demand-severity summary")
print("=" * 80)

print("\nLoaded:")
print(input_path)

print("\nOriginal daily data shape:")
print(df.shape)

print("\nDate range in input file:")
print(df["date"].min(), "to", df["date"].max())

print("\nWinter years in input file:")
print(df["winter_year"].min(), "to", df["winter_year"].max())

print("\nMembers in input file:")
print(sorted(df["member"].unique()))

print("\nDemand column:")
print(DEMAND_COL)

# ============================================================
# Basic checks
# ============================================================

required_cols = [
    "winter_year",
    "member",
    "date",
    "DSN",
    DEMAND_COL,
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

if df[DEMAND_COL].isna().any():
    raise ValueError(f"Missing values found in demand column: {DEMAND_COL}")

# Check that DSN has not been redefined.
# This should show values consistent with Nov 1 = DSN 0.
print("\nDSN range in input file:")
print(df["DSN"].min(), "to", df["DSN"].max())

# ============================================================
# Define severity-analysis window
# ============================================================

df["month_day"] = month_day_string(df["date"])

# Because the winter season crosses calendar years, this condition keeps:
#   Nov 8 - Dec 31 OR Jan 1 - Mar 31
#
# This does NOT redefine the winter season.
# It only restricts the dates used when calculating severity metrics.
df_analysis = df[
    (df["month_day"] >= ANALYSIS_START_MONTH_DAY)
    | (df["month_day"] <= ANALYSIS_END_MONTH_DAY)
].copy()

# Add analysis_day for later time-series plots.
# This is only for plotting/alignment.
# It does NOT replace DSN.
df_analysis["analysis_day"] = compute_analysis_day(df_analysis["date"])

# Add GW version for plotting convenience.
df_analysis["estimated_daily_mean_demand_GW"] = (
    df_analysis[DEMAND_COL] / 1000.0
)

df_analysis = df_analysis.sort_values(
    ["winter_year", "member", "date"]
).reset_index(drop=True)

print("\nAfter analysis-window filtering:")
print("Shape:", df_analysis.shape)
print("Date range:", df_analysis["date"].min(), "to", df_analysis["date"].max())

print("\nAnalysis window used for severity metrics:")
print(f"{ANALYSIS_START_MONTH_DAY} to {ANALYSIS_END_MONTH_DAY}")

# Check how many rows were removed.
n_removed = len(df) - len(df_analysis)
print("\nRows removed by analysis-window filtering:")
print(n_removed)

print("\nAnalysis day range:")
print(df_analysis["analysis_day"].min(), "to", df_analysis["analysis_day"].max())

# ============================================================
# Audit analysis-window days
# ============================================================

audit = (
    df_analysis.groupby(["winter_year", "member"])
    .agg(
        n_days_analysis_window=("date", "nunique"),
        first_analysis_date=("date", "min"),
        last_analysis_date=("date", "max"),
        first_analysis_day=("analysis_day", "min"),
        last_analysis_day=("analysis_day", "max"),
    )
    .reset_index()
)

print("\nAnalysis-window days per winter-member:")
print(audit["n_days_analysis_window"].describe())

print("\nAnalysis-day audit:")
print(audit[["first_analysis_day", "last_analysis_day"]].describe())

# ============================================================
# Save ECMWF daily demand for later time-series plots
# ============================================================

daily_cols_to_save = [
    "winter_year",
    "member",
    "date",
    "DSN",
    "analysis_day",
    DEMAND_COL,
    "estimated_daily_mean_demand_GW",
]

# Keep other useful columns if they exist, but avoid duplicated helper column.
extra_cols = [
    col for col in df_analysis.columns
    if col not in daily_cols_to_save and col not in ["month_day"]
]

daily_to_save = df_analysis[daily_cols_to_save + extra_cols].copy()

daily_to_save.to_csv(daily_output_path, index=False)

print("\nSaved ECMWF daily demand analysis-window file to:")
print(daily_output_path)

print("\nDaily demand file shape:")
print(daily_to_save.shape)

print("\nFirst rows of saved daily demand file:")
print(daily_to_save.head())

# ============================================================
# Calculate severity metrics
# ============================================================

rows = []

for (winter_year, member), group in df_analysis.groupby(["winter_year", "member"]):
    group = group.sort_values("date").copy()

    row = {
        "dataset": "ECMWF",
        "winter_year": winter_year,
        "member": member,
        "n_days_analysis_window": group["date"].nunique(),
        "first_analysis_date": group["date"].min(),
        "last_analysis_date": group["date"].max(),
    }

    demand = group[DEMAND_COL]

    for window in SEVERITY_WINDOWS:
        # This is a rolling mean of already-estimated daily mean demand.
        # The metric represents the maximum average demand sustained over
        # a given duration within the analysis window.
        rolling_mean = demand.rolling(window=window, min_periods=window).mean()

        row[f"max_{window}d_mean_demand_MW"] = rolling_mean.max()

    rows.append(row)

severity = pd.DataFrame(rows)

# ============================================================
# Final checks
# ============================================================

n_winter_member = severity[["winter_year", "member"]].drop_duplicates().shape[0]

print("\nSeverity summary shape:")
print(severity.shape)

print("\nNumber of winter-member combinations:")
print(n_winter_member)

print("\nExpected number if 35 winters × 25 members:")
print(35 * 25)

print("\nFirst rows of severity summary:")
print(severity.head())

print("\nMissing values in severity columns:")
severity_cols = [col for col in severity.columns if col.startswith("max_")]
print(severity[severity_cols].isna().sum())

# ============================================================
# Save severity summary
# ============================================================

severity.to_csv(severity_output_path, index=False)

print("\nSaved severity summary to:")
print(severity_output_path)

print("\nDone.")