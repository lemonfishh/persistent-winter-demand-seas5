from pathlib import Path
import pandas as pd

# ============================================================
# Input / output paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

input_path = (
    PROJECT_DIR
    / "outputs"
    / "daily"
    / "hannah_daily_demand_deterministic_NDJFM.csv"
)

output_dir = PROJECT_DIR / "outputs" / "severity"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "hannah_severity_summary_Nov08.csv"

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
# This only defines the analysis window used when calculating
# demand-severity metrics.
ANALYSIS_START_MONTH_DAY = "11-08"
ANALYSIS_END_MONTH_DAY = "03-31"

# Complete winters only.
# The input file starts on 1950-01-07 and ends on 2020-12-31,
# so winter 1950 and winter 2021 are incomplete and should not
# be used for full-winter severity metrics.
MIN_COMPLETE_WINTER = 1951
MAX_COMPLETE_WINTER = 2020

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(input_path)
df["date"] = pd.to_datetime(df["date"])

print("=" * 80)
print("Making Hannah demand-severity summary")
print("=" * 80)

print("\nLoaded:")
print(input_path)

print("\nOriginal daily data shape:")
print(df.shape)

print("\nDate range in input file:")
print(df["date"].min(), "to", df["date"].max())

print("\nWinter years in input file:")
print(df["winter_year"].min(), "to", df["winter_year"].max())

print("\nDemand column:")
print(DEMAND_COL)

# ============================================================
# Basic checks
# ============================================================

required_cols = [
    "date",
    "winter_year",
    "DSN",
    DEMAND_COL,
]

missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

if df[DEMAND_COL].isna().any():
    raise ValueError(f"Missing values found in demand column: {DEMAND_COL}")

print("\nDSN range in input file:")
print(df["DSN"].min(), "to", df["DSN"].max())

# ============================================================
# Restrict to complete winters
# ============================================================

df = df[
    (df["winter_year"] >= MIN_COMPLETE_WINTER)
    & (df["winter_year"] <= MAX_COMPLETE_WINTER)
].copy()

print("\nAfter restricting to complete winters:")
print("Shape:", df.shape)
print("Winter years:", df["winter_year"].min(), "to", df["winter_year"].max())
print("Number of winters:", df["winter_year"].nunique())

# ============================================================
# Define severity-analysis window
# ============================================================

df["month_day"] = df["date"].dt.strftime("%m-%d")

# Because the winter season crosses calendar years, this condition keeps:
#   Nov 8 - Dec 31 OR Jan 1 - Mar 31
#
# This does NOT redefine the winter season.
# It only restricts the dates used when calculating severity metrics.
df_analysis = df[
    (df["month_day"] >= ANALYSIS_START_MONTH_DAY)
    | (df["month_day"] <= ANALYSIS_END_MONTH_DAY)
].copy()

print("\nAfter analysis-window filtering:")
print("Shape:", df_analysis.shape)
print("Date range:", df_analysis["date"].min(), "to", df_analysis["date"].max())

print("\nAnalysis window used for severity metrics:")
print(f"{ANALYSIS_START_MONTH_DAY} to {ANALYSIS_END_MONTH_DAY}")

# ============================================================
# Audit analysis-window days
# ============================================================

audit = (
    df_analysis.groupby("winter_year")
    .agg(
        n_days_analysis_window=("date", "nunique"),
        first_analysis_date=("date", "min"),
        last_analysis_date=("date", "max"),
    )
    .reset_index()
)

print("\nAnalysis-window days per winter:")
print(audit["n_days_analysis_window"].describe())

print("\nWinters with unusual number of analysis-window days:")
print(
    audit[
        ~audit["n_days_analysis_window"].isin([144, 145])
    ]
)

# ============================================================
# Calculate severity metrics
# ============================================================

rows = []

for winter_year, group in df_analysis.groupby("winter_year"):
    group = group.sort_values("date").copy()

    row = {
        "dataset": "Hannah",
        "winter_year": winter_year,
        "n_days_analysis_window": group["date"].nunique(),
        "first_analysis_date": group["date"].min(),
        "last_analysis_date": group["date"].max(),
    }

    demand = group[DEMAND_COL]

    for window in SEVERITY_WINDOWS:
        rolling_mean = demand.rolling(window=window, min_periods=window).mean()
        row[f"max_{window}d_mean_demand_MW"] = rolling_mean.max()

    rows.append(row)

severity = pd.DataFrame(rows)

# ============================================================
# Final checks
# ============================================================

print("\nSeverity summary shape:")
print(severity.shape)

print("\nNumber of Hannah winters:")
print(severity["winter_year"].nunique())

print("\nFirst rows of severity summary:")
print(severity.head())

print("\nMissing values in severity columns:")
severity_cols = [col for col in severity.columns if col.startswith("max_")]
print(severity[severity_cols].isna().sum())

print("\nDoes benchmark winter 1963 exist?")
print(1963 in set(severity["winter_year"]))

print("\nDoes overlap period 1982-2016 exist?")
overlap = severity[
    (severity["winter_year"] >= 1982)
    & (severity["winter_year"] <= 2016)
]
print("Number of overlap winters:", overlap["winter_year"].nunique())
print("Overlap range:", overlap["winter_year"].min(), "to", overlap["winter_year"].max())

# ============================================================
# Save
# ============================================================

severity.to_csv(output_path, index=False)

print("\nSaved severity summary to:")
print(output_path)