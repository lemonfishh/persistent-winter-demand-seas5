from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    AUDIT_DIR,
    BENCHMARK_WINTER_YEAR,
    PROJECT_DIR,
    SEVERITY_DIR,
    WINTER_END_YEAR,
    WINTER_START_YEAR,
)


# ============================================================
# 15. Add intermediate averaging windows
#
# Purpose:
#   Recalculate Hannah and ECMWF winter demand-severity maxima
#   using weekly averaging windows from 1 day to 84 days.
#
# Main new windows:
#   35 and 42 days, as suggested by Chris.
#
# Additional weekly windows are included to identify the range
# of averaging durations over which the unusual tail shape
# appears.
#
# Important:
#   - Winter/model season remains Nov 1-Mar 31.
#   - DSN remains Nov 1 = 0.
#   - Nov 8-Mar 31 is only the severity-analysis window.
# ============================================================


# ============================================================
# Settings
# ============================================================

DEMAND_COL = "estimated_daily_mean_demand_MW"

EXTENDED_WINDOWS = [
    1,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
    63,
    70,
    77,
    84,
]

ANALYSIS_START_MONTH_DAY = "11-08"
ANALYSIS_END_MONTH_DAY = "03-31"

MIN_COMPLETE_HANNAH_WINTER = 1951
MAX_COMPLETE_HANNAH_WINTER = 2020


# ============================================================
# Input paths
# ============================================================

# This file was created by script 01 and is already restricted
# to the Nov 8-Mar 31 analysis window.
ECMWF_DAILY_PATH = (
    PROJECT_DIR
    / "outputs"
    / "daily"
    / "ecmwf_daily_demand_Nov08_1982_2016.csv"
)

# Try the project folder first, then the original Week 3 path.
HANNAH_DAILY_CANDIDATES = [
    (
        PROJECT_DIR
        / "outputs"
        / "daily"
        / "hannah_daily_demand_deterministic_NDJFM.csv"
    ),
    (
        PROJECT_DIR
        / "outputs"
        / "hannah_daily_demand_deterministic_NDJFM.csv"
    ),
]


# ============================================================
# Output paths
# ============================================================

SEVERITY_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

ECMWF_OUTPUT_PATH = (
    SEVERITY_DIR
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv"
)

HANNAH_OUTPUT_PATH = (
    SEVERITY_DIR
    / "hannah_severity_summary_Nov08_extended_windows.csv"
)

AUDIT_OUTPUT_PATH = (
    AUDIT_DIR
    / "extended_window_severity_audit_Nov08.csv"
)


# ============================================================
# Helper functions
# ============================================================

def find_hannah_daily_file():
    """
    Find the existing Hannah daily deterministic demand file.
    """

    for path in HANNAH_DAILY_CANDIDATES:
        if path.exists():
            return path

    candidate_text = "\n".join(
        f"  - {path}"
        for path in HANNAH_DAILY_CANDIDATES
    )

    raise FileNotFoundError(
        "Could not find the Hannah daily demand file.\n"
        "Checked:\n"
        f"{candidate_text}"
    )


def restrict_to_analysis_window(df):
    """
    Restrict daily data to Nov 8-Mar 31.

    This does not redefine the winter season or DSN.
    """

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df["month_day"] = df["date"].dt.strftime("%m-%d")

    df = df[
        (df["month_day"] >= ANALYSIS_START_MONTH_DAY)
        | (df["month_day"] <= ANALYSIS_END_MONTH_DAY)
    ].copy()

    df = df.drop(columns=["month_day"])

    return df


def check_required_columns(
    df,
    required_columns,
    dataset_name,
):
    """
    Check that all required columns exist.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )


def check_daily_continuity(
    df,
    group_columns,
    dataset_name,
):
    """
    Check that dates are consecutive within each winter or
    winter-member group.
    """

    problem_groups = []

    for group_key, group in df.groupby(group_columns):
        dates = (
            pd.to_datetime(group["date"])
            .sort_values()
            .reset_index(drop=True)
        )

        if dates.duplicated().any():
            problem_groups.append(
                {
                    "group": group_key,
                    "problem": "duplicate dates",
                }
            )
            continue

        day_differences = dates.diff().dt.days.dropna()

        if not day_differences.eq(1).all():
            problem_groups.append(
                {
                    "group": group_key,
                    "problem": "non-consecutive dates",
                }
            )

    if problem_groups:
        raise ValueError(
            f"{dataset_name} contains date-continuity problems.\n"
            f"First problems: {problem_groups[:5]}"
        )


def calculate_severity_summary(
    df,
    group_columns,
    dataset_name,
):
    """
    Calculate maximum rolling-mean demand for all averaging
    windows.

    The maximum value and its start/end dates are retained.
    """

    rows = []

    for group_key, group in df.groupby(group_columns):

        group = (
            group
            .sort_values("date")
            .reset_index(drop=True)
            .copy()
        )

        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        row = {
            "dataset": dataset_name,
        }

        for column, value in zip(
            group_columns,
            group_key,
        ):
            row[column] = value

        row["n_days_analysis_window"] = (
            group["date"].nunique()
        )

        row["first_analysis_date"] = (
            group["date"].min()
        )

        row["last_analysis_date"] = (
            group["date"].max()
        )

        demand = pd.to_numeric(
            group[DEMAND_COL],
            errors="coerce",
        )

        if demand.isna().any():
            raise ValueError(
                f"Missing demand values found for "
                f"{dataset_name}, group {group_key}."
            )

        for window in EXTENDED_WINDOWS:

            if len(group) < window:
                raise ValueError(
                    f"{dataset_name}, group {group_key} "
                    f"contains only {len(group)} rows, "
                    f"which is insufficient for a "
                    f"{window}-day rolling window."
                )

            rolling_mean = demand.rolling(
                window=window,
                min_periods=window,
            ).mean()

            maximum_index = rolling_mean.idxmax()

            maximum_value = float(
                rolling_mean.loc[maximum_index]
            )

            maximum_end_date = pd.to_datetime(
                group.loc[maximum_index, "date"]
            )

            maximum_start_date = (
                maximum_end_date
                - pd.Timedelta(days=window - 1)
            )

            row[
                f"max_{window}d_mean_demand_MW"
            ] = maximum_value

            row[
                f"max_{window}d_window_start"
            ] = maximum_start_date

            row[
                f"max_{window}d_window_end"
            ] = maximum_end_date

        rows.append(row)

    severity = pd.DataFrame(rows)

    sort_columns = [
        column
        for column in group_columns
        if column in severity.columns
    ]

    severity = (
        severity
        .sort_values(sort_columns)
        .reset_index(drop=True)
    )

    return severity


def build_audit_table(
    ecmwf_daily,
    hannah_daily,
    ecmwf_severity,
    hannah_severity,
):
    """
    Create a compact audit table for the extended-window files.
    """

    audit_rows = [
        {
            "dataset": "ECMWF",
            "winter_year_start": (
                ecmwf_daily["winter_year"].min()
            ),
            "winter_year_end": (
                ecmwf_daily["winter_year"].max()
            ),
            "n_winters": (
                ecmwf_daily["winter_year"].nunique()
            ),
            "n_members": (
                ecmwf_daily["member"].nunique()
            ),
            "n_daily_rows": len(ecmwf_daily),
            "n_severity_rows": len(ecmwf_severity),
            "severity_windows": ",".join(
                str(window)
                for window in EXTENDED_WINDOWS
            ),
        },
        {
            "dataset": "Hannah",
            "winter_year_start": (
                hannah_daily["winter_year"].min()
            ),
            "winter_year_end": (
                hannah_daily["winter_year"].max()
            ),
            "n_winters": (
                hannah_daily["winter_year"].nunique()
            ),
            "n_members": np.nan,
            "n_daily_rows": len(hannah_daily),
            "n_severity_rows": len(hannah_severity),
            "severity_windows": ",".join(
                str(window)
                for window in EXTENDED_WINDOWS
            ),
        },
    ]

    return pd.DataFrame(audit_rows)


# ============================================================
# Load ECMWF daily demand
# ============================================================

print("=" * 80)
print("Adding intermediate averaging windows")
print("=" * 80)

print("\nExtended severity windows:")
print(EXTENDED_WINDOWS)

if not ECMWF_DAILY_PATH.exists():
    raise FileNotFoundError(
        "Could not find the ECMWF daily demand file:\n"
        f"{ECMWF_DAILY_PATH}\n\n"
        "Run script 01 first."
    )

ecmwf_daily = pd.read_csv(ECMWF_DAILY_PATH)
ecmwf_daily["date"] = pd.to_datetime(
    ecmwf_daily["date"]
)

check_required_columns(
    ecmwf_daily,
    [
        "winter_year",
        "member",
        "date",
        DEMAND_COL,
    ],
    dataset_name="ECMWF",
)

ecmwf_daily = ecmwf_daily[
    ecmwf_daily["winter_year"].between(
        WINTER_START_YEAR,
        WINTER_END_YEAR,
    )
].copy()

# Apply the filter again defensively, even though script 01
# already produced an analysis-window daily file.
ecmwf_daily = restrict_to_analysis_window(
    ecmwf_daily
)

ecmwf_daily = (
    ecmwf_daily
    .sort_values(
        [
            "winter_year",
            "member",
            "date",
        ]
    )
    .reset_index(drop=True)
)

print("\nLoaded ECMWF daily demand:")
print(ECMWF_DAILY_PATH)
print("Shape:", ecmwf_daily.shape)
print(
    "Winter years:",
    ecmwf_daily["winter_year"].min(),
    "to",
    ecmwf_daily["winter_year"].max(),
)
print(
    "Members:",
    ecmwf_daily["member"].nunique(),
)

check_daily_continuity(
    ecmwf_daily,
    group_columns=[
        "winter_year",
        "member",
    ],
    dataset_name="ECMWF",
)


# ============================================================
# Load Hannah daily demand
# ============================================================

HANNAH_DAILY_PATH = find_hannah_daily_file()

hannah_daily = pd.read_csv(HANNAH_DAILY_PATH)
hannah_daily["date"] = pd.to_datetime(
    hannah_daily["date"]
)

check_required_columns(
    hannah_daily,
    [
        "winter_year",
        "date",
        DEMAND_COL,
    ],
    dataset_name="Hannah",
)

hannah_daily = hannah_daily[
    hannah_daily["winter_year"].between(
        MIN_COMPLETE_HANNAH_WINTER,
        MAX_COMPLETE_HANNAH_WINTER,
    )
].copy()

hannah_daily = restrict_to_analysis_window(
    hannah_daily
)

hannah_daily = (
    hannah_daily
    .sort_values(
        [
            "winter_year",
            "date",
        ]
    )
    .reset_index(drop=True)
)

print("\nLoaded Hannah daily demand:")
print(HANNAH_DAILY_PATH)
print("Shape:", hannah_daily.shape)
print(
    "Winter years:",
    hannah_daily["winter_year"].min(),
    "to",
    hannah_daily["winter_year"].max(),
)
print(
    "Number of complete winters:",
    hannah_daily["winter_year"].nunique(),
)

check_daily_continuity(
    hannah_daily,
    group_columns=[
        "winter_year",
    ],
    dataset_name="Hannah",
)


# ============================================================
# Calculate extended-window severity
# ============================================================

print("\nCalculating ECMWF extended-window severity...")

ecmwf_severity = calculate_severity_summary(
    df=ecmwf_daily,
    group_columns=[
        "winter_year",
        "member",
    ],
    dataset_name="ECMWF",
)

print("ECMWF severity shape:")
print(ecmwf_severity.shape)

print("\nCalculating Hannah extended-window severity...")

hannah_severity = calculate_severity_summary(
    df=hannah_daily,
    group_columns=[
        "winter_year",
    ],
    dataset_name="Hannah",
)

print("Hannah severity shape:")
print(hannah_severity.shape)


# ============================================================
# Hard checks
# ============================================================

if len(ecmwf_severity) != 875:
    raise ValueError(
        "Expected 875 ECMWF winter-member rows, "
        f"but found {len(ecmwf_severity)}."
    )

if (
    ecmwf_severity["winter_year"].nunique()
    != 35
):
    raise ValueError(
        "Expected 35 ECMWF winter years."
    )

if ecmwf_severity["member"].nunique() != 25:
    raise ValueError(
        "Expected 25 ECMWF ensemble members."
    )

if len(hannah_severity) != 70:
    raise ValueError(
        "Expected 70 complete Hannah winters, "
        f"but found {len(hannah_severity)}."
    )

if (
    BENCHMARK_WINTER_YEAR
    not in set(hannah_severity["winter_year"])
):
    raise ValueError(
        f"Hannah benchmark winter "
        f"{BENCHMARK_WINTER_YEAR} is missing."
    )

severity_value_columns = [
    f"max_{window}d_mean_demand_MW"
    for window in EXTENDED_WINDOWS
]

if (
    ecmwf_severity[
        severity_value_columns
    ].isna().any().any()
):
    raise ValueError(
        "Missing values found in ECMWF extended severity."
    )

if (
    hannah_severity[
        severity_value_columns
    ].isna().any().any()
):
    raise ValueError(
        "Missing values found in Hannah extended severity."
    )


# ============================================================
# Validate the original seven windows
# ============================================================

ORIGINAL_WINDOWS = [
    1,
    7,
    14,
    21,
    28,
    56,
    84,
]

ORIGINAL_ECMWF_PATH = (
    SEVERITY_DIR
    / "ecmwf_severity_summary_Nov08_1982_2016.csv"
)

ORIGINAL_HANNAH_PATH = (
    SEVERITY_DIR
    / "hannah_severity_summary_Nov08.csv"
)

if (
    ORIGINAL_ECMWF_PATH.exists()
    and ORIGINAL_HANNAH_PATH.exists()
):

    original_ecmwf = pd.read_csv(
        ORIGINAL_ECMWF_PATH
    )

    original_hannah = pd.read_csv(
        ORIGINAL_HANNAH_PATH
    )

    ecmwf_check = ecmwf_severity.merge(
        original_ecmwf,
        on=[
            "winter_year",
            "member",
        ],
        suffixes=(
            "_extended",
            "_original",
        ),
    )

    hannah_check = hannah_severity.merge(
        original_hannah,
        on=[
            "winter_year",
        ],
        suffixes=(
            "_extended",
            "_original",
        ),
    )

    for window in ORIGINAL_WINDOWS:

        column = (
            f"max_{window}d_mean_demand_MW"
        )

        ecmwf_equal = np.allclose(
            ecmwf_check[
                f"{column}_extended"
            ],
            ecmwf_check[
                f"{column}_original"
            ],
            rtol=0.0,
            atol=1e-8,
        )

        hannah_equal = np.allclose(
            hannah_check[
                f"{column}_extended"
            ],
            hannah_check[
                f"{column}_original"
            ],
            rtol=0.0,
            atol=1e-8,
        )

        if not ecmwf_equal:
            raise ValueError(
                f"ECMWF validation failed for "
                f"{window}-day severity."
            )

        if not hannah_equal:
            raise ValueError(
                f"Hannah validation failed for "
                f"{window}-day severity."
            )

    print(
        "\nValidation passed: all original "
        "1, 7, 14, 21, 28, 56 and 84-day "
        "severity values were reproduced exactly."
    )

else:
    print(
        "\nOriginal severity files were not found, "
        "so the original-window validation was skipped."
    )


# ============================================================
# Save outputs
# ============================================================

ecmwf_severity.to_csv(
    ECMWF_OUTPUT_PATH,
    index=False,
)

hannah_severity.to_csv(
    HANNAH_OUTPUT_PATH,
    index=False,
)

audit_table = build_audit_table(
    ecmwf_daily=ecmwf_daily,
    hannah_daily=hannah_daily,
    ecmwf_severity=ecmwf_severity,
    hannah_severity=hannah_severity,
)

audit_table.to_csv(
    AUDIT_OUTPUT_PATH,
    index=False,
)


# ============================================================
# Print selected new-window summaries
# ============================================================

print("\n" + "=" * 80)
print("Selected new-window summary")
print("=" * 80)

for window in [
    35,
    42,
    49,
    63,
    70,
    77,
]:

    column = (
        f"max_{window}d_mean_demand_MW"
    )

    hannah_overlap = hannah_severity[
        hannah_severity["winter_year"].between(
            WINTER_START_YEAR,
            WINTER_END_YEAR,
        )
    ][column]

    ecmwf_values = ecmwf_severity[column]

    print(f"\n{window}-day severity:")

    print(
        "  Hannah overlap mean: "
        f"{hannah_overlap.mean():.3f} MW"
    )

    print(
        "  Hannah overlap std:  "
        f"{hannah_overlap.std(ddof=1):.3f} MW"
    )

    print(
        "  ECMWF pooled mean:   "
        f"{ecmwf_values.mean():.3f} MW"
    )

    print(
        "  ECMWF pooled std:    "
        f"{ecmwf_values.std(ddof=1):.3f} MW"
    )


# ============================================================
# Final messages
# ============================================================

print("\nSaved ECMWF extended severity to:")
print(ECMWF_OUTPUT_PATH)

print("\nSaved Hannah extended severity to:")
print(HANNAH_OUTPUT_PATH)

print("\nSaved audit table to:")
print(AUDIT_OUTPUT_PATH)

print("\nDone.")