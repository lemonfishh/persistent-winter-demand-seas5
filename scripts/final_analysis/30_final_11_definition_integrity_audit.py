from __future__ import annotations

"""
Final definition-integrity audit for the dissertation pipeline.

Purpose
-------
This script checks the definitions that are most vulnerable to silent
off-by-one, winter-label, window-boundary, unit and hard-coded-constant
errors.

It audits:

1. Single-source configuration values.
2. Hard-coded winter/benchmark constants in Python scripts.
3. ECMWF daily-data winter labels, dates, DSN, analysis_day, units,
   HDD, 7-day HDD and deterministic demand formula.
4. ECMWF member completeness and row counts.
5. Hannah winter labels and complete-winter coverage.
6. Recalculation of all severity-window maxima from daily files.
7. Mean-shift reconstruction, historical maxima and the 1962/63
   benchmark (winter_year=1963).
8. Basic GEV input sizes and return-level difference files, when present.

Run from the project root:

    /usr/bin/python3 scripts/30_final_11_definition_integrity_audit.py

The script writes:

    outputs/final_analysis_clean/11_definition_integrity_audit/
        definition_integrity_audit.csv
        benchmark_recalculation_audit.csv
        script_constant_scan.csv

Any failed core check causes a non-zero exit.
"""

from dataclasses import dataclass
from pathlib import Path
import calendar
import re
import sys

import numpy as np
import pandas as pd

from config import (
    OUTPUT_DIR,
    PROJECT_DIR,
    WINTER_START_YEAR,
    WINTER_END_YEAR,
    VALID_START_MONTH_DAY,
    VALID_END_MONTH_DAY,
    SEVERITY_WINDOWS,
    BENCHMARK_WINTER_YEAR,
)


# =============================================================================
# Frozen dissertation definitions
# =============================================================================

EXPECTED_WINTER_START_YEAR = 1982
EXPECTED_WINTER_END_YEAR = 2016
EXPECTED_BENCHMARK_WINTER_YEAR = 1963

EXPECTED_PRIMARY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]
EXPECTED_EXTENDED_WINDOWS = [
    1, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84
]

EXPECTED_ANALYSIS_START = "11-08"
EXPECTED_ANALYSIS_END = "03-31"

EXPECTED_ECMWF_WINTERS = 35
EXPECTED_ECMWF_MEMBERS_PER_WINTER = 25
EXPECTED_ECMWF_SEVERITY_ROWS = 875
EXPECTED_HANNAH_START_YEAR = 1951
EXPECTED_HANNAH_END_YEAR = 2020
EXPECTED_HANNAH_WINTERS = 70

HDD_BASE_C = 15.5

# Deterministic demand model frozen for the dissertation.
ALPHA = 30983.77
LAMBDA_HDD_CURRENT = 315.31
LAMBDA_HDD_7DAY = 301.14
GAMMA_WIND = 161.41
BETA_DSN = 26.20
BETA_DSN_SQUARED = -0.25
TARGET_YEAR_COEFFICIENT = -399.75
FIXED_TARGET_YEAR_EFFECT = 10.0

DOW_EXPECTED = {
    "Monday": 3876.48,
    "Tuesday": 4369.05,
    "Wednesday": 4389.15,
    "Thursday": 4263.68,
    "Friday": 3629.33,
    "Saturday": 0.0,
    "Sunday": -852.39,
}

ATOL = 1e-6
DEMAND_ATOL_MW = 1e-4


# =============================================================================
# Paths
# =============================================================================

FINAL_ROOT = OUTPUT_DIR / "final_analysis_clean"
AUDIT_DIR = FINAL_ROOT / "11_definition_integrity_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

ECMWF_DAILY_FILE = (
    OUTPUT_DIR / "daily" / "ecmwf_daily_demand_Nov08_1982_2016.csv"
)

HANNAH_DAILY_CANDIDATES = [
    OUTPUT_DIR / "daily" / "hannah_daily_demand_deterministic_NDJFM.csv",
    OUTPUT_DIR / "hannah_daily_demand_deterministic_NDJFM.csv",
]

ECMWF_SEVERITY_FILE = (
    OUTPUT_DIR
    / "severity"
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv"
)

HANNAH_SEVERITY_FILE = (
    OUTPUT_DIR
    / "severity"
    / "hannah_severity_summary_Nov08_extended_windows.csv"
)

MEAN_SHIFT_SUMMARY_FILE = (
    FINAL_ROOT
    / "01_mean_shifted_primary_comparison"
    / "primary_windows_mean_shift_summary.csv"
)

GEV_ORIGINAL_FITS_FILE = (
    FINAL_ROOT
    / "06_stationary_gev_primary_windows"
    / "stationary_gev_original_fits_primary_windows.csv"
)

GEV_DIFF_CI_FILE = (
    FINAL_ROOT
    / "06_stationary_gev_primary_windows"
    / "stationary_gev_bootstrap_return_level_difference_ci.csv"
)

AUDIT_OUTPUT = AUDIT_DIR / "definition_integrity_audit.csv"
BENCHMARK_OUTPUT = AUDIT_DIR / "benchmark_recalculation_audit.csv"
SCRIPT_SCAN_OUTPUT = AUDIT_DIR / "script_constant_scan.csv"


# =============================================================================
# Audit framework
# =============================================================================

@dataclass
class AuditRow:
    section: str
    check: str
    status: str
    detail: str


audit_rows: list[AuditRow] = []


def record(
    section: str,
    check: str,
    passed: bool,
    detail: str,
    *,
    warning: bool = False,
) -> None:
    if passed:
        status = "PASS"
    elif warning:
        status = "WARNING"
    else:
        status = "FAIL"

    audit_rows.append(
        AuditRow(
            section=section,
            check=check,
            status=status,
            detail=detail,
        )
    )

    print(f"[{status:7s}] {section} | {check}")
    print(f"          {detail}")


def require_file(path: Path, label: str) -> bool:
    exists = path.exists()
    record(
        "Files",
        label,
        exists,
        str(path),
    )
    return exists


def allclose(
    left,
    right,
    *,
    atol: float = ATOL,
    rtol: float = 0.0,
) -> bool:
    return bool(
        np.allclose(
            np.asarray(left, dtype=float),
            np.asarray(right, dtype=float),
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )
    )


def expected_winter_year(dates: pd.Series) -> pd.Series:
    dates = pd.to_datetime(dates)
    return pd.Series(
        np.where(
            dates.dt.month >= 11,
            dates.dt.year + 1,
            dates.dt.year,
        ),
        index=dates.index,
        dtype="int64",
    )


def winter_start_date(winter_year: int) -> pd.Timestamp:
    return pd.Timestamp(
        year=int(winter_year) - 1,
        month=11,
        day=8,
    )


def winter_end_date(winter_year: int) -> pd.Timestamp:
    return pd.Timestamp(
        year=int(winter_year),
        month=3,
        day=31,
    )


def expected_analysis_days(winter_year: int) -> int:
    return (
        winter_end_date(winter_year)
        - winter_start_date(winter_year)
    ).days + 1


def calculate_dsn(
    dates: pd.Series,
    winter_years: pd.Series,
) -> pd.Series:
    nov1 = pd.to_datetime(
        {
            "year": pd.to_numeric(winter_years).astype(int) - 1,
            "month": 11,
            "day": 1,
        }
    )
    return (
        pd.to_datetime(dates).reset_index(drop=True)
        - nov1.reset_index(drop=True)
    ).dt.days


def find_hannah_daily_file() -> Path | None:
    matches = [
        path for path in HANNAH_DAILY_CANDIDATES
        if path.exists()
    ]

    if len(matches) == 0:
        record(
            "Files",
            "Hannah daily file",
            False,
            "No Hannah daily candidate was found; Hannah rolling-window "
            "recalculation will be skipped.",
            warning=True,
        )
        return None

    if len(matches) > 1:
        record(
            "Files",
            "Hannah daily file uniqueness",
            False,
            "Several candidates exist: "
            + " | ".join(str(path) for path in matches)
            + ". The first project-local candidate will be used, but "
              "duplicate versions should be removed or archived.",
            warning=True,
        )

    selected = matches[0]
    record(
        "Files",
        "Hannah daily file",
        True,
        str(selected),
    )
    return selected


# =============================================================================
# 1. Configuration and source-code scan
# =============================================================================

def audit_configuration() -> None:
    record(
        "Configuration",
        "winter-year convention start",
        WINTER_START_YEAR == EXPECTED_WINTER_START_YEAR,
        f"config={WINTER_START_YEAR}; expected={EXPECTED_WINTER_START_YEAR}",
    )
    record(
        "Configuration",
        "winter-year convention end",
        WINTER_END_YEAR == EXPECTED_WINTER_END_YEAR,
        f"config={WINTER_END_YEAR}; expected={EXPECTED_WINTER_END_YEAR}",
    )
    record(
        "Configuration",
        "1962/63 benchmark label",
        BENCHMARK_WINTER_YEAR == EXPECTED_BENCHMARK_WINTER_YEAR,
        f"config={BENCHMARK_WINTER_YEAR}; expected=1963 because "
        "winter 1963 means Nov 1962--Mar 1963",
    )
    record(
        "Configuration",
        "analysis start",
        VALID_START_MONTH_DAY == EXPECTED_ANALYSIS_START,
        f"config={VALID_START_MONTH_DAY}; expected={EXPECTED_ANALYSIS_START}",
    )
    record(
        "Configuration",
        "analysis end",
        VALID_END_MONTH_DAY == EXPECTED_ANALYSIS_END,
        f"config={VALID_END_MONTH_DAY}; expected={EXPECTED_ANALYSIS_END}",
    )
    record(
        "Configuration",
        "primary severity windows",
        list(SEVERITY_WINDOWS) == EXPECTED_PRIMARY_WINDOWS,
        f"config={list(SEVERITY_WINDOWS)}; expected={EXPECTED_PRIMARY_WINDOWS}",
    )


def scan_python_constants() -> None:
    scripts_dir = PROJECT_DIR / "scripts"
    rows: list[dict] = []

    suspicious_patterns = [
        (
            "benchmark_assigned_1962",
            re.compile(
                r"(?i)(benchmark[^\n=]*|historical_benchmark[^\n=]*)"
                r"=\s*1962\b"
            ),
        ),
        (
            "winter_year_equal_1962",
            re.compile(r"(?i)winter_year[^\n]*==\s*1962\b"),
        ),
        (
            "wrong_comment_mapping",
            re.compile(
                r"(?i)winter\s*year\s*1962[^\n]*"
                r"(1962[/\-]63|winter\s*1962)"
            ),
        ),
        (
            "external_benchmark_list_1962",
            re.compile(
                r"(?i)external_benchmark[^\n=]*=\s*\[\s*1962\s*\]"
            ),
        ),
    ]

    if not scripts_dir.exists():
        record(
            "Script scan",
            "scripts directory exists",
            False,
            str(scripts_dir),
        )
        return

    for path in sorted(scripts_dir.rglob("*.py")):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            continue

        for line_number, line in enumerate(
            text.splitlines(),
            start=1,
        ):
            for issue, pattern in suspicious_patterns:
                if pattern.search(line):
                    rows.append(
                        {
                            "file": str(path.relative_to(PROJECT_DIR)),
                            "line_number": line_number,
                            "issue": issue,
                            "line": line.strip(),
                        }
                    )

    scan = pd.DataFrame(
        rows,
        columns=[
            "file",
            "line_number",
            "issue",
            "line",
        ],
    )
    scan.to_csv(
        SCRIPT_SCAN_OUTPUT,
        index=False,
    )

    record(
        "Script scan",
        "no hard-coded benchmark=1962 remains",
        len(scan) == 0,
        (
            "No suspicious hard-coded winter-label patterns found."
            if len(scan) == 0
            else f"{len(scan)} suspicious line(s) found; see "
                 f"{SCRIPT_SCAN_OUTPUT}"
        ),
    )


# =============================================================================
# 2. Daily-file audit
# =============================================================================

def audit_daily_group_structure(
    dataframe: pd.DataFrame,
    *,
    dataset: str,
    group_columns: list[str],
    expected_start_year: int,
    expected_end_year: int,
) -> None:
    data = dataframe.copy()
    data["date"] = pd.to_datetime(
        data["date"],
        errors="coerce",
    )
    data["winter_year"] = pd.to_numeric(
        data["winter_year"],
        errors="coerce",
    )

    record(
        dataset,
        "date parsing",
        data["date"].notna().all(),
        f"missing parsed dates={int(data['date'].isna().sum())}",
    )

    expected_labels = expected_winter_year(data["date"])
    actual_labels = data["winter_year"].astype("Int64")

    mismatch = (
        actual_labels.to_numpy(dtype="int64")
        != expected_labels.to_numpy(dtype="int64")
    )

    record(
        dataset,
        "winter_year agrees with dates",
        not mismatch.any(),
        (
            "Every November--December date is labelled by the following "
            "year, and every January--March date by its calendar year."
            if not mismatch.any()
            else f"mismatched rows={int(mismatch.sum())}"
        ),
    )

    months_ok = data["date"].dt.month.isin(
        [11, 12, 1, 2, 3]
    )
    record(
        dataset,
        "months restricted to NDJFM",
        bool(months_ok.all()),
        f"unexpected rows={int((~months_ok).sum())}",
    )

    years = sorted(
        int(value)
        for value in data["winter_year"].dropna().unique()
    )
    expected_years = list(
        range(expected_start_year, expected_end_year + 1)
    )
    record(
        dataset,
        "winter-year coverage is consecutive",
        years == expected_years,
        f"observed={years[0] if years else None}--"
        f"{years[-1] if years else None}, n={len(years)}; "
        f"expected={expected_start_year}--{expected_end_year}, "
        f"n={len(expected_years)}",
    )

    duplicate_count = int(
        data.duplicated(
            subset=group_columns + ["date"]
        ).sum()
    )
    record(
        dataset,
        "no duplicate group-date rows",
        duplicate_count == 0,
        f"duplicates={duplicate_count}",
    )

    bad_start = []
    bad_end = []
    bad_count = []
    nonconsecutive = []

    for key, group in data.groupby(
        group_columns,
        sort=False,
    ):
        group = group.sort_values("date")
        winter_year = int(group["winter_year"].iloc[0])

        expected_start = winter_start_date(winter_year)
        expected_end = winter_end_date(winter_year)
        expected_n = expected_analysis_days(winter_year)

        if group["date"].iloc[0] != expected_start:
            bad_start.append(
                (key, group["date"].iloc[0], expected_start)
            )

        if group["date"].iloc[-1] != expected_end:
            bad_end.append(
                (key, group["date"].iloc[-1], expected_end)
            )

        if len(group) != expected_n:
            bad_count.append(
                (key, len(group), expected_n)
            )

        date_steps = group["date"].diff().dropna().dt.days
        if not (date_steps == 1).all():
            nonconsecutive.append(key)

    record(
        dataset,
        "every group starts on Nov 8",
        len(bad_start) == 0,
        f"bad groups={len(bad_start)}"
        + (f"; first={bad_start[0]}" if bad_start else ""),
    )
    record(
        dataset,
        "every group ends on Mar 31",
        len(bad_end) == 0,
        f"bad groups={len(bad_end)}"
        + (f"; first={bad_end[0]}" if bad_end else ""),
    )
    record(
        dataset,
        "daily row count per winter respects leap years",
        len(bad_count) == 0,
        f"bad groups={len(bad_count)}"
        + (f"; first={bad_count[0]}" if bad_count else ""),
    )
    record(
        dataset,
        "daily dates are consecutive",
        len(nonconsecutive) == 0,
        f"bad groups={len(nonconsecutive)}"
        + (f"; first={nonconsecutive[0]}" if nonconsecutive else ""),
    )


def audit_ecmwf_daily(
    daily: pd.DataFrame,
) -> None:
    required = {
        "winter_year",
        "member",
        "date",
        "DSN",
        "analysis_day",
        "estimated_daily_mean_demand_MW",
        "estimated_daily_mean_demand_GW",
        "daily_mean_t2m_c",
        "daily_mean_wind10m",
        "HDD",
        "HDD_7day_avg",
        "day_name",
        "DOW_effect",
    }
    missing = required - set(daily.columns)
    record(
        "ECMWF daily",
        "required columns",
        len(missing) == 0,
        f"missing={sorted(missing)}",
    )
    if missing:
        return

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["winter_year"] = pd.to_numeric(
        data["winter_year"]
    ).astype(int)
    data["member"] = pd.to_numeric(
        data["member"]
    ).astype(int)

    audit_daily_group_structure(
        data,
        dataset="ECMWF daily",
        group_columns=["winter_year", "member"],
        expected_start_year=EXPECTED_WINTER_START_YEAR,
        expected_end_year=EXPECTED_WINTER_END_YEAR,
    )

    group_counts = (
        data.groupby("winter_year")["member"]
        .nunique()
    )
    record(
        "ECMWF daily",
        "25 members in every winter",
        bool(
            (group_counts == EXPECTED_ECMWF_MEMBERS_PER_WINTER).all()
        ),
        f"minimum={int(group_counts.min())}; "
        f"maximum={int(group_counts.max())}",
    )

    member_sets = (
        data.groupby("winter_year")["member"]
        .apply(lambda values: tuple(sorted(values.unique())))
    )
    record(
        "ECMWF daily",
        "member labels consistent across winters",
        member_sets.nunique() == 1,
        f"member set={member_sets.iloc[0] if len(member_sets) else None}",
    )

    n_groups = data[
        ["winter_year", "member"]
    ].drop_duplicates().shape[0]
    record(
        "ECMWF daily",
        "875 winter-member groups",
        n_groups == EXPECTED_ECMWF_SEVERITY_ROWS,
        f"observed={n_groups}; expected={EXPECTED_ECMWF_SEVERITY_ROWS}",
    )

    expected_total_rows = sum(
        expected_analysis_days(year)
        for year in range(
            EXPECTED_WINTER_START_YEAR,
            EXPECTED_WINTER_END_YEAR + 1,
        )
    ) * EXPECTED_ECMWF_MEMBERS_PER_WINTER

    record(
        "ECMWF daily",
        "total daily row count",
        len(data) == expected_total_rows,
        f"observed={len(data)}; expected={expected_total_rows}",
    )

    expected_dsn = calculate_dsn(
        data["date"],
        data["winter_year"],
    )
    actual_dsn = pd.to_numeric(
        data["DSN"],
        errors="coerce",
    ).reset_index(drop=True)

    record(
        "ECMWF daily",
        "DSN uses Nov 1 = 0",
        allclose(actual_dsn, expected_dsn),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(actual_dsn - expected_dsn))):.6g}",
    )

    actual_analysis_day = pd.to_numeric(
        data["analysis_day"],
        errors="coerce",
    ).reset_index(drop=True)

    record(
        "ECMWF daily",
        "analysis_day uses Nov 8 = 0",
        allclose(
            actual_analysis_day,
            expected_dsn - 7,
        ),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(actual_analysis_day - (expected_dsn - 7)))):.6g}",
    )

    demand_mw = pd.to_numeric(
        data["estimated_daily_mean_demand_MW"],
        errors="coerce",
    )
    demand_gw = pd.to_numeric(
        data["estimated_daily_mean_demand_GW"],
        errors="coerce",
    )
    record(
        "ECMWF daily",
        "MW/GW conversion",
        allclose(
            demand_mw / 1000.0,
            demand_gw,
            atol=ATOL,
        ),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(demand_mw / 1000.0 - demand_gw))):.6g} GW",
    )

    temperature = pd.to_numeric(
        data["daily_mean_t2m_c"],
        errors="coerce",
    )
    hdd = pd.to_numeric(
        data["HDD"],
        errors="coerce",
    )
    expected_hdd = np.maximum(
        HDD_BASE_C - temperature,
        0.0,
    )
    record(
        "ECMWF daily",
        "HDD base temperature is 15.5 C",
        allclose(hdd, expected_hdd),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(hdd - expected_hdd))):.6g}",
    )

    expected_day_name = data["date"].dt.day_name()
    actual_day_name = data["day_name"].astype(str)
    record(
        "ECMWF daily",
        "day_name agrees with date",
        bool((actual_day_name == expected_day_name).all()),
        f"mismatches={int((actual_day_name != expected_day_name).sum())}",
    )

    dow_expected_values = actual_day_name.map(
        DOW_EXPECTED
    ).astype(float)
    actual_dow = pd.to_numeric(
        data["DOW_effect"],
        errors="coerce",
    )
    record(
        "ECMWF daily",
        "day-of-week effects",
        allclose(
            actual_dow,
            dow_expected_values,
        ),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(actual_dow - dow_expected_values))):.6g} MW",
    )

    # Recompute 7-day HDD after the first six rows of the Nov-8 file.
    # Values on Nov 8--13 depend on Nov 2--7, which are not retained in
    # this final analysis-window file, so those rows are deliberately skipped.
    rolling_parts = []
    position_parts = []

    for _, group in data.sort_values(
        ["winter_year", "member", "date"]
    ).groupby(
        ["winter_year", "member"],
        sort=False,
    ):
        rolling_parts.append(
            pd.to_numeric(
                group["HDD"],
                errors="coerce",
            ).rolling(
                window=7,
                min_periods=7,
            ).mean()
        )
        position_parts.append(
            pd.Series(
                np.arange(len(group)),
                index=group.index,
            )
        )

    recomputed_hdd7 = pd.concat(
        rolling_parts
    ).sort_index()
    within_group_position = pd.concat(
        position_parts
    ).sort_index()
    compare_mask = within_group_position >= 6

    actual_hdd7 = pd.to_numeric(
        data["HDD_7day_avg"],
        errors="coerce",
    )
    record(
        "ECMWF daily",
        "7-day HDD rolling mean after Nov 13",
        allclose(
            actual_hdd7.loc[compare_mask],
            recomputed_hdd7.loc[compare_mask],
            atol=ATOL,
        ),
        f"rows compared={int(compare_mask.sum())}; maximum absolute error="
        f"{float(np.nanmax(np.abs(actual_hdd7.loc[compare_mask] - recomputed_hdd7.loc[compare_mask]))):.6g}",
    )

    expected_demand = (
        ALPHA
        + LAMBDA_HDD_CURRENT * hdd
        + LAMBDA_HDD_7DAY * actual_hdd7
        + GAMMA_WIND * pd.to_numeric(
            data["daily_mean_wind10m"],
            errors="coerce",
        )
        + actual_dow
        + BETA_DSN * actual_dsn.to_numpy(dtype=float)
        + BETA_DSN_SQUARED
        * actual_dsn.to_numpy(dtype=float) ** 2
        + TARGET_YEAR_COEFFICIENT * FIXED_TARGET_YEAR_EFFECT
    )

    record(
        "ECMWF daily",
        "deterministic demand formula and fixed target-year effect",
        allclose(
            demand_mw,
            expected_demand,
            atol=DEMAND_ATOL_MW,
        ),
        f"maximum absolute error="
        f"{float(np.nanmax(np.abs(demand_mw - expected_demand))):.6g} MW",
    )


def restrict_hannah_to_analysis_window(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    data = dataframe.copy()
    data["date"] = pd.to_datetime(data["date"])
    month_day = data["date"].dt.strftime("%m-%d")

    return data[
        (month_day >= EXPECTED_ANALYSIS_START)
        | (month_day <= EXPECTED_ANALYSIS_END)
    ].copy()


def audit_hannah_daily(
    daily: pd.DataFrame,
) -> pd.DataFrame:
    required = {
        "winter_year",
        "date",
        "estimated_daily_mean_demand_MW",
    }
    missing = required - set(daily.columns)
    record(
        "Hannah daily",
        "required columns",
        len(missing) == 0,
        f"missing={sorted(missing)}",
    )
    if missing:
        return pd.DataFrame()

    data = daily.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["winter_year"] = pd.to_numeric(
        data["winter_year"],
        errors="coerce",
    )

    data = data[
        data["winter_year"].between(
            EXPECTED_HANNAH_START_YEAR,
            EXPECTED_HANNAH_END_YEAR,
        )
    ].copy()
    data = restrict_hannah_to_analysis_window(data)

    audit_daily_group_structure(
        data,
        dataset="Hannah daily",
        group_columns=["winter_year"],
        expected_start_year=EXPECTED_HANNAH_START_YEAR,
        expected_end_year=EXPECTED_HANNAH_END_YEAR,
    )

    n_winters = int(data["winter_year"].nunique())
    record(
        "Hannah daily",
        "70 complete winters",
        n_winters == EXPECTED_HANNAH_WINTERS,
        f"observed={n_winters}; expected={EXPECTED_HANNAH_WINTERS}",
    )

    benchmark = data[
        data["winter_year"] == EXPECTED_BENCHMARK_WINTER_YEAR
    ]
    wrong_neighbour = data[
        data["winter_year"] == 1962
    ]

    record(
        "Hannah daily",
        "1962/63 row exists as winter_year=1963",
        len(benchmark) > 0,
        f"rows for winter_year=1963: {len(benchmark)}",
    )
    record(
        "Hannah daily",
        "winter_year=1962 is a distinct preceding winter",
        len(wrong_neighbour) > 0,
        f"rows for winter_year=1962: {len(wrong_neighbour)}",
    )

    return data


# =============================================================================
# 3. Severity recalculation
# =============================================================================

def severity_from_daily(
    daily: pd.DataFrame,
    *,
    group_columns: list[str],
    windows: list[int],
) -> pd.DataFrame:
    rows: list[dict] = []

    sorted_daily = daily.sort_values(
        group_columns + ["date"]
    )

    for key, group in sorted_daily.groupby(
        group_columns,
        sort=True,
    ):
        if not isinstance(key, tuple):
            key = (key,)

        row = {
            column: value
            for column, value in zip(group_columns, key)
        }

        demand = pd.to_numeric(
            group["estimated_daily_mean_demand_MW"],
            errors="coerce",
        )
        dates = pd.to_datetime(group["date"])

        for window in windows:
            rolling = demand.rolling(
                window=window,
                min_periods=window,
            ).mean()

            maximum_index = rolling.idxmax()
            maximum_value = float(
                rolling.loc[maximum_index]
            )
            end_date = pd.Timestamp(
                group.loc[maximum_index, "date"]
            )
            start_date = end_date - pd.Timedelta(
                days=window - 1
            )

            row[
                f"max_{window}d_mean_demand_MW"
            ] = maximum_value
            row[
                f"max_{window}d_window_start"
            ] = start_date
            row[
                f"max_{window}d_window_end"
            ] = end_date

        rows.append(row)

    return pd.DataFrame(rows)


def compare_severity(
    recalculated: pd.DataFrame,
    stored: pd.DataFrame,
    *,
    dataset: str,
    key_columns: list[str],
) -> None:
    stored_copy = stored.copy()

    merged = recalculated.merge(
        stored_copy,
        on=key_columns,
        how="outer",
        suffixes=("_recalculated", "_stored"),
        indicator=True,
        validate="one_to_one",
    )

    record(
        dataset,
        "severity keys match daily groups",
        bool((merged["_merge"] == "both").all()),
        merged["_merge"].value_counts().to_dict().__str__(),
    )

    if not (merged["_merge"] == "both").all():
        return

    available_windows = [
        window
        for window in EXPECTED_EXTENDED_WINDOWS
        if (
            f"max_{window}d_mean_demand_MW_recalculated"
            in merged.columns
            and f"max_{window}d_mean_demand_MW_stored"
            in merged.columns
        )
    ]

    record(
        dataset,
        "expected severity windows present",
        available_windows == EXPECTED_EXTENDED_WINDOWS,
        f"available={available_windows}",
    )

    for window in available_windows:
        recalculated_col = (
            f"max_{window}d_mean_demand_MW_recalculated"
        )
        stored_col = (
            f"max_{window}d_mean_demand_MW_stored"
        )

        differences = (
            pd.to_numeric(
                merged[recalculated_col],
                errors="coerce",
            )
            - pd.to_numeric(
                merged[stored_col],
                errors="coerce",
            )
        )

        record(
            dataset,
            f"{window}-day severity maximum",
            bool(
                np.allclose(
                    differences,
                    0.0,
                    atol=DEMAND_ATOL_MW,
                    rtol=0.0,
                )
            ),
            f"maximum absolute error="
            f"{float(np.nanmax(np.abs(differences))):.6g} MW",
        )

        for boundary in ["start", "end"]:
            recalculated_date_col = (
                f"max_{window}d_window_{boundary}_recalculated"
            )
            stored_date_col = (
                f"max_{window}d_window_{boundary}_stored"
            )

            if (
                recalculated_date_col in merged.columns
                and stored_date_col in merged.columns
            ):
                left = pd.to_datetime(
                    merged[recalculated_date_col]
                )
                right = pd.to_datetime(
                    merged[stored_date_col]
                )
                mismatch = left != right

                record(
                    dataset,
                    f"{window}-day maximum window {boundary} date",
                    not bool(mismatch.any()),
                    f"mismatches={int(mismatch.sum())}",
                )


# =============================================================================
# 4. Mean-shift and benchmark audit
# =============================================================================

def audit_mean_shift(
    ecmwf_severity: pd.DataFrame,
    hannah_severity: pd.DataFrame,
) -> None:
    if not MEAN_SHIFT_SUMMARY_FILE.exists():
        record(
            "Mean shift",
            "summary file exists",
            False,
            str(MEAN_SHIFT_SUMMARY_FILE),
        )
        return

    summary = pd.read_csv(
        MEAN_SHIFT_SUMMARY_FILE
    )
    hannah_overlap = hannah_severity[
        hannah_severity["winter_year"].between(
            EXPECTED_WINTER_START_YEAR,
            EXPECTED_WINTER_END_YEAR,
        )
    ].copy()

    benchmark_rows = hannah_severity[
        hannah_severity["winter_year"]
        == EXPECTED_BENCHMARK_WINTER_YEAR
    ]

    record(
        "Mean shift",
        "exactly one 1962/63 benchmark row",
        len(benchmark_rows) == 1,
        f"winter_year=1963 rows={len(benchmark_rows)}",
    )
    if len(benchmark_rows) != 1:
        return

    wrong_rows = hannah_severity[
        hannah_severity["winter_year"] == 1962
    ]
    record(
        "Mean shift",
        "1961/62 and 1962/63 are not conflated",
        len(wrong_rows) == 1,
        f"winter_year=1962 rows={len(wrong_rows)}; "
        "this row must never be labelled 1962/63",
    )

    benchmark_audit_rows = []

    for window in EXPECTED_PRIMARY_WINDOWS:
        column = f"max_{window}d_mean_demand_MW"

        e_values = pd.to_numeric(
            ecmwf_severity[column],
            errors="coerce",
        ) / 1000.0
        h_values = pd.to_numeric(
            hannah_overlap[column],
            errors="coerce",
        ) / 1000.0

        shift = float(
            e_values.mean() - h_values.mean()
        )
        shifted_h = h_values + shift

        benchmark_raw = float(
            pd.to_numeric(
                benchmark_rows.iloc[0][column]
            ) / 1000.0
        )
        benchmark_shifted = (
            benchmark_raw + shift
        )

        row = summary[
            summary["window_days"] == window
        ]

        row_exists = len(row) == 1
        record(
            "Mean shift",
            f"{window}-day summary row",
            row_exists,
            f"rows={len(row)}",
        )
        if not row_exists:
            continue

        row = row.iloc[0]

        checks = {
            "mean_shift_GW": shift,
            "ecmwf_maximum_GW": float(e_values.max()),
            "hannah_raw_maximum_GW": float(h_values.max()),
            "hannah_shifted_maximum_GW": float(shifted_h.max()),
            "hannah_1962_63_raw_GW": benchmark_raw,
            "hannah_1962_63_shifted_GW": benchmark_shifted,
        }

        for field, expected_value in checks.items():
            if field not in summary.columns:
                record(
                    "Mean shift",
                    f"{window}-day {field}",
                    False,
                    "column missing from summary",
                )
                continue

            actual_value = float(row[field])
            record(
                "Mean shift",
                f"{window}-day {field}",
                bool(
                    np.isclose(
                        actual_value,
                        expected_value,
                        atol=ATOL,
                        rtol=0.0,
                    )
                ),
                f"stored={actual_value:.9f}; "
                f"recalculated={expected_value:.9f}",
            )

        benchmark_audit_rows.append(
            {
                "window_days": window,
                "mean_shift_GW": shift,
                "correct_raw_1962_63_GW": benchmark_raw,
                "correct_shifted_1962_63_GW": benchmark_shifted,
                "shifted_historical_overlap_max_GW": float(
                    shifted_h.max()
                ),
                "ecmwf_max_GW": float(
                    e_values.max()
                ),
                "ecmwf_minus_correct_shifted_1962_63_GW": float(
                    e_values.max() - benchmark_shifted
                ),
            }
        )

    pd.DataFrame(
        benchmark_audit_rows
    ).to_csv(
        BENCHMARK_OUTPUT,
        index=False,
    )


# =============================================================================
# 5. Basic GEV output definitions
# =============================================================================

def audit_gev_outputs() -> None:
    if GEV_ORIGINAL_FITS_FILE.exists():
        fits = pd.read_csv(
            GEV_ORIGINAL_FITS_FILE
        )

        record(
            "GEV",
            "14 original fits",
            len(fits) == 14,
            f"observed={len(fits)}; expected=7 windows x 2 samples",
        )

        ecmwf_n = fits.loc[
            fits["sample"] == "ECMWF pooled",
            "n",
        ]
        hannah_n = fits.loc[
            fits["sample"] == "Hannah overlap shifted",
            "n",
        ]

        record(
            "GEV",
            "ECMWF fitted n=875",
            len(ecmwf_n) == 7
            and bool((ecmwf_n == 875).all()),
            f"values={ecmwf_n.tolist()}",
        )
        record(
            "GEV",
            "historical fitted n=35",
            len(hannah_n) == 7
            and bool((hannah_n == 35).all()),
            f"values={hannah_n.tolist()}",
        )
    else:
        record(
            "GEV",
            "original fits file",
            False,
            str(GEV_ORIGINAL_FITS_FILE),
            warning=True,
        )

    if GEV_DIFF_CI_FILE.exists():
        differences = pd.read_csv(
            GEV_DIFF_CI_FILE
        )

        lower_candidates = [
            column for column in differences.columns
            if "lower" in column.lower()
        ]
        upper_candidates = [
            column for column in differences.columns
            if "upper" in column.lower()
        ]

        if lower_candidates and upper_candidates:
            lower = pd.to_numeric(
                differences[lower_candidates[0]],
                errors="coerce",
            )
            upper = pd.to_numeric(
                differences[upper_candidates[0]],
                errors="coerce",
            )
            includes_zero = (
                (lower <= 0.0)
                & (upper >= 0.0)
            )

            record(
                "GEV",
                "all stored difference intervals include zero",
                bool(includes_zero.all()),
                f"rows including zero={int(includes_zero.sum())}/"
                f"{len(includes_zero)}",
            )
        else:
            record(
                "GEV",
                "difference CI columns identified",
                False,
                f"columns={list(differences.columns)}",
                warning=True,
            )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 88)
    print("FINAL DEFINITION-INTEGRITY AUDIT")
    print("=" * 88)

    audit_configuration()
    scan_python_constants()

    required_paths = [
        (ECMWF_DAILY_FILE, "ECMWF final daily file"),
        (ECMWF_SEVERITY_FILE, "ECMWF extended severity file"),
        (HANNAH_SEVERITY_FILE, "Hannah extended severity file"),
        (MEAN_SHIFT_SUMMARY_FILE, "mean-shift summary file"),
    ]

    availability = {
        label: require_file(path, label)
        for path, label in required_paths
    }

    hannah_daily_path = find_hannah_daily_file()

    ecmwf_daily = None
    hannah_daily = None
    ecmwf_severity = None
    hannah_severity = None

    if availability["ECMWF final daily file"]:
        ecmwf_daily = pd.read_csv(
            ECMWF_DAILY_FILE
        )
        audit_ecmwf_daily(
            ecmwf_daily
        )

    if hannah_daily_path is not None:
        hannah_daily_raw = pd.read_csv(
            hannah_daily_path
        )
        hannah_daily = audit_hannah_daily(
            hannah_daily_raw
        )

    if availability["ECMWF extended severity file"]:
        ecmwf_severity = pd.read_csv(
            ECMWF_SEVERITY_FILE
        )
        unique_keys = ecmwf_severity[
            ["winter_year", "member"]
        ].drop_duplicates()
        record(
            "ECMWF severity",
            "875 unique severity rows",
            len(ecmwf_severity) == 875
            and len(unique_keys) == 875,
            f"rows={len(ecmwf_severity)}; "
            f"unique winter-member keys={len(unique_keys)}",
        )

    if availability["Hannah extended severity file"]:
        hannah_severity = pd.read_csv(
            HANNAH_SEVERITY_FILE
        )
        record(
            "Hannah severity",
            "70 unique winter rows",
            len(hannah_severity) == 70
            and hannah_severity["winter_year"].nunique() == 70,
            f"rows={len(hannah_severity)}; "
            f"unique winters={hannah_severity['winter_year'].nunique()}",
        )
        observed_years = sorted(
            pd.to_numeric(
                hannah_severity["winter_year"]
            ).astype(int).unique()
        )
        record(
            "Hannah severity",
            "winter years 1951--2020",
            observed_years == list(
                range(1951, 2021)
            ),
            f"observed={observed_years[0] if observed_years else None}"
            f"--{observed_years[-1] if observed_years else None}",
        )

    if (
        ecmwf_daily is not None
        and ecmwf_severity is not None
    ):
        ecmwf_recalculated = severity_from_daily(
            ecmwf_daily.assign(
                date=pd.to_datetime(
                    ecmwf_daily["date"]
                )
            ),
            group_columns=[
                "winter_year",
                "member",
            ],
            windows=EXPECTED_EXTENDED_WINDOWS,
        )
        compare_severity(
            ecmwf_recalculated,
            ecmwf_severity,
            dataset="ECMWF severity",
            key_columns=[
                "winter_year",
                "member",
            ],
        )

    if (
        hannah_daily is not None
        and len(hannah_daily) > 0
        and hannah_severity is not None
    ):
        hannah_recalculated = severity_from_daily(
            hannah_daily.assign(
                date=pd.to_datetime(
                    hannah_daily["date"]
                )
            ),
            group_columns=[
                "winter_year",
            ],
            windows=EXPECTED_EXTENDED_WINDOWS,
        )
        compare_severity(
            hannah_recalculated,
            hannah_severity,
            dataset="Hannah severity",
            key_columns=[
                "winter_year",
            ],
        )

    if (
        ecmwf_severity is not None
        and hannah_severity is not None
    ):
        audit_mean_shift(
            ecmwf_severity,
            hannah_severity,
        )

    audit_gev_outputs()

    audit_frame = pd.DataFrame(
        [row.__dict__ for row in audit_rows]
    )
    audit_frame.to_csv(
        AUDIT_OUTPUT,
        index=False,
    )

    print("\n" + "=" * 88)
    print("AUDIT SUMMARY")
    print("=" * 88)
    print(
        audit_frame["status"]
        .value_counts()
        .to_string()
    )
    print(f"\nSaved full audit:\n{AUDIT_OUTPUT}")
    print(f"Saved benchmark audit:\n{BENCHMARK_OUTPUT}")
    print(f"Saved script scan:\n{SCRIPT_SCAN_OUTPUT}")

    failures = audit_frame[
        audit_frame["status"] == "FAIL"
    ]

    if len(failures) > 0:
        print("\nCORE FAILURES:")
        print(
            failures[
                [
                    "section",
                    "check",
                    "detail",
                ]
            ].to_string(index=False)
        )
        raise SystemExit(
            "\nDefinition-integrity audit FAILED. "
            "Do not freeze final Results until every core failure is resolved."
        )

    print(
        "\nDefinition-integrity audit PASSED. "
        "Warnings, if any, should still be reviewed."
    )


if __name__ == "__main__":
    main()
