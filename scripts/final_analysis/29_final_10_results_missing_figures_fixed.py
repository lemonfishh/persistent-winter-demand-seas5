from __future__ import annotations

"""
Plot the two remaining Results-chapter figures.

This script creates:

1. demand_gap_size_rank_and_year_concentration.pdf
   - plot-only redraw from frozen final CSV outputs;
   - no gap analysis is recalculated.

2. wy2013_m22_final_demand_hdd_wind_timeseries.pdf
   - reads the final daily SEAS5 demand/weather file;
   - recalculates only the maximum-demand windows for WY2013 member 22;
   - writes an audit CSV containing the selected dates and mean demand.

Run from the project root:

    /usr/bin/python3 scripts/29_final_10_results_missing_figures.py

The script stops rather than silently selecting among duplicate final
CSV files.
"""

from pathlib import Path
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import PercentFormatter

from config import OUTPUT_DIR


# ============================================================================
# Settings
# ============================================================================

WINDOWS = [21, 28, 35, 42, 49, 56]
CASE_WINDOWS = [21, 28, 35, 42]

CASE_WINTER_YEAR = 2013
CASE_MEMBER = 22

HDD_BASE_TEMPERATURE_C = 15.5
FIGURE_DPI = 300

FINAL_ANALYSIS_ROOT = OUTPUT_DIR / "final_analysis_clean"

SOURCE_DIR = (
    FINAL_ANALYSIS_ROOT
    / "05_temperature_tail_and_bump_concentration"
)

OUT_DIR = (
    FINAL_ANALYSIS_ROOT
    / "10_results_missing_figures"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DAILY_FILE = (
    OUTPUT_DIR
    / "daily"
    / "ecmwf_daily_demand_Nov08_1982_2016.csv"
)

GAP_FIGURE_PDF = (
    OUT_DIR
    / "demand_gap_size_rank_and_year_concentration.pdf"
)
GAP_FIGURE_PNG = GAP_FIGURE_PDF.with_suffix(".png")

EVENT_FIGURE_PDF = (
    OUT_DIR
    / "wy2013_m22_final_demand_hdd_wind_timeseries.pdf"
)
EVENT_FIGURE_PNG = EVENT_FIGURE_PDF.with_suffix(".png")

EVENT_AUDIT_CSV = (
    OUT_DIR
    / "wy2013_m22_final_window_summary.csv"
)

GAP_PLOT_AUDIT_CSV = (
    OUT_DIR
    / "demand_gap_size_rank_and_year_concentration_plot_data.csv"
)


# Previous event-window results are used only as a cross-check.
# The final daily file remains the source of truth.
PREVIOUS_EVENT_CHECK = {
    21: ("2012-12-01", "2012-12-21", 42.362995),
    28: ("2012-12-02", "2012-12-29", 42.204350),
    35: ("2012-11-29", "2013-01-02", 41.889081),
    42: ("2012-11-24", "2013-01-04", 41.471621),
}


# ============================================================================
# File and column helpers
# ============================================================================

def require_file(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )
    return path


def find_unique_final_output(filename: str) -> Path:
    """
    Find one exact filename below outputs/final_analysis_clean.

    Stop if no copy or more than one copy is found. This prevents the
    script from silently using an uncertain version.
    """
    matches = sorted(
        FINAL_ANALYSIS_ROOT.rglob(filename)
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} below:\n"
            f"{FINAL_ANALYSIS_ROOT}"
        )

    if len(matches) > 1:
        listed = "\n".join(
            f"  - {path}" for path in matches
        )
        raise RuntimeError(
            f"Several copies of {filename} were found:\n"
            f"{listed}\n\n"
            "Remove or rename duplicate final versions before running."
        )

    return matches[0]


def normalise_column_name(name: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(name).lower(),
    )


def choose_column(
    dataframe: pd.DataFrame,
    *,
    exact_candidates: list[str],
    keyword_groups: list[list[str]],
    exclude_keywords: list[str] | None = None,
) -> str | None:
    """
    Select a column conservatively.

    Exact candidate names are tried first. Keyword matching is used only
    as a fallback. Exclusions prevent, for example, selecting a wind
    contribution term when raw wind speed is required.
    """
    exclude_keywords = exclude_keywords or []

    normalised_map = {
        normalise_column_name(column): column
        for column in dataframe.columns
    }

    for candidate in exact_candidates:
        key = normalise_column_name(candidate)
        if key in normalised_map:
            return normalised_map[key]

    for keyword_group in keyword_groups:
        normalised_keywords = [
            normalise_column_name(keyword)
            for keyword in keyword_group
        ]

        for column in dataframe.columns:
            key = normalise_column_name(column)

            if any(
                normalise_column_name(excluded) in key
                for excluded in exclude_keywords
            ):
                continue

            if all(keyword in key for keyword in normalised_keywords):
                return column

    return None


def to_celsius(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")

    # A robust guard for Kelvin inputs.
    if values.dropna().median() > 100:
        values = values - 273.15

    return values


def demand_to_gw(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    key = normalise_column_name(column_name)

    if "gw" in key and "mw" not in key:
        return values

    # Final file is expected to be MW. The median guard prevents an
    # accidental second conversion if the name is unusual.
    if values.dropna().median() > 100:
        return values / 1000.0

    return values


def detect_daily_columns(
    daily: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    demand_col = choose_column(
        daily,
        exact_candidates=[
            "estimated_daily_mean_demand_MW",
            "estimated_daily_mean_demand_GW",
            "daily_mean_demand_MW",
            "daily_mean_demand_GW",
        ],
        keyword_groups=[
            ["estimated", "daily", "demand"],
            ["daily", "demand"],
        ],
        exclude_keywords=[
            "rolling",
            "maximum",
            "rank",
            "probability",
        ],
    )

    temperature_col = choose_column(
        daily,
        exact_candidates=[
            "daily_mean_t2m_c",
            "daily_mean_t2m_C",
            "t2m_c",
            "temperature_c",
            "daily_mean_temperature_c",
        ],
        keyword_groups=[
            ["daily", "t2m"],
            ["daily", "temperature"],
            ["t2m"],
        ],
    )

    hdd_col = choose_column(
        daily,
        exact_candidates=[
            "HDD",
            "hdd",
            "daily_HDD",
            "daily_hdd",
            "population_weighted_HDD",
            "pop_weighted_HDD",
        ],
        keyword_groups=[
            ["hdd"],
        ],
        exclude_keywords=[
            "7day",
            "7d",
            "lag",
            "rolling",
            "mean7",
        ],
    )

    wind_col = choose_column(
        daily,
        exact_candidates=[
            "daily_mean_wind10m",
            "daily_mean_wind_10m",
            "population_weighted_wind_speed",
            "pop_weighted_wind_speed",
            "daily_mean_wind_speed",
            "daily_mean_wind_speed_ms",
            "wind_speed_10m",
            "wind_speed",
            "wind10m",
            "ws10",
            "si10",
        ],
        keyword_groups=[
            ["daily", "mean", "wind10m"],
            ["wind10m"],
            ["population", "wind", "speed"],
            ["daily", "wind", "speed"],
            ["wind", "speed"],
            ["ws10"],
            ["si10"],
        ],
        exclude_keywords=[
            "term",
            "contribution",
            "effect",
            "coefficient",
            "demand",
        ],
    )

    working = daily.copy()

    if wind_col is None:
        u_col = choose_column(
            working,
            exact_candidates=[
                "u10",
                "10m_u_component_of_wind",
                "u_component_of_wind_10m",
            ],
            keyword_groups=[
                ["u10"],
                ["u", "wind"],
            ],
        )
        v_col = choose_column(
            working,
            exact_candidates=[
                "v10",
                "10m_v_component_of_wind",
                "v_component_of_wind_10m",
            ],
            keyword_groups=[
                ["v10"],
                ["v", "wind"],
            ],
        )

        if u_col is not None and v_col is not None:
            working["derived_wind_speed_10m"] = np.sqrt(
                pd.to_numeric(
                    working[u_col],
                    errors="coerce",
                ) ** 2
                + pd.to_numeric(
                    working[v_col],
                    errors="coerce",
                ) ** 2
            )
            wind_col = "derived_wind_speed_10m"

    if demand_col is None:
        raise ValueError(
            "Could not identify the final daily demand column.\n"
            f"Available columns:\n{list(daily.columns)}"
        )

    if temperature_col is None and hdd_col is None:
        raise ValueError(
            "Could not identify either temperature or HDD.\n"
            f"Available columns:\n{list(daily.columns)}"
        )

    if wind_col is None:
        raise ValueError(
            "Could not identify raw wind speed and could not derive it "
            "from u/v components.\n"
            f"Available columns:\n{list(daily.columns)}"
        )

    if hdd_col is None:
        if temperature_col is None:
            raise ValueError(
                "HDD is missing and cannot be derived without temperature."
            )

        temperature_c = to_celsius(
            working[temperature_col]
        )
        working["derived_daily_HDD_15p5C"] = np.maximum(
            HDD_BASE_TEMPERATURE_C - temperature_c,
            0.0,
        )
        hdd_col = "derived_daily_HDD_15p5C"

    return working, {
        "demand": demand_col,
        "temperature": temperature_col or "",
        "hdd": hdd_col,
        "wind": wind_col,
    }


# ============================================================================
# Figure 1: three-panel gap and winter-year concentration summary
# ============================================================================

def plot_gap_size_rank_and_concentration() -> None:
    gap_file = find_unique_final_output(
        "demand_upper_tail_gap_summary.csv"
    )
    concentration_file = find_unique_final_output(
        "demand_gap_defined_upper_group_concentration_summary.csv"
    )
    loo_file = find_unique_final_output(
        "demand_gap_leave_one_winter_year_out.csv"
    )

    gap = pd.read_csv(gap_file)
    concentration = pd.read_csv(concentration_file)
    loo = pd.read_csv(loo_file)

    required_gap = {
        "window_days",
        "gap_rank",
        "gap_size_standardised",
    }
    required_concentration = {
        "window_days",
        "maximum_single_year_share",
        "top_three_year_share",
    }
    required_loo = {
        "window_days",
        "gap_rank",
        "gap_size_standardised",
    }

    for label, dataframe, required in [
        ("gap summary", gap, required_gap),
        ("concentration summary", concentration, required_concentration),
        ("leave-one-winter-year-out", loo, required_loo),
    ]:
        missing = required - set(dataframe.columns)
        if missing:
            raise ValueError(
                f"Missing columns in {label}: {sorted(missing)}\n"
                f"Available columns: {list(dataframe.columns)}"
            )

    gap = (
        gap[gap["window_days"].isin(WINDOWS)]
        .sort_values("window_days")
        .copy()
    )
    concentration = (
        concentration[
            concentration["window_days"].isin(WINDOWS)
        ]
        .sort_values("window_days")
        .copy()
    )
    loo = (
        loo[loo["window_days"].isin(WINDOWS)]
        .copy()
    )

    if gap["window_days"].tolist() != WINDOWS:
        raise ValueError(
            "Gap summary does not contain exactly the expected windows: "
            f"{WINDOWS}"
        )

    size_summary = (
        loo.groupby("window_days")["gap_size_standardised"]
        .agg(
            loo_size_min="min",
            loo_size_median="median",
            loo_size_max="max",
        )
        .reset_index()
    )

    rank_summary = (
        loo.groupby("window_days")["gap_rank"]
        .agg(
            loo_rank_min="min",
            loo_rank_median="median",
            loo_rank_max="max",
        )
        .reset_index()
    )

    plot_data = (
        gap[
            [
                "window_days",
                "gap_size_standardised",
                "gap_rank",
            ]
        ]
        .merge(
            size_summary,
            on="window_days",
            how="left",
            validate="one_to_one",
        )
        .merge(
            rank_summary,
            on="window_days",
            how="left",
            validate="one_to_one",
        )
        .merge(
            concentration[
                [
                    "window_days",
                    "maximum_single_year_share",
                    "top_three_year_share",
                ]
            ],
            on="window_days",
            how="left",
            validate="one_to_one",
        )
    )

    plot_data.to_csv(
        GAP_PLOT_AUDIT_CSV,
        index=False,
    )

    x = plot_data["window_days"].to_numpy(dtype=float)

    figure = plt.figure(
        figsize=(12.4, 8.4),
        dpi=FIGURE_DPI,
    )
    grid = GridSpec(
        nrows=2,
        ncols=2,
        figure=figure,
        height_ratios=[1.0, 0.95],
        hspace=0.38,
        wspace=0.28,
    )

    ax_size = figure.add_subplot(grid[0, 0])
    ax_rank = figure.add_subplot(grid[0, 1])
    ax_concentration = figure.add_subplot(grid[1, :])

    # (a) Gap size
    ax_size.plot(
        x,
        plot_data["gap_size_standardised"],
        marker="o",
        linewidth=1.7,
        label="Original sample",
    )
    ax_size.fill_between(
        x,
        plot_data["loo_size_min"].to_numpy(dtype=float),
        plot_data["loo_size_max"].to_numpy(dtype=float),
        alpha=0.18,
        label="Leave-one-winter-year-out range",
    )
    ax_size.set_xlabel("Averaging window (days)")
    ax_size.set_ylabel("Largest standardised gap")
    ax_size.set_xticks(WINDOWS)
    ax_size.grid(True, alpha=0.3)
    ax_size.legend(fontsize=8.5)
    ax_size.text(
        0.01,
        0.98,
        "(a) Gap-size sensitivity",
        transform=ax_size.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # (b) Gap rank
    ax_rank.plot(
        x,
        plot_data["gap_rank"],
        marker="o",
        linewidth=1.7,
        label="Original sample",
    )
    ax_rank.fill_between(
        x,
        plot_data["loo_rank_min"].to_numpy(dtype=float),
        plot_data["loo_rank_max"].to_numpy(dtype=float),
        alpha=0.18,
        label="Leave-one-winter-year-out range",
    )
    ax_rank.set_xlabel("Averaging window (days)")
    ax_rank.set_ylabel("Descending rank of selected gap")
    ax_rank.set_xticks(WINDOWS)
    ax_rank.grid(True, alpha=0.3)
    ax_rank.legend(fontsize=8.5)
    ax_rank.text(
        0.01,
        0.98,
        "(b) Gap-location sensitivity",
        transform=ax_rank.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    # (c) Winter-year concentration
    ax_concentration.plot(
        x,
        plot_data["maximum_single_year_share"],
        marker="o",
        linewidth=1.7,
        label="Largest single-year share",
    )
    ax_concentration.plot(
        x,
        plot_data["top_three_year_share"],
        marker="s",
        linewidth=1.7,
        label="Top-three-year share",
    )
    ax_concentration.set_xlabel("Averaging window (days)")
    ax_concentration.set_ylabel("Share of gap-defined upper group")
    ax_concentration.set_xticks(WINDOWS)
    ax_concentration.set_ylim(0.0, 1.0)
    ax_concentration.yaxis.set_major_formatter(
        PercentFormatter(xmax=1.0)
    )
    ax_concentration.grid(True, alpha=0.3)
    ax_concentration.legend(
        fontsize=8.5,
        ncol=2,
        loc="upper right",
    )
    ax_concentration.text(
        0.01,
        0.98,
        "(c) Winter-year concentration",
        transform=ax_concentration.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )

    figure.savefig(
        GAP_FIGURE_PDF,
        bbox_inches="tight",
    )
    figure.savefig(
        GAP_FIGURE_PNG,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nGap figure plot data:")
    print(plot_data.to_string(index=False))
    print(f"\nSaved:\n{GAP_FIGURE_PDF}\n{GAP_FIGURE_PNG}")
    print(f"Audit data:\n{GAP_PLOT_AUDIT_CSV}")


# ============================================================================
# Figure 2: final WY2013 member 22 demand/HDD/wind figure
# ============================================================================

def calculate_case_windows(
    case: pd.DataFrame,
    demand_gw: pd.Series,
) -> pd.DataFrame:
    rows: list[dict] = []

    for window in CASE_WINDOWS:
        rolling = demand_gw.rolling(
            window=window,
            min_periods=window,
        ).mean()

        if rolling.notna().sum() == 0:
            raise ValueError(
                f"No valid {window}-day rolling demand window."
            )

        end_position = int(
            np.nanargmax(rolling.to_numpy(dtype=float))
        )
        start_position = end_position - window + 1

        if start_position < 0:
            raise RuntimeError(
                f"Invalid start position for {window}-day window."
            )

        start_date = case.iloc[start_position]["date"]
        end_date = case.iloc[end_position]["date"]
        mean_demand = float(
            rolling.iloc[end_position]
        )

        previous_start, previous_end, previous_mean = (
            PREVIOUS_EVENT_CHECK[window]
        )

        date_matches_previous = (
            start_date == pd.Timestamp(previous_start)
            and end_date == pd.Timestamp(previous_end)
        )
        mean_matches_previous = bool(
            np.isclose(
                mean_demand,
                previous_mean,
                atol=5e-6,
                rtol=0.0,
            )
        )

        rows.append(
            {
                "winter_year": CASE_WINTER_YEAR,
                "member": CASE_MEMBER,
                "window_days": window,
                "window_start": start_date,
                "window_end": end_date,
                "mean_demand_GW": mean_demand,
                "previous_check_start": previous_start,
                "previous_check_end": previous_end,
                "previous_check_mean_GW": previous_mean,
                "dates_match_previous_check": date_matches_previous,
                "mean_matches_previous_check": mean_matches_previous,
            }
        )

    return pd.DataFrame(rows)


def plot_wy2013_m22_weather() -> None:
    require_file(
        DAILY_FILE,
        "final SEAS5 daily demand/weather file",
    )

    daily_raw = pd.read_csv(DAILY_FILE)

    required_ids = {
        "winter_year",
        "member",
        "date",
    }
    missing_ids = required_ids - set(daily_raw.columns)
    if missing_ids:
        raise ValueError(
            f"Missing identifier columns: {sorted(missing_ids)}\n"
            f"Available columns: {list(daily_raw.columns)}"
        )

    daily_raw["date"] = pd.to_datetime(
        daily_raw["date"],
        errors="coerce",
    )
    daily_raw["winter_year"] = pd.to_numeric(
        daily_raw["winter_year"],
        errors="coerce",
    )
    daily_raw["member"] = pd.to_numeric(
        daily_raw["member"],
        errors="coerce",
    )

    daily, columns = detect_daily_columns(
        daily_raw
    )

    case = daily[
        (daily["winter_year"] == CASE_WINTER_YEAR)
        & (daily["member"] == CASE_MEMBER)
    ].copy()

    case = (
        case.dropna(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(case) == 0:
        raise ValueError(
            f"No rows found for winter_year={CASE_WINTER_YEAR}, "
            f"member={CASE_MEMBER}."
        )

    date_differences = (
        case["date"]
        .diff()
        .dropna()
        .dt.days
    )
    if (date_differences != 1).any():
        bad = case.loc[
            date_differences[date_differences != 1].index,
            "date",
        ]
        raise ValueError(
            "Non-consecutive daily dates found for the case event. "
            f"First affected dates:\n{bad.head().to_string(index=False)}"
        )

    demand_gw = demand_to_gw(
        case[columns["demand"]],
        columns["demand"],
    )
    hdd = pd.to_numeric(
        case[columns["hdd"]],
        errors="coerce",
    )
    wind = pd.to_numeric(
        case[columns["wind"]],
        errors="coerce",
    )

    if demand_gw.isna().any():
        raise ValueError("Missing demand values in the selected case.")
    if hdd.isna().any():
        raise ValueError("Missing HDD values in the selected case.")
    if wind.isna().any():
        raise ValueError("Missing wind-speed values in the selected case.")

    window_summary = calculate_case_windows(
        case=case,
        demand_gw=demand_gw,
    )
    window_summary.to_csv(
        EVENT_AUDIT_CSV,
        index=False,
    )

    if not (
        window_summary["dates_match_previous_check"].all()
        and window_summary["mean_matches_previous_check"].all()
    ):
        warnings.warn(
            "The final daily file does not exactly reproduce every "
            "previous WY2013 M22 event-window result. The newly calculated "
            "values have been retained as the source of truth. Review the "
            "audit CSV before using the old interpretation.",
            stacklevel=2,
        )

    min_start = pd.to_datetime(
        window_summary["window_start"]
    ).min()
    max_end = pd.to_datetime(
        window_summary["window_end"]
    ).max()

    plot_start = min_start - pd.Timedelta(days=14)
    plot_end = max_end + pd.Timedelta(days=14)

    plot_mask = (
        (case["date"] >= plot_start)
        & (case["date"] <= plot_end)
    )
    plot_case = case.loc[plot_mask].copy()

    plot_demand = demand_gw.loc[plot_mask]
    plot_hdd = hdd.loc[plot_mask]
    plot_wind = wind.loc[plot_mask]

    default_colours = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()["color"]

    figure, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(11.8, 9.3),
        dpi=FIGURE_DPI,
        sharex=True,
        gridspec_kw={
            "height_ratios": [1.15, 1.0, 1.0, 0.62],
            "hspace": 0.12,
        },
    )

    axes[0].plot(
        plot_case["date"],
        plot_demand,
        linewidth=1.6,
    )
    axes[0].set_ylabel("Daily demand\n(GW)")
    axes[0].set_title(
        "WY2013 member 22: demand and weather during the "
        "selected medium-duration event"
    )

    axes[1].plot(
        plot_case["date"],
        plot_hdd,
        linewidth=1.6,
    )
    axes[1].set_ylabel("Daily HDD\n(°C)")

    axes[2].plot(
        plot_case["date"],
        plot_wind,
        linewidth=1.6,
    )
    axes[2].set_ylabel("Wind speed\n(m s$^{-1}$)")

    # A separate event-window panel avoids obscuring the three time series
    # with several overlapping shaded regions.
    y_positions = np.arange(
        len(CASE_WINDOWS),
        dtype=float,
    )

    for index, row in window_summary.iterrows():
        colour = default_colours[index % len(default_colours)]
        start = pd.Timestamp(row["window_start"])
        end = pd.Timestamp(row["window_end"])
        duration = (
            end - start
        ).days + 1

        axes[3].barh(
            y=y_positions[index],
            width=duration,
            left=mdates.date2num(start),
            height=0.56,
            color=colour,
            alpha=0.75,
        )

        axes[3].text(
            mdates.date2num(start + (end - start) / 2),
            y_positions[index],
            f"{int(row['window_days'])}d",
            ha="center",
            va="center",
            fontsize=8.5,
        )

        # Light boundary markers across the three physical panels.
        for axis in axes[:3]:
            axis.axvline(
                start,
                linewidth=0.8,
                alpha=0.24,
            )
            axis.axvline(
                end,
                linewidth=0.8,
                alpha=0.24,
            )

    axes[3].set_yticks(y_positions)
    axes[3].set_yticklabels(
        [f"{window}-day" for window in CASE_WINDOWS]
    )
    axes[3].invert_yaxis()
    axes[3].set_ylabel("Maximum-\ndemand window")
    axes[3].set_xlabel("Date")

    for axis in axes:
        axis.grid(True, alpha=0.28)

    axes[3].xaxis_date()
    axes[3].xaxis.set_major_locator(
        mdates.WeekdayLocator(
            interval=1,
        )
    )
    axes[3].xaxis.set_major_formatter(
        mdates.DateFormatter("%d %b")
    )

    figure.autofmt_xdate(
        rotation=30,
        ha="right",
    )

    figure.savefig(
        EVENT_FIGURE_PDF,
        bbox_inches="tight",
    )
    figure.savefig(
        EVENT_FIGURE_PNG,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nDetected daily columns:")
    for name, column in columns.items():
        print(f"  {name}: {column}")

    print("\nFinal WY2013 member 22 window audit:")
    print(window_summary.to_string(index=False))

    print(f"\nSaved:\n{EVENT_FIGURE_PDF}\n{EVENT_FIGURE_PNG}")
    print(f"Audit data:\n{EVENT_AUDIT_CSV}")


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print("=" * 80)
    print("Results missing-figure production")
    print("=" * 80)

    plot_gap_size_rank_and_concentration()
    plot_wy2013_m22_weather()

    print("\nAll requested figures were created successfully.")


if __name__ == "__main__":
    main()
