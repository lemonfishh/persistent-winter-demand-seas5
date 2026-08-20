from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ============================================================
# Final analysis 05
# Raw-temperature tail comparison and demand-bump concentration
#
# Questions:
#   1. Does raw temperature show a similar 21-35-day upper-tail
#      shoulder when analysed using the same rolling-window
#      maxima/minima framework as demand?
#
#   2. Is the demand-tail feature broadly distributed across
#      winters, or concentrated in a small number of winter years
#      and ensemble members?
#
# Analysis:
#   - for each winter-member and duration, calculate:
#       * maximum rolling mean demand;
#       * minimum rolling mean raw temperature;
#       * mean temperature over the maximum-demand window;
#   - define cold severity as minus the minimum rolling mean
#     temperature, so larger values mean colder events;
#   - compare separately standardised top-20% empirical tails;
#   - locate the largest adjacent standardised demand gap within
#     ranks 10-140;
#   - summarise winter-year concentration above that gap;
#   - repeat the gap calculation after leaving out each winter
#     year in turn.
#
# The "gap-defined upper group" is a diagnostic device. It is
# not treated as a formal regime or statistically significant
# breakpoint.
# ============================================================


# ============================================================
# Settings
# ============================================================

OVERLAP_START = 1982
OVERLAP_END = 2016

WINDOWS = [
    21,
    28,
    35,
    42,
    49,
    56,
]

UPPER_TAIL_FRACTION = 0.20

GAP_RANK_MIN = 10
GAP_RANK_MAX = 140

TEMPERATURE_COLUMN = "daily_mean_t2m_c"
DEMAND_COLUMN = "estimated_daily_mean_demand_MW"

# Figure and font settings
FIGURE_DPI = 300

FONT_BASE = 13.0
FONT_AXIS_LABEL = 14.0
FONT_TICK = 12.0
FONT_TITLE = 14.0
FONT_LEGEND = 11.5
FONT_SUPTITLE = 16.0

plt.rcParams.update(
    {
        "font.size": FONT_BASE,
        "axes.labelsize": FONT_AXIS_LABEL,
        "axes.titlesize": FONT_TITLE,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }
)


# ============================================================
# Input and output paths
# ============================================================

ECMWF_DAILY_FILE = (
    OUTPUT_DIR
    / "daily"
    / "ecmwf_daily_demand_Nov08_1982_2016.csv"
)

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "05_temperature_tail_and_bump_concentration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ROLLING_SUMMARY = (
    OUT_DIR
    / "temperature_demand_rolling_extremes_by_winter_member.csv"
)

OUT_STANDARDISATION_SUMMARY = (
    OUT_DIR
    / "temperature_demand_standardisation_summary.csv"
)

OUT_TAIL_DATA = (
    OUT_DIR
    / "temperature_demand_standardised_upper_tail_data.csv"
)

OUT_GAP_SUMMARY = (
    OUT_DIR
    / "demand_upper_tail_gap_summary.csv"
)

OUT_GAP_EVENTS = (
    OUT_DIR
    / "demand_gap_defined_upper_group_events.csv"
)

OUT_YEAR_CONCENTRATION = (
    OUT_DIR
    / "demand_gap_defined_upper_group_year_concentration.csv"
)

OUT_CONCENTRATION_SUMMARY = (
    OUT_DIR
    / "demand_gap_defined_upper_group_concentration_summary.csv"
)

OUT_LEAVE_ONE_YEAR_OUT = (
    OUT_DIR
    / "demand_gap_leave_one_winter_year_out.csv"
)

OUT_TAIL_FIGURE_PNG = (
    OUT_DIR
    / "temperature_vs_demand_standardised_upper_tails.png"
)

OUT_TAIL_FIGURE_PDF = (
    OUT_DIR
    / "temperature_vs_demand_standardised_upper_tails.pdf"
)

OUT_CONCENTRATION_FIGURE_PNG = (
    OUT_DIR
    / "demand_bump_gap_and_year_concentration.png"
)

OUT_CONCENTRATION_FIGURE_PDF = (
    OUT_DIR
    / "demand_bump_gap_and_year_concentration.pdf"
)


# ============================================================
# Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )


def robust_standardise(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    values = np.asarray(
        values,
        dtype=float,
    )

    median_value = np.median(
        values
    )

    q25 = np.quantile(
        values,
        0.25,
    )

    q75 = np.quantile(
        values,
        0.75,
    )

    iqr_value = q75 - q25

    if (
        not np.isfinite(iqr_value)
        or iqr_value <= 0
    ):
        raise ValueError(
            "Cannot standardise a sample with non-positive IQR."
        )

    z = (
        values
        - median_value
    ) / iqr_value

    return (
        z,
        median_value,
        iqr_value,
    )


def empirical_upper_tail(
    values: np.ndarray,
    upper_tail_fraction: float,
) -> pd.DataFrame:
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    values_descending = np.sort(
        values
    )[::-1]

    n = len(
        values_descending
    )

    rank = np.arange(
        1,
        n + 1,
    )

    exceedance_probability = (
        rank
        / (n + 1.0)
    )

    mask = (
        exceedance_probability
        <= upper_tail_fraction
    )

    return pd.DataFrame(
        {
            "standardised_severity": (
                values_descending[
                    mask
                ]
            ),
            "descending_rank": (
                rank[
                    mask
                ]
            ),
            "exceedance_probability": (
                exceedance_probability[
                    mask
                ]
            ),
        }
    )


def rolling_extreme_for_group(
    group: pd.DataFrame,
    window: int,
) -> dict:
    g = (
        group
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )

    winter_year = int(
        g["winter_year"].iloc[0]
    )

    member = int(
        g["member"].iloc[0]
    )

    demand = pd.to_numeric(
        g[DEMAND_COLUMN],
        errors="coerce",
    )

    temperature = pd.to_numeric(
        g[TEMPERATURE_COLUMN],
        errors="coerce",
    )

    demand_roll = demand.rolling(
        window=window,
        min_periods=window,
    ).mean()

    temperature_roll = temperature.rolling(
        window=window,
        min_periods=window,
    ).mean()

    if demand_roll.notna().sum() == 0:
        raise ValueError(
            f"No valid {window}-day demand window for "
            f"winter_year={winter_year}, member={member}."
        )

    if temperature_roll.notna().sum() == 0:
        raise ValueError(
            f"No valid {window}-day temperature window for "
            f"winter_year={winter_year}, member={member}."
        )

    demand_end_index = int(
        demand_roll.idxmax()
    )

    temperature_end_index = int(
        temperature_roll.idxmin()
    )

    demand_start_index = (
        demand_end_index
        - window
        + 1
    )

    temperature_start_index = (
        temperature_end_index
        - window
        + 1
    )

    aligned_temperature = temperature.iloc[
        demand_start_index:
        demand_end_index + 1
    ].mean()

    aligned_min_temperature = temperature.iloc[
        demand_start_index:
        demand_end_index + 1
    ].min()

    return {
        "winter_year": winter_year,
        "member": member,
        "window_days": window,
        "maximum_mean_demand_MW": float(
            demand_roll.loc[
                demand_end_index
            ]
        ),
        "maximum_mean_demand_GW": float(
            demand_roll.loc[
                demand_end_index
            ]
            / 1000.0
        ),
        "demand_window_start": (
            g.loc[
                demand_start_index,
                "date",
            ]
        ),
        "demand_window_end": (
            g.loc[
                demand_end_index,
                "date",
            ]
        ),
        "minimum_mean_temperature_C": float(
            temperature_roll.loc[
                temperature_end_index
            ]
        ),
        "cold_severity_C": float(
            -temperature_roll.loc[
                temperature_end_index
            ]
        ),
        "cold_window_start": (
            g.loc[
                temperature_start_index,
                "date",
            ]
        ),
        "cold_window_end": (
            g.loc[
                temperature_end_index,
                "date",
            ]
        ),
        "mean_temperature_in_max_demand_window_C": float(
            aligned_temperature
        ),
        "minimum_temperature_in_max_demand_window_C": float(
            aligned_min_temperature
        ),
        "aligned_demand_cold_severity_C": float(
            -aligned_temperature
        ),
    }


def locate_largest_gap(
    dataframe: pd.DataFrame,
    severity_column: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Sort descending and locate the largest adjacent gap in the
    selected rank range.

    The returned upper group contains ranks 1 through gap_rank.
    """
    ranked = (
        dataframe
        .sort_values(
            severity_column,
            ascending=False,
        )
        .reset_index(drop=True)
        .copy()
    )

    ranked["descending_rank"] = np.arange(
        1,
        len(ranked) + 1,
    )

    z, median_value, iqr_value = robust_standardise(
        ranked[
            severity_column
        ].to_numpy(dtype=float)
    )

    ranked["standardised_severity"] = z

    ranked["gap_to_next"] = (
        ranked[
            "standardised_severity"
        ]
        - ranked[
            "standardised_severity"
        ].shift(-1)
    )

    candidates = ranked[
        ranked[
            "descending_rank"
        ].between(
            GAP_RANK_MIN,
            GAP_RANK_MAX,
        )
        & ranked[
            "gap_to_next"
        ].notna()
    ].copy()

    if len(candidates) == 0:
        raise ValueError(
            "No candidate ranks available for gap detection."
        )

    gap_row = candidates.loc[
        candidates[
            "gap_to_next"
        ].idxmax()
    ]

    gap_rank = int(
        gap_row[
            "descending_rank"
        ]
    )

    gap_size = float(
        gap_row[
            "gap_to_next"
        ]
    )

    upper_group = ranked[
        ranked[
            "descending_rank"
        ] <= gap_rank
    ].copy()

    summary = {
        "gap_rank": gap_rank,
        "gap_size_standardised": gap_size,
        "severity_above_gap": float(
            ranked.loc[
                gap_rank - 1,
                severity_column,
            ]
        ),
        "severity_below_gap": float(
            ranked.loc[
                gap_rank,
                severity_column,
            ]
        ),
        "median": median_value,
        "IQR": iqr_value,
        "n_upper_group": len(
            upper_group
        ),
    }

    return (
        upper_group,
        summary,
    )


# ============================================================
# Load daily data
# ============================================================

print("=" * 80)
print(
    "Final analysis 05: temperature tail and bump concentration"
)
print("=" * 80)

require_file(
    ECMWF_DAILY_FILE,
    "ECMWF daily demand and weather file",
)

daily = pd.read_csv(
    ECMWF_DAILY_FILE
)

required_columns = {
    "winter_year",
    "member",
    "date",
    TEMPERATURE_COLUMN,
    DEMAND_COLUMN,
}

missing_columns = (
    required_columns
    - set(daily.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        f"{sorted(missing_columns)}"
    )

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce",
)

for column in [
    "winter_year",
    "member",
    TEMPERATURE_COLUMN,
    DEMAND_COLUMN,
]:
    daily[column] = pd.to_numeric(
        daily[column],
        errors="coerce",
    )

daily = daily[
    daily["winter_year"].between(
        OVERLAP_START,
        OVERLAP_END,
    )
].copy()

daily = daily.dropna(
    subset=[
        "winter_year",
        "member",
        "date",
        TEMPERATURE_COLUMN,
        DEMAND_COLUMN,
    ]
).copy()

daily["winter_year"] = (
    daily["winter_year"]
    .astype(int)
)

daily["member"] = (
    daily["member"]
    .astype(int)
)

daily = daily.sort_values(
    [
        "winter_year",
        "member",
        "date",
    ]
).reset_index(
    drop=True
)

winter_years = np.sort(
    daily["winter_year"].unique()
)

print(
    f"\nDaily rows: {len(daily)}"
)

print(
    f"Winter years: {winter_years.min()}-"
    f"{winter_years.max()} (n={len(winter_years)})"
)

print(
    f"Windows: {WINDOWS}"
)


# ============================================================
# Calculate rolling extremes
# ============================================================

rolling_rows = []

grouped = list(
    daily.groupby(
        [
            "winter_year",
            "member",
        ],
        sort=True,
    )
)

for group_index, (
    group_key,
    group,
) in enumerate(
    grouped,
    start=1,
):
    for window in WINDOWS:
        rolling_rows.append(
            rolling_extreme_for_group(
                group=group,
                window=window,
            )
        )

    if (
        group_index % 100 == 0
        or group_index == len(grouped)
    ):
        print(
            f"Processed {group_index}/{len(grouped)} "
            "winter-member groups"
        )

rolling_summary = pd.DataFrame(
    rolling_rows
)

rolling_summary.to_csv(
    OUT_ROLLING_SUMMARY,
    index=False,
)


# ============================================================
# Standardised demand and temperature upper tails
# ============================================================

standardisation_rows = []
tail_frames = []
plot_records = {}
all_tail_values = []

for window in WINDOWS:
    window_data = rolling_summary[
        rolling_summary[
            "window_days"
        ] == window
    ].copy()

    plot_records[
        window
    ] = {}

    variable_settings = [
        (
            "Demand maximum",
            "maximum_mean_demand_GW",
        ),
        (
            "Cold severity",
            "cold_severity_C",
        ),
    ]

    for variable_name, column in variable_settings:
        values = window_data[
            column
        ].to_numpy(dtype=float)

        z, median_value, iqr_value = robust_standardise(
            values
        )

        tail = empirical_upper_tail(
            z,
            UPPER_TAIL_FRACTION,
        )

        tail.insert(
            0,
            "variable",
            variable_name,
        )

        tail.insert(
            0,
            "window_days",
            window,
        )

        tail_frames.append(
            tail
        )

        plot_records[
            window
        ][
            variable_name
        ] = tail

        all_tail_values.extend(
            tail[
                "standardised_severity"
            ].tolist()
        )

        standardisation_rows.append(
            {
                "window_days": window,
                "variable": variable_name,
                "n_full": len(values),
                "n_upper_tail": len(tail),
                "median": median_value,
                "IQR": iqr_value,
            }
        )

standardisation_summary = pd.DataFrame(
    standardisation_rows
)

standardisation_summary.to_csv(
    OUT_STANDARDISATION_SUMMARY,
    index=False,
)

tail_data = pd.concat(
    tail_frames,
    ignore_index=True,
)

tail_data.to_csv(
    OUT_TAIL_DATA,
    index=False,
)


# ============================================================
# Demand gap and winter-year concentration
# ============================================================

gap_summary_rows = []
gap_event_frames = []
year_concentration_frames = []
concentration_summary_rows = []
leave_one_year_out_rows = []

for window in WINDOWS:
    window_data = rolling_summary[
        rolling_summary[
            "window_days"
        ] == window
    ].copy()

    upper_group, gap_summary = locate_largest_gap(
        dataframe=window_data,
        severity_column="maximum_mean_demand_GW",
    )

    gap_summary_rows.append(
        {
            "window_days": window,
            **gap_summary,
        }
    )

    # upper_group is derived from window_data and therefore
    # already contains the window_days column. Do not insert it
    # again, otherwise pandas raises:
    # ValueError: cannot insert window_days, already exists.
    gap_event_frames.append(
        upper_group
    )

    year_counts = (
        upper_group
        .groupby(
            "winter_year"
        )
        .size()
        .reset_index(
            name="n_members"
        )
        .sort_values(
            [
                "n_members",
                "winter_year",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    year_counts.insert(
        0,
        "window_days",
        window,
    )

    year_counts[
        "share_of_upper_group"
    ] = (
        year_counts[
            "n_members"
        ]
        / len(
            upper_group
        )
    )

    year_concentration_frames.append(
        year_counts
    )

    n_distinct_years = (
        upper_group[
            "winter_year"
        ].nunique()
    )

    maximum_year_count = int(
        year_counts[
            "n_members"
        ].max()
    )

    maximum_year_share = float(
        year_counts[
            "share_of_upper_group"
        ].max()
    )

    top_three_year_share = float(
        year_counts[
            "share_of_upper_group"
        ].head(
            3
        ).sum()
    )

    concentration_summary_rows.append(
        {
            "window_days": window,
            "n_upper_group": len(
                upper_group
            ),
            "n_distinct_winter_years": (
                n_distinct_years
            ),
            "maximum_single_year_count": (
                maximum_year_count
            ),
            "maximum_single_year_share": (
                maximum_year_share
            ),
            "top_three_year_share": (
                top_three_year_share
            ),
        }
    )

    for omitted_year in winter_years:
        reduced = window_data[
            window_data[
                "winter_year"
            ] != omitted_year
        ].copy()

        _, reduced_gap_summary = locate_largest_gap(
            dataframe=reduced,
            severity_column="maximum_mean_demand_GW",
        )

        leave_one_year_out_rows.append(
            {
                "window_days": window,
                "omitted_winter_year": int(
                    omitted_year
                ),
                "gap_rank": reduced_gap_summary[
                    "gap_rank"
                ],
                "gap_size_standardised": (
                    reduced_gap_summary[
                        "gap_size_standardised"
                    ]
                ),
                "n_upper_group": reduced_gap_summary[
                    "n_upper_group"
                ],
            }
        )

gap_summary_table = pd.DataFrame(
    gap_summary_rows
)

gap_summary_table.to_csv(
    OUT_GAP_SUMMARY,
    index=False,
)

gap_events = pd.concat(
    gap_event_frames,
    ignore_index=True,
)

gap_events.to_csv(
    OUT_GAP_EVENTS,
    index=False,
)

year_concentration = pd.concat(
    year_concentration_frames,
    ignore_index=True,
)

year_concentration.to_csv(
    OUT_YEAR_CONCENTRATION,
    index=False,
)

concentration_summary = pd.DataFrame(
    concentration_summary_rows
)

concentration_summary.to_csv(
    OUT_CONCENTRATION_SUMMARY,
    index=False,
)

leave_one_year_out = pd.DataFrame(
    leave_one_year_out_rows
)

leave_one_year_out.to_csv(
    OUT_LEAVE_ONE_YEAR_OUT,
    index=False,
)


# ============================================================
# Figure 1: standardised demand versus raw-temperature tails
# ============================================================

all_tail_values = np.asarray(
    all_tail_values,
    dtype=float,
)

x_min = float(
    np.min(
        all_tail_values
    )
)

x_max = float(
    np.max(
        all_tail_values
    )
)

x_padding = max(
    0.05,
    0.03 * (
        x_max
        - x_min
    ),
)

x_lower = (
    x_min
    - x_padding
)

x_upper = (
    x_max
    + x_padding
)

ncols = 2
nrows = math.ceil(
    len(WINDOWS)
    / ncols
)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(
        13.8,
        4.45 * nrows,
    ),
    dpi=FIGURE_DPI,
    sharex=True,
    sharey=True,
)

axes = np.asarray(
    axes
).reshape(-1)

legend_handles = None
legend_labels = None

for index, window in enumerate(
    WINDOWS
):
    ax = axes[
        index
    ]

    demand_tail = plot_records[
        window
    ][
        "Demand maximum"
    ]

    temperature_tail = plot_records[
        window
    ][
        "Cold severity"
    ]

    ax.plot(
        demand_tail[
            "standardised_severity"
        ],
        demand_tail[
            "exceedance_probability"
        ],
        linewidth=1.7,
        marker=".",
        markersize=3.2,
        label="Demand maximum",
    )

    ax.plot(
        temperature_tail[
            "standardised_severity"
        ],
        temperature_tail[
            "exceedance_probability"
        ],
        linewidth=1.7,
        marker=".",
        markersize=3.2,
        label="Raw-temperature cold severity",
    )

    ax.set_yscale(
        "log"
    )

    ax.set_xlim(
        x_lower,
        x_upper,
    )

    ax.set_ylim(
        8e-4,
        UPPER_TAIL_FRACTION * 1.15,
    )

    ax.set_xlabel(
        "Standardised severity\n"
        "(value - sample median) / sample IQR",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Empirical exceedance probability",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        f"{window}-day rolling extreme",
        fontsize=FONT_TITLE,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )

    if legend_handles is None:
        (
            legend_handles,
            legend_labels,
        ) = ax.get_legend_handles_labels()

for index in range(
    len(WINDOWS),
    len(axes),
):
    axes[index].axis(
        "off"
    )

fig.suptitle(
    "ECMWF standardised upper tails: demand versus raw-temperature cold severity\n"
    "Highest 20% of 875 winter-member distributions; shared axes",
    fontsize=FONT_SUPTITLE,
    y=0.995,
)

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=2,
    fontsize=FONT_LEGEND,
    bbox_to_anchor=(
        0.5,
        0.005,
    ),
)

fig.tight_layout(
    rect=[
        0,
        0.075,
        1,
        0.955,
    ],
    h_pad=2.0,
    w_pad=1.5,
)

fig.savefig(
    OUT_TAIL_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_TAIL_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Figure 2: gap stability and year concentration
# ============================================================

loo_summary = (
    leave_one_year_out
    .groupby(
        "window_days"
    )[
        "gap_size_standardised"
    ]
    .agg(
        loo_min="min",
        loo_median="median",
        loo_max="max",
    )
    .reset_index()
)

gap_plot_data = gap_summary_table.merge(
    loo_summary,
    on="window_days",
    how="left",
)

concentration_plot_data = concentration_summary.copy()

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        13.8,
        6.0,
    ),
    dpi=FIGURE_DPI,
)

axes[0].plot(
    gap_plot_data[
        "window_days"
    ],
    gap_plot_data[
        "gap_size_standardised"
    ],
    marker="o",
    markersize=6.0,
    linewidth=1.8,
    label="Original sample",
)

axes[0].fill_between(
    gap_plot_data[
        "window_days"
    ].to_numpy(dtype=float),
    gap_plot_data[
        "loo_min"
    ].to_numpy(dtype=float),
    gap_plot_data[
        "loo_max"
    ].to_numpy(dtype=float),
    alpha=0.16,
    label="Leave-one-winter-year-out range",
)

axes[0].set_xlabel(
    "Averaging window (days)",
    fontsize=FONT_AXIS_LABEL,
)

axes[0].set_ylabel(
    "Largest adjacent standardised demand gap",
    fontsize=FONT_AXIS_LABEL,
)

axes[0].set_title(
    f"Upper-tail gap within ranks "
    f"{GAP_RANK_MIN}-{GAP_RANK_MAX}",
    fontsize=FONT_TITLE,
)

axes[0].set_xticks(
    WINDOWS
)

axes[0].grid(
    True,
    alpha=0.3,
)

axes[0].tick_params(
    axis="both",
    which="major",
    labelsize=FONT_TICK,
)

axes[0].legend(
    fontsize=FONT_LEGEND,
)

axes[1].plot(
    concentration_plot_data[
        "window_days"
    ],
    concentration_plot_data[
        "maximum_single_year_share"
    ],
    marker="o",
    markersize=6.0,
    linewidth=1.8,
    label="Largest single-year share",
)

axes[1].plot(
    concentration_plot_data[
        "window_days"
    ],
    concentration_plot_data[
        "top_three_year_share"
    ],
    marker="s",
    markersize=6.0,
    linewidth=1.8,
    label="Top-three-year share",
)

axes[1].set_xlabel(
    "Averaging window (days)",
    fontsize=FONT_AXIS_LABEL,
)

axes[1].set_ylabel(
    "Share of gap-defined upper group",
    fontsize=FONT_AXIS_LABEL,
)

axes[1].set_title(
    "Winter-year concentration above the detected gap",
    fontsize=FONT_TITLE,
)

axes[1].set_xticks(
    WINDOWS
)

axes[1].set_ylim(
    0,
    1,
)

axes[1].grid(
    True,
    alpha=0.3,
)

axes[1].tick_params(
    axis="both",
    which="major",
    labelsize=FONT_TICK,
)

axes[1].legend(
    fontsize=FONT_LEGEND,
)

fig.suptitle(
    "Demand-tail feature diagnostics: gap stability and winter-year concentration",
    fontsize=FONT_SUPTITLE,
    y=1.02,
)

fig.tight_layout(
    pad=1.3,
    w_pad=2.0,
)

fig.savefig(
    OUT_CONCENTRATION_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_CONCENTRATION_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Print summary
# ============================================================

print("\nGap summary:")
print(
    gap_summary_table.to_string(
        index=False
    )
)

print("\nConcentration summary:")
print(
    concentration_summary.to_string(
        index=False
    )
)

print("\nSaved outputs:")
print(OUT_ROLLING_SUMMARY)
print(OUT_STANDARDISATION_SUMMARY)
print(OUT_TAIL_DATA)
print(OUT_GAP_SUMMARY)
print(OUT_GAP_EVENTS)
print(OUT_YEAR_CONCENTRATION)
print(OUT_CONCENTRATION_SUMMARY)
print(OUT_LEAVE_ONE_YEAR_OUT)
print(OUT_TAIL_FIGURE_PNG)
print(OUT_TAIL_FIGURE_PDF)
print(OUT_CONCENTRATION_FIGURE_PNG)
print(OUT_CONCENTRATION_FIGURE_PDF)

print("\nDone.")
