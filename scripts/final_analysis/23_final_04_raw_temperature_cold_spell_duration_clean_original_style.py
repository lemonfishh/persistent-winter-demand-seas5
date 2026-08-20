from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ============================================================
# Final analysis 04
# Raw daily temperature cold-spell duration analysis
#
# Research question:
#   Do cold spells become markedly less likely to continue
#   beyond approximately four to five weeks?
#
# Main change from the previous HDD-based version:
#   - Cold spells are now defined directly from raw daily mean
#     2-metre temperature.
#   - No 7-day smoothing is used.
#   - No one-day gaps are bridged.
#
# Cold-spell definition:
#   A consecutive run of days on which daily mean temperature
#   is at or below a selected lower-tail temperature threshold.
#
# Threshold sensitivity:
#   20th, 15th and 10th percentiles of pooled ECMWF daily mean
#   temperature over the 1982-2016 Nov 8-Mar 31 sample.
#
# Event inclusion:
#   - minimum duration: 3 consecutive days;
#   - events touching the first or last day of the analysis
#     window are excluded because their true duration may be
#     censored outside Nov 8-Mar 31.
#
# Main quantities:
#
#   Survival:
#       S(n) = P(L >= n | L >= 3)
#
#   Weekly continuation:
#       C(n) = P(L >= n + 7 | L >= n)
#
# Uncertainty:
#   Winter-year cluster bootstrap. Each sampled winter retains
#   all 25 ensemble members and all events from that winter.
#
# Important:
#   This analysis tests literal consecutive cold days. It does
#   not use HDD, a rolling mean or a gap-bridging rule.
# ============================================================


# ============================================================
# Settings
# ============================================================

OVERLAP_START = 1982
OVERLAP_END = 2016

TEMPERATURE_COLUMN = "daily_mean_t2m_c"
DEMAND_COLUMN = "estimated_daily_mean_demand_MW"

# Cold when temperature is at or below each lower percentile.
TEMPERATURE_PERCENTILES = [0.20, 0.15, 0.10]

MIN_EVENT_DAYS = 3

# Exclude spells touching Nov 8 or Mar 31 because their full
# duration may extend outside the available analysis window.
EXCLUDE_BOUNDARY_CENSORED_EVENTS = True

SURVIVAL_DURATIONS = [
    3,
    5,
    7,
    14,
    21,
    28,
    35,
    42,
    49,
    56,
]

CONTINUATION_START_DURATIONS = [
    7,
    14,
    21,
    28,
    35,
    42,
    49,
]

# Continuation estimates with fewer than this many events are
# still retained and plotted. They are shown with open markers
# and labelled with their denominator n.
MIN_CONTINUATION_DENOMINATOR_FOR_PLOT = 10

N_BOOT = 500
RANDOM_SEED = 20260716


# ============================================================
# Input file
# ============================================================

ECMWF_DAILY_FILE = (
    OUTPUT_DIR
    / "daily"
    / "ecmwf_daily_demand_Nov08_1982_2016.csv"
)


# ============================================================
# New clean output folder
# ============================================================

FINAL_OUTPUT_ROOT = OUTPUT_DIR / "final_analysis_clean"

OUT_DIR = (
    FINAL_OUTPUT_ROOT
    / "04_raw_temperature_cold_spell_duration"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

OUT_THRESHOLD_SUMMARY = (
    OUT_DIR
    / "raw_temperature_threshold_summary.csv"
)

OUT_EVENT_TABLE = (
    OUT_DIR
    / "raw_temperature_cold_spell_event_table.csv"
)

OUT_DURATION_COUNTS = (
    OUT_DIR
    / "raw_temperature_cold_spell_duration_counts.csv"
)

OUT_ESTIMATES = (
    OUT_DIR
    / "raw_temperature_survival_continuation_estimates.csv"
)

OUT_BOOTSTRAP_SAMPLES = (
    OUT_DIR
    / "raw_temperature_cluster_bootstrap_samples.csv"
)

OUT_BOOTSTRAP_CI = (
    OUT_DIR
    / "raw_temperature_cluster_bootstrap_ci.csv"
)

OUT_FIGURE_PNG = (
    OUT_DIR
    / "raw_temperature_cold_spell_survival_and_continuation.png"
)

OUT_FIGURE_PDF = (
    OUT_DIR
    / "raw_temperature_cold_spell_survival_and_continuation.pdf"
)


# ============================================================
# Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )


def check_daily_continuity(
    group: pd.DataFrame,
    group_label: str,
) -> None:
    """
    Verify that dates are consecutive within one winter-member.
    """
    date_differences = (
        group["date"]
        .sort_values()
        .diff()
        .dropna()
        .dt.days
    )

    bad_differences = date_differences[
        date_differences != 1
    ]

    if len(bad_differences) > 0:
        raise ValueError(
            "Non-consecutive dates detected for "
            f"{group_label}. Cold-spell runs must not be "
            "continued across missing calendar days."
        )


def extract_events_for_group(
    group: pd.DataFrame,
    threshold_c: float,
    threshold_percentile: float,
) -> list[dict]:
    """
    Extract consecutive raw-temperature cold spells for one
    winter-member group.
    """
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

    group_label = (
        f"winter_year={winter_year}, member={member}"
    )

    check_daily_continuity(
        g,
        group_label,
    )

    temperatures = pd.to_numeric(
        g[TEMPERATURE_COLUMN],
        errors="coerce",
    ).to_numpy(dtype=float)

    if np.any(
        ~np.isfinite(temperatures)
    ):
        raise ValueError(
            f"Missing or non-finite {TEMPERATURE_COLUMN} "
            f"values for {group_label}."
        )

    # Cold spell: raw daily temperature at or below threshold.
    cold_flags = (
        temperatures
        <= threshold_c
    )

    events = []
    n_days = len(g)

    position = 0
    event_number = 0

    while position < n_days:
        if not cold_flags[position]:
            position += 1
            continue

        run_start = position

        while (
            position < n_days
            and cold_flags[position]
        ):
            position += 1

        run_end_exclusive = position
        run_duration = (
            run_end_exclusive
            - run_start
        )

        if run_duration < MIN_EVENT_DAYS:
            continue

        left_censored = (
            run_start == 0
        )

        right_censored = (
            run_end_exclusive == n_days
        )

        if (
            EXCLUDE_BOUNDARY_CENSORED_EVENTS
            and (
                left_censored
                or right_censored
            )
        ):
            continue

        event_number += 1

        event_slice = g.iloc[
            run_start:run_end_exclusive
        ].copy()

        row = {
            "threshold_percentile": (
                threshold_percentile
            ),
            "threshold_label": (
                f"{int(round(threshold_percentile * 100))}th "
                "temperature percentile"
            ),
            "threshold_temperature_C": (
                threshold_c
            ),
            "winter_year": winter_year,
            "member": member,
            "event_number_within_winter_member": (
                event_number
            ),
            "event_start_date": (
                event_slice["date"].iloc[0]
            ),
            "event_end_date": (
                event_slice["date"].iloc[-1]
            ),
            "duration_days": run_duration,
            "left_boundary_censored": left_censored,
            "right_boundary_censored": right_censored,
            "mean_temperature_C": (
                event_slice[
                    TEMPERATURE_COLUMN
                ].mean()
            ),
            "minimum_temperature_C": (
                event_slice[
                    TEMPERATURE_COLUMN
                ].min()
            ),
            "maximum_temperature_C": (
                event_slice[
                    TEMPERATURE_COLUMN
                ].max()
            ),
        }

        if DEMAND_COLUMN in event_slice.columns:
            demand_values = pd.to_numeric(
                event_slice[
                    DEMAND_COLUMN
                ],
                errors="coerce",
            )

            row["mean_demand_GW"] = (
                demand_values.mean()
                / 1000.0
            )

            row["maximum_daily_demand_GW"] = (
                demand_values.max()
                / 1000.0
            )

        events.append(
            row
        )

    return events


def calculate_duration_metrics(
    durations: np.ndarray,
) -> list[dict]:
    """
    Calculate survival and weekly continuation probabilities.
    """
    durations = np.asarray(
        durations,
        dtype=float,
    )

    durations = durations[
        np.isfinite(durations)
    ]

    rows = []

    for duration in SURVIVAL_DURATIONS:
        numerator = int(
            np.sum(
                durations >= duration
            )
        )

        denominator = int(
            len(durations)
        )

        estimate = (
            numerator / denominator
            if denominator > 0
            else np.nan
        )

        rows.append(
            {
                "metric": "survival",
                "duration_days": duration,
                "estimate": estimate,
                "numerator_events": numerator,
                "denominator_events": denominator,
            }
        )

    for duration in CONTINUATION_START_DURATIONS:
        denominator = int(
            np.sum(
                durations >= duration
            )
        )

        numerator = int(
            np.sum(
                durations >= duration + 7
            )
        )

        estimate = (
            numerator / denominator
            if denominator > 0
            else np.nan
        )

        rows.append(
            {
                "metric": "weekly_continuation",
                "duration_days": duration,
                "estimate": estimate,
                "numerator_events": numerator,
                "denominator_events": denominator,
            }
        )

    return rows


def summarise_bootstrap_ci(
    bootstrap_samples: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise bootstrap distributions with percentile intervals.
    """
    rows = []

    group_columns = [
        "threshold_percentile",
        "threshold_label",
        "metric",
        "duration_days",
    ]

    for keys, group in bootstrap_samples.groupby(
        group_columns
    ):
        values = (
            group["estimate"]
            .dropna()
            .to_numpy(dtype=float)
        )

        row = {
            column: key
            for column, key in zip(
                group_columns,
                keys,
            )
        }

        row["n_success"] = len(values)

        if len(values) == 0:
            row["ci_lower"] = np.nan
            row["median"] = np.nan
            row["ci_upper"] = np.nan
            row["bootstrap_mean"] = np.nan
            row["bootstrap_sd"] = np.nan
        else:
            row["ci_lower"] = np.quantile(
                values,
                0.025,
            )

            row["median"] = np.quantile(
                values,
                0.500,
            )

            row["ci_upper"] = np.quantile(
                values,
                0.975,
            )

            row["bootstrap_mean"] = np.mean(
                values
            )

            row["bootstrap_sd"] = (
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Load and validate daily data
# ============================================================

print("=" * 80)
print(
    "Final analysis 04: raw daily temperature "
    "cold-spell duration analysis"
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
}

missing_columns = (
    required_columns
    - set(daily.columns)
)

if missing_columns:
    raise ValueError(
        "Missing required columns in the ECMWF daily file: "
        f"{sorted(missing_columns)}"
    )

daily["date"] = pd.to_datetime(
    daily["date"],
    errors="coerce",
)

daily["winter_year"] = pd.to_numeric(
    daily["winter_year"],
    errors="coerce",
)

daily["member"] = pd.to_numeric(
    daily["member"],
    errors="coerce",
)

daily[TEMPERATURE_COLUMN] = pd.to_numeric(
    daily[TEMPERATURE_COLUMN],
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

members = np.sort(
    daily["member"].unique()
)

print(
    f"\nDaily rows: {len(daily)}"
)

print(
    f"Winter years: "
    f"{winter_years.min()} to {winter_years.max()} "
    f"(n={len(winter_years)})"
)

print(
    f"Members: n={len(members)}"
)

print(
    f"Temperature column: {TEMPERATURE_COLUMN}"
)

print(
    f"Minimum retained event duration: "
    f"{MIN_EVENT_DAYS} days"
)

print(
    "Gap bridging: none"
)

print(
    "Boundary-censored events excluded: "
    f"{EXCLUDE_BOUNDARY_CENSORED_EVENTS}"
)


# ============================================================
# Calculate thresholds and extract events
# ============================================================

threshold_rows = []
all_event_rows = []

temperature_distribution = (
    daily[TEMPERATURE_COLUMN]
    .dropna()
    .to_numpy(dtype=float)
)

for percentile in TEMPERATURE_PERCENTILES:
    threshold_c = np.quantile(
        temperature_distribution,
        percentile,
    )

    threshold_label = (
        f"{int(round(percentile * 100))}th "
        "temperature percentile"
    )

    print(
        f"\nExtracting events for "
        f"{threshold_label}: "
        f"temperature <= {threshold_c:.3f} C"
    )

    threshold_event_rows = []

    for (
        winter_year,
        member,
    ), group in daily.groupby(
        [
            "winter_year",
            "member",
        ],
        sort=True,
    ):
        event_rows = extract_events_for_group(
            group=group,
            threshold_c=threshold_c,
            threshold_percentile=percentile,
        )

        threshold_event_rows.extend(
            event_rows
        )

    threshold_events = pd.DataFrame(
        threshold_event_rows
    )

    if len(threshold_events) == 0:
        warnings.warn(
            f"No events of at least {MIN_EVENT_DAYS} days "
            f"were found for {threshold_label}."
        )

        n_events = 0
        n_years_with_events = 0
        maximum_duration = np.nan
        median_duration = np.nan
    else:
        n_events = len(
            threshold_events
        )

        n_years_with_events = (
            threshold_events[
                "winter_year"
            ].nunique()
        )

        maximum_duration = (
            threshold_events[
                "duration_days"
            ].max()
        )

        median_duration = (
            threshold_events[
                "duration_days"
            ].median()
        )

        all_event_rows.append(
            threshold_events
        )

    threshold_rows.append(
        {
            "threshold_percentile": percentile,
            "threshold_label": threshold_label,
            "threshold_temperature_C": threshold_c,
            "n_events_at_least_3_days": n_events,
            "n_winter_years_with_events": n_years_with_events,
            "median_event_duration_days": median_duration,
            "maximum_event_duration_days": maximum_duration,
        }
    )

threshold_summary = pd.DataFrame(
    threshold_rows
)

threshold_summary.to_csv(
    OUT_THRESHOLD_SUMMARY,
    index=False,
)

if len(all_event_rows) == 0:
    raise ValueError(
        "No raw-temperature cold-spell events were found."
    )

event_table = pd.concat(
    all_event_rows,
    ignore_index=True,
)

event_table.to_csv(
    OUT_EVENT_TABLE,
    index=False,
)


# ============================================================
# Original survival and continuation estimates
# ============================================================

estimate_rows = []
duration_count_rows = []

for percentile in TEMPERATURE_PERCENTILES:
    threshold_label = (
        f"{int(round(percentile * 100))}th "
        "temperature percentile"
    )

    threshold_events = event_table[
        event_table[
            "threshold_percentile"
        ] == percentile
    ].copy()

    durations = threshold_events[
        "duration_days"
    ].to_numpy(dtype=float)

    metric_rows = calculate_duration_metrics(
        durations
    )

    for row in metric_rows:
        row.update(
            {
                "threshold_percentile": percentile,
                "threshold_label": threshold_label,
                "n_events_total": len(durations),
                "n_winter_years_with_events": (
                    threshold_events[
                        "winter_year"
                    ].nunique()
                ),
            }
        )

        estimate_rows.append(row)

    for duration in SURVIVAL_DURATIONS:
        qualifying_events = threshold_events[
            threshold_events[
                "duration_days"
            ] >= duration
        ]

        duration_count_rows.append(
            {
                "threshold_percentile": percentile,
                "threshold_label": threshold_label,
                "duration_days": duration,
                "n_events_at_least_duration": len(
                    qualifying_events
                ),
                "n_events_total": len(
                    durations
                ),
                "n_distinct_winter_years": (
                    qualifying_events[
                        "winter_year"
                    ].nunique()
                ),
                "n_distinct_winter_members": (
                    qualifying_events[
                        [
                            "winter_year",
                            "member",
                        ]
                    ]
                    .drop_duplicates()
                    .shape[0]
                ),
            }
        )

estimates = pd.DataFrame(
    estimate_rows
)

estimates.to_csv(
    OUT_ESTIMATES,
    index=False,
)

duration_counts = pd.DataFrame(
    duration_count_rows
)

duration_counts.to_csv(
    OUT_DURATION_COUNTS,
    index=False,
)


# ============================================================
# Winter-year cluster bootstrap
# ============================================================

print(
    f"\nRunning {N_BOOT} winter-year cluster "
    "bootstrap replicates..."
)

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_rows = []

for percentile in TEMPERATURE_PERCENTILES:
    threshold_label = (
        f"{int(round(percentile * 100))}th "
        "temperature percentile"
    )

    threshold_events = event_table[
        event_table[
            "threshold_percentile"
        ] == percentile
    ].copy()

    durations_by_year = {
        int(year): group[
            "duration_days"
        ].to_numpy(dtype=float)
        for year, group in threshold_events.groupby(
            "winter_year"
        )
    }

    for bootstrap_id in range(
        1,
        N_BOOT + 1,
    ):
        sampled_years = rng.choice(
            winter_years,
            size=len(winter_years),
            replace=True,
        )

        sampled_duration_parts = []

        for sampled_year in sampled_years:
            year_durations = durations_by_year.get(
                int(sampled_year),
                np.asarray(
                    [],
                    dtype=float,
                ),
            )

            if len(year_durations) > 0:
                sampled_duration_parts.append(
                    year_durations
                )

        if len(sampled_duration_parts) == 0:
            sampled_durations = np.asarray(
                [],
                dtype=float,
            )
        else:
            sampled_durations = np.concatenate(
                sampled_duration_parts
            )

        metric_rows = calculate_duration_metrics(
            sampled_durations
        )

        for row in metric_rows:
            bootstrap_rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "threshold_percentile": percentile,
                    "threshold_label": threshold_label,
                    "metric": row["metric"],
                    "duration_days": row["duration_days"],
                    "estimate": row["estimate"],
                    "numerator_events": row[
                        "numerator_events"
                    ],
                    "denominator_events": row[
                        "denominator_events"
                    ],
                    "n_sampled_events": len(
                        sampled_durations
                    ),
                }
            )

bootstrap_samples = pd.DataFrame(
    bootstrap_rows
)

bootstrap_samples.to_csv(
    OUT_BOOTSTRAP_SAMPLES,
    index=False,
)

bootstrap_ci = summarise_bootstrap_ci(
    bootstrap_samples
)

original_estimate_columns = estimates[
    [
        "threshold_percentile",
        "threshold_label",
        "metric",
        "duration_days",
        "estimate",
        "numerator_events",
        "denominator_events",
    ]
].rename(
    columns={
        "estimate": "original_estimate",
        "numerator_events": (
            "original_numerator_events"
        ),
        "denominator_events": (
            "original_denominator_events"
        ),
    }
)

bootstrap_ci = bootstrap_ci.merge(
    original_estimate_columns,
    on=[
        "threshold_percentile",
        "threshold_label",
        "metric",
        "duration_days",
    ],
    how="left",
)

bootstrap_ci.to_csv(
    OUT_BOOTSTRAP_CI,
    index=False,
)


# ============================================================
# Main figure: one row, two panels
#
# Keep the original visual style:
#   - one row, two panels;
#   - linear y-axes;
#   - three threshold curves;
#   - shaded bootstrap intervals;
#   - no n labels;
#   - no NA labels;
#   - small-sample continuation estimates remain as open points.
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        12.8,
        5.2,
    ),
    dpi=300,
)

# ------------------------------------------------------------
# Left panel: survival
# ------------------------------------------------------------

survival_data = bootstrap_ci[
    bootstrap_ci[
        "metric"
    ] == "survival"
].copy()

for percentile in TEMPERATURE_PERCENTILES:
    threshold_data = survival_data[
        survival_data[
            "threshold_percentile"
        ] == percentile
    ].sort_values(
        "duration_days"
    )

    line = axes[0].plot(
        threshold_data[
            "duration_days"
        ],
        threshold_data[
            "original_estimate"
        ],
        marker="o",
        linewidth=1.6,
        label=(
            f"Temperature <= "
            f"{int(round(percentile * 100))}th percentile"
        ),
    )[0]

    axes[0].fill_between(
        threshold_data[
            "duration_days"
        ].to_numpy(dtype=float),
        threshold_data[
            "ci_lower"
        ].to_numpy(dtype=float),
        threshold_data[
            "ci_upper"
        ].to_numpy(dtype=float),
        alpha=0.16,
        color=line.get_color(),
    )

axes[0].set_xlabel(
    "Duration reached (days)"
)

axes[0].set_ylabel(
    r"$P(L \geq n \mid L \geq 3)$"
)

axes[0].set_title(
    "Raw-temperature cold-spell survival"
)

axes[0].set_ylim(
    -0.03,
    1.03,
)

axes[0].set_xticks(
    SURVIVAL_DURATIONS
)

axes[0].grid(
    True,
    alpha=0.3,
)

axes[0].legend(
    fontsize=8.5,
)


# ------------------------------------------------------------
# Right panel: weekly continuation
# ------------------------------------------------------------

continuation_data = bootstrap_ci[
    bootstrap_ci[
        "metric"
    ] == "weekly_continuation"
].copy()

for percentile in TEMPERATURE_PERCENTILES:
    threshold_data = continuation_data[
        continuation_data[
            "threshold_percentile"
        ] == percentile
    ].sort_values(
        "duration_days"
    ).copy()

    finite_data = threshold_data[
        np.isfinite(
            threshold_data[
                "original_estimate"
            ]
        )
    ].copy()

    if len(finite_data) == 0:
        continue

    line = axes[1].plot(
        finite_data[
            "duration_days"
        ],
        finite_data[
            "original_estimate"
        ],
        linewidth=1.6,
        label=(
            f"Temperature <= "
            f"{int(round(percentile * 100))}th percentile"
        ),
    )[0]

    colour = line.get_color()

    interval_data = finite_data[
        np.isfinite(
            finite_data[
                "ci_lower"
            ]
        )
        & np.isfinite(
            finite_data[
                "ci_upper"
            ]
        )
    ].copy()

    if len(interval_data) > 0:
        axes[1].fill_between(
            interval_data[
                "duration_days"
            ].to_numpy(dtype=float),
            interval_data[
                "ci_lower"
            ].to_numpy(dtype=float),
            interval_data[
                "ci_upper"
            ].to_numpy(dtype=float),
            alpha=0.16,
            color=colour,
        )

    reliable_data = finite_data[
        finite_data[
            "original_denominator_events"
        ]
        >= MIN_CONTINUATION_DENOMINATOR_FOR_PLOT
    ].copy()

    small_sample_data = finite_data[
        finite_data[
            "original_denominator_events"
        ]
        < MIN_CONTINUATION_DENOMINATOR_FOR_PLOT
    ].copy()

    if len(reliable_data) > 0:
        axes[1].scatter(
            reliable_data[
                "duration_days"
            ],
            reliable_data[
                "original_estimate"
            ],
            marker="o",
            s=34,
            color=colour,
            zorder=3,
        )

    if len(small_sample_data) > 0:
        axes[1].scatter(
            small_sample_data[
                "duration_days"
            ],
            small_sample_data[
                "original_estimate"
            ],
            marker="o",
            facecolors="white",
            edgecolors=colour,
            s=40,
            linewidths=1.3,
            zorder=4,
        )

axes[1].set_xlabel(
    "Current duration (days)"
)

axes[1].set_ylabel(
    r"$P(L \geq n+7 \mid L \geq n)$"
)

axes[1].set_title(
    "Probability of continuing for another week"
)

axes[1].set_ylim(
    -0.03,
    1.03,
)

axes[1].set_xticks(
    CONTINUATION_START_DURATIONS
)

axes[1].grid(
    True,
    alpha=0.3,
)

axes[1].legend(
    fontsize=8.5,
    loc="upper left",
)

fig.suptitle(
    "ECMWF raw-temperature cold-spell persistence\n"
    "No smoothing or gap bridging; 1982-2016 pooled ensemble; "
    "winter-year cluster-bootstrap 95% intervals",
    fontsize=13.5,
    y=1.03,
)

fig.tight_layout()

fig.savefig(
    OUT_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Print key results
# ============================================================

print("\nThreshold summary:")
print(
    threshold_summary.to_string(
        index=False
    )
)

print("\nEvent counts by minimum duration:")
print(
    duration_counts.to_string(
        index=False
    )
)

print("\nSurvival and continuation estimates:")
print(
    bootstrap_ci[
        [
            "threshold_label",
            "metric",
            "duration_days",
            "original_estimate",
            "ci_lower",
            "ci_upper",
            "original_numerator_events",
            "original_denominator_events",
            "n_success",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved outputs:")
print(OUT_THRESHOLD_SUMMARY)
print(OUT_EVENT_TABLE)
print(OUT_DURATION_COUNTS)
print(OUT_ESTIMATES)
print(OUT_BOOTSTRAP_SAMPLES)
print(OUT_BOOTSTRAP_CI)
print(OUT_FIGURE_PNG)
print(OUT_FIGURE_PDF)

print("\nDone.")
