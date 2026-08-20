from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ============================================================
# Final analysis 08
# Mean residual life diagnostics for pooled SEAS5 severity
#
# Purpose:
#   Examine whether the upper tail is approximately linear in
#   mean excess over a range of thresholds.
#
# Scope:
#   - pooled SEAS5 only;
#   - 28-day and 84-day severity;
#   - thresholds from the 75th to 95th sample percentiles;
#   - 500 winter-year cluster-bootstrap replicates;
#   - all 25 members are retained whenever a winter year is
#     selected in a bootstrap replicate.
#
# Interpretation:
#   This is an exploratory diagnostic. It is not a formal
#   threshold test and does not by itself establish a separate
#   tail regime or justify a GPD model.
# ============================================================


# ============================================================
# Settings
# ============================================================

WINDOWS = [
    28,
    84,
]

OVERLAP_START = 1982
OVERLAP_END = 2016

THRESHOLD_QUANTILES = np.linspace(
    0.75,
    0.95,
    41,
)

N_BOOT = 500
RANDOM_SEED = 20260805

# Figure and font settings
FIGURE_DPI = 300

FONT_BASE = 13.0
FONT_AXIS_LABEL = 14.0
FONT_TICK = 12.0
FONT_TITLE = 14.0
FONT_LEGEND = 11.5
FONT_SUPTITLE = 16.0
FONT_ANNOTATION = 10.5

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

# At the 95th percentile the original n=875 sample has about
# 44 exceedances. A lower minimum is used inside bootstrap
# replicates so that occasional cluster-resampling variation
# does not remove most high-threshold intervals.
MIN_BOOT_EXCEEDANCES = 20


# ============================================================
# Input and output paths
# ============================================================

ECMWF_SEVERITY_FILE = (
    OUTPUT_DIR
    / "severity"
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv"
)

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "08_mean_residual_life_diagnostic"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ESTIMATES = (
    OUT_DIR
    / "mrl_estimates_28day_84day.csv"
)

OUT_BOOTSTRAP_SAMPLES = (
    OUT_DIR
    / "mrl_cluster_bootstrap_samples.csv"
)

OUT_BOOTSTRAP_CI = (
    OUT_DIR
    / "mrl_cluster_bootstrap_ci.csv"
)

OUT_FIGURE_PNG = (
    OUT_DIR
    / "mean_residual_life_28day_84day.png"
)

OUT_FIGURE_PDF = (
    OUT_DIR
    / "mean_residual_life_28day_84day.pdf"
)


# ============================================================
# Helper functions
# ============================================================

def require_file(
    path: Path,
    label: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )


def severity_column(
    window: int,
) -> str:
    return f"max_{window}d_mean_demand_MW"


def finite_values(
    values,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    return values[
        np.isfinite(
            values
        )
    ]


def mean_excess(
    values_gw,
    threshold_gw: float,
    minimum_exceedances: int = 1,
) -> tuple[float, int]:
    """
    Return the empirical mean excess above one threshold.

    The mean excess is:
        mean(values - threshold | values > threshold)
    """
    values_gw = finite_values(
        values_gw
    )

    exceedances = values_gw[
        values_gw
        > threshold_gw
    ]

    n_exceedances = len(
        exceedances
    )

    if n_exceedances < minimum_exceedances:
        return (
            np.nan,
            n_exceedances,
        )

    estimate = float(
        np.mean(
            exceedances
            - threshold_gw
        )
    )

    return (
        estimate,
        n_exceedances,
    )


def build_threshold_table(
    values_gw,
    window: int,
) -> pd.DataFrame:
    """
    Construct a fixed threshold grid from the original sample.

    The same thresholds are used for every cluster-bootstrap
    replicate, so the confidence interval at each x-coordinate
    refers to the same severity threshold.
    """
    values_gw = finite_values(
        values_gw
    )

    threshold_values = np.quantile(
        values_gw,
        THRESHOLD_QUANTILES,
    )

    table = pd.DataFrame(
        {
            "window_days": window,
            "threshold_quantile": (
                THRESHOLD_QUANTILES
            ),
            "threshold_percentile": (
                100.0
                * THRESHOLD_QUANTILES
            ),
            "threshold_GW": (
                threshold_values
            ),
        }
    )

    # Exact duplicate threshold values are unlikely, but can
    # occur in rounded or tied data. Keep only one copy so the
    # plotted x-axis remains strictly increasing.
    table = (
        table
        .drop_duplicates(
            subset=[
                "threshold_GW",
            ],
            keep="first",
        )
        .sort_values(
            "threshold_GW"
        )
        .reset_index(
            drop=True
        )
    )

    estimate_rows = []

    for row in table.itertuples(
        index=False
    ):
        estimate, n_exceedances = mean_excess(
            values_gw,
            float(
                row.threshold_GW
            ),
        )

        estimate_rows.append(
            {
                "window_days": window,
                "threshold_quantile": float(
                    row.threshold_quantile
                ),
                "threshold_percentile": float(
                    row.threshold_percentile
                ),
                "threshold_GW": float(
                    row.threshold_GW
                ),
                "mean_excess_GW": estimate,
                "n_exceedances": n_exceedances,
                "sample_size": len(
                    values_gw
                ),
            }
        )

    return pd.DataFrame(
        estimate_rows
    )


def percentile_summary(
    values,
) -> dict:
    values = finite_values(
        values
    )

    if len(values) == 0:
        return {
            "ci_lower_GW": np.nan,
            "bootstrap_median_GW": np.nan,
            "ci_upper_GW": np.nan,
            "bootstrap_mean_GW": np.nan,
            "bootstrap_sd_GW": np.nan,
            "n_success": 0,
        }

    return {
        "ci_lower_GW": float(
            np.quantile(
                values,
                0.025,
            )
        ),
        "bootstrap_median_GW": float(
            np.quantile(
                values,
                0.5,
            )
        ),
        "ci_upper_GW": float(
            np.quantile(
                values,
                0.975,
            )
        ),
        "bootstrap_mean_GW": float(
            np.mean(
                values
            )
        ),
        "bootstrap_sd_GW": float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else np.nan,
        "n_success": len(
            values
        ),
    }


# ============================================================
# Load and validate data
# ============================================================

print("=" * 80)
print(
    "Final analysis 08: mean residual life diagnostics"
)
print("=" * 80)

require_file(
    ECMWF_SEVERITY_FILE,
    "ECMWF extended-window severity file",
)

ecmwf = pd.read_csv(
    ECMWF_SEVERITY_FILE
)

if "winter_year" not in ecmwf.columns:
    raise ValueError(
        "The ECMWF severity file has no winter_year column."
    )

ecmwf[
    "winter_year"
] = pd.to_numeric(
    ecmwf[
        "winter_year"
    ],
    errors="coerce",
)

ecmwf = ecmwf[
    ecmwf[
        "winter_year"
    ].between(
        OVERLAP_START,
        OVERLAP_END,
    )
].copy()

ecmwf = ecmwf[
    ecmwf[
        "winter_year"
    ].notna()
].copy()

ecmwf[
    "winter_year"
] = ecmwf[
    "winter_year"
].astype(int)

expected_years = np.arange(
    OVERLAP_START,
    OVERLAP_END + 1,
)

observed_years = np.sort(
    ecmwf[
        "winter_year"
    ].unique()
)

if not np.array_equal(
    observed_years,
    expected_years,
):
    raise ValueError(
        "The ECMWF winter years do not match 1982-2016."
    )

counts_by_year = (
    ecmwf
    .groupby(
        "winter_year"
    )
    .size()
)

if not np.all(
    counts_by_year.to_numpy()
    == 25
):
    raise ValueError(
        "The ECMWF file does not contain exactly "
        "25 members for every winter year."
    )

print(
    f"\nECMWF rows: {len(ecmwf)}"
)
print(
    f"Winter years: {len(observed_years)}"
)
print(
    f"Windows: {WINDOWS}"
)
print(
    "Threshold range: "
    f"{100 * THRESHOLD_QUANTILES[0]:.0f}th-"
    f"{100 * THRESHOLD_QUANTILES[-1]:.0f}th percentiles"
)
print(
    f"Bootstrap replicates: {N_BOOT}"
)


# ============================================================
# Original MRL estimates
# ============================================================

estimate_parts = []
window_values = {}
values_by_window_and_year = {}

for window in WINDOWS:
    column = severity_column(
        window
    )

    if column not in ecmwf.columns:
        raise ValueError(
            f"Missing ECMWF severity column: {column}"
        )

    values_gw = (
        pd.to_numeric(
            ecmwf[
                column
            ],
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
        / 1000.0
    )

    if not np.all(
        np.isfinite(
            values_gw
        )
    ):
        raise ValueError(
            f"Non-finite values found in {column}."
        )

    window_values[
        window
    ] = values_gw

    estimate_parts.append(
        build_threshold_table(
            values_gw,
            window,
        )
    )

    values_by_window_and_year[
        window
    ] = {
        int(year): (
            pd.to_numeric(
                group[
                    column
                ],
                errors="coerce",
            )
            .to_numpy(
                dtype=float
            )
            / 1000.0
        )
        for year, group in ecmwf.groupby(
            "winter_year",
            sort=True,
        )
    }

estimates = pd.concat(
    estimate_parts,
    ignore_index=True,
)

estimates.to_csv(
    OUT_ESTIMATES,
    index=False,
)


# ============================================================
# Winter-year cluster bootstrap
# ============================================================

print(
    "\nRunning winter-year cluster bootstrap..."
)

rng = np.random.default_rng(
    RANDOM_SEED
)

bootstrap_rows = []

for bootstrap_id in range(
    1,
    N_BOOT + 1,
):
    sampled_years = rng.choice(
        observed_years,
        size=len(
            observed_years
        ),
        replace=True,
    )

    for window in WINDOWS:
        sampled_values = np.concatenate(
            [
                values_by_window_and_year[
                    window
                ][
                    int(year)
                ]
                for year in sampled_years
            ]
        )

        threshold_subset = estimates[
            estimates[
                "window_days"
            ] == window
        ]

        for row in threshold_subset.itertuples(
            index=False
        ):
            estimate, n_exceedances = mean_excess(
                sampled_values,
                float(
                    row.threshold_GW
                ),
                minimum_exceedances=(
                    MIN_BOOT_EXCEEDANCES
                ),
            )

            bootstrap_rows.append(
                {
                    "bootstrap_id": bootstrap_id,
                    "window_days": window,
                    "threshold_quantile": float(
                        row.threshold_quantile
                    ),
                    "threshold_percentile": float(
                        row.threshold_percentile
                    ),
                    "threshold_GW": float(
                        row.threshold_GW
                    ),
                    "mean_excess_GW": estimate,
                    "n_exceedances": n_exceedances,
                    "n_sampled_winter_years": len(
                        sampled_years
                    ),
                    "n_sampled_values": len(
                        sampled_values
                    ),
                }
            )

    if (
        bootstrap_id % 50
        == 0
    ):
        print(
            f"  completed {bootstrap_id}/{N_BOOT}"
        )

bootstrap_samples = pd.DataFrame(
    bootstrap_rows
)

bootstrap_samples.to_csv(
    OUT_BOOTSTRAP_SAMPLES,
    index=False,
)


# ============================================================
# Bootstrap confidence intervals
# ============================================================

ci_rows = []

group_columns = [
    "window_days",
    "threshold_quantile",
    "threshold_percentile",
    "threshold_GW",
]

for keys, group in bootstrap_samples.groupby(
    group_columns,
    sort=True,
):
    (
        window,
        threshold_quantile,
        threshold_percentile,
        threshold_gw,
    ) = keys

    summary = percentile_summary(
        group[
            "mean_excess_GW"
        ].to_numpy(
            dtype=float
        )
    )

    original_row = estimates[
        (
            estimates[
                "window_days"
            ] == window
        )
        & np.isclose(
            estimates[
                "threshold_GW"
            ],
            threshold_gw,
        )
    ]

    if len(
        original_row
    ) != 1:
        raise RuntimeError(
            "Could not match one original MRL estimate "
            "to its bootstrap threshold."
        )

    original_row = original_row.iloc[
        0
    ]

    ci_rows.append(
        {
            "window_days": int(
                window
            ),
            "threshold_quantile": float(
                threshold_quantile
            ),
            "threshold_percentile": float(
                threshold_percentile
            ),
            "threshold_GW": float(
                threshold_gw
            ),
            "original_mean_excess_GW": float(
                original_row[
                    "mean_excess_GW"
                ]
            ),
            "original_n_exceedances": int(
                original_row[
                    "n_exceedances"
                ]
            ),
            **summary,
        }
    )

bootstrap_ci = pd.DataFrame(
    ci_rows
)

bootstrap_ci.to_csv(
    OUT_BOOTSTRAP_CI,
    index=False,
)


# ============================================================
# Main figure: one row, two panels
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        13.8,
        6.0,
    ),
    dpi=FIGURE_DPI,
)

for ax, window in zip(
    axes,
    WINDOWS,
):
    plot_data = bootstrap_ci[
        bootstrap_ci[
            "window_days"
        ] == window
    ].sort_values(
        "threshold_GW"
    )

    ax.fill_between(
        plot_data[
            "threshold_GW"
        ],
        plot_data[
            "ci_lower_GW"
        ],
        plot_data[
            "ci_upper_GW"
        ],
        alpha=0.22,
        label=(
            "95% winter-year "
            "cluster-bootstrap interval"
        ),
    )

    ax.plot(
        plot_data[
            "threshold_GW"
        ],
        plot_data[
            "original_mean_excess_GW"
        ],
        marker="o",
        markersize=5.2,
        linewidth=1.8,
        label="Empirical mean excess",
    )

    ax.set_xlabel(
        "Threshold (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Mean excess above threshold (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        f"{window}-day severity",
        fontsize=FONT_TITLE,
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )

    ax.legend(
        fontsize=FONT_LEGEND,
    )

    # Add a small, unobtrusive indication of the number of
    # exceedances at the ends of the threshold range.
    first = plot_data.iloc[
        0
    ]
    last = plot_data.iloc[
        -1
    ]

    annotation = (
        "Original exceedances:\n"
        f"{int(first['original_n_exceedances'])} "
        f"at {first['threshold_percentile']:.0f}th percentile\n"
        f"{int(last['original_n_exceedances'])} "
        f"at {last['threshold_percentile']:.0f}th percentile"
    )

    ax.text(
        0.97,
        0.05,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=FONT_ANNOTATION,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.82,
        },
    )

fig.suptitle(
    "Mean residual life diagnostics for pooled SEAS5 demand severity\n"
    "Thresholds from the 75th to 95th sample percentiles",
    fontsize=FONT_SUPTITLE,
    y=1.03,
)

fig.tight_layout(
    pad=1.3,
    w_pad=2.0,
)

fig.savefig(
    OUT_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Print concise results
# ============================================================

print(
    "\nOriginal threshold summaries:"
)

summary_rows = []

for window in WINDOWS:
    window_summary = estimates[
        estimates[
            "window_days"
        ] == window
    ].reset_index(
        drop=True
    )

    selected_indices = [
        0,
        len(
            window_summary
        ) // 2,
        len(
            window_summary
        ) - 1,
    ]

    summary_rows.append(
        window_summary.iloc[
            selected_indices
        ][
            [
                "window_days",
                "threshold_percentile",
                "threshold_GW",
                "mean_excess_GW",
                "n_exceedances",
            ]
        ]
    )

print(
    pd.concat(
        summary_rows,
        ignore_index=True,
    ).to_string(
        index=False
    )
)

print(
    "\nMinimum successful bootstrap replicates "
    "across plotted thresholds:"
)

print(
    bootstrap_ci.groupby(
        "window_days"
    )[
        "n_success"
    ]
    .min()
    .to_string()
)

print(
    "\nSaved outputs:"
)
print(
    OUT_ESTIMATES
)
print(
    OUT_BOOTSTRAP_SAMPLES
)
print(
    OUT_BOOTSTRAP_CI
)
print(
    OUT_FIGURE_PNG
)
print(
    OUT_FIGURE_PDF
)

print(
    "\nDone."
)
