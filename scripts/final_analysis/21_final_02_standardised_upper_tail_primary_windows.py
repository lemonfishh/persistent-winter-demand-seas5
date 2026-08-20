from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ============================================================
# Final analysis 02
# Standardised upper-tail comparison for the primary windows
#
# Primary windows:
#   1, 7, 14, 21, 28, 56 and 84 days
#
# Purpose:
#   Compare the relative upper-tail shapes of Hannah overlap
#   and pooled ECMWF after removing each sample's own location
#   and scale.
#
# Standardisation:
#
#       z = (x - sample median) / sample IQR
#
# Hannah and ECMWF are standardised separately.
#
# Upper tail:
#   Only the highest 20% of each distribution is plotted.
#
# Important:
#   - This is a tail-SHAPE comparison.
#   - It is not an absolute GW comparison.
#   - No explicit mean shift is needed because any constant
#     location shift cancels during median/IQR standardisation.
#   - The figure includes ALL seven primary windows, with no
#     more than two panels per row.
# ============================================================


# ============================================================
# Settings
# ============================================================

PRIMARY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]

OVERLAP_START = 1982
OVERLAP_END = 2016

UPPER_TAIL_FRACTION = 0.20


# ============================================================
# Input files
# ============================================================

SEVERITY_DIR = OUTPUT_DIR / "severity"

ECMWF_SEVERITY_FILE = (
    SEVERITY_DIR
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv"
)

HANNAH_SEVERITY_FILE = (
    SEVERITY_DIR
    / "hannah_severity_summary_Nov08_extended_windows.csv"
)


# ============================================================
# New clean output folder
# ============================================================

FINAL_OUTPUT_ROOT = OUTPUT_DIR / "final_analysis_clean"

OUT_DIR = (
    FINAL_OUTPUT_ROOT
    / "02_standardised_upper_tail_primary_windows"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Output files
# ============================================================

OUT_STANDARDISATION_SUMMARY = (
    OUT_DIR
    / "primary_windows_standardisation_summary.csv"
)

OUT_TAIL_DATA = (
    OUT_DIR
    / "primary_windows_standardised_upper_tail_data.csv"
)

OUT_FIGURE_PNG = (
    OUT_DIR
    / "all_primary_windows_standardised_upper_tail.png"
)

OUT_FIGURE_PDF = (
    OUT_DIR
    / "all_primary_windows_standardised_upper_tail.pdf"
)


# ============================================================
# Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise a clear error when an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )


def severity_column(window: int) -> str:
    """Return the severity column name for one duration."""
    return f"max_{window}d_mean_demand_MW"


def finite_numeric_values(
    dataframe: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Return finite numeric values from a dataframe column."""
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            f"No finite values found in column: {column}"
        )

    return values


def robust_standardise(
    values_gw: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Standardise using the sample median and interquartile range.

        z = (x - median) / IQR
    """
    values_gw = np.asarray(
        values_gw,
        dtype=float,
    )

    median_gw = np.median(
        values_gw
    )

    q25_gw = np.quantile(
        values_gw,
        0.25,
    )

    q75_gw = np.quantile(
        values_gw,
        0.75,
    )

    iqr_gw = q75_gw - q25_gw

    if not np.isfinite(iqr_gw) or iqr_gw <= 0:
        raise ValueError(
            "Cannot standardise a sample with a non-positive IQR."
        )

    standardised_values = (
        values_gw - median_gw
    ) / iqr_gw

    return (
        standardised_values,
        median_gw,
        iqr_gw,
    )


def empirical_upper_tail(
    standardised_values: np.ndarray,
    upper_tail_fraction: float,
) -> pd.DataFrame:
    """
    Construct the empirical exceedance curve and retain only
    the requested upper-tail fraction.

    The largest observation has plotting position:

        1 / (n + 1)
    """
    values = np.asarray(
        standardised_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        raise ValueError(
            "Cannot construct an upper tail from an empty sample."
        )

    values_descending = np.sort(
        values
    )[::-1]

    n = len(
        values_descending
    )

    descending_rank = np.arange(
        1,
        n + 1,
    )

    exceedance_probability = (
        descending_rank
        / (n + 1.0)
    )

    tail_mask = (
        exceedance_probability
        <= upper_tail_fraction
    )

    return pd.DataFrame(
        {
            "standardised_severity": (
                values_descending[
                    tail_mask
                ]
            ),
            "descending_rank": (
                descending_rank[
                    tail_mask
                ]
            ),
            "exceedance_probability": (
                exceedance_probability[
                    tail_mask
                ]
            ),
        }
    )


# ============================================================
# Load data
# ============================================================

print("=" * 80)
print("Final analysis 02: standardised upper tails for primary windows")
print("=" * 80)

require_file(
    ECMWF_SEVERITY_FILE,
    "ECMWF extended-window severity file",
)

require_file(
    HANNAH_SEVERITY_FILE,
    "Hannah extended-window severity file",
)

ecmwf = pd.read_csv(
    ECMWF_SEVERITY_FILE
)

hannah = pd.read_csv(
    HANNAH_SEVERITY_FILE
)

for dataframe in [
    ecmwf,
    hannah,
]:
    if "winter_year" not in dataframe.columns:
        raise ValueError(
            "The input data must contain a winter_year column."
        )

    dataframe["winter_year"] = pd.to_numeric(
        dataframe["winter_year"],
        errors="coerce",
    )


# ============================================================
# Restrict to the common overlap period
# ============================================================

ecmwf_overlap = ecmwf[
    ecmwf["winter_year"].between(
        OVERLAP_START,
        OVERLAP_END,
    )
].copy()

hannah_overlap = hannah[
    hannah["winter_year"].between(
        OVERLAP_START,
        OVERLAP_END,
    )
].copy()

if len(ecmwf_overlap) != 875:
    warnings.warn(
        "Expected 875 ECMWF winter-member realisations "
        f"for 1982-2016, but found {len(ecmwf_overlap)}."
    )

if len(hannah_overlap) != 35:
    warnings.warn(
        "Expected 35 Hannah overlap winters "
        f"for 1982-2016, but found {len(hannah_overlap)}."
    )

print(
    f"\nECMWF overlap rows: {len(ecmwf_overlap)}"
)

print(
    f"Hannah overlap rows: {len(hannah_overlap)}"
)

print(
    f"Primary windows: {PRIMARY_WINDOWS}"
)

print(
    f"Upper-tail fraction: {UPPER_TAIL_FRACTION:.0%}"
)


# ============================================================
# Standardise and extract upper tails
# ============================================================

standardisation_rows = []
tail_rows = []
plot_records = {}

for window in PRIMARY_WINDOWS:
    column = severity_column(
        window
    )

    if column not in ecmwf_overlap.columns:
        raise ValueError(
            f"Missing ECMWF column: {column}"
        )

    if column not in hannah_overlap.columns:
        raise ValueError(
            f"Missing Hannah column: {column}"
        )

    ecmwf_gw = (
        finite_numeric_values(
            ecmwf_overlap,
            column,
        )
        / 1000.0
    )

    hannah_gw = (
        finite_numeric_values(
            hannah_overlap,
            column,
        )
        / 1000.0
    )

    (
        ecmwf_z,
        ecmwf_median_gw,
        ecmwf_iqr_gw,
    ) = robust_standardise(
        ecmwf_gw
    )

    (
        hannah_z,
        hannah_median_gw,
        hannah_iqr_gw,
    ) = robust_standardise(
        hannah_gw
    )

    ecmwf_tail = empirical_upper_tail(
        ecmwf_z,
        UPPER_TAIL_FRACTION,
    )

    hannah_tail = empirical_upper_tail(
        hannah_z,
        UPPER_TAIL_FRACTION,
    )

    standardisation_rows.extend(
        [
            {
                "window_days": window,
                "sample": "ECMWF pooled",
                "n_full": len(ecmwf_gw),
                "n_upper_tail": len(ecmwf_tail),
                "median_GW": ecmwf_median_gw,
                "IQR_GW": ecmwf_iqr_gw,
                "upper_tail_fraction": UPPER_TAIL_FRACTION,
            },
            {
                "window_days": window,
                "sample": "Hannah overlap",
                "n_full": len(hannah_gw),
                "n_upper_tail": len(hannah_tail),
                "median_GW": hannah_median_gw,
                "IQR_GW": hannah_iqr_gw,
                "upper_tail_fraction": UPPER_TAIL_FRACTION,
            },
        ]
    )

    for sample_name, tail_dataframe in [
        ("ECMWF pooled", ecmwf_tail),
        ("Hannah overlap", hannah_tail),
    ]:
        output_tail = tail_dataframe.copy()

        output_tail.insert(
            0,
            "sample",
            sample_name,
        )

        output_tail.insert(
            0,
            "window_days",
            window,
        )

        tail_rows.append(
            output_tail
        )

    plot_records[window] = {
        "ECMWF pooled": ecmwf_tail,
        "Hannah overlap": hannah_tail,
    }


# ============================================================
# Save data tables
# ============================================================

standardisation_summary = pd.DataFrame(
    standardisation_rows
).sort_values(
    [
        "window_days",
        "sample",
    ]
)

standardisation_summary.to_csv(
    OUT_STANDARDISATION_SUMMARY,
    index=False,
)

tail_data = pd.concat(
    tail_rows,
    ignore_index=True,
)

tail_data.to_csv(
    OUT_TAIL_DATA,
    index=False,
)


# ============================================================
# Figure: all primary windows, two panels per row
# ============================================================

n_windows = len(
    PRIMARY_WINDOWS
)

ncols = 2

nrows = math.ceil(
    n_windows / ncols
)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(
        13.5,
        4.2 * nrows,
    ),
    dpi=300,
)

axes = np.asarray(
    axes
).reshape(-1)

legend_handles = None
legend_labels = None

for index, window in enumerate(
    PRIMARY_WINDOWS
):
    ax = axes[
        index
    ]

    for sample_name in [
        "Hannah overlap",
        "ECMWF pooled",
    ]:
        tail = plot_records[
            window
        ][
            sample_name
        ]

        ax.plot(
            tail["standardised_severity"],
            tail["exceedance_probability"],
            marker=(
                "o"
                if sample_name == "Hannah overlap"
                else None
            ),
            markersize=3.0,
            linewidth=1.5,
            label=(
                f"{sample_name} "
                f"(tail n={len(tail)})"
            ),
        )

    ax.set_yscale(
        "log"
    )

    ax.set_xlabel(
        "Standardised severity\n"
        "(value - sample median) / sample IQR"
    )

    ax.set_ylabel(
        "Empirical exceedance probability"
    )

    ax.set_title(
        f"{window}-day severity"
    )

    ax.set_ylim(
        8e-4,
        UPPER_TAIL_FRACTION * 1.15,
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    if legend_handles is None:
        (
            legend_handles,
            legend_labels,
        ) = ax.get_legend_handles_labels()

# Hide the unused eighth panel.
for index in range(
    n_windows,
    len(axes),
):
    axes[index].axis(
        "off"
    )

fig.suptitle(
    "Standardised upper-tail comparison across primary windows\n"
    "Highest 20% of each distribution; "
    "ECMWF pooled versus Hannah overlap, 1982-2016",
    fontsize=15,
    y=0.995,
)

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=2,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(
        0.5,
        0.012,
    ),
)

fig.tight_layout(
    rect=[
        0,
        0.045,
        1,
        0.965,
    ]
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
# Print concise summary
# ============================================================

print("\nStandardisation summary:")
print(
    standardisation_summary[
        [
            "window_days",
            "sample",
            "n_full",
            "n_upper_tail",
            "median_GW",
            "IQR_GW",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved outputs:")
print(OUT_STANDARDISATION_SUMMARY)
print(OUT_TAIL_DATA)
print(OUT_FIGURE_PNG)
print(OUT_FIGURE_PDF)

print("\nDone.")
