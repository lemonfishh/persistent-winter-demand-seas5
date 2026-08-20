from pathlib import Path
import warnings
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config import OUTPUT_DIR


# ============================================================
# Final analysis 01
# Mean-shifted empirical comparison for the primary windows
#
# Primary windows:
#   1, 7, 14, 21, 28, 56 and 84 days
#
# Purpose:
#   1. Calculate a duration-specific mean shift:
#
#          shift_w = mean(ECMWF_w) - mean(Hannah_overlap_w)
#
#   2. Apply the shift to Hannah overlap values.
#   3. Compare mean-shifted Hannah with pooled ECMWF on the
#      original GW scale using empirical exceedance curves.
#   4. Produce a multi-panel figure for ALL primary windows,
#      arranged as two panels per row.
#
# Important:
#   - Mean shifting removes only the location difference.
#   - It does not force the variance or tail shape to agree.
# ============================================================


# ============================================================
# Settings
# ============================================================

PRIMARY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]

OVERLAP_START = 1982
OVERLAP_END = 2016

# Winter year 1962 corresponds to winter 1962/63.
HISTORICAL_BENCHMARK_WINTER = 1963


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
    / "01_mean_shifted_primary_comparison"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Output files
# ============================================================

OUT_SHIFT_SUMMARY = (
    OUT_DIR
    / "primary_windows_mean_shift_summary.csv"
)

OUT_EXCEEDANCE_DATA = (
    OUT_DIR
    / "primary_windows_mean_shifted_exceedance_data.csv"
)

OUT_ALL_WINDOWS_FIGURE_PNG = (
    OUT_DIR
    / "all_primary_windows_mean_shifted_exceedance.png"
)

OUT_ALL_WINDOWS_FIGURE_PDF = (
    OUT_DIR
    / "all_primary_windows_mean_shifted_exceedance.pdf"
)


# ============================================================
# Helper functions
# ============================================================

def require_file(path: Path, label: str) -> None:
    """Raise a clear error if an input file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )


def severity_column(window: int) -> str:
    """Return the severity column for a given duration."""
    return f"max_{window}d_mean_demand_MW"


def finite_numeric_values(
    dataframe: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Read one column as a finite NumPy array."""
    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    ).to_numpy(dtype=float)

    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(
            f"No finite values found in column: {column}"
        )

    return values


def empirical_exceedance(values_gw: np.ndarray) -> pd.DataFrame:
    """
    Return values in ascending order with empirical exceedance
    plotting positions.

    For the largest observation:
        p = 1 / (n + 1)

    For the smallest observation:
        p = n / (n + 1)
    """
    values_gw = np.asarray(values_gw, dtype=float)
    values_gw = values_gw[np.isfinite(values_gw)]

    if len(values_gw) == 0:
        raise ValueError(
            "Cannot calculate exceedance positions for an empty sample."
        )

    sorted_values = np.sort(values_gw)
    n = len(sorted_values)

    ascending_rank = np.arange(1, n + 1)

    exceedance_probability = (
        n - ascending_rank + 1
    ) / (n + 1.0)

    descending_rank = (
        n - ascending_rank + 1
    )

    return pd.DataFrame(
        {
            "severity_GW": sorted_values,
            "ascending_rank": ascending_rank,
            "descending_rank": descending_rank,
            "exceedance_probability": exceedance_probability,
        }
    )


def sample_summary(values_gw: np.ndarray) -> dict:
    """Return compact descriptive statistics."""
    values_gw = np.asarray(values_gw, dtype=float)

    q25 = np.quantile(values_gw, 0.25)
    q75 = np.quantile(values_gw, 0.75)

    return {
        "n": len(values_gw),
        "mean_GW": np.mean(values_gw),
        "sd_GW": np.std(values_gw, ddof=1),
        "median_GW": np.median(values_gw),
        "iqr_GW": q75 - q25,
        "minimum_GW": np.min(values_gw),
        "maximum_GW": np.max(values_gw),
    }


# ============================================================
# Load data
# ============================================================

print("=" * 80)
print("Final analysis 01: mean-shifted primary-window comparison")
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

for dataframe in [ecmwf, hannah]:
    if "winter_year" not in dataframe.columns:
        raise ValueError(
            "The input data must contain a winter_year column."
        )

    dataframe["winter_year"] = pd.to_numeric(
        dataframe["winter_year"],
        errors="coerce",
    )


# ============================================================
# Restrict the main comparison to the common period
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

benchmark_rows = hannah[
    hannah["winter_year"] == HISTORICAL_BENCHMARK_WINTER
].copy()

if len(benchmark_rows) == 0:
    warnings.warn(
        "Hannah winter 1962/63 was not found. "
        "The benchmark line will be omitted."
    )
elif len(benchmark_rows) > 1:
    warnings.warn(
        "More than one Hannah row was found for winter 1962/63. "
        "The first row will be used."
    )

print(f"\nECMWF overlap rows: {len(ecmwf_overlap)}")
print(f"Hannah overlap rows: {len(hannah_overlap)}")
print(f"Primary windows: {PRIMARY_WINDOWS}")


# ============================================================
# Calculate duration-specific mean shifts
# ============================================================

shift_summary_rows = []
exceedance_rows = []

plot_records = {}

for window in PRIMARY_WINDOWS:
    column = severity_column(window)

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

    hannah_raw_gw = (
        finite_numeric_values(
            hannah_overlap,
            column,
        )
        / 1000.0
    )

    mean_shift_gw = (
        np.mean(ecmwf_gw)
        - np.mean(hannah_raw_gw)
    )

    hannah_shifted_gw = (
        hannah_raw_gw
        + mean_shift_gw
    )

    # --------------------------------------------------------
    # Shifted 1962/63 historical benchmark
    # --------------------------------------------------------

    benchmark_raw_gw = np.nan
    benchmark_shifted_gw = np.nan

    if len(benchmark_rows) > 0:
        benchmark_value_mw = pd.to_numeric(
            benchmark_rows.iloc[0][column],
            errors="coerce",
        )

        if np.isfinite(benchmark_value_mw):
            benchmark_raw_gw = (
                float(benchmark_value_mw)
                / 1000.0
            )

            benchmark_shifted_gw = (
                benchmark_raw_gw
                + mean_shift_gw
            )

    # --------------------------------------------------------
    # Save summary statistics
    # --------------------------------------------------------

    e_summary = sample_summary(ecmwf_gw)
    h_raw_summary = sample_summary(hannah_raw_gw)
    h_shift_summary = sample_summary(hannah_shifted_gw)

    shift_summary_rows.append(
        {
            "window_days": window,

            "ecmwf_n": e_summary["n"],
            "hannah_overlap_n": h_raw_summary["n"],

            "ecmwf_mean_GW": e_summary["mean_GW"],
            "hannah_raw_mean_GW": h_raw_summary["mean_GW"],
            "mean_shift_GW": mean_shift_gw,
            "hannah_shifted_mean_GW": h_shift_summary["mean_GW"],

            "ecmwf_sd_GW": e_summary["sd_GW"],
            "hannah_raw_sd_GW": h_raw_summary["sd_GW"],
            "hannah_shifted_sd_GW": h_shift_summary["sd_GW"],

            "ecmwf_median_GW": e_summary["median_GW"],
            "hannah_raw_median_GW": h_raw_summary["median_GW"],
            "hannah_shifted_median_GW": h_shift_summary["median_GW"],

            "ecmwf_iqr_GW": e_summary["iqr_GW"],
            "hannah_raw_iqr_GW": h_raw_summary["iqr_GW"],
            "hannah_shifted_iqr_GW": h_shift_summary["iqr_GW"],

            "ecmwf_maximum_GW": e_summary["maximum_GW"],
            "hannah_raw_maximum_GW": h_raw_summary["maximum_GW"],
            "hannah_shifted_maximum_GW":
                h_shift_summary["maximum_GW"],

            "hannah_1962_63_raw_GW": benchmark_raw_gw,
            "hannah_1962_63_shifted_GW":
                benchmark_shifted_gw,
        }
    )

    # --------------------------------------------------------
    # Save empirical exceedance data for both samples
    # --------------------------------------------------------

    for sample_name, values_gw in [
        ("Mean-shifted Hannah overlap", hannah_shifted_gw),
        ("ECMWF pooled", ecmwf_gw),
    ]:
        curve = empirical_exceedance(values_gw)

        curve.insert(
            0,
            "sample",
            sample_name,
        )

        curve.insert(
            0,
            "window_days",
            window,
        )

        exceedance_rows.append(curve)

    plot_records[window] = {
        "ecmwf_gw": ecmwf_gw,
        "hannah_shifted_gw": hannah_shifted_gw,
        "benchmark_shifted_gw": benchmark_shifted_gw,
    }


# ============================================================
# Save tables
# ============================================================

shift_summary = pd.DataFrame(
    shift_summary_rows
).sort_values(
    "window_days"
)

shift_summary.to_csv(
    OUT_SHIFT_SUMMARY,
    index=False,
)

exceedance_data = pd.concat(
    exceedance_rows,
    ignore_index=True,
)

exceedance_data.to_csv(
    OUT_EXCEEDANCE_DATA,
    index=False,
)


# ============================================================
# Figure: all primary windows, 2 panels per row
# ============================================================

n_windows = len(PRIMARY_WINDOWS)
ncols = 2
nrows = math.ceil(n_windows / ncols)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(13.5, 4.2 * nrows),
    dpi=300,
)

axes = np.array(axes).reshape(-1)

legend_handles = None
legend_labels = None

for i, window in enumerate(PRIMARY_WINDOWS):
    ax = axes[i]
    record = plot_records[window]

    hannah_curve = empirical_exceedance(
        record["hannah_shifted_gw"]
    )

    ecmwf_curve = empirical_exceedance(
        record["ecmwf_gw"]
    )

    ax.step(
        hannah_curve["severity_GW"],
        hannah_curve["exceedance_probability"],
        where="post",
        linewidth=1.5,
        marker="o",
        markersize=2.8,
        markevery=max(1, len(hannah_curve) // 18),
        label=(
            "Mean-shifted Hannah overlap "
            f"(n={len(hannah_curve)})"
        ),
    )

    ax.step(
        ecmwf_curve["severity_GW"],
        ecmwf_curve["exceedance_probability"],
        where="post",
        linewidth=1.5,
        label=(
            "ECMWF pooled "
            f"(n={len(ecmwf_curve)})"
        ),
    )

    benchmark = record["benchmark_shifted_gw"]

    if np.isfinite(benchmark):
        ax.axvline(
            benchmark,
            linestyle=":",
            linewidth=1.4,
            label=(
                "Mean-shifted Hannah 1962/63 benchmark"
            ),
        )

    ax.set_yscale("log")

    ax.set_xlabel(
        f"Maximum {window}-day mean demand (GW)"
    )

    ax.set_ylabel(
        "Empirical exceedance probability"
    )

    ax.set_title(
        f"{window}-day severity"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.25,
    )

    ax.set_ylim(
        8e-4,
        1.15,
    )

    if legend_handles is None:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

# Hide any unused panel
for j in range(n_windows, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    "Mean-shifted empirical severity comparison across primary windows\n"
    "Hannah overlap and pooled ECMWF, 1982-2016; "
    "Nov 8-Mar 31 analysis window",
    fontsize=15,
    y=0.995,
)

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=3,
    fontsize=9,
    frameon=True,
    bbox_to_anchor=(0.5, 0.01),
)

fig.tight_layout(
    rect=[0, 0.04, 1, 0.965]
)

fig.savefig(
    OUT_ALL_WINDOWS_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_ALL_WINDOWS_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# Print concise results
# ============================================================

print("\nDuration-specific mean shifts:")
print(
    shift_summary[
        [
            "window_days",
            "ecmwf_mean_GW",
            "hannah_raw_mean_GW",
            "mean_shift_GW",
            "hannah_shifted_mean_GW",
            "ecmwf_maximum_GW",
            "hannah_shifted_maximum_GW",
            "hannah_1962_63_shifted_GW",
        ]
    ].to_string(index=False)
)

print("\nSaved outputs:")
print(OUT_SHIFT_SUMMARY)
print(OUT_EXCEEDANCE_DATA)
print(OUT_ALL_WINDOWS_FIGURE_PNG)
print(OUT_ALL_WINDOWS_FIGURE_PDF)

print("\nDone.")