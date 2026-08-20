"""
Create separate stationary-versus-shape-trend GEV diagnostic figures for
pooled SEAS5 and the mean-aligned historical sample at all seven primary
averaging windows.

The layout, size and main colour scheme deliberately follow the stationary
diagnostic figures produced by the user's Final analysis 06:

    top left     probability plot
    top right    quantile plot
    bottom left  return-level plot
    bottom right density plot and parameter summary

The stationary fit is shown in Matplotlib blue, as in the earlier figures.
The shape-trend fit is added in orange with a dashed line (or x markers).
The stationary winter-year bootstrap return-level interval is retained as a
light blue band. No interval is drawn for the shape-trend fit because that
model was not cluster-bootstrapped in the original sensitivity analysis.

This is a plot-only script. It does not refit either model.

Run from the project root:

    /usr/bin/python3 scripts/36_final_17_nonstationary_gev_diagnostics.py

This creates 14 figures: seven windows for each of the two samples.

To create selected windows only:

    /usr/bin/python3 scripts/36_final_17_nonstationary_gev_diagnostics.py \
        --windows 28 84

The script is compatible with Python 3.8 and later.
"""

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import genextreme


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

PRIMARY_WINDOWS = (1, 7, 14, 21, 28, 56, 84)
OVERLAP_START = 1982
OVERLAP_END = 2016
YEAR_SCALE = 10.0

RETURN_PERIOD_MIN = 1.05
RETURN_PERIOD_MAX = 200.0
RETURN_PERIOD_POINTS = 280
MARGINAL_GRID_POINTS = 20000
FIGURE_DPI = 300

FONT_BASE = 14.0
FONT_AXIS_LABEL = 15.0
FONT_TICK = 13.0
FONT_TITLE = 15.0
FONT_LEGEND = 11.5
FONT_SUPTITLE = 17.0
FONT_PARAMETER_BOX = 10.5

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

SAMPLE_ECMWF = "ECMWF pooled"
SAMPLE_HISTORICAL = "Hannah overlap shifted"
SAMPLES = (SAMPLE_ECMWF, SAMPLE_HISTORICAL)

DISPLAY_NAMES = {
    SAMPLE_ECMWF: "ECMWF pooled",
    SAMPLE_HISTORICAL: "Hannah overlap shifted",
}

SAMPLE_SLUGS = {
    SAMPLE_ECMWF: "ecmwf_pooled",
    SAMPLE_HISTORICAL: "hannah_overlap_shifted",
}

# Match the stationary plots: the original fit and empirical plotting
# positions use the first Matplotlib colour. The new shape-trend fit is orange.
STATIONARY_COLOUR = "#1f77b4"
SHAPE_TREND_COLOUR = "#ff7f0e"
REFERENCE_COLOUR = "0.35"


# -----------------------------------------------------------------------------
# File and data helpers
# -----------------------------------------------------------------------------

def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Plot stationary versus shape-trend GEV diagnostics for both "
            "samples at the primary averaging windows."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Project outputs directory. If omitted, OUTPUT_DIR is imported "
            "from config.py."
        ),
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(PRIMARY_WINDOWS),
        help=(
            "Windows to plot. The default is all seven primary windows: "
            "1 7 14 21 28 56 84."
        ),
    )
    parser.add_argument("--comparison-csv", type=Path, default=None)
    parser.add_argument("--bootstrap-csv", type=Path, default=None)
    parser.add_argument("--ecmwf-severity-csv", type=Path, default=None)
    parser.add_argument("--historical-severity-csv", type=Path, default=None)
    return parser.parse_args()


def resolve_output_dir(cli_output_dir):
    # type: (Optional[Path]) -> Path
    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve()

    try:
        from config import OUTPUT_DIR  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import OUTPUT_DIR from config.py. Run this script "
            "from the project root or pass --output-dir explicitly."
        ) from exc

    return Path(OUTPUT_DIR).expanduser().resolve()


def require_file(path, label):
    # type: (Path, str) -> Path
    if not path.exists():
        raise FileNotFoundError("Missing {0}:\n{1}".format(label, path))
    return path


def resolve_input_file(explicit_path, expected_path, label):
    # type: (Optional[Path], Path, str) -> Path
    if explicit_path is not None:
        return require_file(explicit_path.expanduser().resolve(), label)
    return require_file(expected_path, label)


def require_columns(frame, columns, label):
    # type: (pd.DataFrame, Iterable[str], str) -> None
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(
            "{0} is missing columns:\n{1}".format(
                label,
                "\n".join("  - {0}".format(column) for column in missing),
            )
        )


def severity_column(window):
    # type: (int) -> str
    return "max_{0}d_mean_demand_MW".format(window)


def clean_values_and_years(values, years):
    # type: (Iterable[float], Iterable[float]) -> Tuple[np.ndarray, np.ndarray]
    value_array = np.asarray(values, dtype=float)
    year_array = np.asarray(years, dtype=float)
    keep = np.isfinite(value_array) & np.isfinite(year_array)
    return value_array[keep], year_array[keep]


def plotting_positions(n):
    # type: (int) -> np.ndarray
    return (np.arange(1, n + 1) - 0.5) / n


def empirical_exceedance(values):
    # type: (Iterable[float]) -> Tuple[np.ndarray, np.ndarray]
    descending = np.sort(np.asarray(values, dtype=float))[::-1]
    ranks = np.arange(1, len(descending) + 1)
    exceedance = ranks / (len(descending) + 1.0)
    return descending, exceedance


def get_fit_row(comparison, window, sample):
    # type: (pd.DataFrame, int, str) -> pd.Series
    rows = comparison.loc[
        (comparison["window_days"] == window)
        & (comparison["sample"] == sample)
    ]
    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one fit row for window={0}, sample={1!r}; "
            "found {2}.".format(window, sample, len(rows))
        )
    return rows.iloc[0]


def get_bootstrap_rows(parameter_samples, window, sample):
    # type: (pd.DataFrame, int, str) -> pd.DataFrame
    rows = parameter_samples.loc[
        (parameter_samples["window_days"] == window)
        & (parameter_samples["sample"] == sample)
    ].copy()
    return rows.loc[
        np.isfinite(rows["scipy_c"])
        & np.isfinite(rows["loc_GW"])
        & np.isfinite(rows["scale_GW"])
        & (rows["scale_GW"] > 0)
    ].copy()


# -----------------------------------------------------------------------------
# Shape-trend GEV helpers
# -----------------------------------------------------------------------------

def conditional_shape(years, fit):
    # type: (Iterable[float], pd.Series) -> np.ndarray
    year_array = np.asarray(years, dtype=float)
    time_covariate = (year_array - OVERLAP_START) / YEAR_SCALE
    return (
        float(fit["xi_1982"])
        + float(fit["xi_change_per_decade"]) * time_covariate
    )


def pooled_marginal_cdf(x_values, years, fit):
    # type: (Iterable[float], Iterable[float], pd.Series) -> np.ndarray
    x_array = np.asarray(x_values, dtype=float)
    unique_years = np.sort(np.unique(np.asarray(years, dtype=float)))
    cdf_rows = []

    for xi_value in conditional_shape(unique_years, fit):
        cdf_rows.append(
            genextreme.cdf(
                x_array,
                c=-float(xi_value),
                loc=float(fit["nonstationary_loc_GW"]),
                scale=float(fit["nonstationary_scale_GW"]),
            )
        )

    return np.mean(np.vstack(cdf_rows), axis=0)


def pooled_marginal_pdf(x_values, years, fit):
    # type: (Iterable[float], Iterable[float], pd.Series) -> np.ndarray
    x_array = np.asarray(x_values, dtype=float)
    unique_years = np.sort(np.unique(np.asarray(years, dtype=float)))
    pdf_rows = []

    for xi_value in conditional_shape(unique_years, fit):
        pdf_rows.append(
            genextreme.pdf(
                x_array,
                c=-float(xi_value),
                loc=float(fit["nonstationary_loc_GW"]),
                scale=float(fit["nonstationary_scale_GW"]),
            )
        )

    return np.mean(np.vstack(pdf_rows), axis=0)


def pooled_marginal_ppf(probabilities, values, years, fit):
    # type: (Iterable[float], np.ndarray, np.ndarray, pd.Series) -> np.ndarray
    probability_array = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-8,
        1.0 - 1e-8,
    )

    sample_sd = float(np.std(values, ddof=1))
    if not np.isfinite(sample_sd) or sample_sd <= 0:
        sample_sd = 1.0

    lower = float(np.min(values) - 3.0 * sample_sd)
    upper = float(np.max(values) + 3.0 * sample_sd)
    target_lower = max(1e-10, float(np.min(probability_array)) / 10.0)
    target_upper = min(
        1.0 - 1e-10,
        1.0 - (1.0 - float(np.max(probability_array))) / 10.0,
    )

    for _ in range(12):
        boundary_cdf = pooled_marginal_cdf([lower, upper], years, fit)
        lower_ok = boundary_cdf[0] <= target_lower
        upper_ok = boundary_cdf[1] >= target_upper
        if lower_ok and upper_ok:
            break
        width = upper - lower
        if not lower_ok:
            lower -= width
        if not upper_ok:
            upper += width

    x_grid = np.linspace(lower, upper, MARGINAL_GRID_POINTS)
    cdf_grid = pooled_marginal_cdf(x_grid, years, fit)
    cdf_grid = np.maximum.accumulate(cdf_grid)
    unique_cdf, unique_indices = np.unique(cdf_grid, return_index=True)
    unique_x = x_grid[unique_indices]

    if unique_cdf[0] > np.min(probability_array):
        raise RuntimeError("The marginal grid does not reach the lower probability.")
    if unique_cdf[-1] < np.max(probability_array):
        raise RuntimeError("The marginal grid does not reach the upper probability.")

    return np.interp(probability_array, unique_cdf, unique_x)


def conditional_pit(values, years, fit):
    # type: (np.ndarray, np.ndarray, pd.Series) -> Tuple[np.ndarray, np.ndarray]
    stationary_pit = genextreme.cdf(
        values,
        c=-float(fit["stationary_xi"]),
        loc=float(fit["stationary_loc_GW"]),
        scale=float(fit["stationary_scale_GW"]),
    )

    shape_trend_pit = genextreme.cdf(
        values,
        c=-conditional_shape(years, fit),
        loc=float(fit["nonstationary_loc_GW"]),
        scale=float(fit["nonstationary_scale_GW"]),
    )

    return (
        np.clip(stationary_pit, 1e-10, 1.0 - 1e-10),
        np.clip(shape_trend_pit, 1e-10, 1.0 - 1e-10),
    )


def stationary_return_level_band(bootstrap_rows, return_probabilities):
    # type: (pd.DataFrame, np.ndarray) -> Tuple[np.ndarray, np.ndarray]
    curves = []

    for row in bootstrap_rows.itertuples(index=False):
        curve = genextreme.ppf(
            return_probabilities,
            float(row.scipy_c),
            loc=float(row.loc_GW),
            scale=float(row.scale_GW),
        )
        if np.all(np.isfinite(curve)):
            curves.append(curve)

    if not curves:
        empty = np.full(len(return_probabilities), np.nan)
        return empty, empty

    matrix = np.vstack(curves)
    return (
        np.quantile(matrix, 0.025, axis=0),
        np.quantile(matrix, 0.975, axis=0),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_single_sample_figure(
    window,
    sample,
    values,
    years,
    fit,
    bootstrap_rows,
    output_path,
):
    values, years = clean_values_and_years(values, years)
    n = len(values)
    marker_size = 18 if sample == SAMPLE_ECMWF else 42

    observed = np.sort(values)
    empirical_probability = plotting_positions(n)
    stationary_pit, shape_trend_pit = conditional_pit(values, years, fit)

    return_period_grid = np.geomspace(
        RETURN_PERIOD_MIN,
        RETURN_PERIOD_MAX,
        RETURN_PERIOD_POINTS,
    )
    return_probabilities = 1.0 - 1.0 / return_period_grid

    all_probabilities = np.concatenate(
        [empirical_probability, return_probabilities]
    )
    all_shape_trend_quantiles = pooled_marginal_ppf(
        all_probabilities,
        values,
        years,
        fit,
    )
    shape_trend_quantiles = all_shape_trend_quantiles[:n]
    shape_trend_return_levels = all_shape_trend_quantiles[n:]

    stationary_quantiles = genextreme.ppf(
        empirical_probability,
        c=-float(fit["stationary_xi"]),
        loc=float(fit["stationary_loc_GW"]),
        scale=float(fit["stationary_scale_GW"]),
    )
    stationary_return_levels = genextreme.ppf(
        return_probabilities,
        c=-float(fit["stationary_xi"]),
        loc=float(fit["stationary_loc_GW"]),
        scale=float(fit["stationary_scale_GW"]),
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.5, 11.0),
        dpi=FIGURE_DPI,
    )

    # Probability plot ---------------------------------------------------------
    ax = axes[0, 0]
    ax.scatter(
        empirical_probability,
        np.sort(stationary_pit),
        s=marker_size,
        color=STATIONARY_COLOUR,
        alpha=0.70,
        label="Stationary GEV",
    )
    ax.scatter(
        empirical_probability,
        np.sort(shape_trend_pit),
        s=marker_size,
        color=SHAPE_TREND_COLOUR,
        marker="x",
        linewidths=1.1,
        alpha=0.78,
        label="Shape-trend GEV",
    )
    ax.plot([0, 1], [0, 1], "--", color=REFERENCE_COLOUR, linewidth=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Empirical cumulative probability")
    ax.set_ylabel("Fitted GEV cumulative probability")
    ax.set_title("Probability plot")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=FONT_LEGEND)

    # Quantile plot ------------------------------------------------------------
    ax = axes[0, 1]
    ax.scatter(
        stationary_quantiles,
        observed,
        s=marker_size,
        color=STATIONARY_COLOUR,
        alpha=0.70,
    )
    ax.scatter(
        shape_trend_quantiles,
        observed,
        s=marker_size,
        color=SHAPE_TREND_COLOUR,
        marker="x",
        linewidths=1.1,
        alpha=0.78,
    )
    combined_quantiles = np.concatenate(
        [stationary_quantiles, shape_trend_quantiles, observed]
    )
    lower = float(np.nanmin(combined_quantiles))
    upper = float(np.nanmax(combined_quantiles))
    padding = max(0.03 * (upper - lower), 0.02)
    limits = (lower - padding, upper + padding)
    ax.plot(limits, limits, "--", color=REFERENCE_COLOUR, linewidth=1.0)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Fitted GEV quantile (GW)")
    ax.set_ylabel("Observed winter maximum (GW)")
    ax.set_title("Quantile plot")
    ax.grid(True, alpha=0.28)

    # Return-level plot --------------------------------------------------------
    ax = axes[1, 0]
    lower_band, upper_band = stationary_return_level_band(
        bootstrap_rows,
        return_probabilities,
    )
    if np.all(np.isfinite(lower_band)) and np.all(np.isfinite(upper_band)):
        ax.fill_between(
            return_period_grid,
            lower_band,
            upper_band,
            color=STATIONARY_COLOUR,
            alpha=0.18,
            label="Stationary 95% winter-year bootstrap interval",
        )
    ax.plot(
        return_period_grid,
        stationary_return_levels,
        color=STATIONARY_COLOUR,
        linewidth=1.9,
        label="Fitted stationary GEV",
    )
    ax.plot(
        return_period_grid,
        shape_trend_return_levels,
        color=SHAPE_TREND_COLOUR,
        linestyle="--",
        linewidth=1.9,
        label="Fitted shape-trend GEV (pooled marginal)",
    )
    descending, exceedance = empirical_exceedance(values)
    empirical_return_period = 1.0 / exceedance
    ax.scatter(
        empirical_return_period,
        descending,
        s=marker_size,
        color=STATIONARY_COLOUR,
        alpha=0.70,
        label="Empirical plotting positions",
        zorder=3,
    )
    ax.set_xscale("log")
    ax.set_xlim(RETURN_PERIOD_MIN, RETURN_PERIOD_MAX)
    ax.set_xlabel("Return period (winter blocks, log scale)")
    ax.set_ylabel("Return level (GW)")
    ax.set_title("Return-level plot")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=FONT_LEGEND)

    # Density plot and parameter box ------------------------------------------
    ax = axes[1, 1]
    ax.hist(
        values,
        bins="auto",
        density=True,
        color=STATIONARY_COLOUR,
        alpha=0.42,
        label="Observed winter maxima",
    )
    sample_sd = float(np.std(values, ddof=1))
    x_grid = np.linspace(
        float(np.min(values) - 0.25 * sample_sd),
        float(np.max(values) + 0.25 * sample_sd),
        600,
    )
    stationary_density = genextreme.pdf(
        x_grid,
        c=-float(fit["stationary_xi"]),
        loc=float(fit["stationary_loc_GW"]),
        scale=float(fit["stationary_scale_GW"]),
    )
    shape_trend_density = pooled_marginal_pdf(x_grid, years, fit)
    ax.plot(
        x_grid,
        stationary_density,
        color=STATIONARY_COLOUR,
        linewidth=1.9,
        label="Fitted stationary GEV density",
    )
    ax.plot(
        x_grid,
        shape_trend_density,
        color=SHAPE_TREND_COLOUR,
        linestyle="--",
        linewidth=1.9,
        label="Fitted shape-trend pooled density",
    )
    rug_height = 0.015 * max(1e-6, float(np.nanmax(stationary_density)))
    ax.vlines(
        values,
        0,
        rug_height,
        color=STATIONARY_COLOUR,
        linewidth=0.5,
        alpha=0.55,
    )

    xi_boot = bootstrap_rows["xi"].dropna().to_numpy(dtype=float)
    if len(xi_boot) > 0:
        xi_ci = "[{0:.3f}, {1:.3f}]".format(
            np.quantile(xi_boot, 0.025),
            np.quantile(xi_boot, 0.975),
        )
    else:
        xi_ci = "not available"

    parameter_text = (
        "Stationary GEV:\n"
        "mu = {0:.3f} GW, sigma = {1:.3f} GW\n"
        "xi = {2:.3f}, 95% bootstrap CI {3}\n\n"
        "Shape-trend GEV:\n"
        "mu = {4:.3f} GW, sigma = {5:.3f} GW\n"
        "xi: {6:.3f} (1982) to {7:.3f} (2016)\n"
        "Delta AIC = {8:.2f}"
    ).format(
        float(fit["stationary_loc_GW"]),
        float(fit["stationary_scale_GW"]),
        float(fit["stationary_xi"]),
        xi_ci,
        float(fit["nonstationary_loc_GW"]),
        float(fit["nonstationary_scale_GW"]),
        float(fit["xi_1982"]),
        float(fit["xi_2016"]),
        float(fit["delta_aic_nonstationary_minus_stationary"]),
    )
    ax.text(
        0.98,
        0.95,
        parameter_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_PARAMETER_BOX,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    ax.set_xlabel("Winter-maximum severity (GW)")
    ax.set_ylabel("Density")
    ax.set_title("Density plot")
    ax.grid(True, alpha=0.28)
    ax.legend(fontsize=10.8, loc="lower left")

    fig.suptitle(
        (
            "Stationary versus shape-trend GEV diagnostics: "
            "{0}, {1}-day severity\n"
            "Nov 8-Mar 31, winters 1982-2016, n={2}"
        ).format(DISPLAY_NAMES[sample], window, n),
        fontsize=FONT_SUPTITLE,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.935], h_pad=2.7, w_pad=2.3)
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    print("Saved {0}".format(output_path))
    print("Saved {0}".format(output_path.with_suffix(".pdf")))


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_arguments()
    output_dir = resolve_output_dir(args.output_dir)

    invalid_windows = sorted(set(args.windows) - set(PRIMARY_WINDOWS))
    if invalid_windows:
        raise ValueError(
            "Windows must be selected from {0}; invalid values: {1}.".format(
                PRIMARY_WINDOWS,
                invalid_windows,
            )
        )
    selected_windows = [
        window for window in PRIMARY_WINDOWS if window in set(args.windows)
    ]

    severity_dir = output_dir / "severity"
    stationary_dir = (
        output_dir
        / "final_analysis_clean"
        / "06_stationary_gev_primary_windows"
    )
    nonstationary_dir = (
        output_dir
        / "final_analysis_clean"
        / "07_nonstationary_shape_trend_sensitivity"
    )

    ecmwf_file = resolve_input_file(
        args.ecmwf_severity_csv,
        severity_dir
        / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv",
        "ECMWF severity summary",
    )
    historical_file = resolve_input_file(
        args.historical_severity_csv,
        severity_dir / "hannah_severity_summary_Nov08_extended_windows.csv",
        "Hannah severity summary",
    )
    comparison_file = resolve_input_file(
        args.comparison_csv,
        nonstationary_dir
        / "stationary_vs_nonstationary_shape_trend_comparison.csv",
        "stationary versus shape-trend comparison table",
    )
    bootstrap_file = resolve_input_file(
        args.bootstrap_csv,
        stationary_dir / "stationary_gev_bootstrap_parameter_samples.csv",
        "stationary GEV bootstrap parameter samples",
    )

    ecmwf = pd.read_csv(ecmwf_file)
    historical = pd.read_csv(historical_file)
    comparison = pd.read_csv(comparison_file)
    parameter_samples = pd.read_csv(bootstrap_file)

    for frame, label in ((ecmwf, "ECMWF"), (historical, "Hannah")):
        require_columns(frame, ["winter_year"], label + " severity summary")
        frame["winter_year"] = pd.to_numeric(
            frame["winter_year"], errors="coerce"
        )

    require_columns(
        comparison,
        [
            "window_days",
            "sample",
            "mean_shift_GW",
            "stationary_xi",
            "stationary_loc_GW",
            "stationary_scale_GW",
            "nonstationary_loc_GW",
            "nonstationary_scale_GW",
            "xi_1982",
            "xi_change_per_decade",
            "xi_2016",
            "delta_aic_nonstationary_minus_stationary",
        ],
        "Stationary versus shape-trend comparison table",
    )
    require_columns(
        parameter_samples,
        ["window_days", "sample", "xi", "scipy_c", "loc_GW", "scale_GW"],
        "Stationary GEV bootstrap parameter samples",
    )

    ecmwf = ecmwf.loc[
        ecmwf["winter_year"].between(OVERLAP_START, OVERLAP_END)
    ].copy()
    historical = historical.loc[
        historical["winter_year"].between(OVERLAP_START, OVERLAP_END)
    ].copy()

    output_root = (
        output_dir
        / "final_analysis_clean"
        / "17_nonstationary_gev_diagnostics"
    )
    sample_directories = {
        SAMPLE_ECMWF: output_root / SAMPLE_SLUGS[SAMPLE_ECMWF],
        SAMPLE_HISTORICAL: output_root / SAMPLE_SLUGS[SAMPLE_HISTORICAL],
    }
    for directory in sample_directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    print("Input ECMWF severity CSV:\n{0}".format(ecmwf_file))
    print("\nInput Hannah severity CSV:\n{0}".format(historical_file))
    print("\nInput comparison CSV:\n{0}".format(comparison_file))
    print("\nInput stationary bootstrap CSV:\n{0}".format(bootstrap_file))
    print("\nWindows plotted: {0}".format(selected_windows))
    print("Samples plotted: {0}\n".format(list(SAMPLES)))

    figure_count = 0

    for window in selected_windows:
        column = severity_column(window)
        require_columns(ecmwf, [column], "ECMWF severity summary")
        require_columns(historical, [column], "Hannah severity summary")

        ecmwf_values, ecmwf_years = clean_values_and_years(
            pd.to_numeric(ecmwf[column], errors="coerce") / 1000.0,
            ecmwf["winter_year"],
        )
        historical_raw, historical_years = clean_values_and_years(
            pd.to_numeric(historical[column], errors="coerce") / 1000.0,
            historical["winter_year"],
        )

        data = {
            SAMPLE_ECMWF: (ecmwf_values, ecmwf_years),
            SAMPLE_HISTORICAL: (historical_raw, historical_years),
        }

        for sample in SAMPLES:
            fit = get_fit_row(comparison, window, sample)
            values, years = data[sample]

            if sample == SAMPLE_HISTORICAL:
                values = values + float(fit["mean_shift_GW"])

            expected_n = 875 if sample == SAMPLE_ECMWF else 35
            if len(values) != expected_n:
                raise ValueError(
                    "Expected {0} values for {1} at {2} days, found {3}.".format(
                        expected_n,
                        sample,
                        window,
                        len(values),
                    )
                )

            bootstrap_rows = get_bootstrap_rows(
                parameter_samples,
                window,
                sample,
            )
            output_path = sample_directories[sample] / (
                "nonstationary_gev_diagnostics_{0}day_{1}.png".format(
                    window,
                    SAMPLE_SLUGS[sample],
                )
            )
            make_single_sample_figure(
                window,
                sample,
                values,
                years,
                fit,
                bootstrap_rows,
                output_path,
            )
            figure_count += 1

    print("\nCreated {0} PNG figures and {0} PDF figures.".format(figure_count))
    print("Output directory:\n{0}".format(output_root))


if __name__ == "__main__":
    main()