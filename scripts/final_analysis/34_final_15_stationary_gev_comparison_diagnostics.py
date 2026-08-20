from __future__ import annotations

"""
Create comparison diagnostics for the stationary GEV fits at all seven
primary averaging windows: 1, 7, 14, 21, 28, 56 and 84 days.

This is a plot-only script. It does not refit the GEV models and does not
rerun the winter-year cluster bootstrap. It reads the frozen outputs created
by the stationary-GEV analysis and writes one 2x2 comparison figure for each
primary averaging window. The 7-, 28- and 84-day figures can be used in the
main text, while the complete set is available for the Appendix.

Expected inputs below OUTPUT_DIR:
    severity/ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv
    severity/hannah_severity_summary_Nov08_extended_windows.csv
    final_analysis_clean/**/stationary_gev_original_fits_primary_windows.csv
    final_analysis_clean/**/stationary_gev_bootstrap_parameter_samples.csv

Run from the project root:
    python scripts/34_final_15_stationary_gev_comparison_diagnostics.py

Or specify the outputs directory explicitly:
    python scripts/34_final_15_stationary_gev_comparison_diagnostics.py \
        --output-dir /path/to/project/outputs
"""

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import genextreme


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

SELECTED_WINDOWS = (1, 7, 14, 21, 28, 56, 84)

OVERLAP_START = 1982
OVERLAP_END = 2016

RETURN_PERIOD_MIN = 1.05
RETURN_PERIOD_MAX = 200.0
RETURN_PERIOD_POINTS = 320

FIGURE_DPI = 300


# -----------------------------------------------------------------------------
# Font settings
# -----------------------------------------------------------------------------

FONT_BASE = 13.0
FONT_AXIS_LABEL = 14.0
FONT_TICK = 12.0
FONT_TITLE = 14.0
FONT_LEGEND = 11.5
FONT_SUPTITLE = 16.0

plt.rcParams.update(
    {
        "font.size": FONT_BASE,
        "axes.titlesize": FONT_TITLE,
        "axes.labelsize": FONT_AXIS_LABEL,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "legend.fontsize": FONT_LEGEND,
    }
)


# -----------------------------------------------------------------------------
# Sample settings
# -----------------------------------------------------------------------------

SAMPLE_ECMWF = "ECMWF pooled"
SAMPLE_HISTORICAL = "Hannah overlap shifted"

SAMPLE_LABELS = {
    SAMPLE_ECMWF: "SEAS5 pooled",
    SAMPLE_HISTORICAL: "Historical overlap shifted",
}

SAMPLE_COLOURS = {
    SAMPLE_ECMWF: "#1f77b4",
    SAMPLE_HISTORICAL: "#d95f02",
}

SAMPLE_MARKERS = {
    SAMPLE_ECMWF: "o",
    SAMPLE_HISTORICAL: "^",
}


# -----------------------------------------------------------------------------
# File and data helpers
# -----------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot stationary-GEV comparison diagnostics for all seven "
            "primary averaging windows."
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

    return parser.parse_args()


def resolve_output_dir(
    cli_output_dir: Path | None,
) -> Path:

    if cli_output_dir is not None:
        return cli_output_dir.expanduser().resolve()

    try:
        from config import OUTPUT_DIR  # type: ignore

    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import OUTPUT_DIR from config.py. Run this script "
            "from the project root or pass --output-dir explicitly."
        ) from exc

    return Path(
        OUTPUT_DIR
    ).expanduser().resolve()


def require_file(
    path: Path,
    label: str,
) -> Path:

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )

    return path


def find_frozen_output(
    root: Path,
    filename: str,
) -> Path:
    """
    Find one frozen result file without depending on a numbered folder.
    """

    matches = sorted(
        root.rglob(
            filename
        )
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} below:\n{root}\n"
            "Run the original stationary-GEV analysis first."
        )

    if len(matches) > 1:

        listed = "\n".join(
            f"  - {path}"
            for path in matches
        )

        raise RuntimeError(
            f"Several copies of {filename} were found:\n{listed}\n"
            "Keep one final copy, or change find_frozen_output() to point "
            "to the intended analysis folder."
        )

    return matches[0]


def require_columns(
    frame: pd.DataFrame,
    columns: Iterable[str],
    label: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{label} is missing columns:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


def finite_array(
    values,
) -> np.ndarray:

    array = np.asarray(
        values,
        dtype=float,
    )

    return array[
        np.isfinite(
            array
        )
    ]


def severity_column(
    window: int,
) -> str:

    return (
        f"max_{window}d_mean_demand_MW"
    )


def get_fit_row(
    fits: pd.DataFrame,
    window: int,
    sample: str,
) -> pd.Series:

    rows = fits.loc[
        (
            fits["window_days"]
            == window
        )
        & (
            fits["sample"]
            == sample
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one stationary GEV fit for "
            f"window={window}, sample={sample!r}; found {len(rows)}."
        )

    return rows.iloc[0]


def empirical_probabilities(
    n: int,
) -> np.ndarray:

    return (
        np.arange(
            1,
            n + 1,
        )
        - 0.5
    ) / n


def empirical_exceedance(
    values,
) -> tuple[np.ndarray, np.ndarray]:

    descending = np.sort(
        finite_array(
            values
        )
    )[::-1]

    ranks = np.arange(
        1,
        len(descending) + 1,
    )

    exceedance = (
        ranks
        / (
            len(descending)
            + 1.0
        )
    )

    return (
        descending,
        exceedance,
    )


def bootstrap_return_level_band(
    parameter_samples: pd.DataFrame,
    window: int,
    sample: str,
    return_period_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:

    subset = parameter_samples.loc[
        (
            parameter_samples["window_days"]
            == window
        )
        & (
            parameter_samples["sample"]
            == sample
        )
    ].copy()

    subset = subset.loc[
        np.isfinite(
            subset["scipy_c"]
        )
        & np.isfinite(
            subset["loc_GW"]
        )
        & np.isfinite(
            subset["scale_GW"]
        )
        & (
            subset["scale_GW"]
            > 0
        )
    ]

    probabilities = (
        1.0
        - 1.0
        / return_period_grid
    )

    curves: list[np.ndarray] = []

    for row in subset.itertuples(
        index=False
    ):

        curve = genextreme.ppf(
            probabilities,
            float(
                row.scipy_c
            ),
            loc=float(
                row.loc_GW
            ),
            scale=float(
                row.scale_GW
            ),
        )

        if np.all(
            np.isfinite(
                curve
            )
        ):
            curves.append(
                curve
            )

    if not curves:

        nan_curve = np.full_like(
            return_period_grid,
            np.nan,
            dtype=float,
        )

        return (
            nan_curve,
            nan_curve,
            0,
        )

    matrix = np.vstack(
        curves
    )

    return (
        np.quantile(
            matrix,
            0.025,
            axis=0,
        ),
        np.quantile(
            matrix,
            0.975,
            axis=0,
        ),
        len(curves),
    )


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def style_axis(
    ax: plt.Axes,
) -> None:

    ax.grid(
        True,
        alpha=0.22,
        linewidth=0.7,
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )


def sample_scatter(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    sample: str,
    size: float,
) -> None:

    colour = SAMPLE_COLOURS[
        sample
    ]

    if sample == SAMPLE_HISTORICAL:

        ax.scatter(
            x,
            y,
            s=size,
            marker=SAMPLE_MARKERS[
                sample
            ],
            facecolors="white",
            edgecolors=colour,
            linewidths=1.25,
            alpha=0.95,
            zorder=4,
        )

    else:

        ax.scatter(
            x,
            y,
            s=size,
            marker=SAMPLE_MARKERS[
                sample
            ],
            color=colour,
            edgecolors="none",
            alpha=0.72,
            zorder=3,
        )


def make_comparison_figure(
    window: int,
    values_by_sample: dict[str, np.ndarray],
    fits: pd.DataFrame,
    parameter_samples: pd.DataFrame,
    output_directory: Path,
) -> None:

    # ---------------------------------------------------------
    # Figure
    # ---------------------------------------------------------

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(8.8, 7.5),
        dpi=FIGURE_DPI,
    )

    fit_by_sample = {
        sample: get_fit_row(
            fits,
            window,
            sample,
        )
        for sample in values_by_sample
    }


    # -------------------------------------------------------------------------
    # Probability plot
    # -------------------------------------------------------------------------

    ax = axes[
        0,
        0,
    ]

    for (
        sample,
        values,
    ) in values_by_sample.items():

        observed = np.sort(
            finite_array(
                values
            )
        )

        empirical = empirical_probabilities(
            len(
                observed
            )
        )

        fit = fit_by_sample[
            sample
        ]

        fitted_probability = genextreme.cdf(
            observed,
            float(
                fit["scipy_c"]
            ),
            loc=float(
                fit["loc_GW"]
            ),
            scale=float(
                fit["scale_GW"]
            ),
        )

        sample_scatter(
            ax,
            empirical,
            fitted_probability,
            sample,
            (
                24
                if sample == SAMPLE_ECMWF
                else 52
            ),
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "--",
        color="0.35",
        linewidth=1.3,
    )

    ax.set(
        xlim=(0, 1),
        ylim=(0, 1),
    )

    ax.set_xlabel(
        "Empirical cumulative probability",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Fitted GEV cumulative probability",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        "Probability plot",
        fontsize=FONT_TITLE,
    )

    style_axis(
        ax
    )


    # -------------------------------------------------------------------------
    # Quantile plot
    # -------------------------------------------------------------------------

    ax = axes[
        0,
        1,
    ]

    all_quantiles: list[np.ndarray] = []

    for (
        sample,
        values,
    ) in values_by_sample.items():

        observed = np.sort(
            finite_array(
                values
            )
        )

        empirical = empirical_probabilities(
            len(
                observed
            )
        )

        fit = fit_by_sample[
            sample
        ]

        fitted_quantiles = genextreme.ppf(
            empirical,
            float(
                fit["scipy_c"]
            ),
            loc=float(
                fit["loc_GW"]
            ),
            scale=float(
                fit["scale_GW"]
            ),
        )

        sample_scatter(
            ax,
            fitted_quantiles,
            observed,
            sample,
            (
                24
                if sample == SAMPLE_ECMWF
                else 52
            ),
        )

        all_quantiles.extend(
            [
                fitted_quantiles,
                observed,
            ]
        )

    combined = np.concatenate(
        all_quantiles
    )

    lower = float(
        np.nanmin(
            combined
        )
    )

    upper = float(
        np.nanmax(
            combined
        )
    )

    padding = max(
        0.05,
        0.035
        * (
            upper
            - lower
        ),
    )

    limits = (
        lower - padding,
        upper + padding,
    )

    ax.plot(
        limits,
        limits,
        "--",
        color="0.35",
        linewidth=1.3,
    )

    ax.set(
        xlim=limits,
        ylim=limits,
    )

    ax.set_xlabel(
        "Fitted GEV quantile (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Observed winter maximum (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        "Quantile plot",
        fontsize=FONT_TITLE,
    )

    style_axis(
        ax
    )


    # -------------------------------------------------------------------------
    # Return-level plot
    # -------------------------------------------------------------------------

    ax = axes[
        1,
        0,
    ]

    return_period_grid = np.geomspace(
        RETURN_PERIOD_MIN,
        RETURN_PERIOD_MAX,
        RETURN_PERIOD_POINTS,
    )

    probabilities = (
        1.0
        - 1.0
        / return_period_grid
    )

    for (
        sample,
        values,
    ) in values_by_sample.items():

        fit = fit_by_sample[
            sample
        ]

        colour = SAMPLE_COLOURS[
            sample
        ]

        fitted_return_level = genextreme.ppf(
            probabilities,
            float(
                fit["scipy_c"]
            ),
            loc=float(
                fit["loc_GW"]
            ),
            scale=float(
                fit["scale_GW"]
            ),
        )

        (
            lower_band,
            upper_band,
            n_curves,
        ) = bootstrap_return_level_band(
            parameter_samples,
            window,
            sample,
            return_period_grid,
        )

        if n_curves:

            ax.fill_between(
                return_period_grid,
                lower_band,
                upper_band,
                color=colour,
                alpha=0.13,
                linewidth=0,
            )

        ax.plot(
            return_period_grid,
            fitted_return_level,
            color=colour,
            linewidth=2.2,
            zorder=2,
        )

        (
            descending,
            exceedance,
        ) = empirical_exceedance(
            values
        )

        empirical_return_period = (
            1.0
            / exceedance
        )

        mask = (
            empirical_return_period
            <= RETURN_PERIOD_MAX
        )

        sample_scatter(
            ax,
            empirical_return_period[
                mask
            ],
            descending[
                mask
            ],
            sample,
            (
                24
                if sample == SAMPLE_ECMWF
                else 52
            ),
        )

    ax.set_xscale(
        "log"
    )

    ax.set_xlim(
        RETURN_PERIOD_MIN,
        RETURN_PERIOD_MAX,
    )

    ax.set_xlabel(
        "Return period (winter blocks, log scale)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Return level (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        "Return-level plot",
        fontsize=FONT_TITLE,
    )

    style_axis(
        ax
    )


    # -------------------------------------------------------------------------
    # Density plot
    # -------------------------------------------------------------------------

    ax = axes[
        1,
        1,
    ]

    combined_values = np.concatenate(
        list(
            values_by_sample.values()
        )
    )

    sample_sd = float(
        np.std(
            combined_values,
            ddof=1,
        )
    )

    x_grid = np.linspace(
        float(
            np.min(
                combined_values
            )
            - 0.20
            * sample_sd
        ),
        float(
            np.max(
                combined_values
            )
            + 0.20
            * sample_sd
        ),
        700,
    )

    for (
        sample,
        values,
    ) in values_by_sample.items():

        fit = fit_by_sample[
            sample
        ]

        colour = SAMPLE_COLOURS[
            sample
        ]

        if sample == SAMPLE_ECMWF:

            ax.hist(
                values,
                bins="fd",
                density=True,
                color=colour,
                alpha=0.16,
                edgecolor=colour,
                linewidth=0.8,
            )

        else:

            ax.hist(
                values,
                bins="auto",
                density=True,
                histtype="step",
                color=colour,
                linewidth=1.7,
                alpha=0.9,
            )

        density = genextreme.pdf(
            x_grid,
            float(
                fit["scipy_c"]
            ),
            loc=float(
                fit["loc_GW"]
            ),
            scale=float(
                fit["scale_GW"]
            ),
        )

        ax.plot(
            x_grid,
            density,
            color=colour,
            linewidth=2.2,
        )

    ax.set_xlabel(
        "Winter-maximum severity (GW)",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_ylabel(
        "Density",
        fontsize=FONT_AXIS_LABEL,
    )

    ax.set_title(
        "Density plot",
        fontsize=FONT_TITLE,
    )

    style_axis(
        ax
    )


    # -------------------------------------------------------------------------
    # Shared title and legend
    # -------------------------------------------------------------------------

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SAMPLE_COLOURS[
                SAMPLE_ECMWF
            ],
            marker=SAMPLE_MARKERS[
                SAMPLE_ECMWF
            ],
            markersize=7,
            linewidth=2.2,
            label=(
                "SEAS5 pooled "
                f"(n={len(values_by_sample[SAMPLE_ECMWF])})"
            ),
        ),
        Line2D(
            [0],
            [0],
            color=SAMPLE_COLOURS[
                SAMPLE_HISTORICAL
            ],
            marker=SAMPLE_MARKERS[
                SAMPLE_HISTORICAL
            ],
            markerfacecolor="white",
            markersize=8,
            linewidth=2.2,
            label=(
                "Historical overlap shifted "
                f"(n={len(values_by_sample[SAMPLE_HISTORICAL])})"
            ),
        ),
        Line2D(
            [0],
            [0],
            color="0.35",
            linestyle="--",
            linewidth=1.3,
            label="Reference line",
        ),
        Patch(
            facecolor="0.55",
            alpha=0.18,
            label=(
                "95% winter-year bootstrap interval"
            ),
        ),
    ]

    fig.suptitle(
        (
            "Stationary GEV diagnostics: "
            f"{window}-day winter-maximum severity"
        ),
        fontsize=FONT_SUPTITLE,
        y=0.985,
    )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
        frameon=False,
        fontsize=FONT_LEGEND,
    )

    fig.subplots_adjust(
        left=0.10,
        right=0.985,
        top=0.90,
        bottom=0.225,
        hspace=0.42,
        wspace=0.31,
    )


    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    stem = (
        f"stationary_gev_diagnostics_"
        f"{window}day_comparison"
    )

    png_path = (
        output_directory
        / f"{stem}.png"
    )

    pdf_path = (
        output_directory
        / f"{stem}.pdf"
    )

    fig.savefig(
        png_path,
        bbox_inches="tight",
        dpi=FIGURE_DPI,
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved {png_path}"
    )

    print(
        f"Saved {pdf_path}"
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:

    args = parse_arguments()

    output_dir = resolve_output_dir(
        args.output_dir
    )

    severity_dir = (
        output_dir
        / "severity"
    )

    analysis_root = (
        output_dir
        / "final_analysis_clean"
    )

    figure_directory = (
        analysis_root
        / "15_stationary_gev_comparison_diagnostics"
    )

    figure_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    ecmwf_file = require_file(
        severity_dir
        / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv",
        "SEAS5 severity summary",
    )

    historical_file = require_file(
        severity_dir
        / "hannah_severity_summary_Nov08_extended_windows.csv",
        "historical severity summary",
    )

    fits_file = find_frozen_output(
        analysis_root,
        "stationary_gev_original_fits_primary_windows.csv",
    )

    parameter_samples_file = find_frozen_output(
        analysis_root,
        "stationary_gev_bootstrap_parameter_samples.csv",
    )

    ecmwf = pd.read_csv(
        ecmwf_file
    )

    historical = pd.read_csv(
        historical_file
    )

    fits = pd.read_csv(
        fits_file
    )

    parameter_samples = pd.read_csv(
        parameter_samples_file
    )

    require_columns(
        ecmwf,
        [
            "winter_year",
        ],
        "SEAS5 severity summary",
    )

    require_columns(
        historical,
        [
            "winter_year",
        ],
        "Historical severity summary",
    )

    require_columns(
        fits,
        [
            "window_days",
            "sample",
            "xi",
            "scipy_c",
            "loc_GW",
            "scale_GW",
            "mean_shift_GW",
        ],
        "Stationary GEV fits",
    )

    require_columns(
        parameter_samples,
        [
            "window_days",
            "sample",
            "xi",
            "scipy_c",
            "loc_GW",
            "scale_GW",
        ],
        "Stationary GEV bootstrap parameter samples",
    )

    for frame in (
        ecmwf,
        historical,
    ):

        frame[
            "winter_year"
        ] = pd.to_numeric(
            frame[
                "winter_year"
            ],
            errors="coerce",
        )

    ecmwf = ecmwf.loc[
        ecmwf[
            "winter_year"
        ].between(
            OVERLAP_START,
            OVERLAP_END,
        )
    ].copy()

    historical = historical.loc[
        historical[
            "winter_year"
        ].between(
            OVERLAP_START,
            OVERLAP_END,
        )
    ].copy()

    if len(ecmwf) != 875:

        raise ValueError(
            "Expected 875 pooled SEAS5 winter-member maxima for 1982-2016, "
            f"but found {len(ecmwf)}."
        )

    if len(historical) != 35:

        raise ValueError(
            "Expected 35 historical winter maxima for 1982-2016, "
            f"but found {len(historical)}."
        )

    for window in SELECTED_WINDOWS:

        column = severity_column(
            window
        )

        require_columns(
            ecmwf,
            [
                column,
            ],
            "SEAS5 severity summary",
        )

        require_columns(
            historical,
            [
                column,
            ],
            "Historical severity summary",
        )

        ecmwf_values_gw = finite_array(
            pd.to_numeric(
                ecmwf[
                    column
                ],
                errors="coerce",
            )
            / 1000.0
        )

        historical_raw_gw = finite_array(
            pd.to_numeric(
                historical[
                    column
                ],
                errors="coerce",
            )
            / 1000.0
        )

        historical_fit = get_fit_row(
            fits,
            window,
            SAMPLE_HISTORICAL,
        )

        shift_gw = float(
            historical_fit[
                "mean_shift_GW"
            ]
        )

        historical_shifted_gw = (
            historical_raw_gw
            + shift_gw
        )

        make_comparison_figure(
            window=window,
            values_by_sample={
                SAMPLE_ECMWF: (
                    ecmwf_values_gw
                ),
                SAMPLE_HISTORICAL: (
                    historical_shifted_gw
                ),
            },
            fits=fits,
            parameter_samples=parameter_samples,
            output_directory=figure_directory,
        )

    print(
        "Done. No GEV model was refitted and no bootstrap was rerun."
    )


if __name__ == "__main__":
    main()