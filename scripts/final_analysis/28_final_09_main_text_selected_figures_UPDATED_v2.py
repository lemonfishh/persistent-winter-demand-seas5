from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import genextreme

from config import OUTPUT_DIR


# ============================================================
# Final analysis 09
# Plot-only figures for the Results chapter
#
# This script DOES NOT:
#   - process weather data;
#   - recalculate demand;
#   - recalculate severity;
#   - refit any GEV model;
#   - rerun any bootstrap.
#
# It only reads the frozen final CSV outputs and creates:
#
#   1. selected_empirical_exceedance_1day_7day_28day_84day.pdf
#   2. selected_extended_upper_tails_21_28_35_42_49_56.pdf
#   3. selected_stationary_gev_diagnostics_1day_28day_84day.pdf
#
# Run from the project root using:
#
#   /usr/bin/python3 scripts/28_final_09_main_text_selected_figures.py
# ============================================================


# ============================================================
# Settings
# ============================================================

SELECTED_EMPIRICAL_WINDOWS = [
    1,
    7,
    28,
    84,
]

SELECTED_EXTENDED_WINDOWS = [
    21,
    28,
    35,
    42,
    49,
    56,
]

SELECTED_GEV_WINDOWS = [
    1,
    28,
    84,
]

UPPER_TAIL_FRACTION = 0.20

RETURN_PERIOD_MIN = 1.05
RETURN_PERIOD_MAX = 50.0
RETURN_PERIOD_POINTS = 260

OVERLAP_START = 1982
OVERLAP_END = 2016

FIGURE_DPI = 300

# Main-text figure typography.
# These values are deliberately larger because the figures are scaled
# down when inserted into the dissertation.
FONT_BASE = 13.0
FONT_AXIS_LABEL = 14.0
FONT_TICK = 12.0
FONT_TITLE = 14.0
FONT_LEGEND = 11.5

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
# Input and output locations
# ============================================================

FINAL_ANALYSIS_ROOT = (
    OUTPUT_DIR
    / "final_analysis_clean"
)

SEVERITY_DIR = (
    OUTPUT_DIR
    / "severity"
)

OUT_DIR = (
    FINAL_ANALYSIS_ROOT
    / "09_main_text_selected_figures"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# File helpers
# ============================================================

def require_file(
    path: Path,
    label: str,
) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {label}:\n{path}"
        )

    return path


def find_final_output(
    filename: str,
) -> Path:
    """
    Find one exact filename below outputs/final_analysis_clean.

    This avoids depending on the exact numbered subfolder name.

    The script stops if several copies are found. It is safer to
    stop than to silently select one of several possible final
    versions.
    """
    matches = sorted(
        FINAL_ANALYSIS_ROOT.rglob(
            filename
        )
    )

    if len(matches) == 0:
        raise FileNotFoundError(
            f"Could not find {filename} below:\n"
            f"{FINAL_ANALYSIS_ROOT}"
        )

    if len(matches) > 1:
        listed = "\n".join(
            f"  - {path}"
            for path in matches
        )

        raise RuntimeError(
            f"Several copies of {filename} were found:\n"
            f"{listed}\n\n"
            "Please remove or rename duplicate final versions "
            "before running this plot-only script."
        )

    return matches[0]


PRIMARY_SUMMARY_FILE = find_final_output(
    "primary_windows_mean_shift_summary.csv"
)

PRIMARY_EXCEEDANCE_FILE = find_final_output(
    "primary_windows_mean_shifted_exceedance_data.csv"
)

EXTENDED_TAIL_FILE = find_final_output(
    "extended_windows_standardised_upper_tail_data.csv"
)

GEV_FITS_FILE = find_final_output(
    "stationary_gev_original_fits_primary_windows.csv"
)

GEV_PARAMETER_SAMPLES_FILE = find_final_output(
    "stationary_gev_bootstrap_parameter_samples.csv"
)

ECMWF_SEVERITY_FILE = require_file(
    SEVERITY_DIR
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv",
    "ECMWF severity summary",
)

HANNAH_SEVERITY_FILE = require_file(
    SEVERITY_DIR
    / "hannah_severity_summary_Nov08_extended_windows.csv",
    "Hannah severity summary",
)


# ============================================================
# Output files
# ============================================================

OUT_EMPIRICAL_PDF = (
    OUT_DIR
    / "selected_empirical_exceedance_1day_7day_28day_84day.pdf"
)

OUT_EMPIRICAL_PNG = (
    OUT_EMPIRICAL_PDF.with_suffix(
        ".png"
    )
)

OUT_EXTENDED_PDF = (
    OUT_DIR
    / "selected_extended_upper_tails_21_28_35_42_49_56.pdf"
)

OUT_EXTENDED_PNG = (
    OUT_EXTENDED_PDF.with_suffix(
        ".png"
    )
)

OUT_GEV_PDF = (
    OUT_DIR
    / "selected_stationary_gev_diagnostics_1day_28day_84day.pdf"
)

OUT_GEV_PNG = (
    OUT_GEV_PDF.with_suffix(
        ".png"
    )
)


# ============================================================
# General helpers
# ============================================================

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

    probabilities = (
        ranks
        / (
            len(descending)
            + 1.0
        )
    )

    return (
        descending,
        probabilities,
    )


def get_fit_row(
    fits: pd.DataFrame,
    window: int,
    sample: str,
) -> pd.Series:
    rows = fits[
        (
            fits[
                "window_days"
            ] == window
        )
        & (
            fits[
                "sample"
            ] == sample
        )
    ]

    if len(rows) != 1:
        raise ValueError(
            "Expected exactly one stationary GEV fit for:\n"
            f"  window={window}\n"
            f"  sample={sample}\n"
            f"Found {len(rows)} rows."
        )

    return rows.iloc[
        0
    ]


def bootstrap_return_level_band(
    parameter_samples: pd.DataFrame,
    window: int,
    sample: str,
    return_period_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    subset = parameter_samples[
        (
            parameter_samples[
                "window_days"
            ] == window
        )
        & (
            parameter_samples[
                "sample"
            ] == sample
        )
    ].copy()

    return_probabilities = (
        1.0
        - 1.0
        / return_period_grid
    )

    curves = []

    for _, row in subset.iterrows():
        scipy_c = float(
            row[
                "scipy_c"
            ]
        )

        loc_gw = float(
            row[
                "loc_GW"
            ]
        )

        scale_gw = float(
            row[
                "scale_GW"
            ]
        )

        valid_parameters = np.isfinite(
            [
                scipy_c,
                loc_gw,
                scale_gw,
            ]
        ).all()

        if (
            not valid_parameters
            or scale_gw <= 0
        ):
            continue

        curve = genextreme.ppf(
            return_probabilities,
            scipy_c,
            loc=loc_gw,
            scale=scale_gw,
        )

        if np.isfinite(
            curve
        ).all():
            curves.append(
                curve
            )

    if len(curves) == 0:
        raise RuntimeError(
            "No valid bootstrap return-level curves for:\n"
            f"  window={window}\n"
            f"  sample={sample}"
        )

    stacked = np.vstack(
        curves
    )

    lower = np.quantile(
        stacked,
        0.025,
        axis=0,
    )

    upper = np.quantile(
        stacked,
        0.975,
        axis=0,
    )

    return (
        lower,
        upper,
        len(curves),
    )


# ============================================================
# Load frozen final results
# ============================================================

primary_summary = pd.read_csv(
    PRIMARY_SUMMARY_FILE
)

primary_exceedance = pd.read_csv(
    PRIMARY_EXCEEDANCE_FILE
)

extended_tail = pd.read_csv(
    EXTENDED_TAIL_FILE
)

gev_fits = pd.read_csv(
    GEV_FITS_FILE
)

gev_parameter_samples = pd.read_csv(
    GEV_PARAMETER_SAMPLES_FILE
)

ecmwf_severity = pd.read_csv(
    ECMWF_SEVERITY_FILE
)

hannah_severity = pd.read_csv(
    HANNAH_SEVERITY_FILE
)


# ============================================================
# Validate columns
# ============================================================

require_columns(
    primary_summary,
    [
        "window_days",
        "hannah_1962_63_shifted_GW",
    ],
    "Primary-window summary",
)

require_columns(
    primary_exceedance,
    [
        "window_days",
        "sample",
        "severity_GW",
        "exceedance_probability",
    ],
    "Primary-window empirical exceedance data",
)

require_columns(
    extended_tail,
    [
        "window_days",
        "sample",
        "standardised_severity",
        "exceedance_probability",
    ],
    "Extended-window upper-tail data",
)

require_columns(
    gev_fits,
    [
        "window_days",
        "sample",
        "mean_shift_GW",
        "scipy_c",
        "loc_GW",
        "scale_GW",
    ],
    "Stationary GEV original fits",
)

require_columns(
    gev_parameter_samples,
    [
        "window_days",
        "sample",
        "scipy_c",
        "loc_GW",
        "scale_GW",
    ],
    "Stationary GEV bootstrap parameter samples",
)

require_columns(
    ecmwf_severity,
    [
        "winter_year",
    ],
    "ECMWF severity summary",
)

require_columns(
    hannah_severity,
    [
        "winter_year",
    ],
    "Hannah severity summary",
)


# ============================================================
# Figure 1
# Selected empirical exceedance curves:
# 1-, 7-, 28- and 84-day severity
# ============================================================

def plot_selected_empirical() -> None:
    print(
        "\nCreating selected empirical exceedance figure..."
    )

    fig, axes_array = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(
            10.2,
            7.6,
        ),
        dpi=FIGURE_DPI,
        sharey=True,
    )

    axes = {
        window: ax
        for window, ax in zip(
            SELECTED_EMPIRICAL_WINDOWS,
            axes_array.flatten(),
        )
    }

    sample_order = [
        "Mean-shifted Hannah overlap",
        "ECMWF pooled",
    ]

    display_labels = {
        "Mean-shifted Hannah overlap":
            "Historical overlap shifted",
        "ECMWF pooled":
            "SEAS5 pooled",
    }

    for index, window in enumerate(
        SELECTED_EMPIRICAL_WINDOWS
    ):
        ax = axes[
            window
        ]

        for sample in sample_order:
            data = primary_exceedance[
                (
                    primary_exceedance[
                        "window_days"
                    ] == window
                )
                & (
                    primary_exceedance[
                        "sample"
                    ] == sample
                )
            ].sort_values(
                "exceedance_probability"
            )

            if data.empty:
                raise ValueError(
                    "No empirical exceedance data for:\n"
                    f"  window={window}\n"
                    f"  sample={sample}"
                )

            ax.plot(
                data[
                    "severity_GW"
                ],
                data[
                    "exceedance_probability"
                ],
                marker=".",
                markersize=(
                    5.8
                    if sample
                    == "Mean-shifted Hannah overlap"
                    else 2.8
                ),
                linewidth=1.35,
                label=display_labels[
                    sample
                ],
            )

        benchmark_row = primary_summary[
            primary_summary[
                "window_days"
            ] == window
        ]

        if len(
            benchmark_row
        ) != 1:
            raise ValueError(
                "Expected exactly one benchmark row "
                f"for {window} days."
            )

        benchmark_gw = float(
            benchmark_row.iloc[
                0
            ][
                "hannah_1962_63_shifted_GW"
            ]
        )

        ax.axvline(
            benchmark_gw,
            linestyle="--",
            linewidth=1.2,
            label="Shifted 1962/63 benchmark",
        )

        ax.set_yscale(
            "log"
        )

        ax.set_ylim(
            8e-4,
            1.05,
        )

        ax.set_xlabel(
            f"Maximum {window}-day mean demand (GW)"
        )

        if index % 2 == 0:
            ax.set_ylabel(
                "Empirical exceedance probability"
            )

        ax.set_title(
            (
                "1 day"
                if window == 1
                else f"{window} days"
            )
        )

        ax.grid(
            True,
            which="both",
            alpha=0.28,
        )

    handles, labels = axes[
        1
    ].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(
            0.5,
            0.015,
        ),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.98,
        bottom=0.14,
        hspace=0.29,
        wspace=0.20,
    )

    fig.savefig(
        OUT_EMPIRICAL_PNG,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_EMPIRICAL_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Figure 2
# Selected extended standardised upper tails:
# 21, 28, 35, 42, 49 and 56 days
# ============================================================

def plot_selected_extended_tails() -> None:
    print(
        "Creating selected extended upper-tail figure..."
    )

    data = extended_tail[
        extended_tail[
            "window_days"
        ].isin(
            SELECTED_EXTENDED_WINDOWS
        )
    ].copy()

    if data.empty:
        raise ValueError(
            "No selected extended-window data were found."
        )

    x_min = float(
        data[
            "standardised_severity"
        ].min()
    )

    x_max = float(
        data[
            "standardised_severity"
        ].max()
    )

    x_padding = max(
        0.05,
        0.03
        * (
            x_max
            - x_min
        ),
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(
            10.2,
            8.8,
        ),
        dpi=FIGURE_DPI,
        sharex=True,
        sharey=True,
    )

    axes = np.asarray(
        axes
    ).reshape(
        -1
    )

    sample_order = [
        "Hannah overlap",
        "ECMWF pooled",
    ]

    for index, window in enumerate(
        SELECTED_EXTENDED_WINDOWS
    ):
        ax = axes[
            index
        ]

        for sample in sample_order:
            subset = data[
                (
                    data[
                        "window_days"
                    ] == window
                )
                & (
                    data[
                        "sample"
                    ] == sample
                )
            ].sort_values(
                "exceedance_probability"
            )

            if subset.empty:
                raise ValueError(
                    "No standardised upper-tail data for:\n"
                    f"  window={window}\n"
                    f"  sample={sample}"
                )

            ax.plot(
                subset[
                    "standardised_severity"
                ],
                subset[
                    "exceedance_probability"
                ],
                marker=".",
                markersize=(
                    5.8
                    if sample
                    == "Hannah overlap"
                    else 2.8
                ),
                linewidth=1.35,
                label=sample,
            )

        ax.set_yscale(
            "log"
        )

        ax.set_xlim(
            x_min
            - x_padding,
            x_max
            + x_padding,
        )

        ax.set_ylim(
            8e-4,
            UPPER_TAIL_FRACTION
            * 1.15,
        )

        ax.set_title(
            f"{window} days",
            fontsize=FONT_TITLE,
            pad=3,
        )

        ax.grid(
            True,
            which="both",
            alpha=0.28,
        )

        # Use shared axis labels for the six-panel figure.
        # Repeating the large y-label on every left-hand panel makes
        # the labels overlap once the figure is placed in the thesis.

    # One shared y-label and one shared x-label keep the text large
    # without crowding the individual panels.
    fig.text(
        0.018,
        0.55,
        "Empirical exceedance probability",
        va="center",
        rotation="vertical",
        fontsize=FONT_AXIS_LABEL,
    )

    fig.text(
        0.535,
        0.085,
        "Standardised severity\n"
        "(value - sample median) / sample IQR",
        ha="center",
        va="center",
        fontsize=FONT_AXIS_LABEL,
    )

    handles, labels = axes[
        0
    ].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=2,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(
            0.5,
            0.012,
        ),
    )

    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        top=0.985,
        bottom=0.17,
        hspace=0.27,
        wspace=0.10,
    )

    fig.savefig(
        OUT_EXTENDED_PNG,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_EXTENDED_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Figure 3
# Selected stationary GEV diagnostics:
# 1-, 28- and 84-day severity
#
# Left column:
#   empirical upper tail and fitted stationary GEV survival curve
#
# Right column:
#   empirical return levels, fitted curve and 95% winter-year
#   cluster-bootstrap interval
# ============================================================

def plot_selected_gev_diagnostics() -> None:
    print(
        "Creating selected stationary GEV diagnostic figure..."
    )

    hannah_overlap = hannah_severity[
        (
            hannah_severity[
                "winter_year"
            ] >= OVERLAP_START
        )
        & (
            hannah_severity[
                "winter_year"
            ] <= OVERLAP_END
        )
    ].copy()

    if len(
        hannah_overlap
    ) != 35:
        raise ValueError(
            "Expected 35 Hannah overlap winters, "
            f"but found {len(hannah_overlap)}."
        )

    default_colors = plt.rcParams[
        "axes.prop_cycle"
    ].by_key()[
        "color"
    ]

    sample_colors = {
        "ECMWF pooled": default_colors[
            0
        ],
        "Hannah overlap shifted": default_colors[
            1
        ],
    }

    sample_labels = {
        "ECMWF pooled": "SEAS5 pooled",
        "Hannah overlap shifted": (
            "Historical overlap shifted"
        ),
    }

    return_period_grid = np.geomspace(
        RETURN_PERIOD_MIN,
        RETURN_PERIOD_MAX,
        RETURN_PERIOD_POINTS,
    )

    return_probabilities = (
        1.0
        - 1.0
        / return_period_grid
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(
            10.6,
            11.0,
        ),
        dpi=FIGURE_DPI,
    )

    for row_index, window in enumerate(
        SELECTED_GEV_WINDOWS
    ):
        column = severity_column(
            window
        )

        require_columns(
            ecmwf_severity,
            [
                column,
            ],
            "ECMWF severity summary",
        )

        require_columns(
            hannah_overlap,
            [
                column,
            ],
            "Hannah overlap severity summary",
        )

        ecmwf_values_gw = finite_array(
            ecmwf_severity[
                column
            ]
            / 1000.0
        )

        hannah_raw_gw = finite_array(
            hannah_overlap[
                column
            ]
            / 1000.0
        )

        hannah_fit = get_fit_row(
            gev_fits,
            window,
            "Hannah overlap shifted",
        )

        mean_shift_gw = float(
            hannah_fit[
                "mean_shift_GW"
            ]
        )

        hannah_shifted_gw = (
            hannah_raw_gw
            + mean_shift_gw
        )

        values_by_sample = {
            "ECMWF pooled": ecmwf_values_gw,
            "Hannah overlap shifted": hannah_shifted_gw,
        }

        # ----------------------------------------------------
        # Left panel: empirical and fitted upper tail
        # ----------------------------------------------------

        ax_tail = axes[
            row_index,
            0,
        ]

        top_tail_values = []

        for values in values_by_sample.values():
            descending, exceedance = empirical_exceedance(
                values
            )

            tail_mask = (
                exceedance
                <= UPPER_TAIL_FRACTION
            )

            top_tail_values.extend(
                descending[
                    tail_mask
                ].tolist()
            )

        tail_x_min = float(
            np.min(
                top_tail_values
            )
        )

        tail_x_max = float(
            np.max(
                top_tail_values
            )
        )

        tail_padding = max(
            0.05,
            0.04
            * (
                tail_x_max
                - tail_x_min
            ),
        )

        tail_grid = np.linspace(
            tail_x_min
            - tail_padding,
            tail_x_max
            + tail_padding,
            500,
        )

        for sample, values in values_by_sample.items():
            fit = get_fit_row(
                gev_fits,
                window,
                sample,
            )

            colour = sample_colors[
                sample
            ]

            descending, exceedance = empirical_exceedance(
                values
            )

            tail_mask = (
                exceedance
                <= UPPER_TAIL_FRACTION
            )

            ax_tail.scatter(
                descending[
                    tail_mask
                ],
                exceedance[
                    tail_mask
                ],
                s=(
                    12
                    if sample
                    == "ECMWF pooled"
                    else 34
                ),
                color=colour,
                alpha=0.85,
                zorder=3,
            )

            fitted_survival = genextreme.sf(
                tail_grid,
                float(
                    fit[
                        "scipy_c"
                    ]
                ),
                loc=float(
                    fit[
                        "loc_GW"
                    ]
                ),
                scale=float(
                    fit[
                        "scale_GW"
                    ]
                ),
            )

            valid = (
                np.isfinite(
                    fitted_survival
                )
                & (
                    fitted_survival
                    > 0
                )
            )

            ax_tail.plot(
                tail_grid[
                    valid
                ],
                fitted_survival[
                    valid
                ],
                color=colour,
                linewidth=1.7,
            )

        ax_tail.set_yscale(
            "log"
        )

        ax_tail.set_ylim(
            8e-4,
            UPPER_TAIL_FRACTION
            * 1.15,
        )

        ax_tail.set_xlabel(
            f"Maximum {window}-day mean demand (GW)"
        )

        ax_tail.set_ylabel(
            "Exceedance probability"
        )

        ax_tail.set_title(
            (
                "1 day: upper tail"
                if window == 1
                else f"{window} days: upper tail"
            ),
            fontsize=FONT_TITLE,
            pad=4,
        )

        ax_tail.grid(
            True,
            which="both",
            alpha=0.28,
        )

        # ----------------------------------------------------
        # Right panel: return-level plot
        # ----------------------------------------------------

        ax_return = axes[
            row_index,
            1,
        ]

        for sample, values in values_by_sample.items():
            fit = get_fit_row(
                gev_fits,
                window,
                sample,
            )

            colour = sample_colors[
                sample
            ]

            fitted_return_levels = genextreme.ppf(
                return_probabilities,
                float(
                    fit[
                        "scipy_c"
                    ]
                ),
                loc=float(
                    fit[
                        "loc_GW"
                    ]
                ),
                scale=float(
                    fit[
                        "scale_GW"
                    ]
                ),
            )

            (
                lower_band,
                upper_band,
                number_valid,
            ) = bootstrap_return_level_band(
                gev_parameter_samples,
                window,
                sample,
                return_period_grid,
            )

            print(
                f"  {window}-day {sample}: "
                f"{number_valid} valid bootstrap curves"
            )

            ax_return.fill_between(
                return_period_grid,
                lower_band,
                upper_band,
                color=colour,
                alpha=0.16,
            )

            ax_return.plot(
                return_period_grid,
                fitted_return_levels,
                color=colour,
                linewidth=1.7,
            )

            descending, exceedance = empirical_exceedance(
                values
            )

            empirical_return_periods = (
                1.0
                / exceedance
            )

            display_mask = (
                empirical_return_periods
                <= RETURN_PERIOD_MAX
            )

            ax_return.scatter(
                empirical_return_periods[
                    display_mask
                ],
                descending[
                    display_mask
                ],
                s=(
                    12
                    if sample
                    == "ECMWF pooled"
                    else 34
                ),
                color=colour,
                alpha=0.85,
                zorder=3,
            )

        ax_return.set_xscale(
            "log"
        )

        ax_return.set_xlim(
            RETURN_PERIOD_MIN,
            RETURN_PERIOD_MAX,
        )

        ax_return.set_xlabel(
            "Return period (winter blocks, log scale)"
        )

        ax_return.set_ylabel(
            "Return level (GW)"
        )

        ax_return.set_title(
            (
                "1 day: return levels"
                if window == 1
                else f"{window} days: return levels"
            ),
            fontsize=FONT_TITLE,
            pad=4,
        )

        ax_return.grid(
            True,
            which="both",
            alpha=0.28,
        )

    legend_handles = [
        Line2D(
            [
                0,
            ],
            [
                0,
            ],
            color=sample_colors[
                "ECMWF pooled"
            ],
            linewidth=2.0,
            label=sample_labels[
                "ECMWF pooled"
            ],
        ),
        Line2D(
            [
                0,
            ],
            [
                0,
            ],
            color=sample_colors[
                "Hannah overlap shifted"
            ],
            linewidth=2.0,
            label=sample_labels[
                "Hannah overlap shifted"
            ],
        ),
        Line2D(
            [
                0,
            ],
            [
                0,
            ],
            color="black",
            linewidth=1.7,
            label="Fitted stationary GEV",
        ),
        Line2D(
            [
                0,
            ],
            [
                0,
            ],
            color="black",
            marker="o",
            linestyle="None",
            markersize=5,
            label="Empirical plotting positions",
        ),
        Patch(
            facecolor="grey",
            alpha=0.16,
            label="95% winter-year bootstrap interval",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(
            0.5,
            0.005,
        ),
    )

    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.985,
        bottom=0.105,
        hspace=0.31,
        wspace=0.18,
    )

    fig.savefig(
        OUT_GEV_PNG,
        bbox_inches="tight",
    )

    fig.savefig(
        OUT_GEV_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "=" * 80
    )

    print(
        "Final analysis 09: plot-only main-text figures"
    )

    print(
        "=" * 80
    )

    plot_selected_empirical()

    plot_selected_extended_tails()

    plot_selected_gev_diagnostics()

    expected_outputs = [
        OUT_EMPIRICAL_PNG,
        OUT_EMPIRICAL_PDF,
        OUT_EXTENDED_PNG,
        OUT_EXTENDED_PDF,
        OUT_GEV_PNG,
        OUT_GEV_PDF,
    ]

    missing_outputs = [
        path
        for path in expected_outputs
        if not path.exists()
    ]

    if missing_outputs:
        raise RuntimeError(
            "Some expected outputs were not created:\n"
            + "\n".join(
                str(path)
                for path in missing_outputs
            )
        )

    print(
        "\nSaved figures:"
    )

    for path in expected_outputs:
        print(
            path
        )

    print(
        "\nThis was a plot-only run."
    )

    print(
        "No weather, demand, severity, GEV or bootstrap "
        "result was recalculated."
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()