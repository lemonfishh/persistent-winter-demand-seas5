from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import OUTPUT_DIR


# ============================================================
# Leave-one-winter-year-out empirical shoulder check
#
# Purpose:
# Examine whether the complete shape of the standardised
# empirical upper tail depends strongly on one winter year.
#
# This script does not use the largest adjacent gap to represent
# the shoulder. Instead, it recalculates and plots the complete
# upper-tail curve after omitting each winter year in turn.
# ============================================================


# ============================================================
# Settings
# ============================================================

WINDOWS = [
    21,
    28,
    35,
    42,
    49,
    56,
]

UPPER_TAIL_FRACTION = 0.20
FIGURE_DPI = 300

YEAR_COLUMN = "winter_year"
MEMBER_COLUMN = "member"
WINDOW_COLUMN = "window_days"
SEVERITY_COLUMN = "maximum_mean_demand_GW"

# Font sizes
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

INPUT_FILE = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "05_temperature_tail_and_bump_concentration"
    / "temperature_demand_rolling_extremes_by_winter_member.csv"
)

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "12_leave_one_winter_out_shoulder_shape"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_FIGURE_PNG = (
    OUT_DIR
    / "leave_one_winter_out_empirical_shoulders.png"
)

OUT_FIGURE_PDF = (
    OUT_DIR
    / "leave_one_winter_out_empirical_shoulders.pdf"
)

OUT_CURVE_DATA = (
    OUT_DIR
    / "leave_one_winter_out_empirical_curve_data.csv"
)


# ============================================================
# Helper functions
# ============================================================

def robust_standardise(
    values: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Standardise values using:

        (value - sample median) / sample IQR

    The median and IQR are recalculated separately for the
    full sample and for every leave-one-winter-year-out sample.
    """

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
            "Cannot standardise values because the IQR is not positive."
        )

    standardised_values = (
        values
        - median_value
    ) / iqr_value

    return (
        standardised_values,
        float(median_value),
        float(iqr_value),
    )


def empirical_upper_tail(
    standardised_values: np.ndarray,
) -> pd.DataFrame:
    """
    Order the standardised values from highest to lowest and
    retain observations with empirical exceedance probability
    no greater than 0.20.
    """

    values = np.asarray(
        standardised_values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    values_descending = np.sort(
        values
    )[::-1]

    sample_size = len(
        values_descending
    )

    descending_rank = np.arange(
        1,
        sample_size + 1,
    )

    exceedance_probability = (
        descending_rank
        / (sample_size + 1.0)
    )

    retain = (
        exceedance_probability
        <= UPPER_TAIL_FRACTION
    )

    return pd.DataFrame(
        {
            "descending_rank": (
                descending_rank[
                    retain
                ]
            ),
            "exceedance_probability": (
                exceedance_probability[
                    retain
                ]
            ),
            "standardised_severity": (
                values_descending[
                    retain
                ]
            ),
        }
    )


def validate_input(
    data: pd.DataFrame,
) -> None:
    """
    Check that the required columns and expected winter-member
    structure are present.
    """

    required_columns = {
        YEAR_COLUMN,
        MEMBER_COLUMN,
        WINDOW_COLUMN,
        SEVERITY_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "The input file is missing these columns: "
            f"{sorted(missing_columns)}"
        )

    required_data = data[
        list(required_columns)
    ]

    if required_data.isna().any().any():
        raise ValueError(
            "The required input columns contain missing values."
        )

    duplicate_key = [
        YEAR_COLUMN,
        MEMBER_COLUMN,
        WINDOW_COLUMN,
    ]

    if data.duplicated(
        duplicate_key
    ).any():
        raise ValueError(
            "The input contains more than one row for the same "
            "winter year, member and averaging window."
        )


# ============================================================
# Calculate the empirical curves
# ============================================================

def calculate_curves(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the full-sample curve and the 35 curves obtained
    after omitting each winter year in turn.
    """

    all_curves = []

    for window in WINDOWS:

        window_data = (
            data.loc[
                data[WINDOW_COLUMN] == window
            ]
            .copy()
        )

        if window_data.empty:
            raise ValueError(
                f"No data were found for the {window}-day window."
            )

        winter_years = np.sort(
            window_data[
                YEAR_COLUMN
            ].unique()
        )

        print(
            f"{window}-day window: "
            f"{len(window_data)} realisations from "
            f"{len(winter_years)} winter years."
        )

        # ----------------------------------------------------
        # Full pooled sample
        # ----------------------------------------------------

        full_values = window_data[
            SEVERITY_COLUMN
        ].to_numpy()

        (
            full_standardised,
            full_median,
            full_iqr,
        ) = robust_standardise(
            full_values
        )

        full_curve = empirical_upper_tail(
            full_standardised
        )

        full_curve[WINDOW_COLUMN] = window
        full_curve["sample_type"] = "Full sample"
        full_curve["omitted_winter_year"] = np.nan
        full_curve["sample_size"] = len(
            full_values
        )
        full_curve["sample_median_GW"] = full_median
        full_curve["sample_IQR_GW"] = full_iqr

        all_curves.append(
            full_curve
        )

        # ----------------------------------------------------
        # Leave one complete winter year out
        # ----------------------------------------------------

        for omitted_year in winter_years:

            reduced_values = (
                window_data.loc[
                    window_data[YEAR_COLUMN]
                    != omitted_year,
                    SEVERITY_COLUMN,
                ]
                .to_numpy()
            )

            (
                reduced_standardised,
                reduced_median,
                reduced_iqr,
            ) = robust_standardise(
                reduced_values
            )

            reduced_curve = empirical_upper_tail(
                reduced_standardised
            )

            reduced_curve[WINDOW_COLUMN] = window
            reduced_curve[
                "sample_type"
            ] = "Leave one winter out"
            reduced_curve[
                "omitted_winter_year"
            ] = int(omitted_year)
            reduced_curve[
                "sample_size"
            ] = len(reduced_values)
            reduced_curve[
                "sample_median_GW"
            ] = reduced_median
            reduced_curve[
                "sample_IQR_GW"
            ] = reduced_iqr

            all_curves.append(
                reduced_curve
            )

    curve_data = pd.concat(
        all_curves,
        ignore_index=True,
    )

    column_order = [
        WINDOW_COLUMN,
        "sample_type",
        "omitted_winter_year",
        "sample_size",
        "sample_median_GW",
        "sample_IQR_GW",
        "descending_rank",
        "exceedance_probability",
        "standardised_severity",
    ]

    return curve_data[
        column_order
    ]


# ============================================================
# Plot the results
# ============================================================

def plot_curves(
    curve_data: pd.DataFrame,
) -> None:
    """
    Produce one six-panel figure.

    Blue line:
        empirical upper tail from the full sample.

    Thin grey lines:
        complete upper-tail curves after omitting each winter.

    Grey area:
        minimum-to-maximum range across the 35 omissions at
        each descending rank.
    """

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12, 13.4),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    axes = axes.ravel()

    full_sample_colour = "#1764AB"
    omitted_curve_colour = "#8F969D"
    envelope_colour = "#B8BEC4"

    for axis, window in zip(
        axes,
        WINDOWS,
    ):

        panel_data = curve_data.loc[
            curve_data[WINDOW_COLUMN]
            == window
        ]

        full_curve = panel_data.loc[
            panel_data["sample_type"]
            == "Full sample"
        ].sort_values(
            "descending_rank"
        )

        omitted_curves = panel_data.loc[
            panel_data["sample_type"]
            == "Leave one winter out"
        ]

        # Calculate the range across all omitted-winter
        # samples at each descending rank.
        envelope = (
            omitted_curves
            .groupby(
                "descending_rank",
                as_index=False,
            )
            .agg(
                exceedance_probability=(
                    "exceedance_probability",
                    "first",
                ),
                minimum_severity=(
                    "standardised_severity",
                    "min",
                ),
                maximum_severity=(
                    "standardised_severity",
                    "max",
                ),
            )
            .sort_values(
                "descending_rank"
            )
        )

        # Range across the omitted samples.
        axis.fill_betweenx(
            envelope[
                "exceedance_probability"
            ].to_numpy(),
            envelope[
                "minimum_severity"
            ].to_numpy(),
            envelope[
                "maximum_severity"
            ].to_numpy(),
            color=envelope_colour,
            alpha=0.32,
            linewidth=0,
            zorder=1,
        )

        # Individual leave-one-winter-out curves.
        for (
            omitted_year,
            one_curve,
        ) in omitted_curves.groupby(
            "omitted_winter_year"
        ):

            one_curve = one_curve.sort_values(
                "descending_rank"
            )

            axis.plot(
                one_curve[
                    "standardised_severity"
                ],
                one_curve[
                    "exceedance_probability"
                ],
                color=omitted_curve_colour,
                alpha=0.24,
                linewidth=0.65,
                zorder=2,
            )

        # Full pooled-sample curve.
        axis.plot(
            full_curve[
                "standardised_severity"
            ],
            full_curve[
                "exceedance_probability"
            ],
            color=full_sample_colour,
            linewidth=2.2,
            zorder=4,
        )

        axis.set_title(
            f"{window}-day rolling extreme",
            fontsize=FONT_TITLE,
        )

        axis.set_yscale(
            "log"
        )

        axis.grid(
            True,
            which="both",
            alpha=0.22,
            linewidth=0.7,
        )

        axis.set_axisbelow(
            True
        )

        axis.tick_params(
            axis="both",
            which="major",
            labelsize=FONT_TICK,
        )

    # X-axis labels only for the bottom row.
    for axis in axes[-2:]:
        axis.set_xlabel(
            "Standardised demand severity\n"
            "(value - sample median) / sample IQR",
            fontsize=FONT_AXIS_LABEL,
            labelpad=4,
        )

    # Y-axis labels only for the left column.
    for axis in axes[::2]:
        axis.set_ylabel(
            "Empirical exceedance probability",
            fontsize=FONT_AXIS_LABEL,
        )

    legend_items = [
        Line2D(
            [0],
            [0],
            color=full_sample_colour,
            linewidth=2.2,
            label="Full sample",
        ),
        Line2D(
            [0],
            [0],
            color=omitted_curve_colour,
            alpha=0.60,
            linewidth=1.0,
            label="Each winter-year omission",
        ),
        Patch(
            facecolor=envelope_colour,
            alpha=0.32,
            edgecolor="none",
            label="Range across winter-year omissions",
        ),
    ]

    fig.suptitle(
        "Sensitivity of the standardised SEAS5 upper-tail shape "
        "to individual winter years",
        fontsize=FONT_SUPTITLE,
    )

    fig.legend(
        handles=legend_items,
        loc="lower center",
        bbox_to_anchor=(
            0.5,
            -0.06,
        ),
        ncol=3,
        frameon=True,
        fontsize=FONT_LEGEND,
    )

    fig.savefig(
        OUT_FIGURE_PNG,
        dpi=FIGURE_DPI,
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
# Run
# ============================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "The rolling-extremes input file could not be found:\n"
            f"{INPUT_FILE}\n\n"
            "Run the temperature-tail and concentration analysis "
            "before running this script."
        )

    data = pd.read_csv(
        INPUT_FILE
    )

    validate_input(
        data
    )

    curve_data = calculate_curves(
        data
    )

    curve_data.to_csv(
        OUT_CURVE_DATA,
        index=False,
    )

    plot_curves(
        curve_data
    )

    print()
    print(
        "Leave-one-winter-year-out shoulder analysis complete."
    )
    print(
        f"Curve data: {OUT_CURVE_DATA}"
    )
    print(
        f"PNG figure: {OUT_FIGURE_PNG}"
    )
    print(
        f"PDF figure: {OUT_FIGURE_PDF}"
    )


if __name__ == "__main__":
    main()