from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR


# ============================================================
# Select the realisations corresponding to the visually
# identified upper-tail shoulder
# ============================================================


# ============================================================
# Settings
# ============================================================

WINDOWS = [
    21,
    28,
    35,
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
    / "13_select_shoulder_points"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_RANGES = (
    OUT_DIR
    / "selected_shoulder_rank_ranges.csv"
)

OUT_SELECTED_POINTS = (
    OUT_DIR
    / "selected_shoulder_realisations.csv"
)

OUT_WINTER_COUNTS = (
    OUT_DIR
    / "selected_shoulder_winter_counts.csv"
)

OUT_SUMMARY = (
    OUT_DIR
    / "selected_shoulder_composition_summary.csv"
)

OUT_FIGURE_PNG = (
    OUT_DIR
    / "selected_shoulder_regions.png"
)

OUT_FIGURE_PDF = (
    OUT_DIR
    / "selected_shoulder_regions.pdf"
)


# ============================================================
# Standardisation and ranking
# ============================================================

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
            "The IQR is not positive."
        )

    standardised = (
        values - median_value
    ) / iqr_value

    return (
        standardised,
        float(median_value),
        float(iqr_value),
    )


def create_ranked_data(
    data: pd.DataFrame,
    window: int,
) -> pd.DataFrame:

    ranked = (
        data.loc[
            data[WINDOW_COLUMN] == window,
            [
                YEAR_COLUMN,
                MEMBER_COLUMN,
                WINDOW_COLUMN,
                SEVERITY_COLUMN,
            ],
        ]
        .copy()
    )

    if ranked.empty:
        raise ValueError(
            f"No data found for the {window}-day window."
        )

    (
        standardised,
        sample_median,
        sample_iqr,
    ) = robust_standardise(
        ranked[SEVERITY_COLUMN].to_numpy()
    )

    ranked["standardised_severity"] = (
        standardised
    )

    ranked = (
        ranked.sort_values(
            "standardised_severity",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranked["descending_rank"] = np.arange(
        1,
        len(ranked) + 1,
    )

    ranked["exceedance_probability"] = (
        ranked["descending_rank"]
        / (len(ranked) + 1.0)
    )

    ranked["sample_median_GW"] = (
        sample_median
    )

    ranked["sample_IQR_GW"] = (
        sample_iqr
    )

    ranked = ranked.loc[
        ranked["exceedance_probability"]
        <= UPPER_TAIL_FRACTION
    ].copy()

    return ranked


# ============================================================
# Match each click to the nearest plotted observation
# ============================================================

def find_nearest_rank(
    axis,
    ranked: pd.DataFrame,
    clicked_point: tuple[float, float],
) -> int:

    curve_coordinates = np.column_stack(
        [
            ranked[
                "standardised_severity"
            ].to_numpy(),
            ranked[
                "exceedance_probability"
            ].to_numpy(),
        ]
    )

    curve_pixels = axis.transData.transform(
        curve_coordinates
    )

    clicked_pixel = axis.transData.transform(
        np.asarray(clicked_point)
    )

    distances = np.sqrt(
        np.sum(
            (
                curve_pixels
                - clicked_pixel
            ) ** 2,
            axis=1,
        )
    )

    nearest_position = int(
        np.argmin(distances)
    )

    return int(
        ranked.iloc[
            nearest_position
        ]["descending_rank"]
    )


# ============================================================
# Interactive shoulder selection
# ============================================================

def select_shoulder_points(
    ranked: pd.DataFrame,
    window: int,
) -> tuple[int, int, pd.DataFrame]:

    fig, axis = plt.subplots(
        figsize=(8.8, 6.4)
    )

    axis.plot(
        ranked["standardised_severity"],
        ranked["exceedance_probability"],
        color="#1764AB",
        linewidth=2.2,
    )

    axis.scatter(
        ranked["standardised_severity"],
        ranked["exceedance_probability"],
        color="#1764AB",
        s=18,
        zorder=3,
    )

    axis.set_yscale(
        "log"
    )

    axis.set_xlabel(
        "Standardised demand severity",
        fontsize=FONT_AXIS_LABEL,
    )

    axis.set_ylabel(
        "Empirical exceedance probability",
        fontsize=FONT_AXIS_LABEL,
    )

    axis.set_title(
        f"{window}-day rolling extreme\n"
        "Click the beginning and end of the shoulder",
        fontsize=FONT_TITLE,
    )

    axis.tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )

    axis.grid(
        True,
        which="both",
        alpha=0.22,
    )

    fig.tight_layout()
    fig.canvas.draw()

    print()
    print(
        f"{window}-day curve:"
    )

    print(
        "Click the two ends of the shoulder region."
    )

    plt.show(
        block=False
    )

    clicked_points = plt.ginput(
        2,
        timeout=-1,
        show_clicks=True,
    )

    if len(clicked_points) != 2:
        plt.close(fig)

        raise RuntimeError(
            "Two shoulder boundaries were not selected."
        )

    clicked_ranks = [
        find_nearest_rank(
            axis,
            ranked,
            clicked_point,
        )
        for clicked_point in clicked_points
    ]

    start_rank = min(
        clicked_ranks
    )

    end_rank = max(
        clicked_ranks
    )

    if start_rank == end_rank:
        plt.close(fig)

        raise ValueError(
            "Both clicks correspond to the same rank."
        )

    selected = ranked.loc[
        ranked["descending_rank"].between(
            start_rank,
            end_rank,
        )
    ].copy()

    axis.scatter(
        selected[
            "standardised_severity"
        ],
        selected[
            "exceedance_probability"
        ],
        color="#E15759",
        s=36,
        zorder=5,
        label=(
            f"Selected ranks "
            f"{start_rank}--{end_rank}"
        ),
    )

    axis.legend(
        fontsize=FONT_LEGEND
    )

    fig.canvas.draw()
    plt.pause(1.2)
    plt.close(fig)

    print(
        f"Selected ranks: "
        f"{start_rank}--{end_rank}"
    )

    return (
        start_rank,
        end_rank,
        selected,
    )


# ============================================================
# Winter-year composition
# ============================================================

def calculate_winter_counts(
    selected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    winter_counts = (
        selected.groupby(
            YEAR_COLUMN
        )
        .size()
        .rename(
            "number_of_realisations"
        )
        .reset_index()
        .sort_values(
            [
                "number_of_realisations",
                YEAR_COLUMN,
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )

    total = len(
        selected
    )

    winter_counts["share"] = (
        winter_counts[
            "number_of_realisations"
        ]
        / total
    )

    winter_counts["share_percent"] = (
        100.0
        * winter_counts["share"]
    )

    summary = {
        "number_of_selected_realisations": total,
        "number_of_winter_years": (
            winter_counts[
                YEAR_COLUMN
            ].nunique()
        ),
        "largest_one_winter_share": (
            winter_counts.head(1)[
                "share"
            ].sum()
        ),
        "largest_three_winter_share": (
            winter_counts.head(3)[
                "share"
            ].sum()
        ),
        "largest_five_winter_share": (
            winter_counts.head(5)[
                "share"
            ].sum()
        ),
        "top_three_winters": str(
            winter_counts.head(3)[
                YEAR_COLUMN
            ].tolist()
        ),
        "top_five_winters": str(
            winter_counts.head(5)[
                YEAR_COLUMN
            ].tolist()
        ),
    }

    return (
        winter_counts,
        summary,
    )


# ============================================================
# Final figure showing the selected observations
# ============================================================

def plot_selected_regions(
    ranked_by_window: dict,
    selected_by_window: dict,
    ranges: dict,
) -> None:

    fig, axes = plt.subplots(
        1,
        len(WINDOWS),
        figsize=(17.0, 5.8),
        sharey=True,
        constrained_layout=True,
    )

    if len(WINDOWS) == 1:
        axes = [axes]

    for axis, window in zip(
        axes,
        WINDOWS,
    ):

        ranked = ranked_by_window[
            window
        ]

        selected = selected_by_window[
            window
        ]

        (
            start_rank,
            end_rank,
        ) = ranges[
            window
        ]

        axis.plot(
            ranked[
                "standardised_severity"
            ],
            ranked[
                "exceedance_probability"
            ],
            color="#1764AB",
            linewidth=2.2,
            label="Full empirical curve",
        )

        axis.scatter(
            selected[
                "standardised_severity"
            ],
            selected[
                "exceedance_probability"
            ],
            color="#E15759",
            s=34,
            zorder=5,
            label=(
                f"Selected ranks "
                f"{start_rank}--{end_rank}"
            ),
        )

        axis.set_yscale(
            "log"
        )

        axis.set_xlabel(
            "Standardised demand severity",
            fontsize=FONT_AXIS_LABEL,
        )

        axis.set_title(
            f"{window}-day rolling extreme",
            fontsize=FONT_TITLE,
        )

        axis.tick_params(
            axis="both",
            which="major",
            labelsize=FONT_TICK,
        )

        axis.grid(
            True,
            which="both",
            alpha=0.22,
        )

        axis.legend(
            fontsize=FONT_LEGEND,
            loc="upper right",
        )

    axes[0].set_ylabel(
        "Empirical exceedance probability",
        fontsize=FONT_AXIS_LABEL,
    )

    fig.suptitle(
        "Realisations selected around the upper-tail shoulder",
        fontsize=FONT_SUPTITLE,
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
            "Input file not found:\n"
            f"{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE
    )

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
            "Missing columns: "
            f"{sorted(missing_columns)}"
        )

    ranked_by_window = {}
    selected_by_window = {}
    ranges = {}

    selected_frames = []
    count_frames = []
    range_rows = []
    summary_rows = []

    for window in WINDOWS:

        ranked = create_ranked_data(
            data,
            window,
        )

        (
            start_rank,
            end_rank,
            selected,
        ) = select_shoulder_points(
            ranked,
            window,
        )

        (
            winter_counts,
            summary,
        ) = calculate_winter_counts(
            selected
        )

        ranked_by_window[
            window
        ] = ranked

        selected_by_window[
            window
        ] = selected

        ranges[
            window
        ] = (
            start_rank,
            end_rank,
        )

        selected[
            "selected_start_rank"
        ] = start_rank

        selected[
            "selected_end_rank"
        ] = end_rank

        winter_counts[
            WINDOW_COLUMN
        ] = window

        summary[
            WINDOW_COLUMN
        ] = window

        summary[
            "start_rank"
        ] = start_rank

        summary[
            "end_rank"
        ] = end_rank

        selected_frames.append(
            selected
        )

        count_frames.append(
            winter_counts
        )

        range_rows.append(
            {
                WINDOW_COLUMN: window,
                "start_rank": start_rank,
                "end_rank": end_rank,
            }
        )

        summary_rows.append(
            summary
        )

    selected_data = pd.concat(
        selected_frames,
        ignore_index=True,
    )

    winter_count_data = pd.concat(
        count_frames,
        ignore_index=True,
    )

    range_data = pd.DataFrame(
        range_rows
    )

    summary_data = pd.DataFrame(
        summary_rows
    )

    selected_data.to_csv(
        OUT_SELECTED_POINTS,
        index=False,
    )

    winter_count_data.to_csv(
        OUT_WINTER_COUNTS,
        index=False,
    )

    range_data.to_csv(
        OUT_RANGES,
        index=False,
    )

    summary_data.to_csv(
        OUT_SUMMARY,
        index=False,
    )

    plot_selected_regions(
        ranked_by_window,
        selected_by_window,
        ranges,
    )

    print()
    print(
        "Shoulder point selection complete."
    )

    print()
    print(
        range_data.to_string(
            index=False
        )
    )

    print()
    print(
        summary_data[
            [
                WINDOW_COLUMN,
                "start_rank",
                "end_rank",
                "number_of_selected_realisations",
                "number_of_winter_years",
                "largest_one_winter_share",
                "largest_three_winter_share",
                "largest_five_winter_share",
                "top_three_winters",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        f"Selected observations:\n"
        f"{OUT_SELECTED_POINTS}"
    )

    print(
        f"Winter-year counts:\n"
        f"{OUT_WINTER_COUNTS}"
    )

    print(
        f"Selected-region figure:\n"
        f"{OUT_FIGURE_PDF}"
    )


if __name__ == "__main__":
    main()