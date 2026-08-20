"""
Joint-winter removal sensitivity for the visually selected upper-tail shoulder.

This script is intended to follow 32_final_13_select_shoulder_points.py.
It does not modify or delete the original data. All removals are temporary
filters applied in memory.

The script:
1. reads the realisations selected around the 21-, 28- and 35-day shoulders;
2. identifies the four winter years that occur most often across those regions;
3. removes those four winters jointly and recalculates the standardisation;
4. compares this targeted removal with random removals of four winters; and
5. saves the comparison curves, summary tables and a three-panel figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUTPUT_DIR


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------

WINDOWS = [21, 28, 35]
TOP_N_WINTERS = 4
UPPER_TAIL_FRACTION = 0.20
N_RANDOM_REMOVALS = 1000
RNG_SEED = 20260813
FIGURE_DPI = 300

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

INPUT_CSV = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "05_temperature_tail_and_bump_concentration"
    / "temperature_demand_rolling_extremes_by_winter_member.csv"
)

SELECTED_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "13_select_shoulder_points"
)

SELECTED_REALISATIONS_CSV = (
    SELECTED_DIR / "selected_shoulder_realisations.csv"
)

SELECTED_RANGES_CSV = (
    SELECTED_DIR / "selected_shoulder_rank_ranges.csv"
)

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "14_joint_winter_removal_sensitivity"
)


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def robust_standardise(values: np.ndarray) -> np.ndarray:
    """Standardise values using the sample median and interquartile range."""
    values = np.asarray(values, dtype=float)
    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25

    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError("The interquartile range is zero or invalid.")

    return (values - median) / iqr


def empirical_upper_tail(data: pd.DataFrame) -> pd.DataFrame:
    """Return a descending empirical upper-tail curve."""
    values = data["maximum_mean_demand_GW"].to_numpy(dtype=float)
    standardised = robust_standardise(values)
    order = np.argsort(standardised)[::-1]

    ranked = data.iloc[order].copy().reset_index(drop=True)
    ranked["standardised_severity"] = standardised[order]
    ranked["descending_rank"] = np.arange(1, len(ranked) + 1)
    ranked["exceedance_probability"] = (
        ranked["descending_rank"] / (len(ranked) + 1)
    )

    number_to_keep = int(np.ceil(UPPER_TAIL_FRACTION * len(ranked)))
    return ranked.iloc[:number_to_keep].copy()


def validate_inputs(
    data: pd.DataFrame,
    selected: pd.DataFrame,
    ranges: pd.DataFrame,
) -> None:
    required_data_columns = {
        "winter_year",
        "member",
        "window_days",
        "maximum_mean_demand_GW",
    }
    required_selected_columns = {
        "winter_year",
        "member",
        "window_days",
        "descending_rank",
    }
    required_range_columns = {"window_days", "start_rank", "end_rank"}

    missing_data = required_data_columns.difference(data.columns)
    missing_selected = required_selected_columns.difference(selected.columns)
    missing_ranges = required_range_columns.difference(ranges.columns)

    if missing_data:
        raise ValueError(f"Input data are missing columns: {sorted(missing_data)}")
    if missing_selected:
        raise ValueError(
            "Selected-realisations file is missing columns: "
            f"{sorted(missing_selected)}"
        )
    if missing_ranges:
        raise ValueError(
            f"Selected-ranges file is missing columns: {sorted(missing_ranges)}"
        )

    for window in WINDOWS:
        window_data = data.loc[data["window_days"] == window]
        window_selected = selected.loc[selected["window_days"] == window]
        window_ranges = ranges.loc[ranges["window_days"] == window]

        if window_data.empty:
            raise ValueError(f"No original data found for the {window}-day window.")
        if window_selected.empty:
            raise ValueError(
                f"No selected shoulder realisations found for {window} days."
            )
        if len(window_ranges) != 1:
            raise ValueError(
                f"Expected one selected rank range for {window} days, "
                f"but found {len(window_ranges)}."
            )

        counts = window_data.groupby("winter_year").size()
        if counts.nunique() != 1:
            raise ValueError(
                f"The {window}-day data are not balanced across winter years."
            )


def build_winter_ranking(selected: pd.DataFrame) -> pd.DataFrame:
    """Rank winters by repeated occurrence in the three selected regions."""
    ranking = (
        selected.groupby("winter_year")
        .agg(
            selected_realisations_across_windows=("winter_year", "size"),
            shoulder_windows_represented=("window_days", "nunique"),
        )
        .reset_index()
        .sort_values(
            [
                "shoulder_windows_represented",
                "selected_realisations_across_windows",
                "winter_year",
            ],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    ranking["combined_rank"] = np.arange(1, len(ranking) + 1)
    ranking["selected_for_joint_removal"] = (
        ranking["combined_rank"] <= TOP_N_WINTERS
    )
    return ranking


def make_random_removal_envelope(
    window_data: pd.DataFrame,
    winter_years: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create a 5th--95th percentile envelope from random four-winter removals."""
    random_curves = []
    probabilities = None
    ranks = None

    for _ in range(N_RANDOM_REMOVALS):
        removed = rng.choice(
            winter_years,
            size=TOP_N_WINTERS,
            replace=False,
        )
        remaining = window_data.loc[
            ~window_data["winter_year"].isin(removed)
        ].copy()
        curve = empirical_upper_tail(remaining)

        if probabilities is None:
            probabilities = curve["exceedance_probability"].to_numpy()
            ranks = curve["descending_rank"].to_numpy()

        random_curves.append(curve["standardised_severity"].to_numpy())

    random_curves = np.vstack(random_curves)

    return pd.DataFrame(
        {
            "descending_rank": ranks,
            "exceedance_probability": probabilities,
            "random_lower_5_percent": np.quantile(
                random_curves, 0.05, axis=0
            ),
            "random_median": np.quantile(random_curves, 0.50, axis=0),
            "random_upper_95_percent": np.quantile(
                random_curves, 0.95, axis=0
            ),
        }
    )


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Original input file not found: {INPUT_CSV}")
    if not SELECTED_REALISATIONS_CSV.exists():
        raise FileNotFoundError(
            "Run script 32 first. File not found: "
            f"{SELECTED_REALISATIONS_CSV}"
        )
    if not SELECTED_RANGES_CSV.exists():
        raise FileNotFoundError(
            "Run script 32 first. File not found: "
            f"{SELECTED_RANGES_CSV}"
        )

    data = pd.read_csv(INPUT_CSV)
    selected = pd.read_csv(SELECTED_REALISATIONS_CSV)
    ranges = pd.read_csv(SELECTED_RANGES_CSV)

    data = data.loc[data["window_days"].isin(WINDOWS)].copy()
    selected = selected.loc[selected["window_days"].isin(WINDOWS)].copy()
    ranges = ranges.loc[ranges["window_days"].isin(WINDOWS)].copy()

    validate_inputs(data, selected, ranges)

    winter_ranking = build_winter_ranking(selected)
    winters_to_remove = (
        winter_ranking.loc[
            winter_ranking["selected_for_joint_removal"], "winter_year"
        ]
        .astype(int)
        .tolist()
    )

    winter_ranking.to_csv(
        OUT_DIR / "combined_shoulder_winter_ranking.csv",
        index=False,
    )

    rng = np.random.default_rng(RNG_SEED)
    all_curve_rows = []
    all_envelope_rows = []
    summary_rows = []

    fig, axes = plt.subplots(
        1,
        len(WINDOWS),
        figsize=(16.2, 6.4),
        sharey=True,
    )

    for ax, window in zip(axes, WINDOWS):
        window_data = data.loc[data["window_days"] == window].copy()
        winter_years = np.sort(window_data["winter_year"].unique())

        full_curve = empirical_upper_tail(window_data)

        targeted_remaining = window_data.loc[
            ~window_data["winter_year"].isin(winters_to_remove)
        ].copy()
        targeted_curve = empirical_upper_tail(targeted_remaining)

        envelope = make_random_removal_envelope(
            window_data,
            winter_years,
            rng,
        )
        envelope["window_days"] = window
        all_envelope_rows.append(envelope)

        full_output = full_curve.copy()
        full_output["window_days"] = window
        full_output["scenario"] = "full_sample"
        full_output["removed_winter_years"] = ""
        all_curve_rows.append(full_output)

        targeted_output = targeted_curve.copy()
        targeted_output["window_days"] = window
        targeted_output["scenario"] = "joint_targeted_removal"
        targeted_output["removed_winter_years"] = ", ".join(
            map(str, winters_to_remove)
        )
        all_curve_rows.append(targeted_output)

        selected_range = ranges.loc[ranges["window_days"] == window].iloc[0]
        start_rank = int(selected_range["start_rank"])
        end_rank = int(selected_range["end_rank"])
        full_n = len(window_data)
        shoulder_probability_low = start_rank / (full_n + 1)
        shoulder_probability_high = end_rank / (full_n + 1)

        ax.axhspan(
            shoulder_probability_low,
            shoulder_probability_high,
            color="#F2C14E",
            alpha=0.16,
            label="Selected shoulder rank range",
            zorder=0,
        )

        ax.fill_betweenx(
            envelope["exceedance_probability"],
            envelope["random_lower_5_percent"],
            envelope["random_upper_95_percent"],
            color="0.75",
            alpha=0.45,
            linewidth=0,
            label="Random removal of four winters: 5--95% range",
            zorder=1,
        )

        ax.plot(
            full_curve["standardised_severity"],
            full_curve["exceedance_probability"],
            color="#2166AC",
            linewidth=2.2,
            label="Full sample",
            zorder=3,
        )

        ax.plot(
            targeted_curve["standardised_severity"],
            targeted_curve["exceedance_probability"],
            color="#D73027",
            linewidth=2.0,
            linestyle="--",
            label=(
                "Remove four most recurrent winters\n"
                + ", ".join(map(str, winters_to_remove))
            ),
            zorder=4,
        )

        ax.set_yscale("log")
        ax.set_title(
            f"{window}-day rolling extreme",
            fontsize=FONT_TITLE,
        )
        ax.set_xlabel(
            "Standardised demand severity",
            fontsize=FONT_AXIS_LABEL,
        )
        ax.tick_params(
            axis="both",
            which="major",
            labelsize=FONT_TICK,
        )
        ax.grid(True, which="both", alpha=0.25)

        summary_rows.append(
            {
                "window_days": window,
                "full_sample_size": len(window_data),
                "targeted_remaining_sample_size": len(targeted_remaining),
                "number_of_removed_winters": len(winters_to_remove),
                "removed_winter_years": ", ".join(
                    map(str, winters_to_remove)
                ),
                "selected_start_rank": start_rank,
                "selected_end_rank": end_rank,
                "number_of_random_removals": N_RANDOM_REMOVALS,
            }
        )

    axes[0].set_ylabel(
        "Empirical exceedance probability",
        fontsize=FONT_AXIS_LABEL,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=2,
        frameon=True,
        fontsize=FONT_LEGEND,
    )

    fig.suptitle(
        "Sensitivity of the upper-tail shoulder to joint winter removal",
        fontsize=FONT_SUPTITLE,
        y=0.98,
    )

    fig.tight_layout(
        rect=[0, 0.14, 1, 0.92],
        w_pad=1.8,
    )

    fig.savefig(
        OUT_DIR / "joint_winter_removal_sensitivity.png",
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT_DIR / "joint_winter_removal_sensitivity.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    curves = pd.concat(all_curve_rows, ignore_index=True)
    envelopes = pd.concat(all_envelope_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)

    curves.to_csv(
        OUT_DIR / "joint_winter_removal_curves.csv",
        index=False,
    )
    envelopes.to_csv(
        OUT_DIR / "random_four_winter_removal_envelope.csv",
        index=False,
    )
    summary.to_csv(
        OUT_DIR / "joint_winter_removal_summary.csv",
        index=False,
    )

    print("\nJoint-winter removal sensitivity completed.")
    print("Winters removed jointly:", winters_to_remove)
    print("Original data were read only and were not modified.")
    print("Output directory:", OUT_DIR)
    print("Figure:", OUT_DIR / "joint_winter_removal_sensitivity.pdf")


if __name__ == "__main__":
    main()