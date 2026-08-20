from pathlib import Path
import argparse
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Final analysis 16
# Main-text summary of the non-stationary GEV sensitivity check
#
# This script does not refit either GEV model. It reads the model-
# comparison CSV produced by the non-stationary analysis and creates
# a two-panel figure corresponding to the stationary GEV summary:
#   1. AIC(shape trend) - AIC(stationary)
#   2. Estimated change in the GEV shape parameter per decade
# ============================================================


PRIMARY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]

SAMPLES = [
    "ECMWF pooled",
    "Hannah overlap shifted",
]

INPUT_FILENAME = (
    "stationary_vs_nonstationary_shape_trend_comparison.csv"
)

# Figure and font settings
FIGURE_DPI = 300

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

try:
    from config import OUTPUT_DIR as PROJECT_OUTPUT_DIR
except ModuleNotFoundError:
    # Allows the script to be tested outside the dissertation project.
    # In the dissertation project, config.OUTPUT_DIR is used instead.
    PROJECT_OUTPUT_DIR = Path.cwd()


DEFAULT_INPUT = (
    PROJECT_OUTPUT_DIR
    / "final_analysis_clean"
    / "07_nonstationary_shape_trend_sensitivity"
    / INPUT_FILENAME
)

DEFAULT_OUT_DIR = (
    PROJECT_OUTPUT_DIR
    / "final_analysis_clean"
    / "16_nonstationary_gev_main_text"
)


REQUIRED_COLUMNS = {
    "window_days",
    "sample",
    "delta_aic_nonstationary_minus_stationary",
    "xi_change_per_decade",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the main-text non-stationary GEV summary figure."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional path to the stationary-versus-non-stationary "
            "comparison CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory in which the PDF and PNG are written.",
    )

    return parser.parse_args()


def find_input_csv(explicit_path: Optional[Path]) -> Path:
    if explicit_path is not None:
        path = explicit_path.expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(
                f"Input CSV not found:\n{path}"
            )

        return path

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    matches = sorted(
        PROJECT_OUTPUT_DIR.rglob(INPUT_FILENAME)
    )

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        raise FileNotFoundError(
            "Could not find the non-stationary comparison CSV.\n"
            f"Expected location:\n{DEFAULT_INPUT}\n\n"
            "Alternatively run this script with:\n"
            "  --input /full/path/to/"
            f"{INPUT_FILENAME}"
        )

    match_text = "\n".join(
        f"  - {path}"
        for path in matches
    )

    raise RuntimeError(
        "More than one comparison CSV was found. "
        "Use --input to select the final file:\n"
        f"{match_text}"
    )


def read_and_validate(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "The comparison CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data = data[
        data["window_days"].isin(PRIMARY_WINDOWS)
        & data["sample"].isin(SAMPLES)
    ].copy()

    duplicate_rows = data.duplicated(
        subset=["window_days", "sample"],
        keep=False,
    )

    if duplicate_rows.any():
        raise ValueError(
            "The comparison CSV contains duplicate sample-window rows."
        )

    expected_pairs = {
        (window, sample)
        for window in PRIMARY_WINDOWS
        for sample in SAMPLES
    }

    observed_pairs = set(
        zip(
            data["window_days"],
            data["sample"],
        )
    )

    missing_pairs = expected_pairs.difference(
        observed_pairs
    )

    if missing_pairs:
        missing_text = ", ".join(
            f"{window} days / {sample}"
            for window, sample in sorted(missing_pairs)
        )

        raise ValueError(
            "The comparison CSV is incomplete: "
            f"{missing_text}"
        )

    numeric_columns = [
        "window_days",
        "delta_aic_nonstationary_minus_stationary",
        "xi_change_per_decade",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    if not np.isfinite(
        data[numeric_columns].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "The comparison CSV contains missing or non-finite values."
        )

    return data.sort_values(
        ["sample", "window_days"]
    )


def make_summary_figure(
    data: pd.DataFrame,
    output_dir: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_png = (
        output_dir
        / "nonstationary_shape_trend_summary.png"
    )

    output_pdf = (
        output_dir
        / "nonstationary_shape_trend_summary.pdf"
    )

    # Enlarged main-text summary figure, matching the dissertation-wide
    # font sizes used for the other final figures.
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(13.8, 6.0),
        dpi=FIGURE_DPI,
    )

    for sample_name in SAMPLES:
        sample_data = data[
            data["sample"] == sample_name
        ].sort_values("window_days")

        axes[0].plot(
            sample_data["window_days"],
            sample_data[
                "delta_aic_nonstationary_minus_stationary"
            ],
            marker="o",
            markersize=6.0,
            linewidth=1.8,
            label=sample_name,
        )

        axes[1].plot(
            sample_data["window_days"],
            sample_data["xi_change_per_decade"],
            marker="o",
            markersize=6.0,
            linewidth=1.8,
            label=sample_name,
        )

    # --------------------------------------------------------
    # Left: model comparison
    # --------------------------------------------------------
    axes[0].axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
    )

    axes[0].set_xlabel(
        "Averaging window (days)",
        fontsize=FONT_AXIS_LABEL,
    )

    axes[0].set_ylabel(
        r"$\Delta$AIC: shape-trend minus stationary",
        fontsize=FONT_AXIS_LABEL,
    )

    axes[0].set_title(
        "AIC comparison",
        fontsize=FONT_TITLE,
    )

    axes[0].set_xticks(
        PRIMARY_WINDOWS
    )

    axes[0].tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend(
        fontsize=FONT_LEGEND,
    )

    # --------------------------------------------------------
    # Right: estimated shape trend
    # --------------------------------------------------------
    axes[1].axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
    )

    axes[1].set_xlabel(
        "Averaging window (days)",
        fontsize=FONT_AXIS_LABEL,
    )

    axes[1].set_ylabel(
        r"Estimated shape change per decade, $\xi_1$",
        fontsize=FONT_AXIS_LABEL,
    )

    axes[1].set_title(
        "Estimated shape trend",
        fontsize=FONT_TITLE,
    )

    axes[1].set_xticks(
        PRIMARY_WINDOWS
    )

    axes[1].tick_params(
        axis="both",
        which="major",
        labelsize=FONT_TICK,
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend(
        fontsize=FONT_LEGEND,
    )

    fig.suptitle(
        "Non-stationary GEV sensitivity across pre-selected primary windows",
        fontsize=FONT_SUPTITLE,
        y=1.02,
    )

    fig.tight_layout(
        pad=1.3,
        w_pad=2.0,
    )

    fig.savefig(
        output_png,
        bbox_inches="tight",
    )

    fig.savefig(
        output_pdf,
        bbox_inches="tight",
    )

    plt.close(fig)

    return output_png, output_pdf


def main() -> None:
    args = parse_args()
    input_csv = find_input_csv(args.input)
    data = read_and_validate(input_csv)

    output_png, output_pdf = make_summary_figure(
        data=data,
        output_dir=args.output_dir,
    )

    print("\nInput comparison CSV:")
    print(input_csv)

    print("\nValues plotted:")
    print(
        data[
            [
                "window_days",
                "sample",
                "delta_aic_nonstationary_minus_stationary",
                "xi_change_per_decade",
            ]
        ].to_string(index=False)
    )

    print("\nSaved figure:")
    print(output_png)
    print(output_pdf)


if __name__ == "__main__":
    main()
