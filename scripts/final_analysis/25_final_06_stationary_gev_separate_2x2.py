from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import genextreme

from config import OUTPUT_DIR


# ============================================================
# Final analysis 06
# Stationary GEV for all pre-selected primary windows
#
# Main comparison:
#   1. ECMWF pooled winter-member maxima, 1982-2016
#   2. Hannah overlap winter maxima, 1982-2016, shifted onto
#      the ECMWF mean scale separately for each duration
#
# Primary windows:
#   1, 7, 14, 21, 28, 56 and 84 days
#
# Important:
#   - All winter maxima are used in each GEV fit.
#   - The 21-35-day empirical shoulder is not removed.
#   - GEV is used as a smooth first-order tail summary.
#   - ECMWF uncertainty uses a winter-year cluster bootstrap:
#     each sampled year retains all 25 members.
#   - Hannah uses the same sampled winter years, one value per year.
#   - The duration-specific mean shift is recomputed inside every
#     bootstrap replicate.
# ============================================================


# ============================================================
# Settings
# ============================================================

PRIMARY_WINDOWS = [
    1,
    7,
    14,
    21,
    28,
    56,
    84,
]

RETURN_PERIODS = [
    10,
    20,
    50,
]

OVERLAP_START = 1982
OVERLAP_END = 2016

N_BOOT = 500
RANDOM_SEED = 20260716

# Figure and font settings
FIGURE_DPI = 300

FONT_BASE = 13.0
FONT_AXIS_LABEL = 14.0
FONT_TICK = 12.0
FONT_TITLE = 14.0
FONT_LEGEND = 11.5
FONT_SUPTITLE = 16.0
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

# Do not assign final quality labels before inspecting the new
# diagnostics. Edit this dictionary after visual review.
DIAGNOSTIC_RATING = {
    1: "to review",
    7: "to review",
    14: "to review",
    21: "to review",
    28: "to review",
    56: "to review",
    84: "to review",
}


# ============================================================
# Input files
# ============================================================

ECMWF_SEVERITY_FILE = (
    OUTPUT_DIR
    / "severity"
    / "ecmwf_severity_summary_Nov08_1982_2016_extended_windows.csv"
)

HANNAH_SEVERITY_FILE = (
    OUTPUT_DIR
    / "severity"
    / "hannah_severity_summary_Nov08_extended_windows.csv"
)


# ============================================================
# Output files
# ============================================================

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "06_stationary_gev_primary_windows"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DIAGNOSTIC_DIR = (
    OUT_DIR
    / "diagnostics"
)

DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ECMWF_DIAGNOSTIC_DIR = (
    DIAGNOSTIC_DIR
    / "ecmwf_pooled"
)

HANNAH_DIAGNOSTIC_DIR = (
    DIAGNOSTIC_DIR
    / "hannah_overlap_shifted"
)

ECMWF_DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HANNAH_DIAGNOSTIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUT_ORIGINAL_FITS = (
    OUT_DIR
    / "stationary_gev_original_fits_primary_windows.csv"
)

OUT_PARAMETER_SAMPLES = (
    OUT_DIR
    / "stationary_gev_bootstrap_parameter_samples.csv"
)

OUT_PARAMETER_CI = (
    OUT_DIR
    / "stationary_gev_bootstrap_parameter_ci.csv"
)

OUT_RL_SAMPLES = (
    OUT_DIR
    / "stationary_gev_bootstrap_return_level_samples.csv"
)

OUT_RL_CI = (
    OUT_DIR
    / "stationary_gev_bootstrap_return_level_ci.csv"
)

OUT_DIFF_SAMPLES = (
    OUT_DIR
    / "stationary_gev_bootstrap_return_level_difference_samples.csv"
)

OUT_DIFF_CI = (
    OUT_DIR
    / "stationary_gev_bootstrap_return_level_difference_ci.csv"
)

OUT_XI_COMPARISON = (
    OUT_DIR
    / "stationary_gev_shape_parameter_comparison.csv"
)

OUT_MAIN_FIGURE_PNG = (
    OUT_DIR
    / "stationary_gev_primary_windows_summary.png"
)

OUT_MAIN_FIGURE_PDF = (
    OUT_DIR
    / "stationary_gev_primary_windows_summary.pdf"
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
        np.isfinite(values)
    ]


def fit_stationary_gev(
    values_gw,
) -> dict:
    """
    Fit a stationary GEV using scipy.stats.genextreme.

    SciPy uses shape c = -xi.
    This function reports the conventional EVT shape xi.
    """
    values_gw = finite_values(
        values_gw
    )

    if len(values_gw) < 10:
        raise ValueError(
            "Too few observations for a GEV fit."
        )

    if np.std(
        values_gw,
        ddof=1,
    ) <= 0:
        raise ValueError(
            "Cannot fit GEV to a zero-variance sample."
        )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore"
        )

        c, loc, scale = genextreme.fit(
            values_gw
        )

    if not np.all(
        np.isfinite(
            [
                c,
                loc,
                scale,
            ]
        )
    ):
        raise ValueError(
            "Non-finite stationary GEV parameters."
        )

    if scale <= 0:
        raise ValueError(
            "Non-positive stationary GEV scale."
        )

    logpdf = genextreme.logpdf(
        values_gw,
        c,
        loc=loc,
        scale=scale,
    )

    if not np.all(
        np.isfinite(
            logpdf
        )
    ):
        raise ValueError(
            "Invalid stationary GEV log likelihood."
        )

    log_likelihood = float(
        np.sum(
            logpdf
        )
    )

    return {
        "xi": float(
            -c
        ),
        "scipy_c": float(
            c
        ),
        "loc_GW": float(
            loc
        ),
        "scale_GW": float(
            scale
        ),
        "log_likelihood": (
            log_likelihood
        ),
        "aic": float(
            2 * 3
            - 2 * log_likelihood
        ),
    }


def return_level(
    fit: dict,
    return_period: float,
) -> float:
    probability = (
        1.0
        - 1.0 / return_period
    )

    value = genextreme.ppf(
        probability,
        fit[
            "scipy_c"
        ],
        loc=fit[
            "loc_GW"
        ],
        scale=fit[
            "scale_GW"
        ],
    )

    return float(
        value
    )


def summarise_ci(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    value_column: str,
) -> pd.DataFrame:
    rows = []

    for keys, group in dataframe.groupby(
        group_columns
    ):
        if not isinstance(
            keys,
            tuple,
        ):
            keys = (
                keys,
            )

        values = (
            group[
                value_column
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        row = {
            column: key
            for column, key in zip(
                group_columns,
                keys,
            )
        }

        row[
            "n_success"
        ] = len(
            values
        )

        if len(values) == 0:
            row[
                "ci_lower"
            ] = np.nan

            row[
                "median"
            ] = np.nan

            row[
                "ci_upper"
            ] = np.nan

            row[
                "bootstrap_mean"
            ] = np.nan

            row[
                "bootstrap_sd"
            ] = np.nan

        else:
            row[
                "ci_lower"
            ] = np.quantile(
                values,
                0.025,
            )

            row[
                "median"
            ] = np.quantile(
                values,
                0.500,
            )

            row[
                "ci_upper"
            ] = np.quantile(
                values,
                0.975,
            )

            row[
                "bootstrap_mean"
            ] = np.mean(
                values
            )

            row[
                "bootstrap_sd"
            ] = (
                np.std(
                    values,
                    ddof=1,
                )
                if len(values) > 1
                else np.nan
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def ecmwf_values_from_sampled_years(
    dataframe: pd.DataFrame,
    sampled_years: np.ndarray,
    column: str,
) -> np.ndarray:
    parts = []

    for year in sampled_years:
        values = dataframe.loc[
            dataframe[
                "winter_year"
            ] == year,
            column,
        ].to_numpy(dtype=float)

        parts.append(
            values
        )

    return np.concatenate(
        parts
    )


def hannah_values_from_sampled_years(
    dataframe: pd.DataFrame,
    sampled_years: np.ndarray,
    column: str,
) -> np.ndarray:
    values = []

    for year in sampled_years:
        matching = dataframe.loc[
            dataframe[
                "winter_year"
            ] == year,
            column,
        ]

        if len(matching) != 1:
            raise ValueError(
                "Expected exactly one Hannah value for "
                f"winter year {year}, found {len(matching)}."
            )

        values.append(
            matching.iloc[
                0
            ]
        )

    return np.asarray(
        values,
        dtype=float,
    )


def empirical_exceedance(
    values,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.sort(
        finite_values(
            values
        )
    )[::-1]

    ranks = np.arange(
        1,
        len(values) + 1,
    )

    probabilities = (
        ranks
        / (
            len(values)
            + 1.0
        )
    )

    return (
        values,
        probabilities,
    )


def empirical_quantile_positions(
    n: int,
) -> np.ndarray:
    return (
        np.arange(
            1,
            n + 1,
        )
        - 0.5
    ) / n


def make_single_sample_stationary_diagnostic(
    window: int,
    sample_name: str,
    values_gw: np.ndarray,
    fit: dict,
    bootstrap_parameters: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Create one stationary 2x2 diagnostic figure for one sample.

    Layout:
        top left     probability plot
        top right    quantile plot
        bottom left  return-level plot with cluster-bootstrap CI
        bottom right density plot and parameter summary
    """
    values_gw = finite_values(
        values_gw
    )

    n = len(
        values_gw
    )

    observed_ascending = np.sort(
        values_gw
    )

    probabilities = empirical_quantile_positions(
        n
    )

    fitted_probability = genextreme.cdf(
        observed_ascending,
        fit[
            "scipy_c"
        ],
        loc=fit[
            "loc_GW"
        ],
        scale=fit[
            "scale_GW"
        ],
    )

    fitted_quantiles = genextreme.ppf(
        probabilities,
        fit[
            "scipy_c"
        ],
        loc=fit[
            "loc_GW"
        ],
        scale=fit[
            "scale_GW"
        ],
    )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(
            13.2,
            10.0,
        ),
        dpi=FIGURE_DPI,
    )

    # --------------------------------------------------------
    # Probability plot
    # --------------------------------------------------------

    ax = axes[
        0,
        0,
    ]

    ax.scatter(
        probabilities,
        fitted_probability,
        s=(
            13
            if sample_name
            == "ECMWF pooled"
            else 34
        ),
    )

    ax.plot(
        [
            0,
            1,
        ],
        [
            0,
            1,
        ],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.set_xlabel(
        "Empirical cumulative probability"
    )

    ax.set_ylabel(
        "Fitted GEV cumulative probability"
    )

    ax.set_title(
        "Probability plot"
    )

    ax.grid(
        True,
        alpha=0.28,
    )

    # --------------------------------------------------------
    # Quantile plot
    # --------------------------------------------------------

    ax = axes[
        0,
        1,
    ]

    ax.scatter(
        fitted_quantiles,
        observed_ascending,
        s=(
            13
            if sample_name
            == "ECMWF pooled"
            else 34
        ),
    )

    combined_quantiles = np.concatenate(
        [
            fitted_quantiles,
            observed_ascending,
        ]
    )

    lower = np.min(
        combined_quantiles
    )

    upper = np.max(
        combined_quantiles
    )

    padding = (
        0.03
        * (
            upper
            - lower
        )
    )

    ax.plot(
        [
            lower
            - padding,
            upper
            + padding,
        ],
        [
            lower
            - padding,
            upper
            + padding,
        ],
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlim(
        lower
        - padding,
        upper
        + padding,
    )

    ax.set_ylim(
        lower
        - padding,
        upper
        + padding,
    )

    ax.set_xlabel(
        "Fitted GEV quantile (GW)"
    )

    ax.set_ylabel(
        "Observed winter maximum (GW)"
    )

    ax.set_title(
        "Quantile plot"
    )

    ax.grid(
        True,
        alpha=0.28,
    )

    # --------------------------------------------------------
    # Return-level plot with winter-year bootstrap interval
    # --------------------------------------------------------

    ax = axes[
        1,
        0,
    ]

    return_period_grid = np.geomspace(
        1.05,
        200,
        280,
    )

    return_probability_grid = (
        1.0
        - 1.0
        / return_period_grid
    )

    fitted_return_level = genextreme.ppf(
        return_probability_grid,
        fit[
            "scipy_c"
        ],
        loc=fit[
            "loc_GW"
        ],
        scale=fit[
            "scale_GW"
        ],
    )

    valid_bootstrap = bootstrap_parameters[
        np.isfinite(
            bootstrap_parameters[
                "scipy_c"
            ]
        )
        & np.isfinite(
            bootstrap_parameters[
                "loc_GW"
            ]
        )
        & np.isfinite(
            bootstrap_parameters[
                "scale_GW"
            ]
        )
        & (
            bootstrap_parameters[
                "scale_GW"
            ]
            > 0
        )
    ].copy()

    bootstrap_return_levels = []

    for _, bootstrap_row in valid_bootstrap.iterrows():
        values = genextreme.ppf(
            return_probability_grid,
            bootstrap_row[
                "scipy_c"
            ],
            loc=bootstrap_row[
                "loc_GW"
            ],
            scale=bootstrap_row[
                "scale_GW"
            ],
        )

        if np.all(
            np.isfinite(
                values
            )
        ):
            bootstrap_return_levels.append(
                values
            )

    if len(
        bootstrap_return_levels
    ) > 0:
        bootstrap_return_levels = np.vstack(
            bootstrap_return_levels
        )

        lower_band = np.quantile(
            bootstrap_return_levels,
            0.025,
            axis=0,
        )

        upper_band = np.quantile(
            bootstrap_return_levels,
            0.975,
            axis=0,
        )

        ax.fill_between(
            return_period_grid,
            lower_band,
            upper_band,
            alpha=0.18,
            label=(
                "95% winter-year bootstrap interval"
            ),
        )

    ax.plot(
        return_period_grid,
        fitted_return_level,
        linewidth=1.7,
        label="Fitted stationary GEV",
    )

    descending, exceedance = empirical_exceedance(
        values_gw
    )

    empirical_return_period = (
        1.0
        / exceedance
    )

    ax.scatter(
        empirical_return_period,
        descending,
        s=(
            13
            if sample_name
            == "ECMWF pooled"
            else 34
        ),
        label="Empirical plotting positions",
        zorder=3,
    )

    ax.set_xscale(
        "log"
    )

    ax.set_xlabel(
        "Return period (winter blocks, log scale)"
    )

    ax.set_ylabel(
        "Return level (GW)"
    )

    ax.set_title(
        "Return-level plot"
    )

    ax.grid(
        True,
        which="both",
        alpha=0.28,
    )

    ax.legend(
        fontsize=FONT_LEGEND,
    )

    # --------------------------------------------------------
    # Density plot and parameter box
    # --------------------------------------------------------

    ax = axes[
        1,
        1,
    ]

    ax.hist(
        values_gw,
        bins="auto",
        density=True,
        alpha=0.42,
        label="Observed winter maxima",
    )

    sample_sd = np.std(
        values_gw,
        ddof=1,
    )

    x_grid = np.linspace(
        np.min(
            values_gw
        )
        - 0.25
        * sample_sd,
        np.max(
            values_gw
        )
        + 0.25
        * sample_sd,
        600,
    )

    density = genextreme.pdf(
        x_grid,
        fit[
            "scipy_c"
        ],
        loc=fit[
            "loc_GW"
        ],
        scale=fit[
            "scale_GW"
        ],
    )

    ax.plot(
        x_grid,
        density,
        linewidth=1.7,
        label="Fitted GEV density",
    )

    # Rug marks.
    rug_height = (
        0.015
        * max(
            1e-6,
            np.nanmax(
                density
            ),
        )
    )

    ax.vlines(
        values_gw,
        0,
        rug_height,
        linewidth=0.5,
        alpha=0.55,
    )

    xi_boot = (
        valid_bootstrap[
            "xi"
        ]
        .dropna()
        .to_numpy(dtype=float)
    )

    if len(
        xi_boot
    ) > 0:
        xi_lower = np.quantile(
            xi_boot,
            0.025,
        )

        xi_upper = np.quantile(
            xi_boot,
            0.975,
        )

        xi_ci_text = (
            f"95% bootstrap CI for xi:\n"
            f"[{xi_lower:.3f}, {xi_upper:.3f}]"
        )
    else:
        xi_ci_text = (
            "95% bootstrap CI for xi:\n"
            "not available"
        )

    parameter_text = (
        f"mu = {fit['loc_GW']:.3f} GW\n"
        f"sigma = {fit['scale_GW']:.3f} GW\n"
        f"xi = {fit['xi']:.3f}\n"
        f"{xi_ci_text}"
    )

    ax.text(
        0.98,
        0.95,
        parameter_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=FONT_PARAMETER_BOX,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
        },
    )

    ax.set_xlabel(
        "Winter-maximum severity (GW)"
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        "Density plot"
    )

    ax.grid(
        True,
        alpha=0.28,
    )

    ax.legend(
        fontsize=FONT_LEGEND,
        loc="upper left",
    )

    display_name = (
        "ECMWF pooled"
        if sample_name
        == "ECMWF pooled"
        else "Hannah overlap shifted"
    )

    fig.suptitle(
        (
            f"Stationary GEV diagnostics: "
            f"{display_name}, {window}-day severity\n"
            f"Nov 8-Mar 31, winters 1982-2016, n={n}"
        ),
        fontsize=FONT_SUPTITLE,
        y=0.995,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.94,
        ],
        h_pad=2.4,
        w_pad=2.0,
    )

    fig.savefig(
        output_path,
        bbox_inches="tight",
    )

    fig.savefig(
        output_path.with_suffix(
            ".pdf"
        ),
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# Load and validate data
# ============================================================

print("=" * 80)
print(
    "Final analysis 06: stationary GEV for primary windows"
)
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
    dataframe[
        "winter_year"
    ] = pd.to_numeric(
        dataframe[
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

hannah_overlap = hannah[
    hannah[
        "winter_year"
    ].between(
        OVERLAP_START,
        OVERLAP_END,
    )
].copy()

years = np.arange(
    OVERLAP_START,
    OVERLAP_END + 1,
)

if len(
    ecmwf
) != 875:
    warnings.warn(
        "Expected 875 ECMWF winter-member realisations, "
        f"found {len(ecmwf)}."
    )

if len(
    hannah_overlap
) != 35:
    warnings.warn(
        "Expected 35 Hannah overlap winters, "
        f"found {len(hannah_overlap)}."
    )

print(
    f"\nECMWF rows: {len(ecmwf)}"
)

print(
    f"Hannah overlap rows: {len(hannah_overlap)}"
)

print(
    f"Primary windows: {PRIMARY_WINDOWS}"
)

print(
    f"Bootstrap replicates: {N_BOOT}"
)


# ============================================================
# Original fits
# ============================================================

original_fit_rows = []
original_data = {}

for window in PRIMARY_WINDOWS:
    column = severity_column(
        window
    )

    if column not in ecmwf.columns:
        raise ValueError(
            f"Missing ECMWF column: {column}"
        )

    if column not in hannah_overlap.columns:
        raise ValueError(
            f"Missing Hannah column: {column}"
        )

    e_values_gw = (
        pd.to_numeric(
            ecmwf[
                column
            ],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        / 1000.0
    )

    h_raw_gw = (
        pd.to_numeric(
            hannah_overlap[
                column
            ],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        / 1000.0
    )

    e_values_gw = finite_values(
        e_values_gw
    )

    h_raw_gw = finite_values(
        h_raw_gw
    )

    shift_gw = (
        np.mean(
            e_values_gw
        )
        - np.mean(
            h_raw_gw
        )
    )

    h_shifted_gw = (
        h_raw_gw
        + shift_gw
    )

    e_fit = fit_stationary_gev(
        e_values_gw
    )

    h_fit = fit_stationary_gev(
        h_shifted_gw
    )

    original_data[
        window
    ] = {
        "ECMWF pooled": {
            "values": e_values_gw,
            "fit": e_fit,
        },
        "Hannah overlap shifted": {
            "values": h_shifted_gw,
            "fit": h_fit,
        },
        "shift_GW": shift_gw,
    }

    for (
        sample_name,
        values_gw,
        fit,
        sample_shift,
    ) in [
        (
            "ECMWF pooled",
            e_values_gw,
            e_fit,
            0.0,
        ),
        (
            "Hannah overlap shifted",
            h_shifted_gw,
            h_fit,
            shift_gw,
        ),
    ]:
        row = {
            "window_days": window,
            "sample": sample_name,
            "n": len(
                values_gw
            ),
            "mean_GW": np.mean(
                values_gw
            ),
            "sd_GW": np.std(
                values_gw,
                ddof=1,
            ),
            "mean_shift_GW": sample_shift,
            **fit,
        }

        for return_period in RETURN_PERIODS:
            row[
                f"return_level_{return_period}block_GW"
            ] = return_level(
                fit,
                return_period,
            )

        original_fit_rows.append(
            row
        )

original_fits = pd.DataFrame(
    original_fit_rows
)

original_fits.to_csv(
    OUT_ORIGINAL_FITS,
    index=False,
)

print("\nOriginal fits:")
print(
    original_fits[
        [
            "window_days",
            "sample",
            "n",
            "xi",
            "loc_GW",
            "scale_GW",
            "return_level_20block_GW",
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# Paired winter-year cluster bootstrap
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

parameter_rows = []
return_level_rows = []
difference_rows = []

for window in PRIMARY_WINDOWS:
    column = severity_column(
        window
    )

    print(
        f"\nBootstrapping {window}-day window..."
    )

    ecmwf_failures = 0
    hannah_failures = 0

    for bootstrap_id in range(
        1,
        N_BOOT + 1,
    ):
        sampled_years = rng.choice(
            years,
            size=len(
                years
            ),
            replace=True,
        )

        e_boot_gw = (
            ecmwf_values_from_sampled_years(
                ecmwf,
                sampled_years,
                column,
            )
            / 1000.0
        )

        h_boot_raw_gw = (
            hannah_values_from_sampled_years(
                hannah_overlap,
                sampled_years,
                column,
            )
            / 1000.0
        )

        shift_boot_gw = (
            np.mean(
                e_boot_gw
            )
            - np.mean(
                h_boot_raw_gw
            )
        )

        h_boot_shifted_gw = (
            h_boot_raw_gw
            + shift_boot_gw
        )

        e_fit = None
        h_fit = None

        try:
            e_fit = fit_stationary_gev(
                e_boot_gw
            )

            parameter_rows.append(
                {
                    "window_days": window,
                    "bootstrap_id": bootstrap_id,
                    "sample": "ECMWF pooled",
                    "mean_shift_GW": 0.0,
                    **e_fit,
                }
            )

            for return_period in RETURN_PERIODS:
                return_level_rows.append(
                    {
                        "window_days": window,
                        "bootstrap_id": bootstrap_id,
                        "sample": "ECMWF pooled",
                        "return_period_blocks": return_period,
                        "return_level_GW": return_level(
                            e_fit,
                            return_period,
                        ),
                    }
                )

        except Exception:
            ecmwf_failures += 1

        try:
            h_fit = fit_stationary_gev(
                h_boot_shifted_gw
            )

            parameter_rows.append(
                {
                    "window_days": window,
                    "bootstrap_id": bootstrap_id,
                    "sample": "Hannah overlap shifted",
                    "mean_shift_GW": shift_boot_gw,
                    **h_fit,
                }
            )

            for return_period in RETURN_PERIODS:
                return_level_rows.append(
                    {
                        "window_days": window,
                        "bootstrap_id": bootstrap_id,
                        "sample": "Hannah overlap shifted",
                        "return_period_blocks": return_period,
                        "return_level_GW": return_level(
                            h_fit,
                            return_period,
                        ),
                    }
                )

        except Exception:
            hannah_failures += 1

        if (
            e_fit is not None
            and h_fit is not None
        ):
            for return_period in RETURN_PERIODS:
                difference_rows.append(
                    {
                        "window_days": window,
                        "bootstrap_id": bootstrap_id,
                        "return_period_blocks": return_period,
                        "difference_ECMWF_minus_Hannah_GW": (
                            return_level(
                                e_fit,
                                return_period,
                            )
                            - return_level(
                                h_fit,
                                return_period,
                            )
                        ),
                    }
                )

    print(
        f"  ECMWF failed fits : "
        f"{ecmwf_failures}/{N_BOOT}"
    )

    print(
        f"  Hannah failed fits: "
        f"{hannah_failures}/{N_BOOT}"
    )


# ============================================================
# Save bootstrap samples and confidence intervals
# ============================================================

parameter_samples = pd.DataFrame(
    parameter_rows
)

return_level_samples = pd.DataFrame(
    return_level_rows
)

difference_samples = pd.DataFrame(
    difference_rows
)

parameter_samples.to_csv(
    OUT_PARAMETER_SAMPLES,
    index=False,
)

return_level_samples.to_csv(
    OUT_RL_SAMPLES,
    index=False,
)

difference_samples.to_csv(
    OUT_DIFF_SAMPLES,
    index=False,
)

parameter_ci_frames = []

for parameter in [
    "xi",
    "loc_GW",
    "scale_GW",
    "mean_shift_GW",
]:
    summary = summarise_ci(
        dataframe=parameter_samples,
        group_columns=[
            "window_days",
            "sample",
        ],
        value_column=parameter,
    )

    summary[
        "parameter"
    ] = parameter

    parameter_ci_frames.append(
        summary
    )

parameter_ci = pd.concat(
    parameter_ci_frames,
    ignore_index=True,
)

parameter_ci.to_csv(
    OUT_PARAMETER_CI,
    index=False,
)

return_level_ci = summarise_ci(
    dataframe=return_level_samples,
    group_columns=[
        "window_days",
        "sample",
        "return_period_blocks",
    ],
    value_column="return_level_GW",
)

original_rl_long_rows = []

for _, row in original_fits.iterrows():
    for return_period in RETURN_PERIODS:
        original_rl_long_rows.append(
            {
                "window_days": row[
                    "window_days"
                ],
                "sample": row[
                    "sample"
                ],
                "return_period_blocks": (
                    return_period
                ),
                "original_return_level_GW": row[
                    f"return_level_{return_period}block_GW"
                ],
            }
        )

original_rl_long = pd.DataFrame(
    original_rl_long_rows
)

return_level_ci = return_level_ci.merge(
    original_rl_long,
    on=[
        "window_days",
        "sample",
        "return_period_blocks",
    ],
    how="left",
)

return_level_ci.to_csv(
    OUT_RL_CI,
    index=False,
)

difference_ci = summarise_ci(
    dataframe=difference_samples,
    group_columns=[
        "window_days",
        "return_period_blocks",
    ],
    value_column=(
        "difference_ECMWF_minus_Hannah_GW"
    ),
)

original_difference_rows = []

for window in PRIMARY_WINDOWS:
    for return_period in RETURN_PERIODS:
        e_rl = original_fits.loc[
            (
                original_fits[
                    "window_days"
                ] == window
            )
            & (
                original_fits[
                    "sample"
                ] == "ECMWF pooled"
            ),
            f"return_level_{return_period}block_GW",
        ].iloc[
            0
        ]

        h_rl = original_fits.loc[
            (
                original_fits[
                    "window_days"
                ] == window
            )
            & (
                original_fits[
                    "sample"
                ] == "Hannah overlap shifted"
            ),
            f"return_level_{return_period}block_GW",
        ].iloc[
            0
        ]

        original_difference_rows.append(
            {
                "window_days": window,
                "return_period_blocks": return_period,
                "original_difference_ECMWF_minus_Hannah_GW": (
                    e_rl
                    - h_rl
                ),
            }
        )

difference_ci = difference_ci.merge(
    pd.DataFrame(
        original_difference_rows
    ),
    on=[
        "window_days",
        "return_period_blocks",
    ],
    how="left",
)

difference_ci.to_csv(
    OUT_DIFF_CI,
    index=False,
)


# ============================================================
# Shape-parameter comparison table
# ============================================================

xi_ci = parameter_ci[
    parameter_ci[
        "parameter"
    ] == "xi"
].copy()

xi_original = original_fits[
    [
        "window_days",
        "sample",
        "xi",
    ]
].rename(
    columns={
        "xi": "xi_original",
    }
)

xi_ci = xi_ci.merge(
    xi_original,
    on=[
        "window_days",
        "sample",
    ],
    how="left",
)

comparison_rows = []

for window in PRIMARY_WINDOWS:
    e_row = xi_ci[
        (
            xi_ci[
                "window_days"
            ] == window
        )
        & (
            xi_ci[
                "sample"
            ] == "ECMWF pooled"
        )
    ].iloc[
        0
    ]

    h_row = xi_ci[
        (
            xi_ci[
                "window_days"
            ] == window
        )
        & (
            xi_ci[
                "sample"
            ] == "Hannah overlap shifted"
        )
    ].iloc[
        0
    ]

    comparison_rows.append(
        {
            "window_days": window,
            "ecmwf_xi": e_row[
                "xi_original"
            ],
            "ecmwf_ci_lower": e_row[
                "ci_lower"
            ],
            "ecmwf_ci_upper": e_row[
                "ci_upper"
            ],
            "hannah_xi": h_row[
                "xi_original"
            ],
            "hannah_ci_lower": h_row[
                "ci_lower"
            ],
            "hannah_ci_upper": h_row[
                "ci_upper"
            ],
            "ecmwf_xi_inside_hannah_ci": bool(
                h_row[
                    "ci_lower"
                ]
                <= e_row[
                    "xi_original"
                ]
                <= h_row[
                    "ci_upper"
                ]
            ),
            "diagnostic_rating": (
                DIAGNOSTIC_RATING[
                    window
                ]
            ),
            "ecmwf_n_success": int(
                e_row[
                    "n_success"
                ]
            ),
            "hannah_n_success": int(
                h_row[
                    "n_success"
                ]
            ),
        }
    )

xi_comparison = pd.DataFrame(
    comparison_rows
)

xi_comparison.to_csv(
    OUT_XI_COMPARISON,
    index=False,
)


# ============================================================
# Separate 2x2 stationary diagnostics for each sample
# ============================================================

for window in PRIMARY_WINDOWS:
    for sample_name, output_directory in [
        (
            "ECMWF pooled",
            ECMWF_DIAGNOSTIC_DIR,
        ),
        (
            "Hannah overlap shifted",
            HANNAH_DIAGNOSTIC_DIR,
        ),
    ]:
        safe_sample = (
            "ecmwf_pooled"
            if sample_name
            == "ECMWF pooled"
            else "hannah_overlap_shifted"
        )

        diagnostic_path = (
            output_directory
            / (
                f"stationary_gev_"
                f"{safe_sample}_"
                f"{window}day_2x2.png"
            )
        )

        bootstrap_subset = parameter_samples[
            (
                parameter_samples[
                    "window_days"
                ] == window
            )
            & (
                parameter_samples[
                    "sample"
                ] == sample_name
            )
        ].copy()

        make_single_sample_stationary_diagnostic(
            window=window,
            sample_name=sample_name,
            values_gw=original_data[
                window
            ][
                sample_name
            ][
                "values"
            ],
            fit=original_data[
                window
            ][
                sample_name
            ][
                "fit"
            ],
            bootstrap_parameters=bootstrap_subset,
            output_path=diagnostic_path,
        )


# ============================================================
# Main summary figure: one row, two panels
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

offsets = {
    "ECMWF pooled": -0.8,
    "Hannah overlap shifted": 0.8,
}

# ------------------------------------------------------------
# Left: xi and bootstrap intervals
# ------------------------------------------------------------

for sample_name in [
    "ECMWF pooled",
    "Hannah overlap shifted",
]:
    sample_ci = xi_ci[
        xi_ci[
            "sample"
        ] == sample_name
    ].sort_values(
        "window_days"
    )

    x = (
        sample_ci[
            "window_days"
        ].to_numpy(dtype=float)
        + offsets[
            sample_name
        ]
    )

    y = sample_ci[
        "xi_original"
    ].to_numpy(dtype=float)

    lower_error = (
        y
        - sample_ci[
            "ci_lower"
        ].to_numpy(dtype=float)
    )

    upper_error = (
        sample_ci[
            "ci_upper"
        ].to_numpy(dtype=float)
        - y
    )

    axes[
        0
    ].errorbar(
        x,
        y,
        yerr=[
            lower_error,
            upper_error,
        ],
        marker="o",
        linewidth=1.3,
        capsize=4,
        label=sample_name,
    )

axes[
    0
].axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

axes[
    0
].set_xlabel(
    "Averaging window (days)"
)

axes[
    0
].set_ylabel(
    r"GEV shape parameter $\xi$"
)

axes[
    0
].set_title(
    "Shape parameter with 95% bootstrap intervals"
)

axes[
    0
].set_xticks(
    PRIMARY_WINDOWS
)

axes[
    0
].grid(
    True,
    alpha=0.3,
)

axes[
    0
].legend(
    fontsize=FONT_LEGEND,
)

# ------------------------------------------------------------
# Right: 20-block return level and intervals
# ------------------------------------------------------------

rl20 = return_level_ci[
    return_level_ci[
        "return_period_blocks"
    ] == 20
].copy()

for sample_name in [
    "ECMWF pooled",
    "Hannah overlap shifted",
]:
    sample_ci = rl20[
        rl20[
            "sample"
        ] == sample_name
    ].sort_values(
        "window_days"
    )

    x = (
        sample_ci[
            "window_days"
        ].to_numpy(dtype=float)
        + offsets[
            sample_name
        ]
    )

    y = sample_ci[
        "original_return_level_GW"
    ].to_numpy(dtype=float)

    lower_error = (
        y
        - sample_ci[
            "ci_lower"
        ].to_numpy(dtype=float)
    )

    upper_error = (
        sample_ci[
            "ci_upper"
        ].to_numpy(dtype=float)
        - y
    )

    axes[
        1
    ].errorbar(
        x,
        y,
        yerr=[
            lower_error,
            upper_error,
        ],
        marker="o",
        linewidth=1.3,
        capsize=4,
        label=sample_name,
    )

axes[
    1
].set_xlabel(
    "Averaging window (days)"
)

axes[
    1
].set_ylabel(
    "20-block return level (GW)"
)

axes[
    1
].set_title(
    "Return-level uncertainty"
)

axes[
    1
].set_xticks(
    PRIMARY_WINDOWS
)

axes[
    1
].grid(
    True,
    alpha=0.3,
)

axes[
    1
].legend(
    fontsize=FONT_LEGEND,
)

fig.suptitle(
    "Stationary GEV summary across pre-selected primary windows",
    fontsize=FONT_SUPTITLE,
    y=1.02,
)

fig.tight_layout(
    pad=1.3,
    w_pad=2.0,
)

fig.savefig(
    OUT_MAIN_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_MAIN_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Print summaries
# ============================================================

print("\nShape-parameter comparison:")
print(
    xi_comparison.to_string(
        index=False
    )
)

print("\n20-block return-level intervals:")
print(
    rl20[
        [
            "window_days",
            "sample",
            "original_return_level_GW",
            "ci_lower",
            "median",
            "ci_upper",
            "n_success",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved outputs:")
print(OUT_ORIGINAL_FITS)
print(OUT_PARAMETER_SAMPLES)
print(OUT_PARAMETER_CI)
print(OUT_RL_SAMPLES)
print(OUT_RL_CI)
print(OUT_DIFF_SAMPLES)
print(OUT_DIFF_CI)
print(OUT_XI_COMPARISON)
print(OUT_MAIN_FIGURE_PNG)
print(OUT_MAIN_FIGURE_PDF)
print(ECMWF_DIAGNOSTIC_DIR)
print(HANNAH_DIAGNOSTIC_DIR)

print("\nDone.")
