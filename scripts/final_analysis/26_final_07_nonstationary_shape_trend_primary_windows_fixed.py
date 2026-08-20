from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from scipy.stats import genextreme, chi2, gumbel_r

from config import OUTPUT_DIR


# ============================================================
# Final analysis 07
# Stationary versus non-stationary GEV sensitivity
# Combined 4-row x 2-column diagnostics
#
# Chris's requested sensitivity model:
#
#   xi_t = xi_0 + xi_1 * ((winter_year - 1982) / 10)
#
# where xi_1 is change in the shape parameter per decade.
#
# Location and scale remain constant.
#
# Primary windows:
#   1, 7, 14, 21, 28, 56 and 84 days
#
# Important:
#   - This is a sensitivity analysis, not the main model.
#   - It does not attempt to model the empirical shoulder as a
#     second regime or mixture distribution.
#   - Negative delta AIC means the shape-trend model has lower AIC.
#   - Likelihood-ratio p-values are descriptive because the 25
#     ensemble members within a winter year are not fully independent.
# ============================================================


# ============================================================
# Settings
# ============================================================

WINDOWS = [
    1,
    7,
    14,
    21,
    28,
    56,
    84,
]

OVERLAP_START = 1982
OVERLAP_END = 2016

YEAR_SCALE = 10.0

XI_ZERO_TOL = 1e-6
SUPPORT_EPS = 1e-10


# ============================================================
# Input and output paths
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

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "07_nonstationary_shape_trend_sensitivity"
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

OUT_MODEL_COMPARISON = (
    OUT_DIR
    / "stationary_vs_nonstationary_shape_trend_comparison.csv"
)

OUT_XI_TRAJECTORIES = (
    OUT_DIR
    / "nonstationary_shape_trajectories.csv"
)

OUT_SUMMARY_FIGURE_PNG = (
    OUT_DIR
    / "nonstationary_shape_trend_summary.png"
)

OUT_SUMMARY_FIGURE_PDF = (
    OUT_DIR
    / "nonstationary_shape_trend_summary.pdf"
)

OUT_TRAJECTORY_FIGURE_PNG = (
    OUT_DIR
    / "nonstationary_shape_trajectories.png"
)

OUT_TRAJECTORY_FIGURE_PDF = (
    OUT_DIR
    / "nonstationary_shape_trajectories.pdf"
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


def time_covariate(
    years,
) -> np.ndarray:
    years = np.asarray(
        years,
        dtype=float,
    )

    return (
        years
        - OVERLAP_START
    ) / YEAR_SCALE


def clean_values_and_years(
    values,
    years,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(
        values,
        dtype=float,
    )

    years = np.asarray(
        years,
        dtype=float,
    )

    keep = (
        np.isfinite(
            values
        )
        & np.isfinite(
            years
        )
    )

    values = values[
        keep
    ]

    years = years[
        keep
    ]

    if len(values) < 10:
        raise ValueError(
            "Too few observations for GEV fitting."
        )

    if np.std(
        values,
        ddof=1,
    ) <= 0:
        raise ValueError(
            "Zero-variance sample."
        )

    return (
        values,
        years,
    )


def fit_stationary_gev(
    values_gw,
) -> dict:
    values_gw = np.asarray(
        values_gw,
        dtype=float,
    )

    values_gw = values_gw[
        np.isfinite(
            values_gw
        )
    ]

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
            "Invalid stationary GEV likelihood."
        )

    log_likelihood = float(
        np.sum(
            logpdf
        )
    )

    return {
        "loc_GW": float(
            loc
        ),
        "scale_GW": float(
            scale
        ),
        "xi": float(
            -c
        ),
        "scipy_c": float(
            c
        ),
        "log_likelihood": (
            log_likelihood
        ),
        "aic": float(
            2 * 3
            - 2 * log_likelihood
        ),
    }


def gev_logpdf_variable_xi(
    values,
    mu,
    sigma,
    xi_values,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    xi_values = np.asarray(
        xi_values,
        dtype=float,
    )

    values, xi_values = np.broadcast_arrays(
        values,
        xi_values,
    )

    if (
        sigma <= 0
        or not np.isfinite(
            sigma
        )
    ):
        return np.full(
            values.shape,
            -np.inf,
        )

    z = (
        values
        - mu
    ) / sigma

    logpdf = np.full(
        values.shape,
        -np.inf,
        dtype=float,
    )

    small_xi = (
        np.abs(
            xi_values
        )
        < XI_ZERO_TOL
    )

    if np.any(
        small_xi
    ):
        z_small = z[
            small_xi
        ]

        logpdf[
            small_xi
        ] = (
            -np.log(
                sigma
            )
            - z_small
            - np.exp(
                -z_small
            )
        )

    regular = (
        ~small_xi
    )

    if np.any(
        regular
    ):
        xi_regular = xi_values[
            regular
        ]

        z_regular = z[
            regular
        ]

        support = (
            1.0
            + xi_regular
            * z_regular
        )

        valid = (
            support
            > SUPPORT_EPS
        )

        regular_indices = np.flatnonzero(
            regular
        )

        if np.any(
            valid
        ):
            xi_valid = xi_regular[
                valid
            ]

            support_valid = support[
                valid
            ]

            log_support = np.log(
                support_valid
            )

            exponent_argument = (
                -log_support
                / xi_valid
            )

            safe = (
                exponent_argument
                < 700.0
            )

            valid_indices = regular_indices[
                valid
            ]

            safe_indices = valid_indices[
                safe
            ]

            if len(
                safe_indices
            ) > 0:
                xi_safe = xi_valid[
                    safe
                ]

                log_support_safe = log_support[
                    safe
                ]

                exponent_safe = exponent_argument[
                    safe
                ]

                logpdf[
                    safe_indices
                ] = (
                    -np.log(
                        sigma
                    )
                    - (
                        1.0
                        / xi_safe
                        + 1.0
                    )
                    * log_support_safe
                    - np.exp(
                        exponent_safe
                    )
                )

    return logpdf


def negative_log_likelihood_shape_trend(
    parameters,
    values_gw,
    time_values,
) -> float:
    mu, log_sigma, xi_0, xi_1 = parameters

    sigma = np.exp(
        log_sigma
    )

    xi_values = (
        xi_0
        + xi_1
        * time_values
    )

    # Avoid unrealistic trajectories that leave the numerical
    # range over the study period.
    if (
        np.min(
            xi_values
        ) < -0.8
        or np.max(
            xi_values
        ) > 0.8
    ):
        return 1e12

    logpdf = gev_logpdf_variable_xi(
        values=values_gw,
        mu=mu,
        sigma=sigma,
        xi_values=xi_values,
    )

    if not np.all(
        np.isfinite(
            logpdf
        )
    ):
        return 1e12

    nll = float(
        -np.sum(
            logpdf
        )
    )

    if not np.isfinite(
        nll
    ):
        return 1e12

    return nll


def fit_shape_trend_gev(
    values_gw,
    years,
) -> dict:
    values_gw, years = clean_values_and_years(
        values_gw,
        years,
    )

    time_values = time_covariate(
        years
    )

    stationary = fit_stationary_gev(
        values_gw
    )

    sample_sd = np.std(
        values_gw,
        ddof=1,
    )

    mu_lower = (
        np.min(
            values_gw
        )
        - 3.0
        * sample_sd
    )

    mu_upper = (
        np.max(
            values_gw
        )
        + 3.0
        * sample_sd
    )

    minimum_scale = max(
        sample_sd
        * 0.05,
        1e-4,
    )

    maximum_scale = max(
        sample_sd
        * 20.0,
        minimum_scale
        * 2.0,
    )

    bounds = [
        (
            mu_lower,
            mu_upper,
        ),
        (
            np.log(
                minimum_scale
            ),
            np.log(
                maximum_scale
            ),
        ),
        (
            -0.8,
            0.8,
        ),
        (
            -0.4,
            0.4,
        ),
    ]

    starts = []

    for xi_adjustment in [
        0.0,
        -0.05,
        0.05,
    ]:
        for slope_start in [
            0.0,
            -0.02,
            0.02,
            -0.05,
            0.05,
        ]:
            starts.append(
                np.asarray(
                    [
                        stationary[
                            "loc_GW"
                        ],
                        np.log(
                            stationary[
                                "scale_GW"
                            ]
                        ),
                        stationary[
                            "xi"
                        ]
                        + xi_adjustment,
                        slope_start,
                    ],
                    dtype=float,
                )
            )

    successful_results = []

    for start in starts:
        result = minimize(
            negative_log_likelihood_shape_trend,
            x0=start,
            args=(
                values_gw,
                time_values,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol": 1e-11,
                "gtol": 1e-7,
            },
        )

        if (
            result.success
            and np.isfinite(
                result.fun
            )
            and result.fun
            < 1e11
        ):
            successful_results.append(
                result
            )

    if len(
        successful_results
    ) == 0:
        raise RuntimeError(
            "All shape-trend optimisations failed."
        )

    best = min(
        successful_results,
        key=lambda result: result.fun,
    )

    mu, log_sigma, xi_0, xi_1 = best.x

    sigma = np.exp(
        log_sigma
    )

    log_likelihood = float(
        -best.fun
    )

    xi_start = float(
        xi_0
    )

    xi_end = float(
        xi_0
        + xi_1
        * time_covariate(
            [
                OVERLAP_END
            ]
        )[
            0
        ]
    )

    close_to_bound = False

    for estimate, bound in zip(
        best.x,
        bounds,
    ):
        lower, upper = bound

        tolerance = (
            1e-4
            * max(
                1.0,
                abs(
                    upper
                    - lower
                ),
            )
        )

        if (
            abs(
                estimate
                - lower
            )
            < tolerance
            or abs(
                estimate
                - upper
            )
            < tolerance
        ):
            close_to_bound = True

    return {
        "loc_GW": float(
            mu
        ),
        "scale_GW": float(
            sigma
        ),
        "xi_1982": (
            xi_start
        ),
        "xi_change_per_decade": float(
            xi_1
        ),
        "xi_2016": (
            xi_end
        ),
        "log_likelihood": (
            log_likelihood
        ),
        "aic": float(
            2 * 4
            - 2 * log_likelihood
        ),
        "optimizer_success": bool(
            best.success
        ),
        "optimizer_message": str(
            best.message
        ),
        "close_to_bound": (
            close_to_bound
        ),
    }


def gev_cdf_variable_xi(
    values,
    mu,
    sigma,
    xi_values,
) -> np.ndarray:
    values = np.asarray(
        values,
        dtype=float,
    )

    xi_values = np.asarray(
        xi_values,
        dtype=float,
    )

    values, xi_values = np.broadcast_arrays(
        values,
        xi_values,
    )

    z = (
        values
        - mu
    ) / sigma

    cdf = np.empty(
        values.shape,
        dtype=float,
    )

    small_xi = (
        np.abs(
            xi_values
        )
        < XI_ZERO_TOL
    )

    if np.any(
        small_xi
    ):
        cdf[
            small_xi
        ] = np.exp(
            -np.exp(
                -z[
                    small_xi
                ]
            )
        )

    regular = (
        ~small_xi
    )

    if np.any(
        regular
    ):
        xi_regular = xi_values[
            regular
        ]

        z_regular = z[
            regular
        ]

        support = (
            1.0
            + xi_regular
            * z_regular
        )

        valid = (
            support
            > 0
        )

        regular_cdf = np.empty(
            support.shape,
            dtype=float,
        )

        if np.any(
            valid
        ):
            regular_cdf[
                valid
            ] = np.exp(
                -np.exp(
                    -np.log(
                        support[
                            valid
                        ]
                    )
                    / xi_regular[
                        valid
                    ]
                )
            )

        invalid = (
            ~valid
        )

        if np.any(
            invalid
        ):
            regular_cdf[
                invalid
                & (
                    xi_regular
                    > 0
                )
            ] = 0.0

            regular_cdf[
                invalid
                & (
                    xi_regular
                    < 0
                )
            ] = 1.0

        cdf[
            regular
        ] = regular_cdf

    return np.clip(
        cdf,
        1e-12,
        1.0
        - 1e-12,
    )


def plotting_positions(
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
    values = np.sort(
        np.asarray(
            values,
            dtype=float,
        )
    )[::-1]

    ranks = np.arange(
        1,
        len(values) + 1,
    )

    exceedance = (
        ranks
        / (
            len(values)
            + 1.0
        )
    )

    return (
        values,
        exceedance,
    )


def conditional_shape_values(
    years,
    nonstationary_fit,
) -> np.ndarray:
    years = np.asarray(
        years,
        dtype=float,
    )

    return (
        nonstationary_fit[
            "xi_1982"
        ]
        + nonstationary_fit[
            "xi_change_per_decade"
        ]
        * time_covariate(
            years
        )
    )


def pooled_marginal_cdf(
    x_values,
    years,
    nonstationary_fit,
) -> np.ndarray:
    """
    Pooled marginal CDF obtained by averaging the conditional
    yearly GEV CDFs across unique winter years.

    Because ECMWF has 25 members in every winter and Hannah has
    one value in every winter, equal weighting by unique winter
    year is also equal weighting over the study period.
    """
    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    unique_years = np.sort(
        np.unique(
            np.asarray(
                years,
                dtype=float,
            )
        )
    )

    cdf_rows = []

    for year in unique_years:
        xi_year = conditional_shape_values(
            [
                year
            ],
            nonstationary_fit,
        )[
            0
        ]

        cdf_rows.append(
            gev_cdf_variable_xi(
                values=x_values,
                mu=nonstationary_fit[
                    "loc_GW"
                ],
                sigma=nonstationary_fit[
                    "scale_GW"
                ],
                xi_values=np.full(
                    len(x_values),
                    xi_year,
                    dtype=float,
                ),
            )
        )

    return np.mean(
        np.vstack(
            cdf_rows
        ),
        axis=0,
    )


def pooled_marginal_pdf(
    x_values,
    years,
    nonstationary_fit,
) -> np.ndarray:
    """
    Pooled marginal density obtained by averaging the
    conditional yearly GEV densities.
    """
    x_values = np.asarray(
        x_values,
        dtype=float,
    )

    unique_years = np.sort(
        np.unique(
            np.asarray(
                years,
                dtype=float,
            )
        )
    )

    pdf_rows = []

    for year in unique_years:
        xi_year = conditional_shape_values(
            [
                year
            ],
            nonstationary_fit,
        )[
            0
        ]

        logpdf = gev_logpdf_variable_xi(
            values=x_values,
            mu=nonstationary_fit[
                "loc_GW"
            ],
            sigma=nonstationary_fit[
                "scale_GW"
            ],
            xi_values=np.full(
                len(x_values),
                xi_year,
                dtype=float,
            ),
        )

        pdf = np.zeros(
            len(x_values),
            dtype=float,
        )

        finite = np.isfinite(
            logpdf
        )

        pdf[
            finite
        ] = np.exp(
            logpdf[
                finite
            ]
        )

        pdf_rows.append(
            pdf
        )

    return np.mean(
        np.vstack(
            pdf_rows
        ),
        axis=0,
    )


def build_marginal_grid(
    values,
    years,
    nonstationary_fit,
    minimum_probability: float,
    maximum_probability: float,
    n_grid: int = 20000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a dense x-grid covering the requested pooled marginal
    probability range. Bounds are expanded until the mixture CDF
    contains the empirical plotting positions.
    """
    values = np.asarray(
        values,
        dtype=float,
    )

    sample_sd = np.std(
        values,
        ddof=1,
    )

    if (
        not np.isfinite(
            sample_sd
        )
        or sample_sd <= 0
    ):
        sample_sd = 1.0

    lower = (
        np.min(
            values
        )
        - 3.0
        * sample_sd
    )

    upper = (
        np.max(
            values
        )
        + 3.0
        * sample_sd
    )

    for _ in range(
        12
    ):
        boundary_cdf = pooled_marginal_cdf(
            [
                lower,
                upper,
            ],
            years,
            nonstationary_fit,
        )

        lower_ok = (
            boundary_cdf[
                0
            ]
            <= minimum_probability
        )

        upper_ok = (
            boundary_cdf[
                1
            ]
            >= maximum_probability
        )

        if (
            lower_ok
            and upper_ok
        ):
            break

        width = (
            upper
            - lower
        )

        if not lower_ok:
            lower -= width

        if not upper_ok:
            upper += width

    x_grid = np.linspace(
        lower,
        upper,
        n_grid,
    )

    cdf_grid = pooled_marginal_cdf(
        x_grid,
        years,
        nonstationary_fit,
    )

    cdf_grid = np.maximum.accumulate(
        cdf_grid
    )

    return (
        x_grid,
        cdf_grid,
    )


def pooled_marginal_ppf(
    probabilities,
    values,
    years,
    nonstationary_fit,
) -> np.ndarray:
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    probabilities = np.clip(
        probabilities,
        1e-8,
        1.0
        - 1e-8,
    )

    x_grid, cdf_grid = build_marginal_grid(
        values=values,
        years=years,
        nonstationary_fit=nonstationary_fit,
        minimum_probability=max(
            1e-10,
            float(
                np.min(
                    probabilities
                )
            )
            / 10.0,
        ),
        maximum_probability=min(
            1.0
            - 1e-10,
            1.0
            - (
                1.0
                - float(
                    np.max(
                        probabilities
                    )
                )
            )
            / 10.0,
        ),
    )

    unique_cdf, unique_indices = np.unique(
        cdf_grid,
        return_index=True,
    )

    unique_x = x_grid[
        unique_indices
    ]

    return np.interp(
        probabilities,
        unique_cdf,
        unique_x,
    )


def conditional_pit_values(
    values,
    years,
    stationary_fit,
    nonstationary_fit,
) -> tuple[np.ndarray, np.ndarray]:
    values, years = clean_values_and_years(
        values,
        years,
    )

    stationary_pit = genextreme.cdf(
        values,
        stationary_fit[
            "scipy_c"
        ],
        loc=stationary_fit[
            "loc_GW"
        ],
        scale=stationary_fit[
            "scale_GW"
        ],
    )

    xi_observation = conditional_shape_values(
        years,
        nonstationary_fit,
    )

    nonstationary_pit = gev_cdf_variable_xi(
        values=values,
        mu=nonstationary_fit[
            "loc_GW"
        ],
        sigma=nonstationary_fit[
            "scale_GW"
        ],
        xi_values=xi_observation,
    )

    return (
        np.clip(
            stationary_pit,
            1e-10,
            1.0
            - 1e-10,
        ),
        np.clip(
            nonstationary_pit,
            1e-10,
            1.0
            - 1e-10,
        ),
    )


def make_combined_diagnostic_figure(
    window: int,
    window_records: dict,
    output_path: Path,
) -> None:
    """
    Create one 4-row x 2-column diagnostic figure.

    Columns:
        ECMWF pooled | Hannah overlap shifted

    Rows:
        1. Distribution and fitted densities
        2. Conditional PIT probability plot
        3. Pooled marginal quantile plot
        4. Empirical and fitted upper tails
    """
    sample_names = [
        "ECMWF pooled",
        "Hannah overlap shifted",
    ]

    # Use a common severity range across both columns so the
    # corresponding panels are directly comparable.
    combined_values = np.concatenate(
        [
            np.asarray(
                window_records[
                    sample_name
                ][
                    "values"
                ],
                dtype=float,
            )
            for sample_name in sample_names
        ]
    )

    combined_sd = np.std(
        combined_values,
        ddof=1,
    )

    common_x_lower = (
        np.min(
            combined_values
        )
        - 0.12
        * combined_sd
    )

    common_x_upper = (
        np.max(
            combined_values
        )
        + 0.18
        * combined_sd
    )

    common_x_grid = np.linspace(
        common_x_lower,
        common_x_upper,
        1000,
    )

    fig, axes = plt.subplots(
        nrows=4,
        ncols=2,
        figsize=(
            13.2,
            17.0,
        ),
        dpi=260,
    )

    # Column headings.
    for column_index, sample_name in enumerate(
        sample_names
    ):
        axes[
            0,
            column_index
        ].text(
            0.5,
            1.19,
            sample_name,
            transform=axes[
                0,
                column_index
            ].transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            weight="bold",
        )

    for column_index, sample_name in enumerate(
        sample_names
    ):
        record = window_records[
            sample_name
        ]

        values, years = clean_values_and_years(
            record[
                "values"
            ],
            record[
                "years"
            ],
        )

        stationary_fit = record[
            "stationary"
        ]

        nonstationary_fit = record[
            "nonstationary"
        ]

        n = len(
            values
        )

        # ----------------------------------------------------
        # Row 1: distribution and fitted densities
        # ----------------------------------------------------

        ax = axes[
            0,
            column_index
        ]

        ax.hist(
            values,
            bins="auto",
            density=True,
            alpha=0.42,
            label=(
                f"Empirical distribution, n={n}"
            ),
        )

        stationary_pdf = genextreme.pdf(
            common_x_grid,
            stationary_fit[
                "scipy_c"
            ],
            loc=stationary_fit[
                "loc_GW"
            ],
            scale=stationary_fit[
                "scale_GW"
            ],
        )

        nonstationary_pdf = pooled_marginal_pdf(
            common_x_grid,
            years,
            nonstationary_fit,
        )

        ax.plot(
            common_x_grid,
            stationary_pdf,
            linewidth=1.6,
            label="Stationary GEV",
        )

        ax.plot(
            common_x_grid,
            nonstationary_pdf,
            linestyle="--",
            linewidth=1.6,
            label=(
                "Non-stationary pooled marginal"
            ),
        )

        ax.set_xlim(
            common_x_lower,
            common_x_upper,
        )

        ax.set_xlabel(
            "Severity (GW)"
        )

        ax.set_ylabel(
            "Density"
        )

        ax.set_title(
            "Distribution and fitted densities"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        ax.legend(
            fontsize=7.8,
        )

        # ----------------------------------------------------
        # Row 2: conditional PIT probability plot
        # ----------------------------------------------------

        ax = axes[
            1,
            column_index
        ]

        stationary_pit, nonstationary_pit = (
            conditional_pit_values(
                values=values,
                years=years,
                stationary_fit=stationary_fit,
                nonstationary_fit=nonstationary_fit,
            )
        )

        empirical_probability = plotting_positions(
            n
        )

        ax.scatter(
            empirical_probability,
            np.sort(
                stationary_pit
            ),
            s=(
                9
                if sample_name
                == "ECMWF pooled"
                else 25
            ),
            label="Stationary GEV",
        )

        ax.scatter(
            empirical_probability,
            np.sort(
                nonstationary_pit
            ),
            s=(
                9
                if sample_name
                == "ECMWF pooled"
                else 25
            ),
            marker="x",
            label="Non-stationary GEV",
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
            label="One-to-one line",
        )

        ax.set_xlim(
            -0.02,
            1.02,
        )

        ax.set_ylim(
            -0.02,
            1.02,
        )

        ax.set_xlabel(
            "Empirical probability"
        )

        ax.set_ylabel(
            "Sorted fitted probability (PIT)"
        )

        ax.set_title(
            "Conditional PIT probability plot"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        if column_index == 0:
            ax.legend(
                fontsize=7.8,
            )

        # ----------------------------------------------------
        # Row 3: pooled marginal quantile plot
        # ----------------------------------------------------

        ax = axes[
            2,
            column_index
        ]

        empirical_quantiles = np.sort(
            values
        )

        probabilities = plotting_positions(
            n
        )

        stationary_quantiles = genextreme.ppf(
            probabilities,
            stationary_fit[
                "scipy_c"
            ],
            loc=stationary_fit[
                "loc_GW"
            ],
            scale=stationary_fit[
                "scale_GW"
            ],
        )

        nonstationary_quantiles = pooled_marginal_ppf(
            probabilities=probabilities,
            values=values,
            years=years,
            nonstationary_fit=nonstationary_fit,
        )

        ax.scatter(
            stationary_quantiles,
            empirical_quantiles,
            s=(
                9
                if sample_name
                == "ECMWF pooled"
                else 25
            ),
            label="Stationary GEV",
        )

        ax.scatter(
            nonstationary_quantiles,
            empirical_quantiles,
            s=(
                9
                if sample_name
                == "ECMWF pooled"
                else 25
            ),
            marker="x",
            label=(
                "Non-stationary pooled marginal"
            ),
        )

        quantile_combined = np.concatenate(
            [
                empirical_quantiles,
                stationary_quantiles,
                nonstationary_quantiles,
            ]
        )

        quantile_lower = np.min(
            quantile_combined
        )

        quantile_upper = np.max(
            quantile_combined
        )

        quantile_padding = (
            0.03
            * (
                quantile_upper
                - quantile_lower
            )
        )

        ax.plot(
            [
                quantile_lower
                - quantile_padding,
                quantile_upper
                + quantile_padding,
            ],
            [
                quantile_lower
                - quantile_padding,
                quantile_upper
                + quantile_padding,
            ],
            linestyle="--",
            linewidth=1.0,
            label="One-to-one line",
        )

        ax.set_xlim(
            quantile_lower
            - quantile_padding,
            quantile_upper
            + quantile_padding,
        )

        ax.set_ylim(
            quantile_lower
            - quantile_padding,
            quantile_upper
            + quantile_padding,
        )

        ax.set_xlabel(
            "Fitted GEV quantile (GW)"
        )

        ax.set_ylabel(
            "Empirical quantile (GW)"
        )

        ax.set_title(
            "Pooled marginal quantile plot"
        )

        ax.grid(
            True,
            alpha=0.25,
        )

        if column_index == 0:
            ax.legend(
                fontsize=7.8,
            )

        # ----------------------------------------------------
        # Row 4: empirical and fitted upper tails
        # ----------------------------------------------------

        ax = axes[
            3,
            column_index
        ]

        descending, exceedance = empirical_exceedance(
            values
        )

        stationary_survival = genextreme.sf(
            common_x_grid,
            stationary_fit[
                "scipy_c"
            ],
            loc=stationary_fit[
                "loc_GW"
            ],
            scale=stationary_fit[
                "scale_GW"
            ],
        )

        nonstationary_survival = (
            1.0
            - pooled_marginal_cdf(
                common_x_grid,
                years,
                nonstationary_fit,
            )
        )

        ax.plot(
            descending,
            exceedance,
            marker="o",
            markersize=(
                2.0
                if sample_name
                == "ECMWF pooled"
                else 4.0
            ),
            linewidth=0.8,
            label="Empirical",
        )

        ax.plot(
            common_x_grid,
            stationary_survival,
            linewidth=1.6,
            label="Stationary GEV",
        )

        ax.plot(
            common_x_grid,
            nonstationary_survival,
            linestyle="--",
            linewidth=1.6,
            label=(
                "Non-stationary pooled marginal"
            ),
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlim(
            common_x_lower,
            common_x_upper,
        )

        # A common lower limit makes the different information
        # content of n=875 and n=35 visually explicit.
        ax.set_ylim(
            8e-4,
            1.15,
        )

        ax.set_xlabel(
            "Severity (GW)"
        )

        ax.set_ylabel(
            "Exceedance probability"
        )

        ax.set_title(
            "Empirical and fitted upper tails"
        )

        ax.grid(
            True,
            which="both",
            alpha=0.25,
        )

        ax.legend(
            fontsize=7.8,
        )

    fig.suptitle(
        (
            f"GEV diagnostics: {window}-day demand severity\n"
            "Stationary versus non-stationary shape-trend models; "
            "ECMWF pooled and Hannah overlap shifted, 1982-2016"
        ),
        fontsize=14.5,
        y=0.997,
    )

    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.965,
        ],
        h_pad=2.2,
        w_pad=1.7,
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
# Load and prepare data
# ============================================================

print("=" * 80)
print(
    "Final analysis 07: non-stationary shape-trend sensitivity"
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

print(
    f"\nECMWF rows: {len(ecmwf)}"
)

print(
    f"Hannah overlap rows: {len(hannah_overlap)}"
)

print(
    f"Primary windows: {WINDOWS}"
)


# ============================================================
# Fit stationary and shape-trend models
# ============================================================

comparison_rows = []
trajectory_rows = []
fit_records = {}

for window in WINDOWS:
    column = severity_column(
        window
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

    e_years = ecmwf[
        "winter_year"
    ].to_numpy(dtype=float)

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

    h_years = hannah_overlap[
        "winter_year"
    ].to_numpy(dtype=float)

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

    fit_records[
        window
    ] = {}

    for (
        sample_name,
        values_gw,
        years,
    ) in [
        (
            "ECMWF pooled",
            e_values_gw,
            e_years,
        ),
        (
            "Hannah overlap shifted",
            h_shifted_gw,
            h_years,
        ),
    ]:
        stationary_fit = fit_stationary_gev(
            values_gw
        )

        nonstationary_fit = fit_shape_trend_gev(
            values_gw,
            years,
        )

        delta_aic = (
            nonstationary_fit[
                "aic"
            ]
            - stationary_fit[
                "aic"
            ]
        )

        likelihood_ratio = max(
            0.0,
            2.0
            * (
                nonstationary_fit[
                    "log_likelihood"
                ]
                - stationary_fit[
                    "log_likelihood"
                ]
            ),
        )

        lr_p_value = chi2.sf(
            likelihood_ratio,
            df=1,
        )

        fit_records[
            window
        ][
            sample_name
        ] = {
            "values": values_gw,
            "years": years,
            "stationary": stationary_fit,
            "nonstationary": nonstationary_fit,
        }

        comparison_rows.append(
            {
                "window_days": window,
                "sample": sample_name,
                "n": len(
                    values_gw
                ),
                "mean_shift_GW": (
                    shift_gw
                    if sample_name
                    == "Hannah overlap shifted"
                    else 0.0
                ),
                "stationary_xi": stationary_fit[
                    "xi"
                ],
                "stationary_loc_GW": stationary_fit[
                    "loc_GW"
                ],
                "stationary_scale_GW": stationary_fit[
                    "scale_GW"
                ],
                "stationary_log_likelihood": stationary_fit[
                    "log_likelihood"
                ],
                "stationary_aic": stationary_fit[
                    "aic"
                ],
                "nonstationary_loc_GW": nonstationary_fit[
                    "loc_GW"
                ],
                "nonstationary_scale_GW": nonstationary_fit[
                    "scale_GW"
                ],
                "xi_1982": nonstationary_fit[
                    "xi_1982"
                ],
                "xi_change_per_decade": nonstationary_fit[
                    "xi_change_per_decade"
                ],
                "xi_2016": nonstationary_fit[
                    "xi_2016"
                ],
                "nonstationary_log_likelihood": nonstationary_fit[
                    "log_likelihood"
                ],
                "nonstationary_aic": nonstationary_fit[
                    "aic"
                ],
                "delta_aic_nonstationary_minus_stationary": (
                    delta_aic
                ),
                "likelihood_ratio_statistic": (
                    likelihood_ratio
                ),
                "likelihood_ratio_p_value_descriptive": (
                    lr_p_value
                ),
                "optimizer_success": nonstationary_fit[
                    "optimizer_success"
                ],
                "close_to_bound": nonstationary_fit[
                    "close_to_bound"
                ],
            }
        )

        for year in range(
            OVERLAP_START,
            OVERLAP_END + 1,
        ):
            trajectory_rows.append(
                {
                    "window_days": window,
                    "sample": sample_name,
                    "winter_year": year,
                    "xi": (
                        nonstationary_fit[
                            "xi_1982"
                        ]
                        + nonstationary_fit[
                            "xi_change_per_decade"
                        ]
                        * time_covariate(
                            [
                                year
                            ]
                        )[
                            0
                        ]
                    ),
                }
            )

    diagnostic_path = (
        DIAGNOSTIC_DIR
        / (
            f"stationary_vs_nonstationary_"
            f"{window}day_4x2.png"
        )
    )

    make_combined_diagnostic_figure(
        window=window,
        window_records=fit_records[
            window
        ],
        output_path=diagnostic_path,
    )


comparison = pd.DataFrame(
    comparison_rows
)

comparison.to_csv(
    OUT_MODEL_COMPARISON,
    index=False,
)

trajectories = pd.DataFrame(
    trajectory_rows
)

trajectories.to_csv(
    OUT_XI_TRAJECTORIES,
    index=False,
)


# ============================================================
# Summary figure: delta AIC and shape change per decade
# ============================================================

fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(
        12.8,
        5.2,
    ),
    dpi=300,
)

offsets = {
    "ECMWF pooled": -0.8,
    "Hannah overlap shifted": 0.8,
}

for sample_name in [
    "ECMWF pooled",
    "Hannah overlap shifted",
]:
    sample_data = comparison[
        comparison[
            "sample"
        ] == sample_name
    ].sort_values(
        "window_days"
    )

    x = (
        sample_data[
            "window_days"
        ].to_numpy(dtype=float)
        + offsets[
            sample_name
        ]
    )

    axes[
        0
    ].plot(
        x,
        sample_data[
            "delta_aic_nonstationary_minus_stationary"
        ],
        marker="o",
        linewidth=1.4,
        label=sample_name,
    )

    axes[
        1
    ].plot(
        x,
        sample_data[
            "xi_change_per_decade"
        ],
        marker="o",
        linewidth=1.4,
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
].axhline(
    -2.0,
    linestyle=":",
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
    "Delta AIC: non-stationary minus stationary"
)

axes[
    0
].set_title(
    "Model comparison"
)

axes[
    0
].set_xticks(
    WINDOWS
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
    fontsize=8.5,
)

axes[
    1
].axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

axes[
    1
].set_xlabel(
    "Averaging window (days)"
)

axes[
    1
].set_ylabel(
    r"Shape change per decade, $\xi_1$"
)

axes[
    1
].set_title(
    "Estimated shape trend"
)

axes[
    1
].set_xticks(
    WINDOWS
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
    fontsize=8.5,
)

fig.suptitle(
    "Shape-trend GEV sensitivity across primary windows",
    fontsize=14,
    y=1.02,
)

fig.tight_layout()

fig.savefig(
    OUT_SUMMARY_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_SUMMARY_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Shape trajectories: no more than two panels per row
# ============================================================

ncols = 2
nrows = math.ceil(
    len(
        WINDOWS
    )
    / ncols
)

fig, axes = plt.subplots(
    nrows=nrows,
    ncols=ncols,
    figsize=(
        13.0,
        3.8
        * nrows,
    ),
    dpi=300,
    sharex=True,
)

axes = np.asarray(
    axes
).reshape(-1)

legend_handles = None
legend_labels = None

for index, window in enumerate(
    WINDOWS
):
    ax = axes[
        index
    ]

    for sample_name in [
        "ECMWF pooled",
        "Hannah overlap shifted",
    ]:
        sample_data = trajectories[
            (
                trajectories[
                    "window_days"
                ] == window
            )
            & (
                trajectories[
                    "sample"
                ] == sample_name
            )
        ]

        ax.plot(
            sample_data[
                "winter_year"
            ],
            sample_data[
                "xi"
            ],
            linewidth=1.5,
            label=sample_name,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    ax.set_xlabel(
        "Winter year"
    )

    ax.set_ylabel(
        r"Conditional shape $\xi_t$"
    )

    ax.set_title(
        f"{window}-day severity"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    if legend_handles is None:
        (
            legend_handles,
            legend_labels,
        ) = ax.get_legend_handles_labels()

for index in range(
    len(
        WINDOWS
    ),
    len(
        axes
    ),
):
    axes[
        index
    ].axis(
        "off"
    )

fig.suptitle(
    "Fitted non-stationary GEV shape trajectories",
    fontsize=14,
    y=0.995,
)

fig.legend(
    legend_handles,
    legend_labels,
    loc="lower center",
    ncol=2,
    fontsize=9,
    bbox_to_anchor=(
        0.5,
        0.01,
    ),
)

fig.tight_layout(
    rect=[
        0,
        0.035,
        1,
        0.97,
    ]
)

fig.savefig(
    OUT_TRAJECTORY_FIGURE_PNG,
    bbox_inches="tight",
)

fig.savefig(
    OUT_TRAJECTORY_FIGURE_PDF,
    bbox_inches="tight",
)

plt.close(
    fig
)


# ============================================================
# Print results
# ============================================================

print("\nModel comparison:")
print(
    comparison[
        [
            "window_days",
            "sample",
            "stationary_xi",
            "xi_1982",
            "xi_change_per_decade",
            "xi_2016",
            "delta_aic_nonstationary_minus_stationary",
            "likelihood_ratio_p_value_descriptive",
            "optimizer_success",
            "close_to_bound",
        ]
    ].to_string(
        index=False
    )
)

print("\nSaved outputs:")
print(OUT_MODEL_COMPARISON)
print(OUT_XI_TRAJECTORIES)
print(OUT_SUMMARY_FIGURE_PNG)
print(OUT_SUMMARY_FIGURE_PDF)
print(OUT_TRAJECTORY_FIGURE_PNG)
print(OUT_TRAJECTORY_FIGURE_PDF)
print(DIAGNOSTIC_DIR)

print("\nDone.")
