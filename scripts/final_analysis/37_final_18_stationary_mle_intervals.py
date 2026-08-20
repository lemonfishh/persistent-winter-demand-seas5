"""
Model-based MLE confidence intervals for the stationary GEV analysis.

This is a supplementary calculation for the stationary GEV fits already used
in the dissertation.  It does not replace the winter-year cluster bootstrap
and it does not redraw the main stationary figure.

For each of the seven primary averaging windows and both samples, the script:
    1. refits the same stationary GEV by maximum likelihood;
    2. calculates the observed Hessian of the negative log-likelihood;
    3. inverts the Hessian to obtain model-based standard errors;
    4. reports approximate 95% Wald intervals for location, scale and shape;
    5. reports a delta-method 95% interval for the 20-block return level.

The intervals are conditional on the fitted GEV likelihood and treat the
observations supplied to each fit as independent.  They therefore do not
account for dependence among the 25 SEAS5 members within a winter year.

The original CSV and figures are not overwritten.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import genextreme

from config import OUTPUT_DIR


# ============================================================
# Settings
# ============================================================

PRIMARY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]
OVERLAP_START = 1982
OVERLAP_END = 2016
RETURN_PERIOD = 20.0
CI_Z = 1.959963984540054


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

ORIGINAL_FITS_FILE = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "06_stationary_gev_primary_windows"
    / "stationary_gev_original_fits_primary_windows.csv"
)

OUT_DIR = (
    OUTPUT_DIR
    / "final_analysis_clean"
    / "18_stationary_gev_mle_intervals"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_INTERVAL_FILE = (
    OUT_DIR
    / "stationary_gev_mle_intervals_primary_windows.csv"
)


# ============================================================
# Helpers
# ============================================================

def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(
            "Missing {0}:\n{1}".format(label, path)
        )


def severity_column(window):
    return "max_{0}d_mean_demand_MW".format(window)


def finite_values(values):
    values = np.asarray(values, dtype=float)
    return values[np.isfinite(values)]


def stationary_negative_log_likelihood(parameters, values_gw):
    """
    Stationary GEV negative log-likelihood.

    parameters = [location, log(scale), conventional EVT shape xi]
    SciPy uses c = -xi.
    """

    location, log_scale, xi = parameters
    scale = np.exp(log_scale)

    if not np.all(np.isfinite(parameters)) or scale <= 0.0:
        return 1e12

    logpdf = genextreme.logpdf(
        values_gw,
        c=-xi,
        loc=location,
        scale=scale,
    )

    if not np.all(np.isfinite(logpdf)):
        return 1e12

    value = float(-np.sum(logpdf))

    if not np.isfinite(value):
        return 1e12

    return value


def return_level_from_parameters(parameters, return_period):
    location, log_scale, xi = parameters
    scale = np.exp(log_scale)
    probability = 1.0 - 1.0 / float(return_period)

    value = genextreme.ppf(
        probability,
        c=-xi,
        loc=location,
        scale=scale,
    )

    return float(value)


def refit_stationary_gev(values_gw, original_row):
    values_gw = finite_values(values_gw)

    if len(values_gw) < 10:
        raise ValueError("Too few observations for a GEV fit.")

    sample_sd = np.std(values_gw, ddof=1)

    if sample_sd <= 0.0:
        raise ValueError("Cannot fit a zero-variance sample.")

    initial = np.asarray(
        [
            float(original_row["loc_GW"]),
            np.log(float(original_row["scale_GW"])),
            float(original_row["xi"]),
        ],
        dtype=float,
    )

    minimum_scale = max(sample_sd * 0.05, 1e-4)
    maximum_scale = max(sample_sd * 20.0, minimum_scale * 2.0)

    bounds = [
        (
            np.min(values_gw) - 3.0 * sample_sd,
            np.max(values_gw) + 3.0 * sample_sd,
        ),
        (np.log(minimum_scale), np.log(maximum_scale)),
        (-0.8, 0.8),
    ]

    starts = [
        initial,
        initial + np.asarray([0.0, 0.0, -0.03]),
        initial + np.asarray([0.0, 0.0, 0.03]),
    ]

    successful_results = []

    for start in starts:
        result = minimize(
            stationary_negative_log_likelihood,
            x0=start,
            args=(values_gw,),
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "maxiter": 5000,
                "ftol": 1e-12,
                "gtol": 1e-8,
            },
        )

        if (
            result.success
            and np.isfinite(result.fun)
            and result.fun < 1e11
        ):
            successful_results.append(result)

    if len(successful_results) == 0:
        raise RuntimeError("All stationary GEV optimisations failed.")

    best = min(successful_results, key=lambda item: item.fun)
    parameters = np.asarray(best.x, dtype=float)

    return {
        "parameters": parameters,
        "values_gw": values_gw,
        "loc_GW": float(parameters[0]),
        "scale_GW": float(np.exp(parameters[1])),
        "xi": float(parameters[2]),
        "return_level_20block_GW": return_level_from_parameters(
            parameters,
            RETURN_PERIOD,
        ),
        "negative_log_likelihood": float(best.fun),
    }


# ============================================================
# Numerical observed Hessian
# ============================================================

def central_hessian(function, point, steps):
    point = np.asarray(point, dtype=float)
    steps = np.asarray(steps, dtype=float)
    dimension = len(point)
    hessian = np.empty((dimension, dimension), dtype=float)

    f_zero = float(function(point))

    if not np.isfinite(f_zero) or f_zero >= 1e11:
        raise ValueError("Invalid likelihood at the fitted parameters.")

    for i in range(dimension):
        plus = point.copy()
        minus = point.copy()
        plus[i] += steps[i]
        minus[i] -= steps[i]

        f_plus = float(function(plus))
        f_minus = float(function(minus))

        if (
            not np.isfinite(f_plus)
            or not np.isfinite(f_minus)
            or f_plus >= 1e11
            or f_minus >= 1e11
        ):
            raise ValueError("Invalid diagonal Hessian perturbation.")

        hessian[i, i] = (
            f_plus - 2.0 * f_zero + f_minus
        ) / (steps[i] ** 2)

        for j in range(i + 1, dimension):
            plus_plus = point.copy()
            plus_minus = point.copy()
            minus_plus = point.copy()
            minus_minus = point.copy()

            plus_plus[i] += steps[i]
            plus_plus[j] += steps[j]

            plus_minus[i] += steps[i]
            plus_minus[j] -= steps[j]

            minus_plus[i] -= steps[i]
            minus_plus[j] += steps[j]

            minus_minus[i] -= steps[i]
            minus_minus[j] -= steps[j]

            function_values = np.asarray(
                [
                    function(plus_plus),
                    function(plus_minus),
                    function(minus_plus),
                    function(minus_minus),
                ],
                dtype=float,
            )

            if (
                not np.all(np.isfinite(function_values))
                or np.any(function_values >= 1e11)
            ):
                raise ValueError("Invalid off-diagonal Hessian perturbation.")

            value = (
                function_values[0]
                - function_values[1]
                - function_values[2]
                + function_values[3]
            ) / (4.0 * steps[i] * steps[j])

            hessian[i, j] = value
            hessian[j, i] = value

    return 0.5 * (hessian + hessian.T)


def numerical_gradient(function, point, steps):
    point = np.asarray(point, dtype=float)
    steps = np.asarray(steps, dtype=float)
    gradient = np.empty(len(point), dtype=float)

    for i in range(len(point)):
        plus = point.copy()
        minus = point.copy()
        plus[i] += steps[i]
        minus[i] -= steps[i]

        gradient[i] = (
            float(function(plus)) - float(function(minus))
        ) / (2.0 * steps[i])

    return gradient


def observed_hessian_covariance(fit):
    parameters = fit["parameters"]
    values_gw = fit["values_gw"]

    objective = lambda pars: stationary_negative_log_likelihood(
        pars,
        values_gw,
    )

    parameter_scales = np.asarray(
        [
            max(fit["scale_GW"], 0.5),
            1.0,
            0.1,
        ],
        dtype=float,
    )

    candidates = []

    for relative_step in [5e-4, 1e-3, 2e-3, 5e-3]:
        steps = parameter_scales * relative_step

        try:
            hessian = central_hessian(
                function=objective,
                point=parameters,
                steps=steps,
            )

            eigenvalues = np.linalg.eigvalsh(hessian)

            if (
                not np.all(np.isfinite(eigenvalues))
                or np.min(eigenvalues) <= 0.0
            ):
                continue

            condition_number = float(np.linalg.cond(hessian))

            if (
                not np.isfinite(condition_number)
                or condition_number > 1e12
            ):
                continue

            covariance = np.linalg.inv(hessian)
            standard_errors = np.sqrt(np.diag(covariance))

            if not np.all(np.isfinite(standard_errors)):
                continue

            candidates.append(
                {
                    "covariance": covariance,
                    "standard_errors": standard_errors,
                    "condition_number": condition_number,
                    "relative_step": relative_step,
                    "steps": steps,
                }
            )

        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            continue

    if len(candidates) == 0:
        return {
            "success": False,
            "covariance": np.full((3, 3), np.nan),
            "standard_errors": np.full(3, np.nan),
            "condition_number": np.nan,
            "relative_step": np.nan,
            "steps": np.full(3, np.nan),
        }

    best = min(candidates, key=lambda item: item["condition_number"])
    best["success"] = True
    return best


def make_interval_row(window, sample_name, fit, covariance_result):
    parameters = fit["parameters"]
    covariance = covariance_result["covariance"]
    standard_errors = covariance_result["standard_errors"]

    loc_estimate = fit["loc_GW"]
    scale_estimate = fit["scale_GW"]
    xi_estimate = fit["xi"]
    rl20_estimate = fit["return_level_20block_GW"]

    if covariance_result["success"]:
        loc_se = standard_errors[0]
        log_scale_se = standard_errors[1]
        xi_se = standard_errors[2]

        loc_lower = loc_estimate - CI_Z * loc_se
        loc_upper = loc_estimate + CI_Z * loc_se

        # The fitted parameter is log(scale), so exponentiating its Wald
        # limits guarantees a positive interval for scale.
        scale_lower = np.exp(parameters[1] - CI_Z * log_scale_se)
        scale_upper = np.exp(parameters[1] + CI_Z * log_scale_se)

        xi_lower = xi_estimate - CI_Z * xi_se
        xi_upper = xi_estimate + CI_Z * xi_se

        rl_function = lambda pars: return_level_from_parameters(
            pars,
            RETURN_PERIOD,
        )

        rl_gradient = numerical_gradient(
            function=rl_function,
            point=parameters,
            steps=covariance_result["steps"],
        )

        rl_variance = float(
            rl_gradient.T.dot(covariance).dot(rl_gradient)
        )

        rl_se = np.sqrt(max(rl_variance, 0.0))
        rl_lower = rl20_estimate - CI_Z * rl_se
        rl_upper = rl20_estimate + CI_Z * rl_se

    else:
        loc_se = np.nan
        log_scale_se = np.nan
        xi_se = np.nan
        rl_se = np.nan
        loc_lower = np.nan
        loc_upper = np.nan
        scale_lower = np.nan
        scale_upper = np.nan
        xi_lower = np.nan
        xi_upper = np.nan
        rl_lower = np.nan
        rl_upper = np.nan

    return {
        "window_days": window,
        "sample": sample_name,
        "n": len(fit["values_gw"]),
        "loc_GW": loc_estimate,
        "loc_mle_se": loc_se,
        "loc_mle_ci_lower_95_GW": loc_lower,
        "loc_mle_ci_upper_95_GW": loc_upper,
        "scale_GW": scale_estimate,
        "log_scale_mle_se": log_scale_se,
        "scale_mle_ci_lower_95_GW": scale_lower,
        "scale_mle_ci_upper_95_GW": scale_upper,
        "xi": xi_estimate,
        "xi_mle_se": xi_se,
        "xi_mle_ci_lower_95": xi_lower,
        "xi_mle_ci_upper_95": xi_upper,
        "xi_mle_ci_contains_zero": bool(
            np.isfinite(xi_lower)
            and xi_lower <= 0.0 <= xi_upper
        ),
        "return_level_20block_GW": rl20_estimate,
        "return_level_20block_mle_se_GW": rl_se,
        "return_level_20block_mle_ci_lower_95_GW": rl_lower,
        "return_level_20block_mle_ci_upper_95_GW": rl_upper,
        "mle_hessian_success": covariance_result["success"],
        "mle_hessian_condition_number": (
            covariance_result["condition_number"]
        ),
        "mle_hessian_relative_step": (
            covariance_result["relative_step"]
        ),
        "mle_interval_type": (
            "Observed-Hessian Wald; independent-observation likelihood"
        ),
    }


# ============================================================
# Read inputs
# ============================================================

require_file(ECMWF_SEVERITY_FILE, "ECMWF severity CSV")
require_file(HANNAH_SEVERITY_FILE, "Hannah severity CSV")
require_file(ORIGINAL_FITS_FILE, "stationary original-fit CSV")

ecmwf = pd.read_csv(ECMWF_SEVERITY_FILE)
hannah = pd.read_csv(HANNAH_SEVERITY_FILE)
original_fits = pd.read_csv(ORIGINAL_FITS_FILE)

for dataframe in [ecmwf, hannah]:
    dataframe["winter_year"] = pd.to_numeric(
        dataframe["winter_year"],
        errors="coerce",
    )

ecmwf = ecmwf[
    ecmwf["winter_year"].between(OVERLAP_START, OVERLAP_END)
].copy()

hannah_overlap = hannah[
    hannah["winter_year"].between(OVERLAP_START, OVERLAP_END)
].copy()


# ============================================================
# Fit and calculate intervals
# ============================================================

interval_rows = []
refit_checks = []

for window in PRIMARY_WINDOWS:
    column = severity_column(window)

    if column not in ecmwf.columns or column not in hannah_overlap.columns:
        raise KeyError("Missing severity column: {0}".format(column))

    e_values_gw = finite_values(
        pd.to_numeric(ecmwf[column], errors="coerce")
        .to_numpy(dtype=float)
        / 1000.0
    )

    h_raw_gw = finite_values(
        pd.to_numeric(hannah_overlap[column], errors="coerce")
        .to_numpy(dtype=float)
        / 1000.0
    )

    mean_shift_gw = np.mean(e_values_gw) - np.mean(h_raw_gw)
    h_shifted_gw = h_raw_gw + mean_shift_gw

    sample_values = [
        ("ECMWF pooled", e_values_gw),
        ("Hannah overlap shifted", h_shifted_gw),
    ]

    for sample_name, values_gw in sample_values:
        selected = original_fits[
            (original_fits["window_days"] == window)
            & (original_fits["sample"] == sample_name)
        ]

        if len(selected) != 1:
            raise ValueError(
                "Expected one original-fit row for {0}-day, {1}; found {2}."
                .format(window, sample_name, len(selected))
            )

        original_row = selected.iloc[0]

        print(
            "Calculating {0}-day, {1} ...".format(window, sample_name)
        )

        fit = refit_stationary_gev(values_gw, original_row)
        covariance_result = observed_hessian_covariance(fit)

        interval_rows.append(
            make_interval_row(
                window=window,
                sample_name=sample_name,
                fit=fit,
                covariance_result=covariance_result,
            )
        )

        refit_checks.append(
            {
                "window_days": window,
                "sample": sample_name,
                "xi_difference": abs(fit["xi"] - original_row["xi"]),
                "loc_difference_GW": abs(
                    fit["loc_GW"] - original_row["loc_GW"]
                ),
                "scale_difference_GW": abs(
                    fit["scale_GW"] - original_row["scale_GW"]
                ),
            }
        )


intervals = pd.DataFrame(interval_rows)
refit_check = pd.DataFrame(refit_checks)

if not intervals["mle_hessian_success"].all():
    failed = intervals.loc[
        ~intervals["mle_hessian_success"],
        ["window_days", "sample"],
    ]
    warnings.warn(
        "Observed Hessian failed for:\n{0}".format(
            failed.to_string(index=False)
        )
    )

maximum_parameter_difference = refit_check[
    ["xi_difference", "loc_difference_GW", "scale_difference_GW"]
].to_numpy(dtype=float).max()

if maximum_parameter_difference > 5e-4:
    raise RuntimeError(
        "The refitted parameters do not reproduce the original stationary "
        "fits closely enough. Maximum absolute difference = {0:.6g}"
        .format(maximum_parameter_difference)
    )

intervals.to_csv(OUT_INTERVAL_FILE, index=False)


# ============================================================
# Console output
# ============================================================

display_columns = [
    "window_days",
    "sample",
    "xi",
    "xi_mle_ci_lower_95",
    "xi_mle_ci_upper_95",
    "xi_mle_ci_contains_zero",
    "return_level_20block_GW",
    "return_level_20block_mle_ci_lower_95_GW",
    "return_level_20block_mle_ci_upper_95_GW",
    "mle_hessian_success",
]

print("\nStationary GEV model-based MLE intervals:")
print(
    intervals[display_columns].to_string(
        index=False,
        float_format=lambda value: "{0:.6f}".format(value),
    )
)

print("\nMaximum absolute difference from the original point estimates:")
print("{0:.8f}".format(maximum_parameter_difference))

print("\nSaved CSV:")
print(OUT_INTERVAL_FILE)

