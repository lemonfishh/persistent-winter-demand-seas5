from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BENCHMARK_WINTER_YEAR,
    CALIBRATION_DIR,
    SEVERITY_DIR,
    SEVERITY_WINDOWS,
    WINTER_END_YEAR,
    WINTER_START_YEAR,
)


# ============================================================
# 13. Mean-shift Hannah severity onto the ECMWF scale
#
# Main logic:
# 1. Use only the common 1982-2016 overlap period to estimate
#    the mean difference between Hannah and pooled ECMWF.
#
# 2. Calculate a separate mean shift for each severity duration.
#
# 3. Apply the overlap-derived shift to the full Hannah record,
#    including winter 1963.
#
# 4. Keep all original Hannah severity values unchanged.
# ============================================================


# ============================================================
# Input paths
# ============================================================

ECMWF_SEVERITY_PATH = (
    SEVERITY_DIR
    / "ecmwf_severity_summary_Nov08_1982_2016.csv"
)

HANNAH_SEVERITY_PATH = (
    SEVERITY_DIR
    / "hannah_severity_summary_Nov08.csv"
)


# ============================================================
# Output paths
# ============================================================

CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

OUT_SHIFT_SUMMARY = (
    CALIBRATION_DIR
    / "hannah_to_ecmwf_mean_shift_summary_Nov08.csv"
)

OUT_HANNAH_SHIFTED = (
    CALIBRATION_DIR
    / "hannah_severity_mean_shifted_to_ecmwf_Nov08.csv"
)


# ============================================================
# Load data
# ============================================================

ecmwf = pd.read_csv(ECMWF_SEVERITY_PATH)
hannah = pd.read_csv(HANNAH_SEVERITY_PATH)

print("=" * 80)
print("Mean-shifting Hannah severity onto the ECMWF scale")
print("=" * 80)

print("\nLoaded ECMWF severity:")
print(ECMWF_SEVERITY_PATH)
print("Shape:", ecmwf.shape)

print("\nLoaded Hannah severity:")
print(HANNAH_SEVERITY_PATH)
print("Shape:", hannah.shape)


# ============================================================
# Define severity columns
# ============================================================

severity_cols = [
    f"max_{window}d_mean_demand_MW"
    for window in SEVERITY_WINDOWS
]

print("\nSeverity windows:")
print(SEVERITY_WINDOWS)

print("\nSeverity columns:")
for col in severity_cols:
    print(col)


# ============================================================
# Basic checks
# ============================================================

required_ecmwf_cols = [
    "dataset",
    "winter_year",
    "member",
] + severity_cols

required_hannah_cols = [
    "dataset",
    "winter_year",
] + severity_cols

missing_ecmwf = [
    col
    for col in required_ecmwf_cols
    if col not in ecmwf.columns
]

missing_hannah = [
    col
    for col in required_hannah_cols
    if col not in hannah.columns
]

if missing_ecmwf:
    raise ValueError(
        f"Missing required ECMWF columns: {missing_ecmwf}"
    )

if missing_hannah:
    raise ValueError(
        f"Missing required Hannah columns: {missing_hannah}"
    )

if ecmwf[severity_cols].isna().any().any():
    raise ValueError(
        "Missing values found in ECMWF severity columns."
    )

if hannah[severity_cols].isna().any().any():
    raise ValueError(
        "Missing values found in Hannah severity columns."
    )


# ============================================================
# Define overlap samples
# ============================================================

# ECMWF pooled ensemble during 1982-2016.
ecmwf_overlap = ecmwf[
    (ecmwf["winter_year"] >= WINTER_START_YEAR)
    & (ecmwf["winter_year"] <= WINTER_END_YEAR)
].copy()

# Hannah historical winters during the same 1982-2016 period.
hannah_overlap = hannah[
    (hannah["winter_year"] >= WINTER_START_YEAR)
    & (hannah["winter_year"] <= WINTER_END_YEAR)
].copy()

# Hannah winter 1963 corresponds to winter 1962/63.
hannah_benchmark = hannah[
    hannah["winter_year"] == BENCHMARK_WINTER_YEAR
].copy()


# ============================================================
# Print sample information
# ============================================================

print("\nCalibration period:")
print(f"{WINTER_START_YEAR}-{WINTER_END_YEAR}")

print("\nSamples used to estimate the mean shifts:")

print(
    "ECMWF pooled winter-member rows:",
    len(ecmwf_overlap),
)

print(
    "ECMWF winters:",
    ecmwf_overlap["winter_year"].nunique(),
)

print(
    "ECMWF members:",
    ecmwf_overlap["member"].nunique(),
)

print(
    "Hannah overlap winters:",
    len(hannah_overlap),
)

print(
    "Hannah benchmark winter:",
    BENCHMARK_WINTER_YEAR,
)


# ============================================================
# Hard sample-size checks
# ============================================================

if len(ecmwf_overlap) != 875:
    raise ValueError(
        "Expected 875 ECMWF winter-member rows "
        f"(35 winters x 25 members), got {len(ecmwf_overlap)}."
    )

if ecmwf_overlap["winter_year"].nunique() != 35:
    raise ValueError(
        "Expected 35 ECMWF winters in the overlap period."
    )

if ecmwf_overlap["member"].nunique() != 25:
    raise ValueError(
        "Expected 25 ECMWF members."
    )

if len(hannah_overlap) != 35:
    raise ValueError(
        "Expected 35 Hannah winters in the overlap period, "
        f"got {len(hannah_overlap)}."
    )

if len(hannah_benchmark) != 1:
    raise ValueError(
        f"Expected one Hannah benchmark row for winter "
        f"{BENCHMARK_WINTER_YEAR}, "
        f"got {len(hannah_benchmark)}."
    )


# ============================================================
# Prepare full Hannah output
# ============================================================

# This dataframe keeps the full Hannah record from 1951-2020.
# The original raw severity columns will not be changed.
hannah_shifted = hannah.copy()

hannah_shifted["calibration_scale"] = (
    "ECMWF pooled mean, calibrated using 1982-2016 overlap"
)

summary_rows = []


# ============================================================
# Calculate duration-specific mean shifts
# ============================================================

for window, severity_col in zip(
    SEVERITY_WINDOWS,
    severity_cols,
):

    print("\n" + "-" * 80)
    print(f"Processing {window}-day severity")
    print("-" * 80)

    # Values used to estimate the correction.
    ecmwf_values = (
        ecmwf_overlap[severity_col]
        .astype(float)
    )

    hannah_values = (
        hannah_overlap[severity_col]
        .astype(float)
    )

    # Mean and standard deviation during the overlap period.
    ecmwf_mean = ecmwf_values.mean()
    hannah_mean = hannah_values.mean()

    ecmwf_std = ecmwf_values.std(ddof=1)
    hannah_std = hannah_values.std(ddof=1)

    # Positive shift means Hannah is moved upwards.
    # Negative shift means Hannah is moved downwards.
    mean_shift_mw = ecmwf_mean - hannah_mean

    # New output column names.
    shift_col = f"mean_shift_{window}d_MW"

    corrected_col = (
        f"max_{window}d_mean_demand_MW"
        f"_shifted_to_ecmwf"
    )

    # Store the same duration-specific shift for every Hannah winter.
    hannah_shifted[shift_col] = mean_shift_mw

    # Apply the shift to the full Hannah record.
    # The original raw severity column remains unchanged.
    hannah_shifted[corrected_col] = (
        hannah_shifted[severity_col]
        + mean_shift_mw
    )

    # ========================================================
    # Validate the correction
    # ========================================================

    shifted_overlap_mean = hannah_shifted.loc[
        (
            hannah_shifted["winter_year"]
            >= WINTER_START_YEAR
        )
        & (
            hannah_shifted["winter_year"]
            <= WINTER_END_YEAR
        ),
        corrected_col,
    ].mean()

    if not np.isclose(
        shifted_overlap_mean,
        ecmwf_mean,
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError(
            f"{window}-day validation failed: "
            "shifted Hannah overlap mean does not equal "
            "the pooled ECMWF mean."
        )

    # ========================================================
    # Extract raw and shifted 1962/63 benchmark
    # ========================================================

    benchmark_raw_mw = float(
        hannah_benchmark[severity_col].iloc[0]
    )

    benchmark_shifted_mw = (
        benchmark_raw_mw
        + mean_shift_mw
    )

    # ========================================================
    # Save summary information
    # ========================================================

    summary_rows.append(
        {
            "severity_window_days": window,

            "severity_column_raw": severity_col,

            "severity_column_shifted": corrected_col,

            "calibration_period": (
                f"{WINTER_START_YEAR}-{WINTER_END_YEAR}"
            ),

            "hannah_overlap_n": len(hannah_values),

            "ecmwf_pooled_n": len(ecmwf_values),

            "hannah_overlap_mean_MW": hannah_mean,

            "ecmwf_pooled_mean_MW": ecmwf_mean,

            "mean_shift_hannah_to_ecmwf_MW": (
                mean_shift_mw
            ),

            "mean_shift_hannah_to_ecmwf_GW": (
                mean_shift_mw / 1000.0
            ),

            "hannah_overlap_std_MW": hannah_std,

            "ecmwf_pooled_std_MW": ecmwf_std,

            "std_ratio_ecmwf_to_hannah": (
                ecmwf_std / hannah_std
                if hannah_std > 0
                else np.nan
            ),

            "hannah_1963_raw_MW": benchmark_raw_mw,

            "hannah_1963_shifted_to_ecmwf_MW": (
                benchmark_shifted_mw
            ),

            "hannah_1963_raw_GW": (
                benchmark_raw_mw / 1000.0
            ),

            "hannah_1963_shifted_to_ecmwf_GW": (
                benchmark_shifted_mw / 1000.0
            ),
        }
    )

    print(
        f"Hannah overlap mean: "
        f"{hannah_mean:.3f} MW"
    )

    print(
        f"ECMWF pooled mean: "
        f"{ecmwf_mean:.3f} MW"
    )

    print(
        f"Hannah-to-ECMWF mean shift: "
        f"{mean_shift_mw:.3f} MW"
    )

    print(
        f"Hannah standard deviation: "
        f"{hannah_std:.3f} MW"
    )

    print(
        f"ECMWF standard deviation: "
        f"{ecmwf_std:.3f} MW"
    )

    print(
        f"1962/63 raw benchmark: "
        f"{benchmark_raw_mw:.3f} MW"
    )

    print(
        f"1962/63 shifted benchmark: "
        f"{benchmark_shifted_mw:.3f} MW"
    )


# ============================================================
# Create summary dataframe
# ============================================================

shift_summary = pd.DataFrame(summary_rows)


# ============================================================
# Save outputs
# ============================================================

shift_summary.to_csv(
    OUT_SHIFT_SUMMARY,
    index=False,
)

hannah_shifted.to_csv(
    OUT_HANNAH_SHIFTED,
    index=False,
)


# ============================================================
# Print summary table
# ============================================================

display_cols = [
    "severity_window_days",
    "hannah_overlap_mean_MW",
    "ecmwf_pooled_mean_MW",
    "mean_shift_hannah_to_ecmwf_MW",
    "hannah_overlap_std_MW",
    "ecmwf_pooled_std_MW",
    "std_ratio_ecmwf_to_hannah",
    "hannah_1963_raw_MW",
    "hannah_1963_shifted_to_ecmwf_MW",
]

print("\n" + "=" * 80)
print("Mean-shift summary")
print("=" * 80)

print(
    shift_summary[display_cols]
    .round(3)
    .to_string(index=False)
)


# ============================================================
# Final output messages
# ============================================================

print("\nSaved mean-shift summary to:")
print(OUT_SHIFT_SUMMARY)

print("\nSaved full Hannah raw and shifted severity record to:")
print(OUT_HANNAH_SHIFTED)

print("\nImportant interpretation:")

print(
    "- The mean shifts were estimated only from "
    "the 1982-2016 overlap period."
)

print(
    "- The pooled ECMWF sample contains "
    "875 winter-member realisations."
)

print(
    "- All original Hannah severity columns were "
    "retained unchanged."
)

print(
    "- The overlap-derived duration-specific shifts were "
    "applied to the full Hannah record, including winter 1963."
)

print(
    "- Raw winter 1963 values remain available for the "
    "historical event benchmark."
)

print("\nDone.")