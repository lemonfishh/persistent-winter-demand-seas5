from pathlib import Path
import pandas as pd

# ============================================================
# Input / output paths
# ============================================================

ECMWF_SEVERITY_PATH = Path("outputs/severity/ecmwf_severity_summary_Nov08_1982_2016.csv")
HANNAH_SEVERITY_PATH = Path("outputs/severity/hannah_severity_summary_Nov08.csv")

OUTPUT_AUDIT_DIR = Path("outputs/audit")
OUTPUT_TABLE_DIR = Path("outputs/tables")

OUTPUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)

OUT_SAMPLE_AUDIT = OUTPUT_AUDIT_DIR / "core_sample_audit_Nov08.csv"
OUT_BENCHMARK_VALUES = OUTPUT_TABLE_DIR / "hannah_1963_benchmark_values_Nov08.csv"
OUT_ECMWF_SAMPLE_SUMMARY = OUTPUT_TABLE_DIR / "ecmwf_single_vs_full_summary_Nov08.csv"

# ============================================================
# Settings
# ============================================================

OVERLAP_START = 1982
OVERLAP_END = 2016
BENCHMARK_WINTER = 1963

SEVERITY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]
SEVERITY_COLS = [f"max_{w}d_mean_demand_MW" for w in SEVERITY_WINDOWS]

# ============================================================
# Load data
# ============================================================

ecmwf = pd.read_csv(ECMWF_SEVERITY_PATH)
hannah = pd.read_csv(HANNAH_SEVERITY_PATH)

print("=" * 80)
print("Preparing core samples for ECMWF-Hannah comparison")
print("=" * 80)

print("\nLoaded ECMWF severity:")
print(ECMWF_SEVERITY_PATH)
print("Shape:", ecmwf.shape)

print("\nLoaded Hannah severity:")
print(HANNAH_SEVERITY_PATH)
print("Shape:", hannah.shape)

# ============================================================
# Basic checks
# ============================================================

required_ecmwf_cols = ["dataset", "winter_year", "member"] + SEVERITY_COLS
required_hannah_cols = ["dataset", "winter_year"] + SEVERITY_COLS

missing_ecmwf = [c for c in required_ecmwf_cols if c not in ecmwf.columns]
missing_hannah = [c for c in required_hannah_cols if c not in hannah.columns]

if missing_ecmwf:
    raise ValueError(f"Missing ECMWF columns: {missing_ecmwf}")

if missing_hannah:
    raise ValueError(f"Missing Hannah columns: {missing_hannah}")

# ============================================================
# Define core samples
# ============================================================

# 1. ECMWF full ensemble, overlap winters 1982-2016
ecmwf_full = ecmwf[
    (ecmwf["winter_year"] >= OVERLAP_START)
    & (ecmwf["winter_year"] <= OVERLAP_END)
].copy()

# 2. ECMWF fixed-member reference sequence
# Working convention: first available member
single_member_id = sorted(ecmwf_full["member"].unique())[0]

ecmwf_single = ecmwf_full[
    ecmwf_full["member"] == single_member_id
].copy()

# 3. Hannah overlap winters 1982-2016
hannah_overlap = hannah[
    (hannah["winter_year"] >= OVERLAP_START)
    & (hannah["winter_year"] <= OVERLAP_END)
].copy()

# 4. Hannah 1962/63 benchmark = winter 1963
hannah_benchmark = hannah[
    hannah["winter_year"] == BENCHMARK_WINTER
].copy()

# ============================================================
# Print sample sizes
# ============================================================

print("\nCore sample sizes:")
print("ECMWF full ensemble rows:", len(ecmwf_full))
print("ECMWF single-member rows:", len(ecmwf_single))
print("ECMWF single member used:", single_member_id)
print("Hannah overlap rows:", len(hannah_overlap))
print("Hannah benchmark rows:", len(hannah_benchmark))

print("\nECMWF full ensemble winters:")
print(ecmwf_full["winter_year"].min(), "to", ecmwf_full["winter_year"].max())
print("Number of winters:", ecmwf_full["winter_year"].nunique())
print("Number of members:", ecmwf_full["member"].nunique())
print("Winter-member combinations:", ecmwf_full[["winter_year", "member"]].drop_duplicates().shape[0])

print("\nECMWF single-member reference winters:")
print(ecmwf_single["winter_year"].min(), "to", ecmwf_single["winter_year"].max())
print("Number of winters:", ecmwf_single["winter_year"].nunique())

print("\nHannah overlap winters:")
print(hannah_overlap["winter_year"].min(), "to", hannah_overlap["winter_year"].max())
print("Number of winters:", hannah_overlap["winter_year"].nunique())

print("\nHannah benchmark winter:")
print(hannah_benchmark[["winter_year"] + SEVERITY_COLS])

# ============================================================
# Hard checks
# ============================================================

if len(ecmwf_full) != 875:
    raise ValueError(f"Expected ECMWF full ensemble to have 875 rows, got {len(ecmwf_full)}")

if len(ecmwf_single) != 35:
    raise ValueError(f"Expected ECMWF single-member reference to have 35 rows, got {len(ecmwf_single)}")

if len(hannah_overlap) != 35:
    raise ValueError(f"Expected Hannah overlap to have 35 rows, got {len(hannah_overlap)}")

if len(hannah_benchmark) != 1:
    raise ValueError(f"Expected Hannah benchmark winter {BENCHMARK_WINTER} to have 1 row, got {len(hannah_benchmark)}")

# ============================================================
# Save sample audit table
# ============================================================

sample_audit = pd.DataFrame(
    [
        {
            "sample": "ECMWF full ensemble",
            "purpose": "full alternative-weather ensemble for tail comparison",
            "winter_years": f"{OVERLAP_START}-{OVERLAP_END}",
            "members": "all 25 members",
            "n_rows": len(ecmwf_full),
            "n_winters": ecmwf_full["winter_year"].nunique(),
            "n_members": ecmwf_full["member"].nunique(),
        },
        {
            "sample": "ECMWF fixed-member reference",
            "purpose": "historical-length ECMWF reference sequence",
            "winter_years": f"{OVERLAP_START}-{OVERLAP_END}",
            "members": f"member {single_member_id}",
            "n_rows": len(ecmwf_single),
            "n_winters": ecmwf_single["winter_year"].nunique(),
            "n_members": ecmwf_single["member"].nunique(),
        },
        {
            "sample": "Hannah overlap",
            "purpose": "historical reference sample for calibration",
            "winter_years": f"{OVERLAP_START}-{OVERLAP_END}",
            "members": "not applicable",
            "n_rows": len(hannah_overlap),
            "n_winters": hannah_overlap["winter_year"].nunique(),
            "n_members": None,
        },
        {
            "sample": "Hannah 1962/63 benchmark",
            "purpose": "out-of-overlap severe historical benchmark",
            "winter_years": str(BENCHMARK_WINTER),
            "members": "not applicable",
            "n_rows": len(hannah_benchmark),
            "n_winters": hannah_benchmark["winter_year"].nunique(),
            "n_members": None,
        },
    ]
)

sample_audit.to_csv(OUT_SAMPLE_AUDIT, index=False)

# ============================================================
# Save Hannah 1962/63 benchmark values
# ============================================================

benchmark_values = hannah_benchmark[
    ["dataset", "winter_year"] + SEVERITY_COLS
].copy()

benchmark_values.to_csv(OUT_BENCHMARK_VALUES, index=False)

# ============================================================
# Save ECMWF single vs full summary by severity window
# ============================================================

summary_rows = []

for window, col in zip(SEVERITY_WINDOWS, SEVERITY_COLS):
    full_values = ecmwf_full[col]
    single_values = ecmwf_single[col]
    benchmark_value = float(hannah_benchmark[col].iloc[0])

    summary_rows.append(
        {
            "severity_window_days": window,
            "severity_column": col,

            "ecmwf_single_n": len(single_values),
            "ecmwf_single_mean_MW": single_values.mean(),
            "ecmwf_single_std_MW": single_values.std(),
            "ecmwf_single_min_MW": single_values.min(),
            "ecmwf_single_max_MW": single_values.max(),

            "ecmwf_full_n": len(full_values),
            "ecmwf_full_mean_MW": full_values.mean(),
            "ecmwf_full_std_MW": full_values.std(),
            "ecmwf_full_min_MW": full_values.min(),
            "ecmwf_full_max_MW": full_values.max(),

            "hannah_1963_raw_MW": benchmark_value,
            "n_ecmwf_full_exceeding_hannah_1963_raw": int((full_values > benchmark_value).sum()),
            "pct_ecmwf_full_exceeding_hannah_1963_raw": 100.0 * (full_values > benchmark_value).mean(),
        }
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT_ECMWF_SAMPLE_SUMMARY, index=False)

print("\nSaved sample audit to:")
print(OUT_SAMPLE_AUDIT)

print("\nSaved Hannah 1962/63 benchmark values to:")
print(OUT_BENCHMARK_VALUES)

print("\nSaved ECMWF single vs full summary to:")
print(OUT_ECMWF_SAMPLE_SUMMARY)

print("\nECMWF single vs full summary:")
print(summary)