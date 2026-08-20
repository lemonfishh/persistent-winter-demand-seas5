from pathlib import Path

# ============================================================
# Project paths
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = PROJECT_DIR / "outputs"
AUDIT_DIR = OUTPUT_DIR / "audit"
SEVERITY_DIR = OUTPUT_DIR / "severity"
CALIBRATION_DIR = OUTPUT_DIR / "calibration"
FIGURE_DIR = OUTPUT_DIR / "figures"

for directory in [
    OUTPUT_DIR,
    AUDIT_DIR,
    SEVERITY_DIR,
    CALIBRATION_DIR,
    FIGURE_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# Dissertation definitions
# ============================================================

WINTER_START_YEAR = 1982
WINTER_END_YEAR = 2016

VALID_START_MONTH_DAY = "11-08"
VALID_END_MONTH_DAY = "03-31"

SEVERITY_WINDOWS = [1, 7, 14, 21, 28, 56, 84]

BENCHMARK_WINTER_YEAR = 1963
