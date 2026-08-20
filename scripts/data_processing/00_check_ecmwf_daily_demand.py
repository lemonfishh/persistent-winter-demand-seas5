from pathlib import Path
import pandas as pd

# ============================================================
# Input file
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[2]

path = (
    PROJECT_DIR
    / "outputs"
    / "daily"
    / "ecmwf_daily_demand_1982_2016.csv"
)

output_dir = PROJECT_DIR / "outputs" / "audit"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("Checking ECMWF daily demand file")
print("=" * 80)
print("Path:", path)

if not path.exists():
    raise FileNotFoundError(f"File not found: {path}")

df = pd.read_csv(path)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

# ============================================================
# Basic date check
# ============================================================

if "date" not in df.columns:
    raise ValueError("No 'date' column found.")

df["date"] = pd.to_datetime(df["date"])

print("\nDate range:")
print("min date:", df["date"].min())
print("max date:", df["date"].max())

# ============================================================
# Winter-year check
# ============================================================

if "winter_year" in df.columns:
    winter_years = sorted(df["winter_year"].dropna().unique())

    print("\nWinter years:")
    print(winter_years)
    print("number of winter years:", len(winter_years))
else:
    print("\nNo winter_year column found.")

# ============================================================
# Member check
# ============================================================

if "member" in df.columns:
    members = sorted(df["member"].dropna().unique())

    print("\nMembers:")
    print(members)
    print("number of members:", len(members))

    if "winter_year" in df.columns:
        member_audit = (
            df.groupby("winter_year")
            .agg(
                n_members=("member", "nunique"),
                first_date=("date", "min"),
                last_date=("date", "max"),
                n_unique_dates=("date", "nunique"),
                n_rows=("date", "size"),
            )
            .reset_index()
        )

        print("\nMember/date audit by winter:")
        print(member_audit)

        member_audit.to_csv(output_dir / "ecmwf_daily_demand_audit_by_winter.csv", index=False)
        print("\nSaved:")
        print(output_dir / "ecmwf_daily_demand_audit_by_winter.csv")
else:
    print("\nNo member column found.")

# ============================================================
# Demand-column check
# ============================================================

possible_demand_cols = [c for c in df.columns if "demand" in c.lower()]

print("\nPossible demand columns:")
print(possible_demand_cols)

# ============================================================
# Missing-value check for key columns
# ============================================================

key_cols = []
for c in ["date", "winter_year", "member"]:
    if c in df.columns:
        key_cols.append(c)

key_cols += possible_demand_cols

print("\nMissing values in key columns:")
print(df[key_cols].isna().sum())

# ============================================================
# Quick expected-size check
# ============================================================

if "winter_year" in df.columns and "member" in df.columns:
    n_winter_member = df[["winter_year", "member"]].drop_duplicates().shape[0]
    print("\nNumber of winter-member combinations:")
    print(n_winter_member)

    print("\nIf this is 35 winters × 25 members, expected value is 875.")