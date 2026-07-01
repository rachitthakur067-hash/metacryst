# =============================================================================
# MATERIALS INFORMATICS — DATA INGESTION & CLEANING PIPELINE
# =============================================================================
# Purpose : Ingest raw alloy composition data, clean it, validate it
#           metallurgically, then export feature matrix X and target y.
# Usage   : Fill in the CONFIGURATION block below, then run cell-by-cell
#           in VS Code Jupyter or execute as a plain Python script.
# Deps    : pandas, numpy  (pip install pandas numpy)
# =============================================================================

import pandas as pd
import numpy as np
import re

# =============================================================================
# ⚙️  CONFIGURATION  — edit these before running
# =============================================================================

RAW_FILE        = "alloy_properties_raw.csv"   # path to your raw CSV
TARGET_COL      = "Tensile Strength: Ultimate (UTS) (psi)"                # y column name
ELEMENTAL_COLS  = ["Fe", "Ni", "Co", "Cr", "Mn", "C", "Mo", "Si", "Cu", "Al", "W", "V", "Ti", "Nb"]  # feature columns (X)

# Base-element mode
# Set BASE_ELEMENT to the symbol string (e.g. "Fe") if that element is the
# balance and is NOT already in the file; set to None if all elements are
# explicitly listed and no back-calculation is needed.
BASE_ELEMENT    = None   # e.g.  "Fe"  or  None

# Validation tolerance: rows whose elemental sum deviates from 100 % by more
# than this threshold (in percentage points) will be flagged.
SUM_TOLERANCE   = 2.0    # pp  (e.g. 2.0 means 98–102 % is acceptable)

OUTPUT_FILE     = "clean_pipeline_data.csv"    # final export path

# =============================================================================
# STEP 1 — DATA LOADING
# =============================================================================

print("=" * 70)
print("STEP 1 │ Loading raw data")
print("=" * 70)

df_raw = pd.read_csv(RAW_FILE)

print(f"  ✔  Loaded '{RAW_FILE}'")
print(f"     Shape  : {df_raw.shape[0]} rows × {df_raw.shape[1]} columns")
print(f"     Columns: {list(df_raw.columns)}\n")
print("--- Head (first 5 rows) ---")
print(df_raw.head())
print()

# Work on a copy so the raw frame is preserved for reference
df = df_raw.copy()

# =============================================================================
# STEP 2 — NaN HANDLING
# =============================================================================
# Missing composition values mean the element is absent from that alloy → 0 %.

print("=" * 70)
print("STEP 2 │ NaN handling in elemental columns")
print("=" * 70)

nan_report = df[ELEMENTAL_COLS].isna().sum()
total_nans = nan_report.sum()

print("  NaN counts per element (before fill):")
for col, cnt in nan_report.items():
    marker = "⚠ " if cnt > 0 else "  "
    print(f"    {marker}{col}: {cnt}")

df[ELEMENTAL_COLS] = df[ELEMENTAL_COLS].fillna(0.0)

print(f"\n  ✔  Filled {total_nans} NaN(s) with 0.0\n")

# =============================================================================
# STEP 3 — DATA-TYPE CLEANING
# =============================================================================
# Strip '%', whitespace, footnote markers, and any other non-numeric
# characters, then cast to float64.

print("=" * 70)
print("STEP 3 │ Data-type cleaning")
print("=" * 70)

def to_float(value) -> float:
    """
    Convert a raw cell (str, int, float, NaN) to a clean float.
    Removes any characters that are not digits, dots, minus signs,
    or scientific-notation markers (e, E, +).
    Returns np.nan if conversion fails after stripping.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = re.sub(r"[^0-9.\-eE+]", "", value.strip())
        try:
            return float(cleaned)
        except ValueError:
            return np.nan
    return np.nan


for col in ELEMENTAL_COLS:
    original_dtype = df[col].dtype
    df[col] = df[col].apply(to_float)
    print(f"  {col}: dtype {original_dtype} → {df[col].dtype}")

print(f"\n  ✔  All elemental columns cast to float64\n")

# =============================================================================
# STEP 3b — BASE-ELEMENT BACK-CALCULATION (optional)
# =============================================================================
# If the balance element is not stored explicitly, compute it as the remainder.

if BASE_ELEMENT is not None:
    print("=" * 70)
    print(f"STEP 3b │ Computing balance for base element '{BASE_ELEMENT}'")
    print("=" * 70)

    non_base = [c for c in ELEMENTAL_COLS if c != BASE_ELEMENT]
    df[BASE_ELEMENT] = 100.0 - df[non_base].sum(axis=1)

    # Clamp to [0, 100] — negative values indicate data issues caught in Step 4
    df[BASE_ELEMENT] = df[BASE_ELEMENT].clip(lower=0.0)
    print(f"  ✔  '{BASE_ELEMENT}' computed; negative values clamped to 0.0\n")

# =============================================================================
# STEP 4 — LOGIC VALIDATION (metallurgical sanity check)
# =============================================================================
# Each row's elemental percentages should sum to ≈ 100 %.

print("=" * 70)
print("STEP 4 │ Metallurgical sanity check (row-wise compositional sum)")
print("=" * 70)

df["_elem_sum"] = df[ELEMENTAL_COLS].sum(axis=1)
df["_sum_ok"]   = df["_elem_sum"].between(100.0 - SUM_TOLERANCE,
                                           100.0 + SUM_TOLERANCE)

n_pass = df["_sum_ok"].sum()
n_fail = (~df["_sum_ok"]).sum()

print(f"  Tolerance band : 100 ± {SUM_TOLERANCE} pp  "
      f"({100-SUM_TOLERANCE:.1f} – {100+SUM_TOLERANCE:.1f} %)")
print(f"  Rows passing   : {n_pass} / {len(df)}")
print(f"  Rows failing   : {n_fail} / {len(df)}")

if n_fail > 0:
    flagged = df.loc[~df["_sum_ok"], ELEMENTAL_COLS + ["_elem_sum"]]
    print(f"\n  ⚠  Flagged rows (review before modelling):")
    print(flagged.to_string(index=True))
    print()
    print("  Action: these rows are retained but tagged. Consider dropping or")
    print("  re-examining them prior to model training.\n")
else:
    print("\n  ✔  All rows pass the compositional sum check.\n")

# Attach flag to df so downstream users can filter if needed
df["compositional_sum_flag"] = np.where(df["_sum_ok"], "PASS", "FAIL")
df.drop(columns=["_elem_sum", "_sum_ok"], inplace=True)

# =============================================================================
# STEP 5 — ISOLATION & EXPORT
# =============================================================================

print("=" * 70)
print("STEP 5 │ Isolation of X / y and export")
print("=" * 70)

# --- Feature matrix and target vector ----------------------------------------
X = df[ELEMENTAL_COLS].copy()
y = df[TARGET_COL].copy()

print(f"  Feature matrix X : {X.shape}  →  columns: {list(X.columns)}")
print(f"  Target vector  y : {y.shape}  →  column : '{y.name}'")
print(f"\n  X dtypes:\n{X.dtypes.to_string()}")
print(f"\n  y dtype : {y.dtype}")
print(f"\n  y descriptive stats:\n{y.describe().to_string()}")

# --- Export ------------------------------------------------------------------
export_df = X.copy()
export_df[TARGET_COL] = y.values
export_df["compositional_sum_flag"] = df["compositional_sum_flag"].values

export_df.to_csv(OUTPUT_FILE, index=False)

print(f"\n  ✔  Clean data exported → '{OUTPUT_FILE}'")
print(f"     Columns in file: {list(export_df.columns)}")
print(f"     Rows           : {len(export_df)}")

# =============================================================================
# PIPELINE SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("PIPELINE COMPLETE — Summary")
print("=" * 70)
print(f"  Raw file        : {RAW_FILE}")
print(f"  Output file     : {OUTPUT_FILE}")
print(f"  Total rows      : {len(df)}")
print(f"  NaNs filled     : {total_nans}")
print(f"  Composition flags — PASS: {n_pass}  |  FAIL: {n_fail}")
print(f"  Features (X)    : {ELEMENTAL_COLS}")
print(f"  Target   (y)    : {TARGET_COL}")
print("=" * 70)
print("  X and y are available as pandas objects for immediate use.")
print("  e.g.  from sklearn.model_selection import train_test_split")
print("        X_train, X_test, y_train, y_test = train_test_split(X, y)")
print("=" * 70)
