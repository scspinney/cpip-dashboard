import numpy as np
import pandas as pd
from typing import Optional

# Income utilities from the shared helpers module
from ..util.helpers import (
    _as_int_or_nan,
    closest_code_from_dollars,
    SPECIAL_INCOME,
    _BRACKET_MEDIAN,
)


def merge_income_row(row: pd.Series) -> Optional[int]:
    """
    Merge caregiver + partner income into a single REDCap bracket code.

    Rules:
    -------
    • If either caregiver or partner income is a special code (777/888),
      return that code.
    • If only one is available → return that one.
    • If both present → convert medians to dollars, sum, then map back to closest code.
    • If neither present → return NaN.
    """
    cg = _as_int_or_nan(row.get("demo_cg_income"))
    pt = _as_int_or_nan(row.get("demo_cg_prtnr_income"))

    # Special-case overrides (Unknown / Prefer not to answer)
    if cg in SPECIAL_INCOME:
        return cg
    if pt in SPECIAL_INCOME:
        return pt

    # Full-missing
    if pd.isna(cg) and pd.isna(pt):
        return np.nan

    # Single-respondent cases
    if pd.isna(pt):
        return cg
    if pd.isna(cg):
        return pt

    # Merge via dollar medians
    total_dollars = _BRACKET_MEDIAN[int(cg)] + _BRACKET_MEDIAN[int(pt)]
    return closest_code_from_dollars(total_dollars)


def add_merged_income(non_repeat: pd.DataFrame) -> pd.DataFrame:
    """
    Add/replace `demo_cg_income` with merged caregiver + partner income.

    Handles the following dataset variants:
    ------------------------------------------------
    1) Both caregiver & partner income columns exist
    2) Only partner column exists
    3) Only caregiver column exists
    4) Neither exists → output column filled with NaN

    Returns a NEW DataFrame.
    """
    df = non_repeat.copy()
    cols = set(df.columns)

    if {"demo_cg_income", "demo_cg_prtnr_income"} & cols:
        # At least one income field exists — perform merge logic
        df["demo_cg_income"] = df.apply(merge_income_row, axis=1)

    elif "demo_cg_prtnr_income" in cols:
        # Only partner information exists
        df["demo_cg_income"] = df["demo_cg_prtnr_income"].apply(_as_int_or_nan)

    elif "demo_cg_income" in cols:
        # Only caregiver info exists — just clean
        df["demo_cg_income"] = df["demo_cg_income"].apply(_as_int_or_nan)

    else:
        # Neither column exists → define clean column
        df["demo_cg_income"] = np.nan

    return df
