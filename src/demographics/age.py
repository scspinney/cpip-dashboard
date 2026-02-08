import pandas as pd
import numpy as np
import re
from typing import Optional

# ---------------------------------------------------------
# Add continuous age (in years)
# ---------------------------------------------------------
def add_age_years(baseline: pd.DataFrame) -> pd.DataFrame:
    """Compute participant age in years from youth_age_y + youth_age_m."""
    def _age_years(row):
        y, m = row.get("youth_age_y"), row.get("youth_age_m")
        if pd.isna(y) and pd.isna(m):
            return np.nan
        return (0 if pd.isna(y) else y) + (0 if pd.isna(m) else m) / 12

    df = baseline.copy()
    df["age_years"] = df.apply(_age_years, axis=1)
    return df


# ---------------------------------------------------------
# Map raw age values to broad developmental blocks
# These bins are used in many plots and must remain consistent.
# ---------------------------------------------------------
AGE_BLOCK_ORDER = ["0-2", "2-5", "6-9", "10-12", "13-15", "16-18"]

def to_age_block(a: float) -> Optional[str]:
    """Map numeric age (years) to inclusive developmental brackets."""
    if pd.isna(a):
        return None
    if a <= 2:   return "0-2"
    if a <= 5:   return "2-5"
    if a <= 9:   return "6-9"
    if a <= 12:  return "10-12"
    if a <= 15:  return "13-15"
    if a <= 18:  return "16-18"
    return None


# ---------------------------------------------------------
# Convert *raw age-bin labels* (from Fil’s target CSVs)
# into the same age blocks as above.
# ---------------------------------------------------------
def bin_label_to_block(lbl: str) -> Optional[str]:
    """
    Convert an age-bin label like "0-1", "1-2", "10-11", "<5wks", "18"
    into the standard developmental block label.
    """
    if lbl is None:
        return None

    # special early bins
    if lbl in ("<5wks", "0-1", "1-2"):
        return "0-2"

    # terminal case
    if lbl == "18":
        return "16-18"

    # pattern-based bins
    m = re.match(r"^(\d+)-(\d+)$", str(lbl))
    if not m:
        return None

    k = int(m.group(1))  # lower edge

    if k < 2:   return "0-2"
    if k < 6:   return "2-5"
    if k < 10:  return "6-9"
    if k < 13:  return "10-12"
    if k < 16:  return "13-15"
    return "16-18"
