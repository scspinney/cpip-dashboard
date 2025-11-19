"""
analytics/site_counts.py

Compute per-site participant counts used for dashboard panels:

• Total baseline counts  
• Non-missing counts per demographic dimension  
• Missing counts per demographic dimension (with effective sex missingness)

This module expects the `baseline` dataframe already enriched with:
    - age_years
    - gender_lbl
    - sex_missing_effective  (True/False)
    - income_lbl
    - ethnicity checkboxes OR eth_long melted form
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Optional


# ──────────────────────────────────────────────────────────────
# Helper: generic missingness counter
# ──────────────────────────────────────────────────────────────

def _count_non_na(df: pd.DataFrame, column: str) -> pd.Series:
    """Return number of participants with non-NA values by site."""
    return df[df[column].notna()].groupby("site")["record_id"].nunique()


# ──────────────────────────────────────────────────────────────
# Main function: full site-level counts dataframe
# ──────────────────────────────────────────────────────────────

def compute_site_counts(
    baseline: pd.DataFrame,
    eth_cols: Optional[List[str]] = None,
    gender_col: Optional[str] = None,
    income_col: str = "demo_cg_income",
) -> pd.DataFrame:
    """
    Compute site-level NON-MISSING counts for:
        • Age (age_years)
        • Sex (effective missingness → sex_missing_effective)
        • Gender (gender_col provided)
        • Income (income_lbl)
        • Ethnicity (eth_cols list)

    Returns a tidy DataFrame with:
        Total available counts
        Age non-NA
        Sex non-NA  (effective)
        Gender non-NA
        Income non-NA
        Ethnicity non-NA
    """
    # Total participants per site
    total = baseline.groupby("site")["record_id"].nunique()

    # Age
    age_non_na = _count_non_na(baseline, "age_years")

    # Sex (effective):
    # sex_missing_effective = True means NA → so non-NA = count(False)
    sex_non_na = (
        baseline.groupby("site")["sex_missing_effective"]
        .apply(lambda s: (~s).sum())
    )

    # Gender
    if gender_col and (gender_col in baseline.columns):
        gender_non_na = _count_non_na(baseline, gender_col)
    else:
        gender_non_na = pd.Series(0, index=total.index)

    # Income
    if income_col in baseline.columns:
        income_non_na = _count_non_na(baseline, income_col)
    else:
        income_non_na = pd.Series(0, index=total.index)

    # Ethnicity
    if eth_cols:
        # non-NA means: at least one ethnicity checkbox marked 1
        eth_non_na = (
            baseline[["site"] + eth_cols]
            .assign(any_eth=lambda d: (d[eth_cols].fillna(0).sum(axis=1) > 0))
            .groupby("site")["any_eth"].sum()
        )
    else:
        eth_non_na = pd.Series(0, index=total.index)

    # Combine all
    out = pd.DataFrame({
        "Total available counts": total,
        "Age non-NA": age_non_na.reindex(total.index, fill_value=0),
        "Sex non-NA (effective)": sex_non_na.reindex(total.index, fill_value=0),
        "Gender non-NA": gender_non_na.reindex(total.index, fill_value=0),
        "Income non-NA": income_non_na.reindex(total.index, fill_value=0),
        "Ethnicity non-NA": eth_non_na.reindex(total.index, fill_value=0),
    })

    return out


# ──────────────────────────────────────────────────────────────
# Missingness wrapper
# ──────────────────────────────────────────────────────────────

def compute_missing_counts(
    site_counts: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert NON-MISSING counts into MISSING counts.

    Inputs:
        site_counts (from compute_site_counts)

    Output columns:
        • Age NA
        • Sex NA
        • Gender NA
        • Income NA
        • Ethnicity NA
    """
    total = site_counts["Total available counts"]

    out = pd.DataFrame(index=site_counts.index)

    out["Total available counts"] = total
    out["Age NA"]       = total - site_counts["Age non-NA"]
    out["Sex NA"]       = total - site_counts["Sex non-NA (effective)"]
    out["Gender NA"]    = total - site_counts["Gender non-NA"]
    out["Income NA"]    = total - site_counts["Income non-NA"]
    out["Ethnicity NA"] = total - site_counts["Ethnicity non-NA"]

    return out


# ──────────────────────────────────────────────────────────────
# Combined convenience function
# ──────────────────────────────────────────────────────────────

def compute_all_counts(
    baseline: pd.DataFrame,
    eth_cols: Optional[List[str]],
    gender_col: Optional[str],
    income_col: str = "demo_cg_income"
) -> (pd.DataFrame, pd.DataFrame):
    """
    Return both:
        (site_counts, missing_counts)
    """
    sc = compute_site_counts(
        baseline,
        eth_cols=eth_cols,
        gender_col=gender_col,
        income_col=income_col
    )
    mc = compute_missing_counts(sc)
    return sc, mc
