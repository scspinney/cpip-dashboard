"""
analytics/targets.py

Utilities for loading and transforming C-PIP target distributions:
- Age × Sex (blocks)
- Age × Ethnicity (blocks)
- Totals by Ethnicity
- Observed sex/age, ethnicity/age, ethnicity totals

Reads:
    configs/age_bins.py
    configs/target_totals.py
    util/helpers.py
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

# ---------------------------------------------------------------------
# CORRECT RELATIVE IMPORTS
# ---------------------------------------------------------------------
from ..configs.age_bins import (
    AGE_BIN_ORDER,
    AGE_BLOCK_ORDER,
    bin_label_to_block,
)

from ..configs.target_totals import (
    TARGET_TOTALS,
    ETHNICITY_ORDER_TARGET,
)

from ..util.helpers import derive_site


# =====================================================================
# INTERNAL HELPERS
# =====================================================================

def _find_age_file(stem: str) -> Optional[Path]:
    """
    Look for a file matching patterns such as:
        *calgary*age_by_group*.csv
    Expected to be located in project root.
    """
    stem = stem.lower()
    for p in Path(".").glob(f"*{stem}*age_by_group*.csv"):
        return p
    return None


def _load_single_target_csv(p: Path) -> pd.DataFrame:
    """
    Load a single site-level target CSV (Age × Group)
    and convert to long format.
    """
    df = pd.read_csv(p, comment="#")

    # Standardize a known column naming edge-case
    df = df.rename(columns={"5 wks pre": "<5wks"})

    # Columns to retain
    keep_cols = ["Group"] + AGE_BIN_ORDER + ["Total"]
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()

    value_cols = [c for c in AGE_BIN_ORDER if c in df.columns]

    # Melt to long
    long = df.melt(
        id_vars=["Group"],
        value_vars=value_cols,
        var_name="age_bin_label",
        value_name="Target_N",
    )

    long["site"] = derive_site(p)
    long["Target_N"] = pd.to_numeric(long["Target_N"], errors="coerce").fillna(0)

    # Block conversion
    long["age_block"] = long["age_bin_label"].map(bin_label_to_block)

    # Fix ethnicity label mismatch
    long["Group"] = long["Group"].replace({"Middle Eastern": "Middle East"})

    return long[["site", "Group", "age_bin_label", "age_block", "Target_N"]]


# =====================================================================
# TARGET LOADING
# =====================================================================

def load_targets_age_by_group() -> pd.DataFrame:
    """
    Load targets for Calgary, Montreal, Toronto.
    """
    out = []
    for stem in ("calgary", "montreal", "toronto"):
        p = _find_age_file(stem)
        if p is None:
            print(f"⚠️ Missing age-by-group CSV for: {stem}")
            continue
        out.append(_load_single_target_csv(p))

    if not out:
        return pd.DataFrame(
            columns=["site", "Group", "age_bin_label", "age_block", "Target_N"]
        )

    df = pd.concat(out, ignore_index=True)

    df["age_block"] = pd.Categorical(
        df["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )

    return df


# =====================================================================
# TARGETS — AGE × SEX
# =====================================================================

def targets_age_by_sex(targets_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Target counts: site × sex × age-block.
    """
    sex_targets = (
        targets_long
        .loc[targets_long["Group"].isin(["Male", "Female"])]
        .dropna(subset=["age_block"])
        .groupby(["site", "Group", "age_block"], observed=False)["Target_N"]
        .sum()
        .reset_index()
        .rename(columns={"Group": "sex_lbl"})
    )

    sex_targets["age_block"] = pd.Categorical(
        sex_targets["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )

    return sex_targets


# =====================================================================
# TARGETS — AGE × ETHNICITY
# =====================================================================

def targets_age_by_ethnicity(targets_long: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Target counts: site × Ethnicity × age-block.
    """
    eth_targets = (
        targets_long
        .loc[targets_long["Group"].isin(ETHNICITY_ORDER_TARGET)]
        .dropna(subset=["age_block"])
        .groupby(["site", "Group", "age_block"], observed=False)["Target_N"]
        .sum()
        .reset_index()
        .rename(columns={"Group": "Ethnicity"})
    )

    eth_targets["Ethnicity"] = pd.Categorical(
        eth_targets["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )

    eth_targets["age_block"] = pd.Categorical(
        eth_targets["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )

    return eth_targets


# =====================================================================
# TARGET TOTALS — ETHNICITY (site-level)
# =====================================================================

def targets_ethnicity_totals() -> pd.DataFrame:
    """
    Compute Target totals: site × Ethnicity.
    """
    rows = []
    for site, payload in TARGET_TOTALS.items():
        eth_dict = payload.get("Ethnicity", {})
        for eth, n in eth_dict.items():
            rows.append({
                "site": site,
                "Ethnicity": eth,
                "Target_N": int(n),
            })

    df = pd.DataFrame(rows)

    df["Ethnicity"] = pd.Categorical(
        df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )

    return df.sort_values(["site", "Ethnicity"]).reset_index(drop=True)


# =====================================================================
# OBSERVED — SEX × AGE (blocks)
# =====================================================================

def build_obs_sex_age_blocks(baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Observed counts: site × sex_lbl × age_block.
    Requires:
        baseline["site"]
        baseline["age_block"]
        baseline["sex_effective_label"]
    """

    if "age_block" not in baseline.columns:
        raise ValueError("baseline must contain 'age_block'")

    # Determine column containing sex label
    if "sex_effective_label" in baseline.columns:
        sex_col = "sex_effective_label"
    elif "sex" in baseline.columns:
        sex_col = "sex"
    else:
        raise ValueError("Missing sex label column for observed counts.")

    df = (
        baseline
        .dropna(subset=["age_block", sex_col])
        .groupby(["site", sex_col, "age_block"], observed=False)
        .size()
        .reset_index(name="Observed_N")
    )

    df = df.rename(columns={sex_col: "sex_lbl"})

    df["age_block"] = pd.Categorical(
        df["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )

    return df


# =====================================================================
# OBSERVED — ETHNICITY × AGE (blocks)
# =====================================================================

def build_obs_eth_age_blocks(baseline: pd.DataFrame,
                             eth_long: pd.DataFrame) -> pd.DataFrame:
    """
    Observed ethnicity × age-block.
    eth_long schema must be:
        site | Ethnicity | age_block | count
    """

    required = {"site", "Ethnicity", "age_block", "count"}
    missing = required - set(eth_long.columns)
    if missing:
        raise ValueError(f"eth_long missing required columns: {missing}")

    df = (
        eth_long
        .dropna(subset=["Ethnicity", "age_block"])
        .groupby(["site", "Ethnicity", "age_block"], observed=False)["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "Observed_N"})
    )

    df["age_block"] = pd.Categorical(
        df["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )
    df["Ethnicity"] = pd.Categorical(
        df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )

    return df


# =====================================================================
# OBSERVED — ETHNICITY TOTALS (site-level)
# =====================================================================

def build_obs_eth_totals(baseline: pd.DataFrame,
                         eth_long: pd.DataFrame) -> pd.DataFrame:
    """
    Observed ethnicity totals: site × Ethnicity.
    """

    df = (
        eth_long
        .groupby(["site", "Ethnicity"], observed=False)["count"]
        .sum()
        .reset_index()
        .rename(columns={"count": "Observed_N"})
    )

    df["Ethnicity"] = pd.Categorical(
        df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )

    return df
