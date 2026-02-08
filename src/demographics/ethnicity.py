import numpy as np
import pandas as pd
from typing import Optional, List, Dict

# ---------------------------------------------------------------------
# Canonical ethnicity labels (as in REDCap CFQ and demo_ethnicity___X)
# ---------------------------------------------------------------------
ETH_LABELS = {
    1: "Black",
    2: "East Asian",
    3: "First Nations",
    4: "Inuk/Inuit",
    5: "Métis",
    6: "Latin American",
    7: "Middle Eastern",
    8: "South Asian",
    9: "Southeast Asian",
    10: "White",
    11: "Other",
    888: "Prefer not to say",
}

ASIAN = {"East Asian", "South Asian", "Southeast Asian"}
INDIG = {"First Nations", "Inuk/Inuit", "Métis"}


# ---------------------------------------------------------------------
# Build long-form ethnicity table
# ---------------------------------------------------------------------
def build_eth_long(baseline: pd.DataFrame, eth_cols: List[str]) -> pd.DataFrame:
    """
    Returns long-form ethnicity DataFrame:
        record_id | site | variable | val | label
    """
    if not eth_cols:
        return pd.DataFrame()

    df = baseline.copy()
    df[eth_cols] = df[eth_cols].fillna(0)

    # lookup for demo_ethnicity___X → label
    def _extract_code(col: str) -> Optional[int]:
        try:
            return int(col.split("___")[-1])
        except Exception:
            return None

    eth_lookup = {
        col: ETH_LABELS.get(code)
        for col in eth_cols
        if (code := _extract_code(col)) in ETH_LABELS
    }

    eth_long = (
        df[["record_id", "site"] + eth_cols]
        .melt(id_vars=["record_id", "site"], value_name="val")
        .query("val == 1")
        .assign(label=lambda d: d["variable"].map(eth_lookup))
    )

    return eth_long


# ---------------------------------------------------------------------
# White vs Non-white panel (wide format)
# ---------------------------------------------------------------------
def build_white_nonwhite_panel(baseline: pd.DataFrame, eth_long: pd.DataFrame) -> pd.DataFrame:
    if eth_long.empty:
        return pd.DataFrame()

    tmp = (
        eth_long.groupby(["site", "record_id"])["label"]
        .apply(lambda s: sorted(set(s.dropna())))
        .reset_index(name="labels")
    )

    tmp["is_white_only"] = tmp["labels"].apply(
        lambda L: ("White" in L) and (len(set(L)) == 1)
    )
    tmp["has_nonwhite"] = tmp["labels"].apply(
        lambda L: any(lbl != "White" for lbl in set(L))
    )

    eth_wide = tmp[["site", "record_id", "is_white_only", "has_nonwhite"]].copy()
    eth_wide["White"] = eth_wide["is_white_only"].map({True: "White", False: np.nan})
    eth_wide["Non-white"] = eth_wide["has_nonwhite"].map({True: "Non-white", False: np.nan})

    return eth_wide[["site", "record_id", "White", "Non-white"]]


# ---------------------------------------------------------------------
# Collapse to target-bin ethnicity for block-based overlays
# ---------------------------------------------------------------------
def collapse_eth_for_blocks(lbl: str) -> Optional[str]:
    """
    Collapses raw ethnicity labels into:
        ["White", "Asian", "Black", "Indigenous", "Middle East", "Latin American", "Other"]
    """
    if lbl in ASIAN:
        return "Asian"
    if lbl in INDIG:
        return "Indigenous"
    if lbl == "Middle Eastern":
        return "Middle East"
    if lbl in {"Black", "Latin American", "Other", "White"}:
        return lbl
    return None


# ---------------------------------------------------------------------
# Collapse for TOTAL-level target vs observed comparisons (slightly different rules)
# ---------------------------------------------------------------------
def collapse_eth_for_totals(lbl: str) -> Optional[str]:
    """
    Same as blocks, but excludes 'Prefer not to say' and keeps only canonical target bins.
    """
    if lbl in ASIAN:
        return "Asian"
    if lbl in INDIG:
        return "Indigenous"
    if lbl == "Middle Eastern":
        return "Middle East"
    if lbl in {"Black", "Latin American", "White", "Other"}:
        return lbl
    return None


# ---------------------------------------------------------------------
# Utility for merging baseline to ethnicity for block panels
# ---------------------------------------------------------------------
def add_age_block(df: pd.DataFrame, age_block_func) -> pd.DataFrame:
    """
    Adds an `age_block` column using a provided mapping function (e.g. to_age_block).
    Deeply used in:
       - observed vs target overlays (age × ethnicity)
       - MRI-age plots
    """
    out = df.copy()
    out["age_block"] = out["age_years"].apply(age_block_func)
    return out
