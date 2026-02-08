import pandas as pd
from typing import List, Optional


def build_key_vars(
    youth_sex_col: Optional[str],
    parent_sex_col: Optional[str],
    gender_col: Optional[str],
    eth_cols: List[str]
) -> List[str]:
    """
    Construct the list of 'key variables' used for scoring baseline rows.
    These determine which T1 row is the most complete per participant.
    """
    key_vars = ["youth_age_y", "youth_age_m", "demo_cg_income"]

    if gender_col:
        key_vars.append(gender_col)

    # Ethnicity checkbox columns
    key_vars.extend(eth_cols)

    # Sex variables
    if youth_sex_col:
        key_vars.append(youth_sex_col)
    if parent_sex_col:
        key_vars.append(parent_sex_col)

    return key_vars


def _ensure_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    """
    Filter out key_vars that do not appear in the dataframe.
    Logs nothing; the pipeline will handle NA scoring implicitly.

    Returns only the subset of columns that actually exist.
    """
    return [c for c in cols if c in df.columns]


def best_baseline_rows(df: pd.DataFrame, key_vars: List[str]) -> pd.DataFrame:
    """
    Select the single best baseline (T1) row per participant.

    Logic:
    - Restrict to redcap_event_name == 't1_arm_1'
    - Score each row by number of non-missing key variables
    - Pick the highest-scoring row for each record_id

    Missing columns among key_vars are silently ignored.
    """
    # Restrict key_vars to existing columns
    usable_vars = _ensure_columns(df, key_vars)

    # Extract T1 rows
    t1 = df[df["redcap_event_name"] == "t1_arm_1"].copy()

    # Score completeness
    if usable_vars:
        t1["_score"] = t1[usable_vars].notna().sum(axis=1)
    else:
        # Edge case: no usable vars → score = 0 for all rows
        t1["_score"] = 0

    # Sort within each record_id by completeness (descending)
    # but maintain ascending record_id ordering across participants
    t1 = t1.sort_values(["record_id", "_score"], ascending=[True, False])

    # Pick the highest scoring row per participant
    best = t1.drop_duplicates(subset="record_id").drop(columns="_score")

    return best.reset_index(drop=True)
