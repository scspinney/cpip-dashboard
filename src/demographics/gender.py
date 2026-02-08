import pandas as pd
import numpy as np

# Mapping from REDCap integer codes → labels
gender_map = {
    1: "Boy/Man",
    2: "Girl/Woman",
    3: "Non-binary",
    4: "Two-spirit",
    5: "Another",
    6: "Questioning",
    777: "Unknown",
    888: "Prefer not to answer",
}

def _norm_int(x):
    """
    Safely normalize REDCap values to int or NaN.
    Avoids mapping failures when REDCap exports floats or strings.
    """
    try:
        return int(x)
    except Exception:
        return np.nan


def add_gender_label(
    df: pd.DataFrame,
    gender_col: str | None,
    compute_missing: bool = False,
) -> pd.DataFrame:
    """
    Add a `gender_lbl` column using REDCap-coded gender values.

    Parameters
    ----------
    df : pd.DataFrame
        Baseline dataframe (or subset).
    gender_col : str | None
        Column name containing the REDCap gender integer code.
    compute_missing : bool
        Future hook: whether to compute gender_missing_effective.

    Returns
    -------
    pd.DataFrame
        Dataframe with a new `gender_lbl` column.
    """

    out = df.copy()

    if gender_col is None:
        out["gender_lbl"] = pd.NA
        if compute_missing:
            out["gender_missing_effective"] = False
        return out

    norm = out[gender_col].apply(_norm_int)

    out["gender_lbl"] = norm.map(gender_map)
    out.loc[norm.isna(), "gender_lbl"] = pd.NA

    if compute_missing:
        # “True missingness” = field exists but is not a valid code
        out["gender_missing_effective"] = norm.isna()

    return out
