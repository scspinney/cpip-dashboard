import numpy as np
import pandas as pd
from ..util.helpers import _norm_int

sex_map = {
    1: "Male",
    2: "Female",
    3: "Intersex / Non-binary",
    777: "Unknown",
    888: "Prefer not to answer"
}

SEX_VALID = {1, 2, 3, 777, 888}


def add_effective_sex(
    baseline: pd.DataFrame,
    youth_sex_col: str | None,
    parent_sex_col: str | None,
    youth_form_col: str | None,
    parent_form_col: str | None,
) -> pd.DataFrame:
    """
    Compute EFFECTIVE sex assigned at birth following CPIP instrument logic:

    RULES:
    - If age >= 11:
          expected reporter = youth
          use youth sex ONLY if youth form is complete AND a valid value exists
    - If age < 11:
          expected reporter = parent
          use parent sex ONLY if parent form is complete AND a valid value exists
    - Fallbacks:
          if expected form incomplete, DO NOT mark as missing
          if youth missing but parent answered → use parent
    - Missing only when:
          expected form is completed AND no valid value exists
    """

    df = baseline.copy()

    # ------------------------------
    # Normalize raw sex values
    # ------------------------------
    df["sex_youth_raw"] = (
        df[youth_sex_col].apply(_norm_int) if youth_sex_col else np.nan
    )
    df["sex_parent_raw"] = (
        df[parent_sex_col].apply(_norm_int) if parent_sex_col else np.nan
    )

    # ------------------------------
    # Expected reporter by age
    # ------------------------------
    df["expected_youth"] = df["age_years"] >= 11
    df["expected_parent"] = df["age_years"] < 11

    # ------------------------------
    # Completion flags
    # ------------------------------
    def _form_completed(x):
        try:
            return int(x) in (1, 3)    # REDCap: 1=Unverified, 3=Complete
        except Exception:
            return False

    df["youth_form_complete"] = (
        df[youth_form_col].apply(_form_completed)
        if (youth_form_col and youth_form_col in df.columns) else False
    )
    df["parent_form_complete"] = (
        df[parent_form_col].apply(_form_completed)
        if (parent_form_col and parent_form_col in df.columns) else False
    )

    # ------------------------------
    # Effective sex logic
    # ------------------------------
    def _effective_sex(row):
        # Expected youth reporter
        if row["expected_youth"]:
            if row["youth_form_complete"] and row["sex_youth_raw"] in SEX_VALID:
                return int(row["sex_youth_raw"])

        # Expected parent reporter
        if row["expected_parent"]:
            if row["parent_form_complete"] and row["sex_parent_raw"] in SEX_VALID:
                return int(row["sex_parent_raw"])

        # Fallback: parent answered anyway
        if row["sex_parent_raw"] in SEX_VALID:
            return int(row["sex_parent_raw"])

        return np.nan

    df["sex_code_effective"] = df.apply(_effective_sex, axis=1)
    df["sex_lbl"] = df["sex_code_effective"].map(sex_map)

    # ------------------------------
    # TRUE missing logic
    # ------------------------------
    def _true_missing(row):
        # Not missing if ANY valid value exists
        if (row["sex_youth_raw"] in SEX_VALID) or (row["sex_parent_raw"] in SEX_VALID):
            return False

        # Expected youth form was completed but no answer
        if row["expected_youth"]:
            return bool(row["youth_form_complete"])

        # Expected parent form completed but no answer
        if row["expected_parent"]:
            return bool(row["parent_form_complete"])

        return False

    df["sex_missing_effective"] = df.apply(_true_missing, axis=1)
    df["sex_present_effective"] = ~df["sex_missing_effective"]

    return df
