import numpy as np
import pandas as pd
from pandas import DataFrame
from ..util.constants import MIN_T1_DATE, MRI_INSTRUMENT


def build_mri_baseline(df_main: DataFrame) -> DataFrame:
    """
    Extract the baseline MRI rows (t1_arm_1 × mri instrument),
    normalize dates, apply completion logic,
    and filter to valid rows only.
    """

    # Ensure correct instrument & event
    mri_base = df_main.loc[
        (df_main["redcap_event_name"] == "t1_arm_1") &
        (df_main["redcap_repeat_instrument"]
             .fillna("")
             .str.strip()
             .str.lower()
         == MRI_INSTRUMENT)
    ].copy()

    # Date normalization + minimum date
    mri_base["mri_date"] = (
        pd.to_datetime(mri_base["mri_date"], errors="coerce")
        .dt.normalize()
    )
    mri_base = mri_base.dropna(subset=["mri_date"])
    mri_base = mri_base.loc[mri_base["mri_date"] >= MIN_T1_DATE]

    today = pd.Timestamp.now().normalize()

    # Status logic (same as original pipeline)
    mri_base["status"] = np.select(
        [
            mri_base["mri_ses_outcome"].isin([1, 3]),                    # Completed
            (mri_base["mri_ses_outcome"] == 2) | (mri_base["mri_date"] > today),  # Rescheduled or future
        ],
        ["Completed", "Scheduled"],
        default="Unknown"
    )

    # Collapse partial → completed
    mri_base["status_collapsed"] = mri_base["status"].replace(
        {"Partial": "Completed"}
    )

    # Deduplicate per participant per date
    mri_base = (
        mri_base.sort_values(["record_id", "mri_date"])
                .drop_duplicates(subset=["record_id", "mri_date"])
    )

    # Exclude weird/undefined statuses before timeline logic
    mri_base = mri_base[
        mri_base["status_collapsed"].isin(["Completed", "Scheduled"])
    ].copy()

    return mri_base


def build_timeline(mri_base: DataFrame) -> DataFrame:
    """
    Build cumulative MRI completion timeline by site.

    Output columns:
        - site
        - mri_date
        - cum (cumulative count)
        - status ("Completed" or "Scheduled")
    """

    if mri_base.empty:
        return pd.DataFrame(columns=["site", "mri_date", "cum", "status"])

    parts = []
    for status_label in ["Completed", "Scheduled"]:
        subset = mri_base[mri_base["status_collapsed"] == status_label]
        if subset.empty:
            continue

        tmp = (
            subset.groupby(["site", "mri_date"])["record_id"]
                  .nunique()
                  .groupby("site")
                  .cumsum()
                  .rename("cum")
                  .reset_index()
                  .assign(status=status_label)
        )
        parts.append(tmp)

    if not parts:
        return pd.DataFrame(columns=["site", "mri_date", "cum", "status"])

    timeline = pd.concat(parts, ignore_index=True)
    return timeline
