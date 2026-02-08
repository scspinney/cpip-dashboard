import pandas as pd

from ..configs.age_bins import AGE_BLOCK_ORDER, to_age_block


def build_mri_age_counts(
    mri_base: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build MRI-by-age counts per site and status.

    Parameters
    ----------
    mri_base : pd.DataFrame
        Output of mri.timeline.build_mri_baseline(), expected to contain:
        - record_id
        - site
        - mri_date
        - status_collapsed  (e.g. 'Completed' / 'Scheduled')

    baseline : pd.DataFrame
        Baseline participant table, expected to contain:
        - record_id
        - site
        - age_years  (already derived in demographics.age.add_age_years)

    Returns
    -------
    pd.DataFrame
        Tidy table with one row per site × status_collapsed × age_block:
        - site
        - status_collapsed
        - age_block   (categorical with AGE_BLOCK_ORDER)
        - Count       (number of unique participants)
    """
    # Merge MRI rows with baseline age + site
    mri_age = mri_base.merge(
        baseline[["record_id", "age_years", "site"]],
        on=["record_id", "site"],
        how="left",
    )

    # Map continuous age to developmental blocks
    mri_age["age_block"] = mri_age["age_years"].apply(to_age_block)

    # Drop rows with no valid age block
    mri_age = mri_age.dropna(subset=["age_block"])

    # Aggregate to unique participants per site × status × age_block
    mri_age_counts = (
        mri_age.groupby(["site", "status_collapsed", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )

    # Enforce canonical ordering of age blocks
    mri_age_counts["age_block"] = pd.Categorical(
        mri_age_counts["age_block"],
        categories=AGE_BLOCK_ORDER,
        ordered=True,
    )

    return mri_age_counts
