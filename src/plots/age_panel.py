# src/plots/age_panel.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..demographics.age import to_age_block
from ..configs.age_bins import AGE_BLOCK_ORDER
from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER


def compute_age_block_counts(baseline: pd.DataFrame) -> pd.DataFrame:
    """
    Compute participant counts per site × age_block.

    baseline must already contain:
        - age_years
        - site
        - record_id
    """
    df = baseline.copy()

    # Assign developmental age block
    df["age_block"] = df["age_years"].apply(to_age_block)

    # Remove missing blocks
    df = df.dropna(subset=["age_block"])

    # Count unique participants by site × block
    out = (
        df.groupby(["site", "age_block"], observed=False)["record_id"]
            .nunique()
            .reset_index(name="Count")
    )

    # Ordered categorical for age blocks
    out["age_block"] = pd.Categorical(
        out["age_block"],
        categories=AGE_BLOCK_ORDER,
        ordered=True
    )

    return out


def plot_age_panel(
    baseline: pd.DataFrame,
    ax: plt.Axes,
    site_order=SITE_ORDER,
    site_colours=SITE_COLOURS,
):
    """
    Plot the age panel **onto an existing matplotlib axis**.
    This matches the usage in src/pipeline.py.

    Parameters
    ----------
    baseline : DataFrame
        Participant baseline table.
    ax : matplotlib axis
        Axis to draw the panel.
    site_order : list
        Order of sites for plotting.
    site_colours : dict
        Mapping of site → colour.
    """

    age_counts = compute_age_block_counts(baseline)

    sns.barplot(
        data=age_counts,
        x="age_block",
        y="Count",
        hue="site",
        ax=ax,
        hue_order=site_order,
        palette=site_colours,
    )

    ax.set_title("Age groups by site (baseline)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Age group")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=0)


def build_and_plot_age_panel(baseline: pd.DataFrame, ax: plt.Axes):
    """
    Adapter for consistency with older code.
    """
    plot_age_panel(baseline, ax)
