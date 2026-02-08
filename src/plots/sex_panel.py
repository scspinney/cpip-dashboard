"""
plots/sex_panel.py

Sex-at-birth (effective) panel.

Input requirements:
    baseline: DataFrame containing at least:
        - site
        - record_id
        - sex_lbl   (effective sex label)

Output:
    Saves sex distribution panel → figs/sex_panel.png
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER
from ..configs.color_palettes import MEASURE_COLOURS, MEASURE_ORDER  # not used but consistent
from ..configs.age_bins import AGE_BLOCK_ORDER                       # not used but consistent

FIGDIR = Path("figs")


def plot_sex_panel(baseline: pd.DataFrame, *, outfile: Path | None = None):
    """
    Generate the Sex-at-birth panel as a countplot.

    Parameters
    ----------
    baseline : DataFrame
        Must include:
            - sex_lbl
            - site
    outfile : Path or None
        Where the plot will be written. Default: figs/sex_panel.png
    """
    if outfile is None:
        outfile = FIGDIR / "sex_panel.png"

    FIGDIR.mkdir(exist_ok=True)

    # Data check
    if "sex_lbl" not in baseline.columns:
        raise ValueError("baseline DataFrame is missing required column: sex_lbl")

    # Ensure ordering: same as original dashboard
    order_sex = [
        "Male",
        "Female",
        "Intersex / Non-binary",
        "Unknown",
        "Prefer not to answer",
    ]

    # Plot
    sns.set(style="whitegrid")
    sns.set_context("talk")
    plt.rcParams.update({
        "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 12,
        "axes.titleweight": "bold", "axes.labelweight": "bold",
    })

    plt.figure(figsize=(12, 7))
    sns.countplot(
        data=baseline,
        x="sex_lbl",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=order_sex,
    )

    plt.title("Sex assigned at birth (effective: youth ≥11 else parent)")
    plt.xlabel("Sex at birth")
    plt.ylabel("Participant count")
    plt.xticks(rotation=25)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"✓ Saved sex panel → {outfile}")
