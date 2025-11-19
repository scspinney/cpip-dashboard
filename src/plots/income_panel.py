# plots/income_panel.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER
from ..configs.color_palettes import MEASURE_COLOURS, MEASURE_ORDER

# -----------------------------------------------------------------------------
# Income label ordering — fixed (match REDCap bracket order)
# -----------------------------------------------------------------------------

INCOME_ORDER = [
    "Under $10k", "$10-19k", "$20-39k", "$40-59k", "$60-79k",
    "$80-99k", "$100-124k", "$125-149k", "$150-199k",
    "$200-299k", "$300-399k", "$400k+",
    "Unknown", "Prefer not to say"
]

# -----------------------------------------------------------------------------
# Main plotting function
# -----------------------------------------------------------------------------

def plot_income_panel(baseline: pd.DataFrame, figdir: Path) -> Path:
    """
    Generates the baseline household-income panel.

    Parameters
    ----------
    baseline : pd.DataFrame
        Must contain:
        • site
        • income_lbl (already mapped from numeric -> human labels)

    figdir : Path
        Directory where the figure will be saved.

    Returns
    -------
    Path : path to saved PNG.
    """

    figdir.mkdir(exist_ok=True, parents=True)
    out_path = figdir / "income_panel.png"

    # ------------------------------------------------------------------
    # Validate availability
    # ------------------------------------------------------------------
    if "income_lbl" not in baseline.columns:
        print("⚠️ Income panel skipped — no 'income_lbl' column found.")
        return out_path

    df = baseline.dropna(subset=["income_lbl"]).copy()
    if df.empty:
        print("⚠️ Income panel skipped — all income values are NA.")
        return out_path

    # ------------------------------------------------------------------
    # Set visualization context
    # ------------------------------------------------------------------
    sns.set(style="whitegrid")
    sns.set_context("talk")

    order_available = [x for x in INCOME_ORDER if x in df["income_lbl"].unique()]

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plt.figure(figsize=(18, 8))
    ax = sns.countplot(
        data=df,
        x="income_lbl",
        order=order_available,
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS
    )

    ax.set_title("Household Income (Baseline)", fontsize=20, weight="bold")
    ax.set_xlabel("Household income bracket")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=35, labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.legend(title="Site", fontsize=12)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"✓ Saved income panel → {out_path}")
    return out_path
