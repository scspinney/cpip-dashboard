"""
Gender Panel Plot
-----------------
Produces the cross-site bar plot of gender identity at baseline.

Inputs:
    baseline : pd.DataFrame with columns:
        - site
        - gender_lbl : already mapped to human-readable strings

Outputs:
    PNG saved to FIGDIR / "gender_panel.png"
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configs
from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER

# ---------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------

def plot_gender_panel(baseline: pd.DataFrame, outdir: Path):
    """
    Generate the 'Gender identity (baseline)' panel.

    Parameters
    ----------
    baseline : pd.DataFrame
        Cleaned baseline dataframe. Must include:
            - site
            - gender_lbl
    outdir : Path
        Directory where output PNG is saved.
    """

    outdir.mkdir(exist_ok=True, parents=True)

    # --- Validate required columns ---
    needed = {"site", "gender_lbl"}
    missing = needed - set(baseline.columns)
    if missing:
        raise ValueError(f"Missing required columns for gender panel: {missing}")

    # Category order (consistent across sites)
    gender_order = [
        "Boy/Man", "Girl/Woman", "Non-binary", "Two-spirit",
        "Another", "Questioning", "Unknown", "Prefer not to answer"
    ]

    # Add explicit NA category if needed
    tmp = baseline.copy()
    tmp["gender_lbl"] = tmp["gender_lbl"].fillna("NA")
    if "NA" not in gender_order:
        gender_order.append("NA")

    # --- Plot ---
    sns.set(style="whitegrid")
    sns.set_context("talk")

    plt.figure(figsize=(14, 7))
    ax = sns.countplot(
        data=tmp,
        x="gender_lbl",
        hue="site",
        hue_order=SITE_ORDER,
        order=gender_order,
        palette=SITE_COLOURS
    )

    ax.set_title("Current gender identity (baseline)")
    ax.set_xlabel("Gender identity")
    ax.set_ylabel("Participant count")
    ax.tick_params(axis="x", rotation=35)

    # Save
    outpath = outdir / "gender_panel.png"
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"✓ Saved gender panel → {outpath}")


# ---------------------------------------------------------------------
# Optional CLI-style execution
# ---------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pickle

    parser = argparse.ArgumentParser(description="Generate gender panel plot.")
    parser.add_argument("--baseline", type=Path, required=True,
                        help="Pickled baseline DataFrame")
    parser.add_argument("--outdir", type=Path, default=Path("figs"))
    args = parser.parse_args()

    baseline = pd.read_pickle(args.baseline)
    plot_gender_panel(baseline, args.outdir)
