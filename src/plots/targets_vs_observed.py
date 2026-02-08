"""
Observed vs Target recruitment comparison panels.

Produces:
- observed vs target Age × Sex
- observed vs target Age × Ethnicity
- observed vs target Ethnicity totals

Inputs are passed explicitly from the orchestrator script.
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ..configs.age_bins import (
    AGE_BLOCK_ORDER,
    AGE_BIN_ORDER,
    bin_label_to_block,
    to_age_block,
)
from ..configs.color_palettes import (
    SITE_COLOURS,
    SITE_ORDER,
    MEASURE_ORDER,
    MEASURE_COLOURS,
)
from ..configs.target_totals import TARGET_TOTALS, ETHNICITY_ORDER_TARGET


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------

def ensure_figdir(figdir: Path):
    figdir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Panel A — Observed vs Target: Age × Sex
# ---------------------------------------------------------

def plot_age_sex(
    baseline: pd.DataFrame,
    targets_age_group: pd.DataFrame,
    figdir: Path,
):
    ensure_figdir(figdir)

    # Observed
    obs = (
        baseline.dropna(subset=["sex_lbl", "age_years"])
                .assign(age_block=lambda d: d["age_years"].apply(to_age_block))
                .dropna(subset=["age_block"])
                .query("sex_lbl in ['Male','Female']")
                .groupby(["site", "sex_lbl", "age_block"])["record_id"]
                .nunique()
                .reset_index(name="Observed_N")
    )

    if targets_age_group.empty or obs.empty:
        print("ℹ️  Skipping Age×Sex overlay — missing data.")
        return

    # Target
    targ = targets_age_group[targets_age_group["Group"].isin(["Male", "Female"])].copy()
    targ["age_block"] = targ["age_bin_label"].map(bin_label_to_block)
    targ = (
        targ.dropna(subset=["age_block"])
            .groupby(["site", "Group", "age_block"])["Target_N"]
            .sum()
            .reset_index()
            .rename(columns={"Group": "sex_lbl"})
    )

    # Long format
    obs_long = obs.rename(columns={"Observed_N": "N"}).assign(Measure="Observed")
    tgt_long = targ.rename(columns={"Target_N": "N"}).assign(Measure="Target")
    long = pd.concat([tgt_long, obs_long], ignore_index=True)

    long["age_block"] = pd.Categorical(long["age_block"], categories=AGE_BLOCK_ORDER, ordered=True)

    g = sns.catplot(
        data=long,
        kind="bar",
        x="age_block",
        y="N",
        hue="Measure",
        hue_order=MEASURE_ORDER,
        palette=MEASURE_COLOURS,
        col="site",
        row="sex_lbl",
        height=2.8,
        aspect=1.7,
        sharey=True,
        sharex=False,
        legend_out=True,
    )

    g.set_axis_labels("Age group", "Participants")
    g.set_titles("{row_name} — {col_name}")
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="x", labelrotation=0)
    g.fig.subplots_adjust(top=0.88, hspace=0.35)
    g.fig.suptitle("Observed vs Target by Age × Sex, per site", fontsize=16)

    out = figdir / "target_vs_observed_age_by_sex.png"
    g.savefig(out, dpi=300)
    plt.close(g.fig)
    print("✓ Saved:", out)


# ---------------------------------------------------------
# Panel B — Observed vs Target: Age × Ethnicity
# ---------------------------------------------------------

def plot_age_ethnicity(
    obs_eth_age_blocks: pd.DataFrame,
    targets_age_group: pd.DataFrame,
    figdir: Path,
):
    ensure_figdir(figdir)

    if targets_age_group.empty or obs_eth_age_blocks.empty:
        print("ℹ️  Skipping Age×Ethnicity overlay — missing data.")
        return

    targ_eth = targets_age_group[
        targets_age_group["Group"].isin(ETHNICITY_ORDER_TARGET)
    ].copy()

    targ_eth["age_block"] = targ_eth["age_bin_label"].map(bin_label_to_block)
    targ_eth = (
        targ_eth.dropna(subset=["age_block"])
                .groupby(["site", "Group", "age_block"])["Target_N"]
                .sum()
                .reset_index()
                .rename(columns={"Group": "Ethnicity"})
    )

    obs_long = (
        obs_eth_age_blocks.rename(columns={"Observed_N": "N"})
                          .assign(Measure="Observed")
    )
    tgt_long = targ_eth.rename(columns={"Target_N": "N"}).assign(Measure="Target")

    df = pd.concat([tgt_long, obs_long], ignore_index=True)

    df["Ethnicity"] = pd.Categorical(df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True)
    df["age_block"] = pd.Categorical(df["age_block"], categories=AGE_BLOCK_ORDER, ordered=True)
    df = df.sort_values(["Ethnicity", "site", "age_block", "Measure"])

    plt.rcParams.update({
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.titleweight": "bold",
        "axes.labelweight": "bold",
    })

    g = sns.catplot(
        data=df,
        kind="bar",
        x="age_block",
        y="N",
        hue="Measure",
        hue_order=MEASURE_ORDER,
        palette=MEASURE_COLOURS,
        col="site",
        row="Ethnicity",
        height=2.1,
        aspect=1.9,
        sharey="row",
        sharex=False,
        legend_out=True,
    )

    g.set_axis_labels("Age group", "Participants")
    g.set_titles("{row_name} — {col_name}")
    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="x", labelrotation=0)
    g.fig.subplots_adjust(top=0.92, hspace=0.45)
    g.fig.suptitle("Observed vs Target by Age × Ethnicity, per site", fontsize=16)

    out = figdir / "target_vs_observed_age_by_ethnicity.png"
    g.savefig(out, dpi=300)
    plt.close(g.fig)
    print("✓ Saved:", out)


# ---------------------------------------------------------
# Panel C — Observed vs Target: Ethnicity totals
# ---------------------------------------------------------

def plot_ethnicity_totals(
    eth_long: pd.DataFrame,
    baseline: pd.DataFrame,
    figdir: Path,
):
    ensure_figdir(figdir)

    if eth_long.empty:
        print("ℹ️  No ethnicity data; skipping totals comparison.")
        return

    # Collapse ethnicity
    ASIAN = {"East Asian", "South Asian", "Southeast Asian"}
    INDIG = {"First Nations", "Inuk/Inuit", "Métis"}

    def collapse(lbl):
        if lbl in ASIAN:
            return "Asian"
        if lbl in INDIG:
            return "Indigenous"
        if lbl == "Middle Eastern":
            return "Middle East"
        if lbl in {"Black", "Middle East", "Latin American", "Other", "White"}:
            return lbl
        return None

    df = eth_long.copy()
    df["Ethnicity"] = df["label"].map(collapse)
    df = df.dropna(subset=["Ethnicity"])

    # Observed totals
    observed = (
        df.groupby(["site", "Ethnicity"])["record_id"]
          .nunique()
          .reset_index(name="Observed_N")
    )

    # Expected (Target totals)
    rows = []
    for site, payload in TARGET_TOTALS.items():
        for eth, n in payload["Ethnicity"].items():
            rows.append({"site": site, "Ethnicity": eth, "Target_N": int(n)})
    target = pd.DataFrame(rows)

    # Merge
    cmp = (
        target.merge(observed, on=["site", "Ethnicity"], how="left")
              .fillna({"Observed_N": 0})
    )

    cmp["Ethnicity"] = pd.Categorical(
        cmp["Ethnicity"],
        categories=ETHNICITY_ORDER_TARGET,
        ordered=True
    )

    long = pd.concat([
        cmp.rename(columns={"Target_N": "N"}).assign(Measure="Target")[["site", "Ethnicity", "N", "Measure"]],
        cmp.rename(columns={"Observed_N": "N"}).assign(Measure="Observed")[["site", "Ethnicity", "N", "Measure"]],
    ], ignore_index=True)

    g = sns.catplot(
        data=long,
        kind="bar",
        x="Ethnicity",
        y="N",
        hue="Measure",
        hue_order=MEASURE_ORDER,
        palette=MEASURE_COLOURS,
        col="site",
        order=ETHNICITY_ORDER_TARGET,
        height=4.5,
        aspect=1.25,
        sharey=True,
    )

    g.set_xticklabels(rotation=35)
    g.set_axis_labels("Ethnicity (collapsed)", "Participants")
    g.fig.subplots_adjust(top=0.86)
    g.fig.suptitle("Observed vs Target recruitment by Ethnicity and Site", fontsize=16)

    out = figdir / "ethnicity_observed_vs_target.png"
    g.savefig(out, dpi=300)
    plt.close(g.fig)
    print("✓ Saved:", out)


# ---------------------------------------------------------
# Master entry point
# ---------------------------------------------------------

def produce_all_targets_vs_observed(
    baseline: pd.DataFrame,
    eth_long: pd.DataFrame,
    obs_eth_age_blocks: pd.DataFrame,
    targets_age_group: pd.DataFrame,
    figdir: Path,
):
    """Generate all observed vs target plots."""
    plot_age_sex(baseline, targets_age_group, figdir)
    plot_age_ethnicity(obs_eth_age_blocks, targets_age_group, figdir)
    plot_ethnicity_totals(eth_long, baseline, figdir)
