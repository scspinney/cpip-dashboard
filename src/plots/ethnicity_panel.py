"""
Ethnicity panel plots for the C-PIP dashboard.

Inputs:
    baseline  – one-row-per-participant T1 baseline DataFrame
    eth_long  – long-form ethnicity labels DataFrame:
                columns ["record_id","site","variable","val","label"]
    eth_wide  – collapsed White / Non-white DataFrame:
                ["site","record_id","White","Non-white"]

Outputs:
    figs/ethnicity_full.png
    figs/ethnicity_white_nonwhite.png
    figs/ethnicity_observed_vs_target.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Config imports ─────────────────────────────────────────
from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER, MEASURE_ORDER, MEASURE_COLOURS
from ..configs.target_totals import TARGET_TOTALS, ETHNICITY_ORDER_TARGET

FIGDIR = Path("figs")
FIGDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------

def _collapse_for_totals(lbl: str):
    """Collapse raw ethnicity labels into the target bins."""
    ASIAN = {"East Asian","South Asian","Southeast Asian"}
    INDIG = {"First Nations","Inuk/Inuit","Métis"}

    if lbl in ASIAN: return "Asian"
    if lbl in INDIG: return "Indigenous"
    if lbl == "Middle Eastern": return "Middle East"
    if lbl in {"Black","Middle East","Latin American","Other","White"}:
        return lbl
    return None


# ------------------------------------------------------------
# 1. Full Ethnicity Panel (all labels as entered)
# ------------------------------------------------------------

def plot_full_ethnicity(eth_long: pd.DataFrame):
    """Plot raw full-label ethnicity counts."""
    out = FIGDIR / "ethnicity_full.png"

    if eth_long.empty:
        print("ℹ️ Ethnicity N/A → skipping full ethnicity panel")
        return

    order_eth_full = eth_long["label"].value_counts().index.tolist()

    plt.figure(figsize=(20, 8))
    sns.countplot(
        data=eth_long,
        x="label",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=order_eth_full,
    )
    plt.xticks(rotation=35, ha="right")
    plt.title("Ethnicity (baseline)")
    plt.ylabel("Participant count")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✓ Saved → {out}")


# ------------------------------------------------------------
# 2. White vs Non-white Panel
# ------------------------------------------------------------

def plot_white_nonwhite(eth_wide: pd.DataFrame):
    """Plot White vs Non-white per site."""
    out = FIGDIR / "ethnicity_white_nonwhite.png"

    if eth_wide.empty:
        print("ℹ️ No White/Non-white data → skipping ethnicity collapse panel")
        return

    eth_long2 = (
        eth_wide
        .melt(id_vars=["site","record_id"], value_vars=["White","Non-white"],
              var_name="Ethnicity", value_name="Label")
        .dropna(subset=["Label"])
    )

    eth_counts = (
        eth_long2
        .groupby(["site","Ethnicity"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=eth_counts,
        x="Ethnicity",
        y="Count",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=["White", "Non-white"],
    )
    plt.title("Ethnicity (White / Non-white) by site")
    plt.xlabel("Ethnicity group")
    plt.ylabel("Participant count")
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"✓ Saved → {out}")


# ------------------------------------------------------------
# 3. Observed vs Target totals by ethnicity
# ------------------------------------------------------------

def plot_observed_vs_target_totals(eth_long: pd.DataFrame):
    """Compare observed vs target totals (collapsed ethnicity)."""
    out = FIGDIR / "ethnicity_observed_vs_target.png"

    if eth_long.empty:
        print("ℹ️ No observed ethnicity → skipping observed vs target totals")
        return

    # ---- Build observed collapsed ethnicity ----
    by_rec = eth_long.copy()
    by_rec["Ethnicity"] = by_rec["label"].map(_collapse_for_totals)
    by_rec = by_rec.dropna(subset=["Ethnicity"])

    # White-only group
    AS_WHITE = (
        by_rec.groupby(["site","record_id"])["Ethnicity"]
              .apply(lambda s: set(s) == {"White"})
              .reset_index(name="is_white")
    )

    white_recs = AS_WHITE[AS_WHITE["is_white"]]["record_id"]

    white_rows = (
        by_rec[by_rec["record_id"].isin(white_recs)]
        [["site","record_id"]]
        .drop_duplicates()
        .assign(Ethnicity="White")
    )

    # Nonwhite rows — remove White
    nonwhite_rows = (
        by_rec[by_rec["Ethnicity"] != "White"]
        [["site","record_id","Ethnicity"]]
        .drop_duplicates()
    )

    obs_all = pd.concat([white_rows, nonwhite_rows], ignore_index=True)

    obs_counts = (
        obs_all.groupby(["site","Ethnicity"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Observed_N")
    )

    # ---- Build target totals ----
    tgt_rows = []
    for site, payload in TARGET_TOTALS.items():
        for eth, n in payload["Ethnicity"].items():
            tgt_rows.append({"site":site, "Ethnicity":eth, "Target_N":int(n)})

    target_df = pd.DataFrame(tgt_rows)

    compare = (
        target_df.merge(obs_counts, on=["site","Ethnicity"], how="left")
                 .fillna({"Observed_N":0})
                 .astype({"Observed_N":int})
    )

    compare["Ethnicity"] = pd.Categorical(
        compare["Ethnicity"],
        categories=ETHNICITY_ORDER_TARGET,
        ordered=True
    )
    compare = compare.sort_values(["site","Ethnicity"])

    long = pd.concat([
        compare.rename(columns={"Target_N":"N"})
               [["site","Ethnicity","N"]].assign(Measure="Target"),
        compare.rename(columns={"Observed_N":"N"})
               [["site","Ethnicity","N"]].assign(Measure="Observed")
    ], ignore_index=True)

    # ---- Plot ----
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
        height=5,
        aspect=1.3,
        sharey=True
    )

    g.set_xticklabels(rotation=35)
    g.set_axis_labels("Ethnicity (target bins)", "Participants")
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle("Observed vs Target recruitment by Ethnicity and Site", fontsize=16)

    g.savefig(out, dpi=300)
    plt.close(g.fig)
    print(f"✓ Saved → {out}")


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def build_ethnicity_panels(baseline: pd.DataFrame, eth_long: pd.DataFrame, eth_wide: pd.DataFrame):
    """
    Main entrypoint used by the pipeline.
    Produces all ethnicity-related figures.
    """
    print("\n=== Building Ethnicity Panels ===")
    plot_full_ethnicity(eth_long)
    plot_white_nonwhite(eth_wide)
    plot_observed_vs_target_totals(eth_long)
    print("✓ Ethnicity panels complete.\n")
