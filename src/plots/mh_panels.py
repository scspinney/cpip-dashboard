# plots/mh_panel.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List

from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER

FIGDIR = Path("figs")


# ---------------------------------------------------------------------
# CFQ variable definitions (copy from existing logic)
# ---------------------------------------------------------------------
CFQ_VARS = [
    "cfq_diag_asd_part","cfq_diag_id_part","cfq_adhd_part","cfq_fasd_part",
    "cfq_ld_part","cfq_lcd_part","cfq_md_part","cfq_diag_other_part",
    "cfq_ment_ad_part","cfq_ment_dd_part","cfq_ment_bd_part","cfq_ment_ocd_part",
    "cfq_ment_ts_part","cfq_ment_psyep_part","cfq_ment_schizo_part",
    "cfq_ment_sa_part","cfq_ment_epilepsy_part","cfq_ment_other_part"
]

CFQ_LABELS = {
    "cfq_diag_asd_part": "Autism Spectrum Disorder",
    "cfq_diag_id_part":  "Cognitive Impair/IDD",
    "cfq_adhd_part":      "ADHD",
    "cfq_fasd_part":      "FASD",
    "cfq_ld_part":        "Learning Disability",
    "cfq_lcd_part":       "Language Disorder",
    "cfq_md_part":        "Motor Disorder",
    "cfq_diag_other_part":"Other Neurodev.",
    "cfq_ment_ad_part":   "Anxiety Disorder",
    "cfq_ment_dd_part":   "Depressive Disorder",
    "cfq_ment_bd_part":   "Bipolar Disorder",
    "cfq_ment_ocd_part":  "OCD",
    "cfq_ment_ts_part":   "Tourette’s",
    "cfq_ment_psyep_part":"Psychosis",
    "cfq_ment_schizo_part":"Schizophrenia",
    "cfq_ment_sa_part":   "Substance Abuse",
    "cfq_ment_epilepsy_part":"Epilepsy",
    "cfq_ment_other_part":"Other (MH)"
}


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def plot_mh_panels(baseline: pd.DataFrame) -> None:
    """
    Build all Mental Health diagnostic plots:
    - Barplot: counts of diagnoses by site
    - Heatmap (counts)
    - Heatmap with explicit 'No diagnosis'
    - Percentage heatmap
    - Co-occurrence heatmap + correlation
    """

    FIGDIR.mkdir(exist_ok=True)

    # Only keep CFQ variables that exist in baseline
    avail = [c for c in CFQ_VARS if c in baseline.columns]
    if not avail:
        print("⚠ No CFQ mental health variables found — skipping MH panel.")
        return

    # Identify participants with *any* CFQ data
    cfq_any = baseline[avail].notna().any(axis=1)
    total_yes = baseline[avail].fillna(0).sum(axis=1)

    # Long form table of “diagnosis present”
    mh_long = (
        baseline.loc[cfq_any, ["record_id","site"] + avail]
        .melt(id_vars=["record_id","site"], value_name="val")
        .query("val == 1")
        .assign(Diagnosis=lambda d: d["variable"].map(CFQ_LABELS))
    )

    # Counts per site × diagnosis
    mh_counts = (
        mh_long.groupby(["site","Diagnosis"], observed=False)["record_id"]
        .nunique().reset_index(name="Count")
    )

    # Participants with no diagnosis
    no_dx_mask = cfq_any & (total_yes == 0)
    no_dx_counts = (
        baseline.loc[no_dx_mask]
        .groupby("site")["record_id"]
        .nunique()
        .reindex(SITE_ORDER, fill_value=0)
    )

    # Total # of participants with any CFQ data
    cfq_site_totals = (
        baseline.loc[cfq_any]
        .groupby("site")["record_id"]
        .nunique()
        .reindex(SITE_ORDER, fill_value=0)
    )

    # -------------------------- BARPLOT ------------------------------
    if not mh_counts.empty:
        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=mh_counts,
            x="Diagnosis",
            y="Count",
            hue="site",
            hue_order=SITE_ORDER,
            palette=SITE_COLOURS
        )
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Number of participants")
        plt.title("Child mental health & neurodevelopmental diagnoses", fontsize=16, pad=28)
        plt.tight_layout()
        plt.savefig(FIGDIR / "mh_counts.png", dpi=300)
        plt.savefig(FIGDIR / "dashboard_mh.png", dpi=300)
        plt.close()

    # -------------------------- HEATMAP (counts) ---------------------
    heat = mh_counts.pivot(index="Diagnosis", columns="site", values="Count").fillna(0)

    plt.figure(figsize=(10, 10))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="Reds", cbar_kws={'label':'Count'})
    plt.title("Counts of diagnoses by site", fontsize=16)
    plt.tight_layout()
    plt.savefig(FIGDIR / "mh_heatmap_counts.png", dpi=300)
    plt.savefig(FIGDIR / "dashboard_mh_heatmap.png", dpi=300)
    plt.close()

    # ----------------- HEATMAP with NO DIAGNOSIS row -----------------
    no_row = pd.DataFrame(
        [no_dx_counts.reindex(heat.columns, fill_value=0)], index=["No diagnosis"]
    )
    heat_all = pd.concat([heat, no_row], axis=0)

    # Two masks: highlight "No diagnosis" separately
    row_is_no_dx = (heat_all.index.to_numpy() == "No diagnosis")
    mask_main = np.tile(row_is_no_dx[:, None], (1, heat_all.shape[1]))
    mask_nodx = ~mask_main

    # Use 2-colormap overlay trick
    from matplotlib.colors import ListedColormap
    fixed_nodx_cmap = ListedColormap(["#2b0000"])

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(
        heat_all,
        mask=mask_main,
        annot=True, fmt=".0f",
        cmap="Reds",
        cbar=True,
        cbar_kws={'label':'Number of participants'},
        linewidths=.5,
        linecolor='gray',
        ax=ax
    )
    sns.heatmap(
        heat_all,
        mask=mask_nodx,
        annot=True, fmt=".0f",
        cmap=fixed_nodx_cmap,
        cbar=False,
        linewidths=.5,
        linecolor='gray',
        ax=ax
    )
    ax.set_title("Counts of diagnoses by site", fontsize=16, pad=22)
    ax.text(
        0.5,
        1.025,
        "Participants can have multiple diagnoses",
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=11,
        color="black",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / "mh_heatmap_counts_with_no_dx.png", dpi=300)
    fig.savefig(FIGDIR / "dashboard_mh_heatmap_update.png", dpi=300)
    plt.close(fig)

    # -------------------------- PERCENT HEATMAP ----------------------
    denom = cfq_site_totals.replace(0, np.nan)
    heat_pct = heat.div(denom, axis=1) * 100

    no_dx_pct = (no_dx_counts / denom) * 100
    heat_pct_all = pd.concat([heat_pct, pd.DataFrame([no_dx_pct], index=["No diagnosis"])], axis=0)

    # Same row-order trick
    row_is_nodx = (heat_pct_all.index == "No diagnosis")
    mask_no2 = np.tile(~row_is_nodx[:, None], (1, heat_pct_all.shape[1]))
    mask_rest2 = ~mask_no2

    max_color = plt.cm.Blues(1.0)
    nodx_cmap2 = ListedColormap([max_color])

    fig, ax = plt.subplots(figsize=(10, 11))
    sns.heatmap(
        heat_pct_all,
        mask=mask_rest2,
        annot=True, fmt=".1f",
        cmap="Blues",
        cbar=True,
        cbar_kws={'label':'% of participants'},
        ax=ax
    )
    sns.heatmap(
        heat_pct_all,
        mask=mask_no2,
        annot=True, fmt=".1f",
        cmap=nodx_cmap2,
        cbar=False,
        linewidths=.5,
        linecolor='gray',
        ax=ax
    )
    ax.set_title("Diagnoses by site (% of participants)", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIGDIR / "mh_heatmap_percent.png", dpi=300)
    fig.savefig(FIGDIR / "dashboard_mh_heatmap_percent.png", dpi=300)
    plt.close()

    # ------------------ CO-OCCURRENCE & CORRELATION ------------------
    cfq_bin = baseline[avail].fillna(0).astype(int)
    co_matrix = cfq_bin.T.dot(cfq_bin)
    co_matrix_labeled = co_matrix.rename(index=CFQ_LABELS, columns=CFQ_LABELS)

    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix_labeled, annot=True, fmt=".0f", cmap="Reds",
                cbar_kws={'label':'Number of participants'})
    plt.title("Co-occurrence of diagnoses (counts)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "mh_cooccurrence_counts.png", dpi=300)
    plt.savefig(FIGDIR / "dashboard_mh_cooccurrence.png", dpi=300)
    plt.close()

    corr_matrix = cfq_bin.corr(method="pearson")

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, cbar_kws={'label':'Phi correlation'})
    plt.title("Correlation between diagnoses")
    plt.tight_layout()
    plt.savefig(FIGDIR / "mh_corr.png", dpi=300)
    plt.savefig(FIGDIR / "dashboard_mh_corr.png", dpi=300)
    plt.close()

    print("✓ Mental health diagnostic plots saved.")


# ---------------------------------------------------------------------
# Optional usage example
# ---------------------------------------------------------------------
if __name__ == "__main__":
    raise SystemExit("Import this module and call plot_mh_panels(baseline)")
