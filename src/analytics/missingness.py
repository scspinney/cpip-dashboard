# src/analytics/missingness.py

"""
Compute “effective” missingness across demographic panels.
"""

from __future__ import annotations
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Optional

from ..configs.color_palettes import SITE_ORDER, SITE_COLOURS
from ..util.constants import FIGDIR


# ──────────────────────────────────────────────────────────────
# Internal missingness computation
# ──────────────────────────────────────────────────────────────

def compute_missingness(
    baseline: pd.DataFrame,
    eth_cols: List[str],
    gender_col: Optional[str]
) -> pd.DataFrame:
    """Internal function: compute missingness per site."""
    rows = []
    for site in sorted(baseline["site"].unique()):
        df = baseline[baseline["site"] == site]

        total = len(df)
        age_na = df["age_years"].isna().sum()
        sex_na = df["sex_missing_effective"].sum()

        # gender
        if gender_col and gender_col in df.columns:
            gender_na = df[gender_col].isna().sum()
        else:
            gender_na = 0

        # income
        if "demo_cg_income" in df.columns:
            income_na = df["demo_cg_income"].isna().sum()
        else:
            income_na = 0

        # ethnicity
        if eth_cols:
            eth_mask = (df[eth_cols].sum(axis=1) == 0) | df[eth_cols].isna().all(axis=1)
            eth_na = eth_mask.sum()
        else:
            eth_na = 0

        rows.append({
            "site": site,
            "Total available counts": total,
            "Age NA": age_na,
            "Sex NA": sex_na,
            "Gender NA": gender_na,
            "Income NA": income_na,
            "Ethnicity NA": eth_na,
        })

    return pd.DataFrame(rows).set_index("site")


# ──────────────────────────────────────────────────────────────
# Public API function expected by pipeline.py
# ──────────────────────────────────────────────────────────────

def compute_effective_na_counts(
    baseline: pd.DataFrame,
    eth_cols: List[str],
    site_order: list,
    gender_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Wrapper matching pipeline expectations.
    Automatically orders rows by SITE_ORDER.
    """
    df = compute_missingness(baseline, eth_cols, gender_col)
    return df.reindex(site_order)


# ──────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────

def plot_missingness_panel(
    na_counts: pd.DataFrame,
    ax,
    site_order,
    site_colours,
    *,
    title="Missing counts per panel (Effective missingness)"
):
    """Plot into an existing Axes."""
    if na_counts.empty:
        ax.text(0.5, 0.5, "No missingness data", ha="center", va="center")
        ax.axis("off")
        return

    na_long = (
        na_counts.reset_index()
        .melt(id_vars="site", var_name="Metric", value_name="Count")
    )

    label_map = {
        "Age NA": "Age",
        "Sex NA": "Sex",
        "Gender NA": "Gender",
        "Income NA": "Income",
        "Ethnicity NA": "Ethnicity",
    }
    na_long = na_long[na_long["Metric"].isin(label_map.keys())].copy()
    na_long["Metric"] = na_long["Metric"].map(label_map)

    metric_order = [
        "Age",
        "Sex",
        "Gender",
        "Income",
        "Ethnicity",
    ]

    na_long["Metric"] = pd.Categorical(
        na_long["Metric"], categories=metric_order, ordered=True
    )

    sns.barplot(
        data=na_long,
        x="Metric",
        y="Count",
        hue="site",
        hue_order=site_order,
        palette=site_colours,
        ax=ax,
    )

    ax.set_title(title, fontsize=16, weight="bold")
    ax.set_xlabel("Panel")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Site")
