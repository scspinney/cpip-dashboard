# plots/mri_panels.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from ..configs.color_palettes import SITE_COLOURS, SITE_ORDER
from ..configs.age_bins import AGE_BLOCK_ORDER, to_age_block
from ..mri.timeline import build_mri_baseline, build_timeline


def add_mri_timeline_panel(
    ax: plt.Axes,
    df_main: pd.DataFrame,
    *,
    status_filter: str = "Completed",
) -> None:
    """
    Add the MRI cumulative timeline panel to an existing Axes.

    Parameters
    ----------
    ax : plt.Axes
        Matplotlib Axes to draw into (e.g., one cell of the 4×2 dashboard).
    df_main : pd.DataFrame
        Full REDCap dataframe (all rows). We will derive MRI baseline rows
        using the central logic in mri.timeline.build_mri_baseline().
    status_filter : {"Completed", "Scheduled", "all"}, default "Completed"
        Which status to display:
        - "Completed" → plot only completed sessions (1 or 3)
        - "Scheduled" → plot only scheduled sessions (2 or future-dated)
        - "all"       → plot all statuses combined

    Notes
    -----
    - If no valid MRI rows are found, the Axes is annotated with "MRI timeline N/A".
    """
    # Derive MRI baseline rows and timeline from the canonical helpers
    mri_base = build_mri_baseline(df_main)
    if mri_base.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "MRI timeline\nN/A",
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        return

    timeline = build_timeline(mri_base)
    if timeline.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "MRI timeline\nN/A",
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        return

    # Optionally filter by status
    if status_filter.lower() != "all":
        timeline = timeline.query("status == @status_filter")
        if timeline.empty:
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"MRI timeline\n({status_filter} only)\nN/A",
                ha="center",
                va="center",
                fontsize=12,
                color="grey",
                transform=ax.transAxes,
            )
            return

    # Safety: ensure site ordering & sorting
    timeline = timeline.copy()
    if "site" in timeline.columns:
        timeline["site"] = pd.Categorical(
            timeline["site"], categories=SITE_ORDER, ordered=True
        )

    sns.lineplot(
        data=timeline,
        x="mri_date",
        y="cum",
        hue="site",
        hue_order=SITE_ORDER,
        style="status",
        palette=SITE_COLOURS,
        dashes={"Completed": "", "Scheduled": [2, 2]},
        ax=ax,
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.set_ylabel("Cumulative T1 MRIs")
    ax.set_xlabel("Date")

    if status_filter.lower() == "all":
        title = "Baseline MRI visits by site (Completed + Scheduled)"
    else:
        title = f"Baseline MRI visits by site ({status_filter})"

    ax.set_title(title)

    # Tidy legend
    leg = ax.legend(title="Site / Status", loc="upper left")
    if leg is not None:
        leg.set_frame_on(True)


def save_mri_by_age_figure(
    df_main: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    figdir: Path,
    filename: str = "dashboard_mri_by_age_fixed.png",
) -> Path:
    """
    Build the vertical MRI-by-age-by-site facet plot and save to disk.

    This reproduces the monolithic script's final MRI-by-age figure:
    - rows = sites (Calgary, Montreal, Toronto)
    - x     = age_block (0–2, 2–5, …, 16–18)
    - hue   = status_collapsed (Completed vs Scheduled)
    - y     = participant count

    Parameters
    ----------
    df_main : pd.DataFrame
        Full REDCap dataframe (all rows). Used to derive MRI baseline rows.
    baseline : pd.DataFrame
        Baseline (one row per participant) with at least:
        - record_id
        - site
        - age_years
    figdir : Path
        Directory where the figure will be saved.
    filename : str, default "dashboard_mri_by_age_fixed.png"
        Output filename.

    Returns
    -------
    Path
        Full path of the saved PNG.
    """
    figdir = Path(figdir)
    figdir.mkdir(exist_ok=True, parents=True)

    # Build MRI baseline rows with canonical logic
    mri_base = build_mri_baseline(df_main)
    if mri_base.empty:
        out = figdir / filename
        # Nothing to plot; create a small "blank" figure so downstream scripts
        # don't explode if they expect a file to exist.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "MRI-by-age\nN/A (no MRI data)",
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # Merge MRI rows with age + site from baseline
    required_cols = {"record_id", "site", "age_years"}
    missing = required_cols - set(baseline.columns)
    if missing:
        raise ValueError(
            f"baseline is missing required columns for MRI age plot: {sorted(missing)}"
        )

    mri_age = mri_base.merge(
        baseline[["record_id", "site", "age_years"]],
        on=["record_id", "site"],
        how="left",
    )

    # Age blocks
    mri_age["age_block"] = mri_age["age_years"].apply(to_age_block)

    # Collapse to counts per site × status × age_block
    mri_age_counts = (
        mri_age.dropna(subset=["age_block"])
        .groupby(["site", "status_collapsed", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )

    if mri_age_counts.empty:
        out = figdir / filename
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "MRI-by-age\nN/A (no valid age blocks)",
            ha="center",
            va="center",
            fontsize=12,
            color="grey",
            transform=ax.transAxes,
        )
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    # Consistent ordering
    mri_age_counts["site"] = pd.Categorical(
        mri_age_counts["site"], categories=SITE_ORDER, ordered=True
    )
    mri_age_counts["age_block"] = pd.Categorical(
        mri_age_counts["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )

    # Default palette: Completed = blue, Scheduled = grey, Unknown = light grey
    status_palette: Dict[str, Any] = {
        "Completed": "#4c78a8",
        "Scheduled": "#999999",
        "Unknown": "#cccccc",
    }

    sns.set(style="whitegrid")
    g = sns.catplot(
        data=mri_age_counts,
        kind="bar",
        x="age_block",
        y="Count",
        hue="status_collapsed",
        row="site",
        row_order=SITE_ORDER,
        palette=status_palette,
        order=AGE_BLOCK_ORDER,
        height=3.5,
        aspect=2.2,
        sharex=True,
        sharey=True,
        legend=True,
    )

    # Axes cosmetics
    for ax in g.axes.flatten():
        if ax is None:
            continue
        ax.set_xlabel("Age group")
        ax.tick_params(axis="x", labelbottom=True)
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(0)

    g.set_axis_labels("Age group", "Participants")
    g.set_titles("{row_name}")

    # Central legend (on the middle axis if we can find one)
    handles, labels = None, None
    if getattr(g, "_legend", None) is not None:
        handles = g._legend.legend_handles
        labels = [t.get_text() for t in g._legend.texts]
        g._legend.remove()

    if not handles or not labels:
        # Fallback: try to grab from a middle axis
        nrows = g.axes.shape[0]
        mid_ax = g.axes[nrows // 2, 0]
        handles, labels = mid_ax.get_legend_handles_labels()

    if handles and labels:
        nrows = g.axes.shape[0]
        mid_ax = g.axes[nrows // 2, 0]
        mid_ax.legend(
            handles,
            labels,
            title="Status",
            loc="upper center",
            bbox_to_anchor=(1.18, 0.6),
            ncol=1,
            frameon=True,
            borderaxespad=0.6,
        )

    g.fig.subplots_adjust(hspace=0.4, top=0.90, bottom=0.12)
    g.fig.suptitle(
        "MRI sessions completed (1 or 3) vs scheduled by age group and site",
        fontsize=16,
    )

    out_path = figdir / filename
    g.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    return out_path
