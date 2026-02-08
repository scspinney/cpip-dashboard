# src/pipeline.py
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

# Allow running as `python src/pipeline.py` by wiring up the package path.
if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    __package__ = "src"

# ──────────────────────────────────────────────────────────────
# CONFIG IMPORTS
# ──────────────────────────────────────────────────────────────
from .configs.color_palettes import (
    SITE_COLOURS,
    SITE_ORDER,
)

# ──────────────────────────────────────────────────────────────
# UTILS
# ──────────────────────────────────────────────────────────────
from .util.constants import FIGDIR
from .util.helpers import (
    add_cover_page,
    add_image_page,
    pick_first,
    find_complete_col,
)

# ──────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────
from .data_loading.load_redcap import discover_site_paths, load_redcap_exports
from .data_loading.merge_income import add_merged_income
from .data_loading.baseline_selector import build_key_vars, best_baseline_rows

# ──────────────────────────────────────────────────────────────
# DEMOGRAPHICS
# ──────────────────────────────────────────────────────────────
from .demographics.age import add_age_years, to_age_block
from .demographics.sex_effective import add_effective_sex
from .demographics.gender import add_gender_label
from .demographics.ethnicity import (
    build_eth_long,
    build_white_nonwhite_panel,
    collapse_eth_for_blocks,
)

# ──────────────────────────────────────────────────────────────
# ANALYTICS
# ──────────────────────────────────────────────────────────────
from .analytics.site_counts import compute_site_counts
from .analytics.missingness import compute_effective_na_counts
from .analytics.targets import load_targets_age_by_group

# ──────────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────────
from .plots.age_panel import plot_age_panel
from .plots.sex_panel import plot_sex_panel
from .plots.gender_panel import plot_gender_panel
from .plots.ethnicity_panel import build_ethnicity_panels
from .plots.income_panel import plot_income_panel, INCOME_ORDER
from .plots.mri_panels import save_mri_by_age_figure
from .plots.targets_vs_observed import plot_age_sex, plot_age_ethnicity
from .plots.mh_panels import plot_mh_panels
from .mri.timeline import build_mri_baseline, build_timeline


def _build_obs_eth_age_counts(
    baseline: pd.DataFrame, eth_long: pd.DataFrame
) -> pd.DataFrame:
    """Create observed counts for age × ethnicity overlays (monolith parity)."""
    if eth_long.empty or "record_id" not in baseline.columns:
        return pd.DataFrame(columns=["site", "Ethnicity", "age_block", "Observed_N"])

    by_rec = (
        eth_long.groupby(["site", "record_id"])["label"]
        .apply(lambda s: sorted(set(s.dropna())))
        .reset_index(name="labels")
    )

    white_ids = set(
        by_rec.loc[
            by_rec["labels"].apply(
                lambda L: ("White" in set(L)) and (len(set(L)) == 1)
            ),
            "record_id",
        ]
    )

    white_only_df = baseline.loc[
        baseline["record_id"].isin(white_ids), ["record_id", "site", "age_years"]
    ].assign(Ethnicity="White", age_block=lambda d: d["age_years"].apply(to_age_block))

    nonwhite = by_rec.loc[
        by_rec["labels"].apply(lambda L: any(l != "White" for l in set(L)))
    ].copy()
    nonwhite = (
        nonwhite.assign(
            EthnicityList=nonwhite["labels"].apply(lambda L: [l for l in L if l != "White"])
        )
        .explode("EthnicityList")
        .rename(columns={"EthnicityList": "Ethnicity"})
        .drop(columns="labels")
    )
    nonwhite["Ethnicity"] = nonwhite["Ethnicity"].map(collapse_eth_for_blocks)
    nonwhite = nonwhite.dropna(subset=["Ethnicity"])
    nonwhite = (
        nonwhite.merge(
            baseline[["record_id", "age_years"]],
            on="record_id",
            how="left",
        ).assign(age_block=lambda d: d["age_years"].apply(to_age_block))
    )

    parts = []
    if not white_only_df.empty:
        parts.append(white_only_df[["site", "Ethnicity", "age_block", "record_id"]])
    if not nonwhite.empty:
        parts.append(nonwhite[["site", "Ethnicity", "age_block", "record_id"]])
    if not parts:
        return pd.DataFrame(columns=["site", "Ethnicity", "age_block", "Observed_N"])

    obs_eth_age_blocks = (
        pd.concat(parts, ignore_index=True)
        .dropna(subset=["age_block"])
        .groupby(["site", "Ethnicity", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Observed_N")
    )
    return obs_eth_age_blocks


SEX_ORDER = [
    "Male",
    "Female",
    "Intersex / Non-binary",
    "Unknown",
    "Prefer not to answer",
]

GENDER_ORDER = [
    "Boy/Man",
    "Girl/Woman",
    "Non-binary",
    "Two-spirit",
    "Another",
    "Questioning",
    "Unknown",
    "Prefer not to answer",
    "NA",
]


def _plot_mri_timeline_axis(ax, timeline: pd.DataFrame):
    if timeline is None or timeline.empty:
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

    data = timeline[timeline["status"].str.lower() == "completed"].copy()
    if data.empty:
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

    data["site"] = pd.Categorical(data["site"], categories=SITE_ORDER, ordered=True)

    sns.lineplot(
        data=data,
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
    ax.set_title("Baseline MRI visits by site (Completed = 1 or 3)")
    leg = ax.legend(title="Site / Status", loc="upper left")
    if leg is not None:
        leg.set_frame_on(True)


def _plot_sex_axis(ax, baseline: pd.DataFrame):
    if "sex_lbl" not in baseline.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "Sex data\nN/A", ha="center", va="center")
        return

    sns.countplot(
        data=baseline,
        x="sex_lbl",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=SEX_ORDER,
        ax=ax,
    )
    ax.set_title("Sex assigned at birth (effective: youth ≥11 else parent)")
    ax.set_xlabel("Sex at birth")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=35)


def _plot_gender_axis(ax, baseline: pd.DataFrame):
    if "gender_lbl" not in baseline.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "Gender data\nN/A", ha="center", va="center")
        return

    df = baseline.copy()
    df["gender_lbl"] = df["gender_lbl"].fillna("NA")

    sns.countplot(
        data=df,
        x="gender_lbl",
        hue="site",
        hue_order=SITE_ORDER,
        order=GENDER_ORDER,
        palette=SITE_COLOURS,
        ax=ax,
    )
    ax.set_title("Current gender identity (baseline)")
    ax.set_xlabel("Gender identity")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=35)


def _plot_ethnicity_full_axis(ax, eth_long: pd.DataFrame):
    if eth_long.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Ethnicity\nN/A", ha="center", va="center")
        return

    order_eth_full = eth_long["label"].value_counts().index.tolist()

    sns.countplot(
        data=eth_long,
        x="label",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=order_eth_full,
        ax=ax,
    )
    ax.set_title("Ethnicity (baseline)")
    ax.set_xlabel("Ethnicity")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=35)


def _plot_ethnicity_white_nonwhite_axis(ax, eth_wide: pd.DataFrame):
    if eth_wide.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Ethnicity\nN/A", ha="center", va="center")
        return

    eth_long2 = (
        eth_wide
        .melt(
            id_vars=["site", "record_id"],
            value_vars=["White", "Non-white"],
            var_name="Ethnicity",
            value_name="Label",
        )
        .dropna(subset=["Label"])
    )

    eth_counts = (
        eth_long2.groupby(["site", "Ethnicity"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )

    sns.barplot(
        data=eth_counts,
        x="Ethnicity",
        y="Count",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=["White", "Non-white"],
        ax=ax,
    )
    ax.set_title("Ethnicity (White / Non-white) by site")
    ax.set_xlabel("Ethnicity group")
    ax.set_ylabel("Participant count")


def _plot_income_axis(ax, baseline: pd.DataFrame):
    if "income_lbl" not in baseline.columns:
        ax.axis("off")
        ax.text(0.5, 0.5, "Income\nN/A", ha="center", va="center")
        return

    df = baseline.dropna(subset=["income_lbl"])
    if df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Income\nN/A", ha="center", va="center")
        return

    order_available = [lbl for lbl in INCOME_ORDER if lbl in df["income_lbl"].unique()]

    sns.countplot(
        data=df,
        x="income_lbl",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        order=order_available,
        ax=ax,
    )
    ax.set_title("Household income (baseline)")
    ax.set_xlabel("Household income bracket")
    ax.set_ylabel("Participants")
    ax.tick_params(axis="x", rotation=35)


def _plot_missingness_axis(ax, missing_df: pd.DataFrame):
    if missing_df is None or missing_df.empty:
        ax.axis("off")
        ax.text(0.5, 0.5, "Missing counts\nN/A", ha="center", va="center")
        return

    na_long = (
        missing_df.reset_index()
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
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        ax=ax,
    )
    ax.set_title("Missing counts per panel")
    ax.set_xlabel("Panel")
    ax.set_ylabel("Missing count")
    ax.tick_params(axis="x", rotation=25)


def build_dashboard_figure(
    baseline: pd.DataFrame,
    eth_long: pd.DataFrame,
    eth_wide: pd.DataFrame,
    missing_df: pd.DataFrame,
    timeline: pd.DataFrame,
):
    sns.set(style="whitegrid")
    sns.set_context("talk")
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 21,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 12,
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
        }
    )

    fig, axes = plt.subplots(4, 2, figsize=(22, 20))
    (ax_cum, ax_age), (ax_gender, ax_sex), (ax_eth, ax_eth_wnw), (ax_inc, ax_na) = axes
    fig.subplots_adjust(wspace=0.35, hspace=0.3)

    _plot_mri_timeline_axis(ax_cum, timeline)
    plot_age_panel(baseline, ax_age, SITE_ORDER, SITE_COLOURS)
    ax_age.set_title("Age groups by site (baseline)")
    ax_age.set_xlabel("Age group")
    ax_age.tick_params(axis="x", rotation=0)

    _plot_gender_axis(ax_gender, baseline)
    _plot_sex_axis(ax_sex, baseline)
    _plot_ethnicity_full_axis(ax_eth, eth_long)
    _plot_ethnicity_white_nonwhite_axis(ax_eth_wnw, eth_wide)
    _plot_income_axis(ax_inc, baseline)
    _plot_missingness_axis(ax_na, missing_df)

    fig.tight_layout()
    FIGDIR.mkdir(exist_ok=True)
    dash_path = FIGDIR / "dashboard.png"
    fig.savefig(dash_path, dpi=300)
    plt.close(fig)
    print(f"✓ Dashboard saved → {dash_path}")


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main(argv=None):
    FIGDIR.mkdir(exist_ok=True)

    # ──────────────────────────────────────────────────────────────
    # 1) LOAD REDCap
    # ──────────────────────────────────────────────────────────────
    paths = discover_site_paths(argv or sys.argv[1:])
    df_main, non_repeat = load_redcap_exports(paths)

    # Demographic columns
    youth_sex_col = "sex" if "sex" in non_repeat.columns else None
    parent_sex_col = "demo_sex" if "demo_sex" in non_repeat.columns else None
    gender_col = pick_first(non_repeat, ["gender", "demo_gender", "gender_oth"])
    gender_label_col = "demo_gender" if "demo_gender" in non_repeat.columns else gender_col
    eth_cols = [c for c in non_repeat.columns if "ethnicity___" in c]

    # Form-completion columns
    YOUTH_COMPLETE_CAND_REGEXES = [
        r"^youth.*demo.*baseline.*_complete$",
        r"^youth.*demo.*_complete$",
        r"^demo.*youth.*_complete$",
    ]
    YOUTH_COMPLETE_FALLBACKS = [
        "youth_demographics_baseline_complete",
        "youth_demographics_complete",
        "youth_demo_complete",
    ]

    PARENT_CHILD_COMPLETE_CAND_REGEXES = [
        r"^caregiver.*demo.*about.*child.*_complete$",
        r"^caregiver.*demo.*child.*_complete$",
        r"^caregiver.*demograph.*_complete$",
        r"^demo.*child.*_complete$",
    ]
    PARENT_CHILD_COMPLETE_FALLBACKS = [
        "caregiver_demographics_baseline_about_child_complete",
        "caregiver_demographics_child_complete",
        "caregiver_demographics_complete",
        "demo_child_complete",
    ]

    youth_form_col = find_complete_col(
        non_repeat, YOUTH_COMPLETE_CAND_REGEXES, YOUTH_COMPLETE_FALLBACKS
    )
    parent_form_col = find_complete_col(
        non_repeat, PARENT_CHILD_COMPLETE_CAND_REGEXES, PARENT_CHILD_COMPLETE_FALLBACKS
    )

    # ──────────────────────────────────────────────────────────────
    # 2) INCOME MERGE
    # ──────────────────────────────────────────────────────────────
    non_repeat = add_merged_income(non_repeat)

    # ──────────────────────────────────────────────────────────────
    # 3) BASELINE (one T1 per participant)
    # ──────────────────────────────────────────────────────────────
    key_vars = build_key_vars(
        youth_sex_col,
        parent_sex_col,
        gender_col,
        eth_cols,
    )
    baseline = best_baseline_rows(non_repeat, key_vars)

    # ──────────────────────────────────────────────────────────────
    # 4) DERIVED DEMOGRAPHICS
    # ──────────────────────────────────────────────────────────────
    baseline = add_age_years(baseline)
    baseline = add_effective_sex(
        baseline,
        youth_sex_col,
        parent_sex_col,
        youth_form_col,
        parent_form_col,
    )
    baseline = add_gender_label(baseline, gender_label_col)

    income_map = {
        1: "Under $10k",
        2: "$10-19k",
        3: "$20-39k",
        4: "$40-59k",
        5: "$60-79k",
        6: "$80-99k",
        7: "$100-124k",
        8: "$125-149k",
        9: "$150-199k",
        10: "$200-299k",
        11: "$300-399k",
        12: "$400k+",
        777: "Unknown",
        888: "Prefer not to say",
    }
    if "demo_cg_income" in baseline.columns:
        baseline["income_lbl"] = baseline["demo_cg_income"].map(income_map)
    else:
        baseline["income_lbl"] = None

    eth_long = build_eth_long(baseline, eth_cols)
    eth_wide = build_white_nonwhite_panel(baseline, eth_long)

    mri_base = build_mri_baseline(df_main)
    timeline = build_timeline(mri_base)

    # ──────────────────────────────────────────────────────────────
    # 5) ANALYTICS
    # ──────────────────────────────────────────────────────────────
    site_counts = compute_site_counts(
        baseline,
        eth_cols=eth_cols,
        gender_col=gender_col,
        income_col="demo_cg_income",
    )
    site_counts = site_counts.reindex(SITE_ORDER, fill_value=0).fillna(0).astype(int)
    site_counts.to_csv(FIGDIR / "dashboard_counts.csv")

    missing_df = compute_effective_na_counts(
        baseline,
        eth_cols,
        SITE_ORDER,
        "gender_lbl", #gender_col,
    )
    missing_df = missing_df.fillna(0).astype(int)
    missing_df.to_csv(FIGDIR / "dashboard_counts_na.csv")
    missing_df.to_csv(FIGDIR / "dashboard_missingness.csv")

    # ──────────────────────────────────────────────────────────────
    # 6) PLOTS & PANELS
    # ──────────────────────────────────────────────────────────────
    build_dashboard_figure(baseline, eth_long, eth_wide, missing_df, timeline)

    # Standalone panel PNGs for flexibility
    plot_sex_panel(baseline)
    plot_gender_panel(baseline, FIGDIR)
    plot_income_panel(baseline, FIGDIR)
    build_ethnicity_panels(baseline, eth_long, eth_wide)
    save_mri_by_age_figure(df_main, baseline, figdir=FIGDIR)
    plot_mh_panels(baseline)

    # ──────────────────────────────────────────────────────────────
    # 7) TARGET OVERLAYS
    # ──────────────────────────────────────────────────────────────
    targets_long = load_targets_age_by_group()
    if targets_long.empty:
        print("ℹ️ Target distributions missing — skipping overlay plots.")
    else:
        plot_age_sex(baseline, targets_long, FIGDIR)
        obs_eth_age = _build_obs_eth_age_counts(baseline, eth_long)
        plot_age_ethnicity(obs_eth_age, targets_long, FIGDIR)

    # ──────────────────────────────────────────────────────────────
    # 8) PDF
    # ──────────────────────────────────────────────────────────────
    preferred = [
        ("dashboard.png", ""),
        ("dashboard_mri_by_age_fixed.png", "MRI sessions by age group"),
        ("target_vs_observed_age_by_sex.png", "Observed vs Target — Age × Sex"),
        (
            "target_vs_observed_age_by_ethnicity.png",
            "Observed vs Target — Age × Ethnicity",
        ),
        ("ethnicity_observed_vs_target.png", "Observed vs Target — Ethnicity totals"),
        (
            "dashboard_mh_heatmap_update.png",
            "Diagnoses by site (heatmap + ‘No diagnosis’ row)",
        ),
    ]

    outpdf = FIGDIR / "cpip_dashboard_all_figs.pdf"
    with PdfPages(outpdf) as pp:
        add_cover_page(pp)
        for fname, ttl in preferred:
            p = FIGDIR / fname
            if p.exists():
                add_image_page(pp, p, ttl)
            else:
                print(f"ℹ️ Missing figure → {fname}")

    print(f"✅ PDF saved → {outpdf}")


if __name__ == "__main__":
    main()
