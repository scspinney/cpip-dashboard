# src/pipeline.py
import sys
import re
from pathlib import Path
from typing import List, Optional

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
from .demographics.age import add_age_years, to_age_block, AGE_BLOCK_ORDER
from .demographics.sex_effective import add_effective_sex
from .demographics.gender import add_gender_label
from .demographics.ethnicity import (
    build_eth_long,
    build_white_nonwhite_panel,
    collapse_eth_for_blocks,
    collapse_eth_for_totals,
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
from .plots.mh_panels import plot_mh_panels, CFQ_LABELS, CFQ_VARS
from .mri.timeline import build_mri_baseline, build_timeline
from .configs.target_totals import TARGET_TOTALS, ETHNICITY_ORDER_TARGET


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


def _print_subject_codes_age_0_to_2(baseline: pd.DataFrame):
    if "age_years" not in baseline.columns:
        print("ℹ️ age_years missing — cannot list subjects age 0-2.")
        return

    def _first_non_empty(row: pd.Series, cols: List[str]) -> Optional[str]:
        for col in cols:
            if col not in row.index:
                continue
            val = row.get(col)
            if pd.isna(val):
                continue
            s = str(val).strip()
            if s and s.lower() != "nan":
                return s
        return None

    def _digits(s: Optional[str]) -> Optional[str]:
        if s is None:
            return None
        m = re.findall(r"\d+", s)
        return m[0] if m else None

    def _subject_num(row: pd.Series) -> Optional[str]:
        raw = _first_non_empty(row, ["subject_id", "study_id", "numeric_id"])
        if raw is not None:
            return _digits(raw)

        rec = _first_non_empty(row, ["record_id"])
        if rec is None:
            return None
        m = re.match(r"^\d+-(\d+)$", rec)
        return m.group(1) if m else _digits(rec)

    def _family_num(row: pd.Series) -> str:
        raw = _first_non_empty(row, ["fam_id", "family_id", "fam_numeric_id"])
        out = _digits(raw) if raw is not None else None
        return out if out is not None else "unknown"

    def _site_num(row: pd.Series) -> str:
        raw = _first_non_empty(row, ["site_id", "site_code"])
        out = _digits(raw) if raw is not None else None
        if out is not None:
            return out

        rec = _first_non_empty(row, ["record_id"])
        if rec is not None:
            m = re.match(r"^(\d+)-", rec)
            if m:
                return m.group(1)

        site_name = _first_non_empty(row, ["site"])
        site_map = {
            "Montreal": "1644",
            "Calgary": "1643",
            "Toronto": "1645",
            "montreal": "1644",
            "calgary": "1643",
            "toronto": "1645",
        }
        return site_map.get(site_name, "0000")

    def _subject_code(row: pd.Series) -> str:
        existing = _first_non_empty(row, ["subject_code"])
        if existing is not None:
            return existing

        sub = _subject_num(row)
        if sub is None:
            fallback = _first_non_empty(row, ["subject_id", "study_id", "record_id"])
            return fallback if fallback is not None else "unknown_subject"

        fam = _family_num(row)
        site = _site_num(row)
        # Match existing demo naming style: sub_<study>_ses_1a_fam_<family>_site_<site>
        return f"sub_{sub}_ses_1a_fam_{fam}_site_{site}"

    mask = baseline["age_years"].between(0, 2, inclusive="both")
    subject_codes = (
        baseline.loc[mask]
        .apply(_subject_code, axis=1)
        .dropna()
        .drop_duplicates()
        .sort_values()
        .astype(str)
        .tolist()
    )

    print(f"Subjects age 0-2 years ({len(subject_codes)}):")
    if not subject_codes:
        print("  (none)")
        return
    for code in subject_codes:
        print(f"  - {code}")


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


def _export_age_blocks_by_site(baseline: pd.DataFrame) -> None:
    age_counts = (
        baseline.assign(age_block=lambda d: d["age_years"].apply(to_age_block))
        .dropna(subset=["age_block"])
        .groupby(["site", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )
    age_counts["site"] = pd.Categorical(
        age_counts["site"], categories=SITE_ORDER, ordered=True
    )
    age_counts["age_block"] = pd.Categorical(
        age_counts["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )
    age_counts = age_counts.sort_values(["site", "age_block"]).reset_index(drop=True)
    age_counts.to_csv(FIGDIR / "age_blocks_by_site.csv", index=False)


def _export_counts_by_site(
    baseline: pd.DataFrame,
    *,
    label_col: str,
    order: List[str],
    filename: str,
) -> None:
    if label_col not in baseline.columns:
        return

    df = baseline.copy()
    if label_col == "gender_lbl":
        df[label_col] = df[label_col].fillna("NA")

    counts = (
        df.dropna(subset=[label_col])
        .groupby(["site", label_col], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )
    counts["site"] = pd.Categorical(counts["site"], categories=SITE_ORDER, ordered=True)
    counts[label_col] = pd.Categorical(counts[label_col], categories=order, ordered=True)
    counts = counts.sort_values(["site", label_col]).reset_index(drop=True)
    counts.to_csv(FIGDIR / filename, index=False)


def _export_mri_by_age_status_site(
    df_main: pd.DataFrame,
    baseline: pd.DataFrame,
) -> pd.DataFrame:
    mri_base = build_mri_baseline(df_main)
    if mri_base.empty:
        out = pd.DataFrame(columns=["site", "status_collapsed", "age_block", "Count"])
        out.to_csv(FIGDIR / "mri_by_age_status_site.csv", index=False)
        return mri_base

    mri_age = mri_base.merge(
        baseline[["record_id", "site", "age_years"]],
        on=["record_id", "site"],
        how="left",
    )
    mri_age["age_block"] = mri_age["age_years"].apply(to_age_block)

    counts = (
        mri_age.dropna(subset=["age_block"])
        .groupby(["site", "status_collapsed", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Count")
    )
    counts["site"] = pd.Categorical(counts["site"], categories=SITE_ORDER, ordered=True)
    counts["status_collapsed"] = pd.Categorical(
        counts["status_collapsed"],
        categories=["Completed", "Scheduled"],
        ordered=True,
    )
    counts["age_block"] = pd.Categorical(
        counts["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )
    counts = counts.sort_values(["site", "status_collapsed", "age_block"]).reset_index(
        drop=True
    )
    counts.to_csv(FIGDIR / "mri_by_age_status_site.csv", index=False)
    return mri_base


def _export_scanning_rate_last3months(mri_base: pd.DataFrame) -> None:
    if mri_base.empty:
        out = pd.DataFrame(
            columns=[
                "site",
                "participants_last_3mo",
                "weeks_in_period",
                "active_weeks_in_period",
                "rate_per_week",
                "rate_per_active_week",
            ]
        )
        out.to_csv(FIGDIR / "scanning_rate_last3months.csv", index=False)
        return

    today = pd.Timestamp.now().normalize()
    start_3mo = today - pd.DateOffset(months=3)
    recent = mri_base.loc[
        (mri_base["mri_date"] >= start_3mo)
        & (mri_base["mri_date"] <= today)
        & (mri_base["status_collapsed"] == "Completed")
    ].copy()

    weeks_in_period = (today - start_3mo).days / 7
    active_weeks_in_period = weeks_in_period * (5 / 7)

    rate_df = (
        recent.groupby("site")["record_id"]
        .nunique()
        .reindex(SITE_ORDER, fill_value=0)
        .reset_index(name="participants_last_3mo")
        .assign(
            weeks_in_period=weeks_in_period,
            active_weeks_in_period=active_weeks_in_period,
            rate_per_week=lambda d: d["participants_last_3mo"] / d["weeks_in_period"],
            rate_per_active_week=lambda d: d["participants_last_3mo"]
            / d["active_weeks_in_period"],
        )
    )
    rate_df.to_csv(FIGDIR / "scanning_rate_last3months.csv", index=False)


def _export_mh_cooccurrence_counts(baseline: pd.DataFrame) -> None:
    avail = [c for c in CFQ_VARS if c in baseline.columns]
    if not avail:
        return

    cfq_bin = baseline[avail].fillna(0).astype(int)
    co_matrix = cfq_bin.T.dot(cfq_bin)
    co_matrix.to_csv(FIGDIR / "dashboard_mh_cooccurrence_counts.csv")
    co_matrix.rename(index=CFQ_LABELS, columns=CFQ_LABELS).to_csv(
        FIGDIR / "dashboard_mh_cooccurrence_counts_labeled.csv"
    )


def _export_target_vs_observed_age_by_sex(
    baseline: pd.DataFrame,
    targets_long: pd.DataFrame,
) -> None:
    obs = (
        baseline.dropna(subset=["sex_lbl", "age_years"])
        .assign(age_block=lambda d: d["age_years"].apply(to_age_block))
        .dropna(subset=["age_block"])
        .query("sex_lbl in ['Male', 'Female']")
        .groupby(["site", "sex_lbl", "age_block"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Observed_N")
    )
    targ = (
        targets_long.loc[targets_long["Group"].isin(["Male", "Female"])]
        .dropna(subset=["age_block"])
        .groupby(["site", "Group", "age_block"], observed=False)["Target_N"]
        .sum()
        .reset_index()
        .rename(columns={"Group": "sex_lbl"})
    )
    long = pd.concat(
        [
            targ.rename(columns={"Target_N": "N"}).assign(Measure="Target"),
            obs.rename(columns={"Observed_N": "N"}).assign(Measure="Observed"),
        ],
        ignore_index=True,
    )
    long["site"] = pd.Categorical(long["site"], categories=SITE_ORDER, ordered=True)
    long["sex_lbl"] = pd.Categorical(
        long["sex_lbl"], categories=["Female", "Male"], ordered=True
    )
    long["age_block"] = pd.Categorical(
        long["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )
    long["Measure"] = pd.Categorical(
        long["Measure"], categories=["Target", "Observed"], ordered=True
    )
    long = long.sort_values(["site", "sex_lbl", "age_block", "Measure"]).reset_index(
        drop=True
    )
    long.to_csv(FIGDIR / "target_vs_observed_age_by_sex.csv", index=False)


def _export_target_vs_observed_age_by_ethnicity(
    baseline: pd.DataFrame,
    eth_long: pd.DataFrame,
    targets_long: pd.DataFrame,
) -> None:
    obs_eth_age = _build_obs_eth_age_counts(baseline, eth_long)
    targ_eth = (
        targets_long.loc[targets_long["Group"].isin(ETHNICITY_ORDER_TARGET)]
        .dropna(subset=["age_block"])
        .groupby(["site", "Group", "age_block"], observed=False)["Target_N"]
        .sum()
        .reset_index()
        .rename(columns={"Group": "Ethnicity"})
    )
    df = pd.concat(
        [
            targ_eth.rename(columns={"Target_N": "N"}).assign(Measure="Target"),
            obs_eth_age.rename(columns={"Observed_N": "N"}).assign(Measure="Observed"),
        ],
        ignore_index=True,
    )
    df["site"] = pd.Categorical(df["site"], categories=SITE_ORDER, ordered=True)
    df["Ethnicity"] = pd.Categorical(
        df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )
    df["age_block"] = pd.Categorical(
        df["age_block"], categories=AGE_BLOCK_ORDER, ordered=True
    )
    df["Measure"] = pd.Categorical(
        df["Measure"], categories=["Target", "Observed"], ordered=True
    )
    df = df.sort_values(["site", "Ethnicity", "age_block", "Measure"]).reset_index(
        drop=True
    )
    df.to_csv(FIGDIR / "target_vs_observed_age_by_ethnicity.csv", index=False)


def _export_target_vs_observed_ethnicity_totals(eth_long: pd.DataFrame) -> None:
    if eth_long.empty:
        return

    observed = (
        eth_long.assign(Ethnicity=lambda d: d["label"].map(collapse_eth_for_totals))
        .dropna(subset=["Ethnicity"])
        .groupby(["site", "Ethnicity"], observed=False)["record_id"]
        .nunique()
        .reset_index(name="Observed_N")
    )

    target_rows = []
    for site, payload in TARGET_TOTALS.items():
        for eth, n in payload.get("Ethnicity", {}).items():
            target_rows.append({"site": site, "Ethnicity": eth, "Target_N": int(n)})
    target = pd.DataFrame(target_rows)

    cmp = target.merge(observed, on=["site", "Ethnicity"], how="left").fillna(
        {"Observed_N": 0}
    )
    long = pd.concat(
        [
            cmp.rename(columns={"Target_N": "N"}).assign(Measure="Target")[
                ["site", "Ethnicity", "N", "Measure"]
            ],
            cmp.rename(columns={"Observed_N": "N"}).assign(Measure="Observed")[
                ["site", "Ethnicity", "N", "Measure"]
            ],
        ],
        ignore_index=True,
    )
    long["site"] = pd.Categorical(long["site"], categories=SITE_ORDER, ordered=True)
    long["Ethnicity"] = pd.Categorical(
        long["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True
    )
    long["Measure"] = pd.Categorical(
        long["Measure"], categories=["Target", "Observed"], ordered=True
    )
    long = long.sort_values(["site", "Ethnicity", "Measure"]).reset_index(drop=True)
    long.to_csv(FIGDIR / "target_vs_observed_ethnicity_totals.csv", index=False)


def _export_legacy_csvs(
    baseline: pd.DataFrame,
    df_main: pd.DataFrame,
    eth_long: pd.DataFrame,
    targets_long: pd.DataFrame,
) -> None:
    _export_age_blocks_by_site(baseline)
    _export_counts_by_site(
        baseline,
        label_col="sex_lbl",
        order=SEX_ORDER,
        filename="sex_effective_by_site.csv",
    )
    _export_counts_by_site(
        baseline,
        label_col="gender_lbl",
        order=GENDER_ORDER,
        filename="gender_by_site.csv",
    )
    _export_counts_by_site(
        baseline,
        label_col="income_lbl",
        order=INCOME_ORDER,
        filename="income_by_site.csv",
    )
    mri_base = _export_mri_by_age_status_site(df_main, baseline)
    _export_scanning_rate_last3months(mri_base)
    _export_mh_cooccurrence_counts(baseline)
    _export_target_vs_observed_ethnicity_totals(eth_long)

    if not targets_long.empty:
        _export_target_vs_observed_age_by_sex(baseline, targets_long)
        _export_target_vs_observed_age_by_ethnicity(baseline, eth_long, targets_long)


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
    _print_subject_codes_age_0_to_2(baseline)
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

    _export_legacy_csvs(baseline, df_main, eth_long, targets_long)

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
