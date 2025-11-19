#!/usr/bin/env python3
"""
C-PIP multi-site dashboard  (main REDCap export only)

• Baseline (T1) rows only
• Exactly one row per participant – the row that answers the *most* key fields
• Outputs PNGs in ./figs and a consolidated PDF.

Targets / overlays:
- We read *numbers* (not %) straight from your screenshot CSVs:
    calgary_age_by_group.csv
    montreal_age_by_group.csv
    toronto_age_by_group.csv
  Each has columns: Group, 5 wks pre, 0-1, …, 17-18, 18, Total

Axis label policy:
- Show every age-bin label (rotated 45°) where used.
- The first bin is displayed as "<5wks" (we still read "5 wks pre" from files).

Color policy:
- Hue levels are fixed and ordered as: ["Target", "Observed"].
- Colors are fixed and consistent everywhere:
    Target   → #f58518  (orange)
    Observed → #4c78a8  (blue)
"""

import sys, glob, re, warnings
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────── CONFIG ─────────────────────────
FIGDIR       = Path("figs")
MIN_T1_DATE  = pd.Timestamp("2023-01-01")
SITE_COLOURS = {"Montreal": "#1f77b4", "Calgary": "#ff0e0e", "Toronto": "#2ca02c"}
SITE_ORDER   = ["Calgary", "Montreal", "Toronto"]
MRI_INSTRUMENT = "mri"   # repeat-instrument (lower-case)

# Fixed hue order + colors for Target vs Observed overlays
MEASURE_ORDER   = ["Target", "Observed"]
MEASURE_COLOURS = {"Target": "#f58518", "Observed": "#4c78a8"}

# Ethnicity order for consistent axes (collapsed bins where used)
ETHNICITY_ORDER_TARGET = ["White", "Asian", "Black", "Indigenous", "Middle East", "Latin American", "Other"]

# Age-bin labels used in target CSV parsing (first bin renamed to "<5wks")
AGE_BIN_ORDER = ["<5wks"] + [f"{i}-{i+1}" for i in range(0, 18)] + ["18"]

# ─────────────── Fil’s target totals (kept for totals plot) ───────────────
TARGET_TOTALS = {
    "Calgary": {
        "Total": 263,
        "Sex": {"Male":132, "Female":132},
        "Ethnicity": {"White":153,"Asian":63,"Black":11,"Indigenous":8,"Middle East":8,"Latin American":5,"Other":16},
        "UrbanRural": {"Urban":213,"Rural":50},
        "SES": {"MidHigh":223,"Low":41},
    },
    "Toronto": {
        "Total": 263,
        "Sex": {"Male":132, "Female":132},
        "Ethnicity": {"White":124,"Asian":87,"Black":24,"Indigenous":3,"Middle East":11,"Latin American":8,"Other":8},
        "UrbanRural": {"Urban":226,"Rural":37},
        "SES": {"MidHigh":201,"Low":63},
    },
    "Montreal": {
        "Total": 263,
        "Sex": {"Male":132, "Female":132},
        "Ethnicity": {"White":171,"Asian":29,"Black":26,"Indigenous":3,"Middle East":21,"Latin American":11,"Other":3},
        "UrbanRural": {"Urban":211,"Rural":53},
        "SES": {"MidHigh":183,"Low":80},
    }
}

# ───────── Age blocks for the main age panel (short labels) ─────────
AGE_BLOCK_ORDER = ["0-2", "2-5", "6-9", "10-12", "13-15", "16-18"]

def to_age_block(a: float) -> Optional[str]:
    """Map age (years) to inclusive blocks with short labels.
       Blocks: [0,2], (2,5], (5,9], (9,12], (12,15], (15,18]"""
    if pd.isna(a): return None
    if a <= 2:   return "0-2"
    if a <= 5:   return "2-5"
    if a <= 9:   return "6-9"
    if a <= 12:  return "10-12"
    if a <= 15:  return "13-15"
    if a <= 18:  return "16-18"
    return None  # outside scope

def bin_label_to_block(lbl: str) -> Optional[str]:
    if lbl is None:
        return None
    if lbl in ("<5wks", "0-1", "1-2"):
        return "0-2"
    if lbl == "18":
        return "16-18"
    m = re.match(r"^(\d+)-(\d+)$", str(lbl))
    if not m:
        return None
    k = int(m.group(1))  # lower edge of [k, k+1)
    if k < 2:   return "0-2"
    if k < 6:   return "2-5"
    if k < 10:  return "6-9"
    if k < 13:  return "10-12"
    if k < 16:  return "13-15"
    return "16-18"

# ───────────────────────── HELPERS ────────────────────────
def derive_site(p: Path) -> str:
    s = p.stem.lower()
    if   "montreal" in s: return "Montreal"
    elif "calgary"  in s: return "Calgary"
    elif "toronto"  in s: return "Toronto"
    raise ValueError(f"Cannot infer site from filename: {p.name}")

def pick_first(df: pd.DataFrame, names: List[str], *, required=False) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise ValueError(f"None of {names} found")
    return None

def find_complete_col(df: pd.DataFrame, regexes: List[str], fallbacks: List[str]) -> Optional[str]:
    cols = [c for c in df.columns if c.endswith("_complete")]
    for r in regexes:
        pat = re.compile(r, flags=re.IGNORECASE)
        for c in cols:
            if pat.search(c):
                return c
    for fb in fallbacks:
        if fb in df.columns:
            return fb
    return None

def load_csv(p: Path) -> pd.DataFrame:
    d = pd.read_csv(p, low_memory=False)
    d["site"] = derive_site(p)
    return d

# ───────────────────────── LOAD REDCap DATA ──────────────────────
paths = [Path(p) for p in (sys.argv[1:] if len(sys.argv) > 1 else
                           [f for g in ("*montreal*.csv","*calgary*.csv","*toronto*.csv")
                            for f in glob.glob(g)])]
if not paths:
    sys.exit("❌ No site CSVs found.")

df_main = pd.concat([load_csv(p) for p in paths], ignore_index=True)
non_repeat = df_main[df_main["redcap_repeat_instrument"].isna()].copy()

# Columns used later (raw fields; we will compute an EFFECTIVE sex below)
YOUTH_SEX_COL    = "sex"       if "sex" in non_repeat.columns       else None
PARENT_SEX_COL   = "demo_sex"  if "demo_sex" in non_repeat.columns  else None
GENDER_COL       = pick_first(non_repeat, ["gender","demo_gender","gender_oth"], required=False)
ETH_COLS         = [c for c in non_repeat.columns if re.match(r"(demo_)?ethnicity___\d+", c)]

# Try to discover the two completion flags
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

YOUTH_FORM_COL  = find_complete_col(non_repeat, YOUTH_COMPLETE_CAND_REGEXES, YOUTH_COMPLETE_FALLBACKS)
PARENT_FORM_COL = find_complete_col(non_repeat, PARENT_CHILD_COMPLETE_CAND_REGEXES, PARENT_CHILD_COMPLETE_FALLBACKS)

# ────────── Household-income merge (round to nearest REDCap bracket) ──────────
_bracket_median = {
     1:   5_000,  2:  15_000,  3:  30_000,  4:  50_000,
     5:  70_000,  6:  90_000,  7: 112_500,  8: 137_500,
     9: 175_000, 10: 250_000, 11: 350_000, 12: 450_000
}
SPECIAL_INCOME = {777, 888}

def closest_code_from_dollars(amount: float) -> int:
    return int(min(_bracket_median, key=lambda c: abs(_bracket_median[c] - amount)))

def _as_int_or_nan(x):
    try:
        return int(x)
    except Exception:
        return np.nan

def merge_income_row(row: pd.Series):
    cg = _as_int_or_nan(row.get("demo_cg_income"))
    pt = _as_int_or_nan(row.get("demo_cg_prtnr_income"))
    if cg in SPECIAL_INCOME: return cg
    if pt in SPECIAL_INCOME: return pt
    if pd.isna(cg) and pd.isna(pt): return np.nan
    if pd.isna(pt): return cg
    if pd.isna(cg): return pt
    total = _bracket_median[int(cg)] + _bracket_median[int(pt)]
    return closest_code_from_dollars(total)

# Apply merge logic
if {"demo_cg_income", "demo_cg_prtnr_income"} & set(non_repeat.columns):
    non_repeat["demo_cg_income"] = non_repeat.apply(merge_income_row, axis=1)
elif "demo_cg_prtnr_income" in non_repeat.columns:
    non_repeat["demo_cg_income"] = non_repeat["demo_cg_prtnr_income"].apply(_as_int_or_nan)
else:
    non_repeat["demo_cg_income"] = non_repeat.get("demo_cg_income", np.nan).apply(_as_int_or_nan)

# ────────────────── PICK BEST BASELINE ROW PER PARTICIPANT ─────────────────
KEY_VARS = ["youth_age_y", "youth_age_m", "demo_cg_income"] + ([GENDER_COL] if GENDER_COL else []) + ETH_COLS
if YOUTH_SEX_COL:  KEY_VARS.append(YOUTH_SEX_COL)
if PARENT_SEX_COL: KEY_VARS.append(PARENT_SEX_COL)

def best_baseline_rows(df: pd.DataFrame) -> pd.DataFrame:
    t1 = df[df["redcap_event_name"] == "t1_arm_1"].copy()
    t1["_score"] = t1[KEY_VARS].notna().sum(axis=1)
    t1 = t1.sort_values(["record_id", "_score"], ascending=[True, False])
    best = t1.drop_duplicates(subset="record_id").drop(columns="_score")
    return best.reset_index(drop=True)

baseline = best_baseline_rows(non_repeat)

# ────────────────── DERIVED DEMOGRAPHIC FIELDS ─────────────────────────────
def _age_years(row):
    y, m = row.get("youth_age_y"), row.get("youth_age_m")
    if pd.isna(y) and pd.isna(m): return np.nan
    return (0 if pd.isna(y) else y) + (0 if pd.isna(m) else m) / 12

baseline["age_years"] = baseline.apply(_age_years, axis=1)

# ────────── EFFECTIVE sex using instrument completeness (FINAL) ──────────
# ────────── EFFECTIVE sex using instrument completeness (FIXED LOGIC) ──────────

sex_map    = {1:"Male",2:"Female",3:"Intersex / Non-binary",777:"Unknown",888:"Prefer not to answer"}
gender_map = {1:"Boy/Man",2:"Girl/Woman",3:"Non-binary",4:"Two-spirit",5:"Another",
              6:"Questioning",777:"Unknown",888:"Prefer not to answer"}
income_map = {1:"Under $10k",2:"$10-19k",3:"$20-39k",4:"$40-59k",5:"$60-79k",
              6:"$80-99k",7:"$100-124k",8:"$125-149k",9:"$150-199k",
              10:"$200-299k",11:"$300-399k",12:"$400k+",777:"Unknown",888:"Prefer not to say"}

SEX_VALID = {1, 2, 3, 777, 888}

# Normalize raw integers
def _norm_int(x):
    try:
        return int(x)
    except:
        return np.nan

baseline["sex_youth_raw"]  = baseline[YOUTH_SEX_COL].apply(_norm_int)  if YOUTH_SEX_COL  else np.nan
baseline["sex_parent_raw"] = baseline[PARENT_SEX_COL].apply(_norm_int) if PARENT_SEX_COL else np.nan

# Identify expected reporter
baseline["expected_youth"]  = baseline["age_years"] >= 11
baseline["expected_parent"] = baseline["age_years"] < 11

# Form completed flags (1 = unverified, 3 = complete)
def _form_completed(x):
    try:
        return int(x) in (1, 3)
    except:
        return False

baseline["youth_form_complete"] = baseline[YOUTH_FORM_COL].apply(_form_completed) \
                                  if (YOUTH_FORM_COL and YOUTH_FORM_COL in baseline.columns) else False

baseline["parent_form_complete"] = baseline[PARENT_FORM_COL].apply(_form_completed) \
                                   if (PARENT_FORM_COL and PARENT_FORM_COL in baseline.columns) else False

# EFFECTIVE SEX (presence-first logic)
def _effective_sex(row):
    # 1. If youth provides valid sex → use it
    if row["sex_youth_raw"] in SEX_VALID:
        return int(row["sex_youth_raw"])

    # 2. If youth is blank but parent provides valid sex → use parent
    if row["sex_parent_raw"] in SEX_VALID:
        return int(row["sex_parent_raw"])

    # 3. Neither provided → missing (for now)
    return np.nan

baseline["sex_code_effective"] = baseline.apply(_effective_sex, axis=1)
baseline["sex_lbl"] = baseline["sex_code_effective"].map(sex_map)

# TRUE MISSINGNESS (Sean’s intended logic)
def _true_missing(row):

    # If sex exists anywhere → NOT missing
    if (row["sex_youth_raw"] in SEX_VALID) or (row["sex_parent_raw"] in SEX_VALID):
        return False

    # Now sex is blank everywhere. Determine if this is 'missing' or simply 'not entered yet'.

    # If youth expected
    if row["expected_youth"]:
        # Missing only if youth form COMPLETE
        return row["youth_form_complete"]

    # If parent expected
    if row["expected_parent"]:
        # Missing only if parent form COMPLETE
        return row["parent_form_complete"]

    # No valid branch (should not happen)
    return False

baseline["sex_missing_effective"] = baseline.apply(_true_missing, axis=1)
baseline["sex_present_effective"] = ~baseline["sex_missing_effective"]

# Gender / income labels
baseline["gender_lbl"] = baseline[GENDER_COL].map(gender_map) if GENDER_COL else np.nan
baseline["income_lbl"] = baseline["demo_cg_income"].map(income_map)


# ───────── Validation prints for sex missingness ─────────
print("\n──────── VALIDATION: Sex Missingness vs Form Completion ────────")
to_show = baseline.copy()
to_show = to_show.rename(columns={"sex_code_effective":"sex"})
disp_cols = ["record_id","site","age_years","sex",
             "youth_form_complete","parent_form_complete",
             "sex_missing_effective","sex_present_effective"]
print(to_show[disp_cols].head(40).to_string(index=False))

mtl_true_missing = baseline.loc[
    (baseline["site"]=="Montreal") & (baseline["sex_missing_effective"])
, "record_id"].astype(str).tolist()

print("\n──────── Montreal: TRUE Missing Sex (record_id only) ────────")
print(mtl_true_missing)
print(f"\nTotal Montreal TRUE missing sex: {len(mtl_true_missing)}")

# ────────── Ethnicity (checkbox melt; single build) ──────────
if ETH_COLS:
    baseline[ETH_COLS] = baseline[ETH_COLS].fillna(0)
    eth_lookup = {f"{p}{i}": lbl
                  for p in ("demo_ethnicity___","demo_cg_ethnicity___")
                  for i,lbl in {
                      1:"Black",2:"East Asian",3:"First Nations",4:"Inuk/Inuit",5:"Métis",
                      6:"Latin American",7:"Middle Eastern",8:"South Asian",9:"Southeast Asian",
                      10:"White",11:"Other",888:"Prefer not to say"}.items()}
    eth_long = (baseline[["record_id","site"]+ETH_COLS]
                .melt(id_vars=["record_id","site"], value_name="val")
                .query("val == 1")
                .assign(label=lambda d: d["variable"].map(eth_lookup)))
else:
    eth_long = pd.DataFrame()

# Build White/Non-white wide form (for one of the panels)
eth_wide = pd.DataFrame()
if not eth_long.empty:
    tmp = (eth_long.groupby(["site","record_id"])["label"]
           .apply(lambda s: sorted(set(s.dropna()))).reset_index(name="labels"))
    tmp["is_white_only"] = tmp["labels"].apply(lambda L: ("White" in set(L)) and (len(set(L))==1))
    tmp["has_nonwhite"]  = tmp["labels"].apply(lambda L: any(l!="White" for l in set(L)))
    eth_wide = tmp[["site","record_id","is_white_only","has_nonwhite"]].copy()
    eth_wide["White"] = eth_wide["is_white_only"].map({True:"White", False:np.nan})
    eth_wide["Non-white"] = eth_wide["has_nonwhite"].map({True:"Non-white", False:np.nan})
    eth_wide = eth_wide[["site","record_id","White","Non-white"]]

# ────────── MRI baseline rows using mri_ses_outcome (RA logic) ──────────
mri_base = df_main.loc[
    (df_main["redcap_event_name"] == "t1_arm_1") &
    (df_main["redcap_repeat_instrument"].fillna("").str.strip().str.lower() == MRI_INSTRUMENT)
].copy()

mri_base["mri_date"] = pd.to_datetime(mri_base["mri_date"], errors="coerce").dt.normalize()
mri_base = mri_base.dropna(subset=["mri_date"])
mri_base = mri_base.loc[mri_base["mri_date"] >= MIN_T1_DATE]
today = pd.Timestamp.now().normalize()

# Fixed logic: Completed = (1 or 3), Scheduled = (2 or future-dated)
mri_base["status"] = np.select(
    [
        mri_base["mri_ses_outcome"].isin([1, 3]),
        (mri_base["mri_ses_outcome"] == 2) | (mri_base["mri_date"] > today)
    ],
    ["Completed", "Scheduled"],
    default="Unknown"
)

mri_base["status_collapsed"] = mri_base["status"].replace({"Partial": "Completed"})
mri_base = mri_base.sort_values(["record_id","mri_date"]).drop_duplicates(subset=["record_id","mri_date"])
mri_base = mri_base[mri_base["status_collapsed"].isin(["Completed", "Scheduled"])]

timeline_parts = []
for status_label in ["Completed", "Scheduled"]:
    tmp = (
        mri_base.loc[mri_base["status_collapsed"] == status_label]
        .groupby(["site", "mri_date"])["record_id"]
        .nunique()
        .groupby("site")
        .cumsum()
        .rename("cum")
        .reset_index()
        .assign(status=status_label)
    )
    timeline_parts.append(tmp)
timeline = pd.concat(timeline_parts, ignore_index=True)

# ────────── sanity-check counts ──────────
def _count_non_na(column: str) -> pd.Series:
    return (baseline[baseline[column].notna()].groupby("site")["record_id"].nunique())

site_totals = baseline.groupby("site")["record_id"].nunique()

counts = pd.DataFrame({
    "Total available counts": site_totals,
    "Age non-NA":     _count_non_na("age_years"),
    "Sex non-NA":     _count_non_na("sex_lbl"),
    "Gender non-NA":  _count_non_na("gender_lbl"),
    "Income non-NA":  _count_non_na("income_lbl"),
})

if not eth_long.empty:
    eth_non_na = (eth_long.groupby("site")["record_id"].nunique().rename("Ethnicity non-NA"))
    counts = counts.join(eth_non_na, how="left")
else:
    eth_non_na = pd.Series(0, index=site_totals.index, name="Ethnicity non-NA")

counts = counts.fillna(0).astype(int)

na_counts = pd.DataFrame({
    "Total available counts": site_totals,
    "Age NA":     (site_totals - _count_non_na("age_years")),
    "Sex NA":     (site_totals - _count_non_na("sex_lbl")),
    "Gender NA":  (site_totals - _count_non_na("gender_lbl")),
    "Income NA":  (site_totals - _count_non_na("income_lbl")),
    "Ethnicity NA": (site_totals - eth_non_na.reindex(site_totals.index).fillna(0).astype(int)),
}).fillna(0).astype(int)

print("\nParticipant counts used in each baseline panel")
print(counts.to_string())
print("\nParticipant counts with missing values (NA)")
print(na_counts.to_string())
print("\nNOTE: All numbers refer to *unique* baseline participants after deduplication (one T1 row per participant).")

FIGDIR.mkdir(exist_ok=True)
counts.to_csv(FIGDIR / "dashboard_counts.csv")
na_counts.to_csv(FIGDIR / "dashboard_counts_na.csv")

# ────────── PLOTS ──────────
sns.set(style="whitegrid")
sns.set_context("talk")
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 18, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
})

fig, axes = plt.subplots(4, 2, figsize=(22, 20))
(ax_cum, ax_age), (ax_gender, ax_sex), (ax_eth, ax_eth_wnw), (ax_inc, ax_nas) = axes
fig.subplots_adjust(wspace=0.35, hspace=0.3)

# ── Age (developmental blocks with short labels) ──
age_block_counts = (
    baseline.assign(age_block=lambda d: d["age_years"].apply(to_age_block))
            .dropna(subset=["age_block"])
            .groupby(["site","age_block"], observed=False)["record_id"]
            .nunique()
            .reset_index(name="Count")
)
age_block_counts["age_block"] = pd.Categorical(age_block_counts["age_block"],
                                               categories=AGE_BLOCK_ORDER, ordered=True)
sns.barplot(data=age_block_counts, x="age_block", y="Count",
            hue="site", hue_order=SITE_ORDER, palette=SITE_COLOURS, ax=ax_age)
ax_age.set_title("Age groups by site (baseline)")
ax_age.set_xlabel("Age group")
ax_age.tick_params(axis="x", rotation=0)

# Sex (effective)
sns.countplot(data=baseline, x="sex_lbl", hue="site", hue_order=SITE_ORDER,
              palette=SITE_COLOURS, order=list(sex_map.values()), ax=ax_sex)
ax_sex.set_title("Sex assigned at birth (effective, youth ≥11 else parent)")
ax_sex.tick_params(axis="x", rotation=35)

# Gender
order_g = ["Boy/Man","Girl/Woman","Non-binary","Two-spirit",
           "Another","Questioning","Unknown","Prefer not to answer","NA"]
sns.countplot(data=baseline, x="gender_lbl", hue="site", hue_order=SITE_ORDER,
              palette=SITE_COLOURS, order=order_g, ax=ax_gender)
ax_gender.set_title("Current gender identity (baseline)")
ax_gender.tick_params(axis="x", rotation=35)

# Ethnicity (all labels as entered)
if not eth_long.empty:
    order_eth_full = eth_long["label"].value_counts().index.tolist()
    sns.countplot(data=eth_long, x="label", hue="site", hue_order=SITE_ORDER,
                  palette=SITE_COLOURS, order=order_eth_full, ax=ax_eth)
    ax_eth.set_title("Ethnicity (baseline)")
    ax_eth.tick_params(axis="x", rotation=35)
else:
    ax_eth.axis("off"); ax_eth.text(0.5,0.5,"Ethnicity\nN/A", ha="center", va="center",
                                    fontsize=12, color="grey", transform=ax_eth.transAxes)

# Ethnicity (White/Non-white only)
if not eth_wide.empty:
    eth_wide_long = eth_wide.melt(id_vars=["site", "record_id"],
                                  value_vars=["White", "Non-white"],
                                  var_name="Ethnicity", value_name="Label")
    eth_wide_long = eth_wide_long[eth_wide_long["Label"].notna() & (eth_wide_long["Label"] != "")]
    eth_counts_panel = (eth_wide_long.groupby(["site", "Ethnicity"], observed=False)["record_id"]
                        .nunique().reset_index(name="Count"))
    sns.barplot(data=eth_counts_panel, x="Ethnicity", y="Count", hue="site", hue_order=SITE_ORDER,
                palette=SITE_COLOURS, ax=ax_eth_wnw, order=["White","Non-white"])
    ax_eth_wnw.set_title("Ethnicity (White/Non-white) by site")
    ax_eth_wnw.set_ylabel("Participant count")
    ax_eth_wnw.set_xlabel("Ethnicity group")
else:
    ax_eth_wnw.axis("off"); ax_eth_wnw.text(0.5,0.5,"Ethnicity\nN/A", ha="center", va="center",
                                            fontsize=12, color="grey", transform=ax_eth_wnw.transAxes)

# MRI timeline
if not timeline.empty:
    sns.lineplot(
        data=timeline.query("status == 'Completed'"), x="mri_date", y="cum",
        hue="site", hue_order=SITE_ORDER, style="status",
        palette=SITE_COLOURS,
        dashes={"Completed":"", "Scheduled":[2,2]},
        ax=ax_cum
    )
    ax_cum.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax_cum.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax_cum.get_xticklabels(), rotation=45, ha="right")
    ax_cum.set_ylabel("Cumulative T1 MRIs")
    ax_cum.set_xlabel("Date")
    ax_cum.set_title("Baseline MRI visits by site (Completed = 1 or 3)")
    ax_cum.legend(title="Site / Status", loc="upper left")
else:
    ax_cum.axis("off")
    ax_cum.text(0.5,0.5,"MRI timeline\nN/A",ha="center",va="center",
                fontsize=12,color="grey",transform=ax_cum.transAxes)

# Income
if not baseline["income_lbl"].dropna().empty:
    order_inc = [income_map[i] for i in range(1, 13)] + [income_map[777], income_map[888]]
    sns.countplot(data=baseline.dropna(subset=["income_lbl"]),
                  x="income_lbl", hue="site", hue_order=SITE_ORDER, palette=SITE_COLOURS,
                  order=order_inc, ax=ax_inc)
    ax_inc.set_title("Household income (baseline)")
    ax_inc.tick_params(axis="x", rotation=35)
else:
    ax_inc.axis("off"); ax_inc.text(0.5,0.5,"Income\nN/A", ha="center", va="center",
                                    fontsize=12, color="grey", transform=ax_inc.transAxes)

# ============================================================
# EFFECTIVE MISSINGNESS PANEL (only Sex is replaced)
# ============================================================

# We assume baseline already contains:
# - sex_missing_effective
# - sex_present_effective
# - youth_form_complete
# - parent_form_complete
# - age_years
# - income, ethnicity, gender vars exist as before

# ---- 1. Compute missing per site ----
# All sites present in baseline
sites = baseline["site"].unique()

rows = []
for site in sites:
    df = baseline[baseline["site"] == site]

    total = len(df)

    age_na = df["age_years"].isna().sum()
    gender_na = df[GENDER_COL].isna().sum() if GENDER_COL in df.columns else 0

    income_na = df["demo_cg_income"].isna().sum() if "demo_cg_income" in df.columns else 0

    # Ethnicity NA = all checkboxes empty OR all NA
    if len(ETH_COLS) > 0:
        eth_mask = (df[ETH_COLS].sum(axis=1) == 0) | df[ETH_COLS].isna().all(axis=1)
        eth_na = eth_mask.sum()
    else:
        eth_na = 0

    # ✅ Replace Sex NA with EFFECTIVE missingness
    sex_na = df["sex_missing_effective"].sum()

    rows.append({
        "site": site,
        "Total available counts": total,
        "Age NA": age_na,
        "Sex NA": sex_na,                  # ✅ EFFECTIVE missing
        "Gender NA": gender_na,
        "Income NA": income_na,
        "Ethnicity NA": eth_na,
    })

na_counts = pd.DataFrame(rows).set_index("site")

# ============================================================
# ---- 2. Plot the Missing counts per panel (unchanged style) ----
# ============================================================

if not na_counts.empty:
    na_counts_long = (
        na_counts.reset_index()
        .melt(id_vars="site", var_name="Metric", value_name="Count")
    )

    metric_order = [
        "Total available counts",
        "Age NA",
        "Sex NA",       # ✅ now effective
        "Gender NA",
        "Income NA",
        "Ethnicity NA"
    ]

    na_counts_long["Metric"] = pd.Categorical(
        na_counts_long["Metric"],
        categories=metric_order,
        ordered=True
    )

    sns.barplot(
        data=na_counts_long,
        x="Metric",
        y="Count",
        hue="site",
        hue_order=SITE_ORDER,
        palette=SITE_COLOURS,
        ax=ax_nas
    )

    ax_nas.set_title("Missing counts per panel (Sex = Effective Missingness)")
    ax_nas.set_ylabel("Count")
    ax_nas.set_xlabel("Panel")
    ax_nas.tick_params(axis="x", rotation=25)

else:
    ax_nas.axis("off")
    ax_nas.text(
        0.5, 0.5,
        "Missing counts\nN/A",
        ha="center", va="center",
        fontsize=12, color="grey",
        transform=ax_nas.transAxes
    )

fig.tight_layout()
FIGDIR.mkdir(exist_ok=True)
fig.savefig(FIGDIR / "dashboard.png", dpi=300)
plt.close(fig)
print(f"✓ Dashboard saved → {FIGDIR/'dashboard.png'}")


# ────────── MRI-by-age by site (vertical layout) ──────────
mri_age = mri_base.merge(
    baseline[["record_id", "age_years", "site"]],
    on=["record_id", "site"], how="left"
)
mri_age["age_block"] = mri_age["age_years"].apply(to_age_block)

mri_age_counts = (
    mri_age.dropna(subset=["age_block"])
    .groupby(["site", "status_collapsed", "age_block"], observed=False)["record_id"]
    .nunique()
    .reset_index(name="Count")
)

g = sns.catplot(
    data=mri_age_counts,
    kind="bar",
    x="age_block", y="Count",
    hue="status_collapsed",
    row="site",
    row_order=SITE_ORDER,
    palette={"Completed": "#4c78a8", "Scheduled": "#999999", "Unknown": "#cccccc"},
    order=AGE_BLOCK_ORDER,
    height=3.5, aspect=2.2, sharex=True, sharey=True,
    legend=True
)

for ax in g.axes.flatten():
    ax.set_xlabel("Age group")
    ax.tick_params(axis="x", labelbottom=True)
    for label in ax.get_xticklabels():
        label.set_rotation(0)

g.set_axis_labels("Age group", "Participants")
g.set_titles("{row_name}")

handles, labels = None, None
if getattr(g, "_legend", None) is not None:
    handles = g._legend.legend_handles
    labels  = [t.get_text() for t in g._legend.texts]
    g._legend.remove()

if not handles or not labels:
    nrows = g.axes.shape[0]
    mid_ax = g.axes[nrows // 2, 0]
    handles, labels = mid_ax.get_legend_handles_labels()

if handles and labels:
    nrows = g.axes.shape[0]
    mid_ax = g.axes[nrows // 2, 0]
    mid_ax.legend(
        handles, labels, title="Status",
        loc="upper center",
        bbox_to_anchor=(1.18, 0.6),
        ncol=1,
        frameon=True,
        borderaxespad=0.6
    )
else:
    print("⚠️ Warning: No legend handles found; skipping legend creation.")

g.fig.subplots_adjust(hspace=0.4, top=0.90, bottom=0.12)
g.fig.suptitle("MRI sessions completed (1 or 3) vs scheduled by age group and site", fontsize=16)

out_path = FIGDIR / "dashboard_mri_by_age_fixed.png"
g.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close(g.fig)
print(f"✓ Final MRI-by-age-by-site → {out_path}")

# ────────── CFQ mental health diagnostics ──────────
cfq_vars = [
    "cfq_diag_asd_part","cfq_diag_id_part","cfq_adhd_part","cfq_fasd_part",
    "cfq_ld_part","cfq_lcd_part","cfq_md_part","cfq_diag_other_part",
    "cfq_ment_ad_part","cfq_ment_dd_part","cfq_ment_bd_part","cfq_ment_ocd_part",
    "cfq_ment_ts_part","cfq_ment_psyep_part","cfq_ment_schizo_part",
    "cfq_ment_sa_part","cfq_ment_epilepsy_part","cfq_ment_other_part"
]
cfq_labels = {
    "cfq_diag_asd_part": "Autism Spectrum Disorder",
    "cfq_diag_id_part": "Cognitive Impair/IDD",
    "cfq_adhd_part": "ADHD",
    "cfq_fasd_part": "FASD",
    "cfq_ld_part": "Learning Disability",
    "cfq_lcd_part": "Language Disorder",
    "cfq_md_part": "Motor Disorder",
    "cfq_diag_other_part": "Other Neurodev.",
    "cfq_ment_ad_part": "Anxiety Disorder",
    "cfq_ment_dd_part": "Depressive Disorder",
    "cfq_ment_bd_part": "Bipolar Disorder",
    "cfq_ment_ocd_part": "OCD",
    "cfq_ment_ts_part": "Tourette’s",
    "cfq_ment_psyep_part": "Psychosis",
    "cfq_ment_schizo_part": "Schizophrenia",
    "cfq_ment_sa_part": "Substance Abuse",
    "cfq_ment_epilepsy_part": "Epilepsy",
    "cfq_ment_other_part": "Other (MH)"
}

avail_cfq = [c for c in cfq_vars if c in baseline.columns]
if avail_cfq:
    cfq_any   = baseline[avail_cfq].notna().any(axis=1)
    total_yes = baseline[avail_cfq].fillna(0).sum(axis=1)

    mh_long = (baseline.loc[cfq_any, ["record_id","site"] + avail_cfq]
               .melt(id_vars=["record_id","site"], value_name="val")
               .query("val == 1")
               .assign(Diagnosis=lambda d: d["variable"].map(cfq_labels)))

    mh_counts = (mh_long.groupby(["site","Diagnosis"], observed=False)["record_id"]
                 .nunique().reset_index(name="Count"))

    no_dx_mask   = cfq_any & (total_yes == 0)
    no_dx_counts = (baseline.loc[no_dx_mask].groupby("site")["record_id"]
                    .nunique().reindex(SITE_COLOURS.keys(), fill_value=0))
    cfq_site_totals = (baseline.loc[cfq_any].groupby("site")["record_id"]
                       .nunique().reindex(SITE_COLOURS.keys(), fill_value=0))

    if not mh_counts.empty:
        plt.figure(figsize=(14, 8))
        sns.barplot(data=mh_counts, x="Diagnosis", y="Count", hue="site", hue_order=SITE_ORDER, palette=SITE_COLOURS)
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("Number of participants")
        plt.title("Child mental health & neurodevelopmental diagnoses (participant report)", fontsize=16, pad=28)
        plt.tight_layout()
        plt.savefig(FIGDIR / "dashboard_mh.png", dpi=300)
        plt.close()

        heat = mh_counts.pivot(index="Diagnosis", columns="site", values="Count").fillna(0)
        plt.figure(figsize=(10, 10))
        sns.heatmap(heat, annot=True, fmt=".0f", cmap="Reds", cbar_kws={'label':'Number of participants'})
        plt.title("Counts of diagnoses by site", fontsize=16, pad=28)
        plt.tight_layout()
        plt.savefig(FIGDIR / "dashboard_mh_heatmap.png", dpi=300)
        plt.close()

        no_row = pd.DataFrame([no_dx_counts.reindex(heat.columns, fill_value=0)], index=["No diagnosis"])
        heat_all = pd.concat([heat, no_row], axis=0)
        row_is_no_dx = (heat_all.index.to_numpy() == "No diagnosis")
        mask_main = np.tile(row_is_no_dx[:, None], (1, heat_all.shape[1]))
        mask_nodx = ~mask_main
        from matplotlib.colors import ListedColormap
        fixed_nodx_cmap = ListedColormap(["#2b0000"])
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(heat_all, mask=mask_main, annot=True, fmt=".0f", cmap="Reds",
                    cbar=True, cbar_kws={'label':'Number of participants'},
                    linewidths=.5, linecolor='gray', ax=ax)
        sns.heatmap(heat_all, mask=mask_nodx, annot=True, fmt=".0f", cmap=fixed_nodx_cmap,
                    cbar=False, linewidths=.5, linecolor='gray', ax=ax)
        ax.set_title("Counts of diagnoses by site", fontsize=16, pad=22)
        ax.text(0.5, 1.025, "Participants can have multiple diagnoses", ha="center", va="top",
                transform=ax.transAxes, fontsize=11, color="black")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(FIGDIR / "dashboard_mh_heatmap_update.png", dpi=300)
        plt.close(fig)

    # Percent heatmap
    if not mh_counts.empty:
        heat_counts = (mh_counts
                    .pivot(index="Diagnosis", columns="site", values="Count")
                    .fillna(0)
                    .reindex(columns=SITE_ORDER, fill_value=0))

        denom = cfq_site_totals.reindex(SITE_ORDER, fill_value=0)
        denom_safe = denom.replace(0, np.nan)

        heat_pct = heat_counts.div(denom_safe, axis=1) * 100
        no_dx_pct = (baseline.loc[no_dx_mask].groupby("site")["record_id"]
                     .nunique().reindex(SITE_ORDER, fill_value=0)
                     .div(denom_safe) * 100)
        heat_pct_all = pd.concat([heat_pct, pd.DataFrame([no_dx_pct], index=["No diagnosis"])], axis=0)

        ordered_rows = sorted([r for r in heat_pct.index.tolist()]) + ["No diagnosis"]
        heat_pct_all = heat_pct_all.reindex(ordered_rows)

        row_is_nodx = (heat_pct_all.index.to_numpy() == "No diagnosis")
        mask_main   = np.tile(row_is_nodx[:, None], (1, heat_pct_all.shape[1]))
        mask_nodx   = ~mask_main

        from matplotlib.colors import ListedColormap
        max_color = plt.cm.Blues(1.0)
        nodx_cmap = ListedColormap([max_color])

        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        sns.heatmap(heat_pct_all, mask=mask_main, annot=True, fmt=".1f", cmap="Blues",
                    cbar=True, cbar_kws={'label': '% of participants'}, ax=ax)
        sns.heatmap(heat_pct_all, mask=mask_nodx, annot=True, fmt=".1f", cmap=nodx_cmap,
                    cbar=False, linewidths=.5, linecolor='gray', ax=ax)
        ax.set_title("Diagnoses by site: % of participants (multiple diagnoses possible)",
                    fontsize=14, pad=18)
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        out = FIGDIR / "dashboard_mh_heatmap_percent.png"
        plt.savefig(out, dpi=300)
        plt.close()
        print("✓ Saved →", out)

    # Correlation & co-occurrence
    cfq_data = baseline[["record_id","site"] + avail_cfq].copy()
    cfq_data = cfq_data.dropna(how="all", subset=avail_cfq)
    cfq_bin = cfq_data[avail_cfq].fillna(0).astype(int)
    co_matrix = cfq_bin.T.dot(cfq_bin)
    co_matrix.to_csv(FIGDIR / "dashboard_mh_cooccurrence_counts.csv")
    corr_matrix = cfq_bin.corr(method="pearson")
    plt.figure(figsize=(12,10))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, cbar_kws={'label':'Phi correlation'})
    plt.title("Correlation between diagnoses")
    plt.tight_layout()
    plt.savefig(FIGDIR / "dashboard_mh_corr.png", dpi=300)
    plt.close()

    co_matrix = co_matrix.rename(index=cfq_labels, columns=cfq_labels)
    co_matrix.to_csv(FIGDIR / "dashboard_mh_cooccurrence_counts_labeled.csv")
    plt.figure(figsize=(12,10))
    sns.heatmap(co_matrix, annot=True, fmt=".0f", cmap="Reds",
                cbar_kws={'label':'Number of participants'})
    plt.title("Co-occurrence of diagnoses (counts)")
    plt.tight_layout()
    plt.savefig(FIGDIR / "dashboard_mh_cooccurrence.png", dpi=300)
    plt.close()
    print("✓ Mental health diagnostics visualizations saved.")
else:
    print("⚠ No CFQ mental health variables found in baseline.")

# ────────── TARGETS: read numbers from age-by-group CSVs ──────────
def _find_age_file(stem: str) -> Optional[Path]:
    for p in Path(".").glob(f"*{stem.lower()}*age_by_group*.csv"):
        return p
    return None

def _site_from_filename(fname: Path) -> str:
    return derive_site(fname)

def load_targets_age_by_group_numbers() -> pd.DataFrame:
    rows = []
    for stem in ("calgary", "montreal", "toronto"):
        p = _find_age_file(stem)
        if p is None:
            print(f"ℹ️  Missing age-by-group CSV for: {stem} (skipping overlays for this site)")
            continue
        site = _site_from_filename(p)
        t = pd.read_csv(p, comment="#")
        t = t.rename(columns={"5 wks pre": "<5wks"})
        keep_cols = ["Group"] + AGE_BIN_ORDER + ["Total"]
        cols = [c for c in keep_cols if c in t.columns]
        t = t[cols].copy()
        value_cols = [c for c in AGE_BIN_ORDER if c in t.columns]
        long = (t.melt(id_vars=["Group"], value_vars=value_cols,
                       var_name="age_bin_label", value_name="Target_N"))
        long["site"] = site
        long["Group"] = long["Group"].replace({"Middle Eastern":"Middle East"})
        rows.append(long[["site","Group","age_bin_label","Target_N"]])
    if not rows:
        return pd.DataFrame(columns=["site","Group","age_bin_label","Target_N"])
    out = pd.concat(rows, ignore_index=True)
    out["Target_N"] = pd.to_numeric(out["Target_N"], errors="coerce").fillna(0.0)
    out["age_bin_label"] = pd.Categorical(out["age_bin_label"], categories=AGE_BIN_ORDER, ordered=True)
    return out

targets_age_group = load_targets_age_by_group_numbers()

# ────────── Build observed Sex×Age and Ethnicity×Age tables (BLOCKS) ──────────
# Observed: Sex × Age (limit to Male/Female to match targets)
obs_sex_age_blocks = (
    baseline.dropna(subset=["sex_lbl","age_years"])
            .assign(age_block=lambda d: d["age_years"].apply(to_age_block))
            .dropna(subset=["age_block"])
            .query("sex_lbl in ['Male','Female']")
            .groupby(["site","sex_lbl","age_block"], observed=False)["record_id"]
            .nunique().reset_index(name="Observed_N")
)

# Observed: Ethnicity × Age with collapsing and white-only logic
ASIAN = {"East Asian","South Asian","Southeast Asian"}
INDIG = {"First Nations","Inuk/Inuit","Métis"}

def collapse_eth(lbl: str) -> Optional[str]:
    if lbl in ASIAN: return "Asian"
    if lbl in INDIG: return "Indigenous"
    if lbl == "Middle Eastern": return "Middle East"
    if lbl in {"Black","Middle East","Latin American","Other","White"}: return lbl
    return None

white_only_df = pd.DataFrame()
nonwhite = pd.DataFrame()
if not eth_long.empty:
    by_rec = (eth_long.groupby(["site","record_id"])["label"]
              .apply(lambda s: sorted(set(s.dropna()))).reset_index(name="labels"))
    white_ids = set(by_rec.loc[by_rec["labels"].apply(lambda L: ("White" in set(L)) and (len(set(L))==1)),
                               "record_id"])
    white_only_df = (baseline.loc[baseline["record_id"].isin(white_ids),
                                  ["record_id","site","age_years"]]
                     .assign(Ethnicity="White",
                             age_block=lambda d: d["age_years"].apply(to_age_block)))

    nonwhite = by_rec.loc[by_rec["labels"].apply(lambda L: any(l!="White" for l in set(L)))].copy()
    nonwhite = (nonwhite.assign(EthnicityList=nonwhite["labels"].apply(lambda L: [l for l in L if l!="White"]))
                        .explode("EthnicityList")
                        .rename(columns={"EthnicityList":"Ethnicity"})
                        .drop(columns="labels"))
    nonwhite["Ethnicity"] = nonwhite["Ethnicity"].map(collapse_eth)
    nonwhite = nonwhite.dropna(subset=["Ethnicity"])
    nonwhite = (nonwhite.merge(baseline[["record_id","age_years"]], on="record_id", how="left")
                        .assign(age_block=lambda d: d["age_years"].apply(to_age_block)))

# Observed counts by Ethnicity × Age block
obs_eth_age_blocks = pd.DataFrame()
parts = []
if not white_only_df.empty:
    parts.append(white_only_df[["site","Ethnicity","age_block","record_id"]])
if not nonwhite.empty:
    parts.append(nonwhite[["site","Ethnicity","age_block","record_id"]])
if parts:
    obs_eth_age_blocks = (pd.concat(parts, ignore_index=True)
                            .dropna(subset=["age_block"])
                            .groupby(["site","Ethnicity","age_block"], observed=False)["record_id"]
                            .nunique().reset_index(name="Observed_N"))

# ────────── PLOT: Observed vs Target — Age × Sex (per site; BLOCKS) ──────────
if not targets_age_group.empty:
    targ_sex = targets_age_group[targets_age_group["Group"].isin(["Male","Female"])].copy()
    targ_sex["age_block"] = targ_sex["age_bin_label"].map(bin_label_to_block)
    targ_sex = (targ_sex.dropna(subset=["age_block"])
                        .groupby(["site","Group","age_block"], observed=False)["Target_N"]
                        .sum().reset_index())
    targ_sex = targ_sex.rename(columns={"Group":"sex_lbl"})

    sx_obs = obs_sex_age_blocks.rename(columns={"Observed_N":"N"}).assign(Measure="Observed")
    sx_tgt = targ_sex.rename(columns={"Target_N":"N"}).assign(Measure="Target")
    sx_long = pd.concat([sx_tgt, sx_obs], ignore_index=True)

    sx_long["age_block"] = pd.Categorical(sx_long["age_block"], categories=AGE_BLOCK_ORDER, ordered=True)

    g_sex = sns.catplot(
        data=sx_long, kind="bar",
        x="age_block", y="N", hue="Measure", hue_order=MEASURE_ORDER, palette=MEASURE_COLOURS,
        col="site", row="sex_lbl",
        height=2.8, aspect=1.7, sharey=True, sharex=False, legend_out=True
    )
    g_sex.set_axis_labels("Age group", "Participants")
    g_sex.set_titles("{row_name} — {col_name}")
    for ax in g_sex.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="x", labelrotation=0)
    g_sex.fig.subplots_adjust(top=0.88, hspace=0.35)
    g_sex.fig.suptitle("Observed vs Target by Age × Sex, per site", fontsize=16)
    g_sex.savefig(FIGDIR / "target_vs_observed_age_by_sex.png", dpi=300)
    plt.close(g_sex.fig)
    print("✓ Saved →", FIGDIR / "target_vs_observed_age_by_sex.png")
else:
    print("ℹ️  No age-by-group target CSVs found → skipped 'Age × Sex' overlay figure.")

# ────────── PLOT: Observed vs Target — Age × Ethnicity (per site; BLOCKS) ──────────
plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 16,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
    "axes.titleweight": "bold", "axes.labelweight": "bold",
})

if (not targets_age_group.empty) and (not obs_eth_age_blocks.empty):
    targ_eth = targets_age_group[targets_age_group["Group"].isin(ETHNICITY_ORDER_TARGET)].copy()
    targ_eth["age_block"] = targ_eth["age_bin_label"].map(bin_label_to_block)
    targ_eth = (targ_eth.dropna(subset=["age_block"])
                        .groupby(["site","Group","age_block"], observed=False)["Target_N"]
                        .sum().reset_index()
                        .rename(columns={"Group":"Ethnicity"}))

    et_obs = obs_eth_age_blocks.rename(columns={"Observed_N":"N"}).assign(Measure="Observed")
    et_tgt = targ_eth.rename(columns={"Target_N":"N"}).assign(Measure="Target")
    et_long = pd.concat([et_tgt, et_obs], ignore_index=True)

    et_long["Ethnicity"] = pd.Categorical(et_long["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True)
    et_long["age_block"] = pd.Categorical(et_long["age_block"], categories=AGE_BLOCK_ORDER, ordered=True)
    et_long = et_long.sort_values(["Ethnicity","site","age_block","Measure"])

    g_eth = sns.catplot(
        data=et_long, kind="bar",
        x="age_block", y="N",
        hue="Measure", hue_order=MEASURE_ORDER, palette=MEASURE_COLOURS,
        col="site", row="Ethnicity",
        height=2.1, aspect=1.9, sharey="row", sharex=False, legend_out=True
    )
    g_eth.set_axis_labels("Age group", "Participants")
    g_eth.set_titles("{row_name} — {col_name}")
    for ax in g_eth.axes.flatten():
        if ax is not None:
            ax.tick_params(axis="x", labelrotation=0)
    g_eth.fig.subplots_adjust(top=0.92, hspace=0.45)
    g_eth.fig.suptitle("Observed vs Target by Age × Ethnicity, per site", fontsize=16)
    g_eth.savefig(FIGDIR / "target_vs_observed_age_by_ethnicity.png", dpi=300)
    plt.close(g_eth.fig)
    print("✓ Saved →", FIGDIR / "target_vs_observed_age_by_ethnicity.png")
else:
    print("ℹ️  Missing target or observed ethnicity-by-age data → skipped 'Age × Ethnicity' overlay figure.")

# ────────── Observed vs Target (totals by ethnicity) ──────────
def _collapse_for_totals(lbl: str) -> Optional[str]:
    if lbl in ASIAN: return "Asian"
    if lbl in INDIG: return "Indigenous"
    if lbl == "Middle Eastern": return "Middle East"
    if lbl in {"Black","Middle East","Latin American","Other","White"}: return lbl
    return None

obs_eth_tot = pd.DataFrame()
if not eth_long.empty:
    by_rec2 = eth_long.copy()
    by_rec2["Ethnicity"] = by_rec2["label"].map(_collapse_for_totals)
    by_rec2 = by_rec2.dropna(subset=["Ethnicity"])
    white_tot = pd.DataFrame(columns=["site","record_id","Ethnicity"])
    if not white_only_df.empty:
        white_tot = white_only_df[["site","record_id"]].assign(Ethnicity="White")
    nonwhite_tot = by_rec2[by_rec2["Ethnicity"]!="White"][["site","record_id","Ethnicity"]].drop_duplicates()
    obs_all = pd.concat([white_tot, nonwhite_tot], ignore_index=True)
    obs_eth_tot = (obs_all.groupby(["site","Ethnicity"], observed=False)["record_id"]
                   .nunique().reset_index(name="Observed_N"))

rows = []
for site, payload in TARGET_TOTALS.items():
    for eth, n in payload["Ethnicity"].items():
        rows.append({"site":site, "Ethnicity":eth, "Target_N": int(n)})
targets_tot_eth = pd.DataFrame(rows)

if not obs_eth_tot.empty:
    compare_df = (targets_tot_eth.merge(obs_eth_tot, on=["site","Ethnicity"], how="left")
                                 .fillna({"Observed_N":0}).astype({"Observed_N":int}))
    compare_df["Ethnicity"] = pd.Categorical(compare_df["Ethnicity"], categories=ETHNICITY_ORDER_TARGET, ordered=True)
    compare_df = compare_df.sort_values(["site","Ethnicity"])
    cmp_long = pd.concat([
        compare_df.rename(columns={"Target_N":"N"})[["site","Ethnicity","N"]].assign(Measure="Target"),
        compare_df.rename(columns={"Observed_N":"N"})[["site","Ethnicity","N"]].assign(Measure="Observed")
    ], ignore_index=True)
    g3 = sns.catplot(
        data=cmp_long, kind="bar",
        x="Ethnicity", y="N", hue="Measure", hue_order=MEASURE_ORDER, palette=MEASURE_COLOURS,
        col="site",
        order=ETHNICITY_ORDER_TARGET,
        height=4.8, aspect=1.25, sharey=True
    )
    g3.set_xticklabels(rotation=35)
    g3.set_axis_labels("Ethnicity (target bins)", "Participants")
    g3.fig.subplots_adjust(top=0.86)
    g3.fig.suptitle("Observed vs Target recruitment by Ethnicity and Site", fontsize=16)
    g3.savefig(FIGDIR / "ethnicity_observed_vs_target.png", dpi=300)
    plt.close(g3.fig)
    print("✓ Saved ethnicity totals comparison →", FIGDIR / "ethnicity_observed_vs_target.png")
else:
    print("ℹ️  No observed ethnicity data; skipped totals comparison plot.")

# ────────── COMBINE ALL FIGURES INTO A SINGLE PDF ──────────
def _add_image_page(pp: PdfPages, img_path: Path, title: Optional[str] = None):
    import matplotlib.image as mpimg
    try:
        img = mpimg.imread(img_path)
    except Exception as e:
        print(f"⚠️  Skipping {img_path.name}: {e}")
        return
    fig, ax = plt.subplots(figsize=(11, 8.5))  # US Letter landscape
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.imshow(img); ax.axis("off")
    if title:
        fig.suptitle(title, y=0.98, fontsize=12)
        fig.subplots_adjust(top=0.95)
    pp.savefig(fig, dpi=300)
    plt.close(fig)

def _add_cover_page(pp: PdfPages):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis("off")
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    ax.text(0.5, 0.6, "C-PIP Multi-site Dashboard\n\n" + f"Generated: {now}",
            ha="center", va="center", fontsize=18)
    pp.savefig(fig); plt.close(fig)

preferred = [
    ("dashboard.png",                         "6-panel overview"),
    ("dashboard_mri_by_age_fixed.png",        "MRI sessions by age group (Completed + Partial)"),
    ("target_vs_observed_age_by_sex.png",     "Observed vs Target — Age × Sex"),
    ("target_vs_observed_age_by_ethnicity.png","Observed vs Target — Age × Ethnicity"),
    ("ethnicity_observed_vs_target.png",      "Observed vs Target — Ethnicity totals"),
    ("dashboard_mh_heatmap_update.png",       "Diagnoses by site (heatmap + ‘No diagnosis’ row)"),
]

OUTPDF = FIGDIR / "cpip_dashboard_all_figs.pdf"
with PdfPages(OUTPDF) as pp:
    _add_cover_page(pp)
    for fname, ttl in preferred:
        p = FIGDIR / fname
        if p.exists():
            _add_image_page(pp, p, ttl)
        else:
            print(f"ℹ️  Missing (skipped): {fname}")

print(f"✅ PDF saved → {OUTPDF}")


################### DEBUG CODE #################################
# ────────── DEBUG: check MRI outcome distribution by site ──────────
print("\n──────── MRI session outcome summary ────────")
outcome_summary = (
    mri_base.groupby(["site", "mri_ses_outcome"])["record_id"]
    .nunique()
    .reset_index()
    .rename(columns={"record_id": "Participants"})
)
print(outcome_summary.to_string(index=False))

print("\n──────── Scheduled MRI sessions (future-dated) ────────")
future_sched = mri_base.query("mri_ses_outcome == 2 and mri_date > @today")
if future_sched.empty:
    print("⚠️  No future-dated (status=2) MRI sessions found.")
else:
    print(future_sched[["site","record_id","mri_date","mri_ses_outcome"]]
          .sort_values(["site","mri_date"]).to_string(index=False))

print("\n──────── Calgary 'Scheduled' entries (past-dated) ────────")
print(
    mri_base.query("site == 'Calgary' and mri_ses_outcome == 2")[
        ["record_id", "mri_date", "mri_ses_outcome"]
    ].sort_values("mri_date").to_string(index=False)
)
print("\n──────── Inconsistent MRI entries (date present but outcome != 1/3) ────────")
mismatch = mri_base.query("mri_date <= @today and mri_ses_outcome not in [1,3]")
if mismatch.empty:
    print("✅ No inconsistencies found.")
else:
    print(mismatch[["site","record_id","mri_date","mri_ses_outcome"]]
          .sort_values(["site","mri_date"]).to_string(index=False))


# --- Compute scanning rates for the last 3 months ---
today = pd.Timestamp.now().normalize()
start_3mo = today - pd.DateOffset(months=3)

recent = (
    mri_base
    .query("mri_date >= @start_3mo and mri_date <= @today and status_collapsed == 'Completed'")
    .copy()
)

weeks_in_period = (today - start_3mo).days / 7
active_weeks_in_period = weeks_in_period * (5/7)

rate_df = (
    recent.groupby("site")["record_id"]
    .nunique()
    .reset_index(name="participants_last_3mo")
    .assign(
        weeks_in_period=weeks_in_period,
        active_weeks_in_period=active_weeks_in_period,
        rate_per_week=lambda d: d["participants_last_3mo"] / d["weeks_in_period"],
        rate_per_active_week=lambda d: d["participants_last_3mo"] / d["active_weeks_in_period"]
    )
)


print("\n──────── Scanning Rate Summary (Last 3 Months) ────────")
print(rate_df.to_string(index=False))
rate_df.to_csv(FIGDIR / "scanning_rate_last3months.csv", index=False)


# ────────── Debug: check if Sex NAs are truly missing (T1 date passed) ──────────
print("\n──────── Debug: Checking if Sex NAs are truly missing (T1 date passed) ────────")

# Handle both real NaN and string "NA"
na_mask = baseline["sex_lbl"].isna() | baseline["sex_lbl"].eq("NA")
na_sex_rows = baseline.loc[na_mask, ["record_id", "site", "sex_code_effective"]].copy()

# Merge with MRI info
t1_dates = (
    df_main.loc[df_main["redcap_event_name"] == "t1_arm_1",
                ["record_id", "site", "mri_date", "mri_ses_outcome"]]
    .copy()
)
t1_dates["mri_date"] = pd.to_datetime(t1_dates["mri_date"], errors="coerce")
t1_dates = t1_dates.dropna(subset=["mri_date"]).sort_values(["record_id", "mri_date"])
t1_latest = t1_dates.drop_duplicates(subset="record_id", keep="last")

today = pd.Timestamp.now().normalize()
na_sex_with_date = na_sex_rows.merge(t1_latest, on=["record_id", "site"], how="left")
na_sex_with_date["T1_passed"] = na_sex_with_date["mri_date"] <= today

# Summary
total_na = len(na_sex_with_date)
filled_dates = na_sex_with_date["mri_date"].notna().sum()
true_missing = na_sex_with_date["T1_passed"].sum()

print(f"Total sex NAs (including string 'NA'): {total_na}")
print(f"→ With T1 MRI date recorded: {filled_dates}")
print(f"→ T1 date has passed (true missing responses): {true_missing}")

if true_missing > 0:
    print("\nExamples of true missing cases:")
    print(
        na_sex_with_date.loc[
            na_sex_with_date["T1_passed"],
            ["record_id", "site", "mri_date", "sex_code_effective"]
        ].head(10).to_string(index=False)
    )
else:
    print("\nNo participants with missing sex data *after* T1 date.")


print("\n\n============================")
print(" SEX VARIABLE DEBUG SUMMARY ")
print("============================\n")

# 1. Raw value distribution BEFORE mapping
print("1) Raw values before mapping (baseline rows):")
raw_vals = baseline["sex_code_effective"].value_counts(dropna=False)
print(raw_vals.to_string())
print()

print("\n2) Raw values across ALL T1 rows:")

# Identify the raw sex column in df_main
RAW_SEX_COL = YOUTH_SEX_COL or PARENT_SEX_COL

# Extract all baseline T1 rows from df_main (raw REDCap)
t1_all = df_main[df_main["redcap_event_name"] == "t1_arm_1"][
    ["record_id", "site", RAW_SEX_COL]
].copy()

# Print raw distribution of sex entries at T1
print(t1_all[RAW_SEX_COL].value_counts(dropna=False).to_string())
print()

# Merge EFFECTIVE sex from baseline → t1_all
# (baseline contains the deduplicated T1 rows + sex_code_effective)
all_t1 = t1_all.merge(
    baseline[["record_id", "sex_code_effective"]],
    on="record_id",
    how="left"
)

print("Effective sex across ALL T1 rows (baseline-matched):")
print(all_t1["sex_code_effective"].value_counts(dropna=False).to_string())
print()



# 3. Mapped distribution AFTER mapping
print("3) Values AFTER mapping (baseline):")
print(baseline["sex_lbl"].value_counts(dropna=False).to_string())
print()

# 4. Detect TRUE NAs (real missing, not special codes)
true_na_mask = baseline["sex_lbl"].isna()
true_na = baseline.loc[true_na_mask, ["record_id", "site", "sex_code_effective"]]
print(f"4) Number of TRUE missing sex values at baseline: {len(true_na)}")
print(true_na.head(20).to_string(index=False))
print()

# 5. Check if any of these participants have sex filled in ANY T1 row
all_t1 = t1_all.merge(
    baseline[["record_id", "sex_code_effective"]],
    on="record_id",
    how="left"
)[["record_id", "sex_code_effective"]]

any_filled = (
    all_t1.groupby("record_id")["sex_code_effective"]
    .apply(lambda x: x.notna().any())
)

merged = true_na.merge(any_filled.rename("any_filled"), on="record_id", how="left")

print("5) For baseline-missing sex participants, did ANY T1 row have a sex value?")
print(merged[["record_id", "site", "sex_code_effective", "any_filled"]].to_string(index=False))
print()

# 6. Identify 'ghosted' cases where sex existed but the chosen baseline row lost it
ghosted_ids = merged.loc[merged["any_filled"] == True, "record_id"].unique()

if len(ghosted_ids) > 0:
    print("6) ⚠️ Participants where SEX EXISTS in REDCap T1 but the selected baseline row is missing it:")
    ghost_df = t1_all[t1_all["record_id"].isin(ghosted_ids)][["record_id", "site", "sex_code_effective"]]
    print(ghost_df.sort_values(["record_id", "site"]).to_string(index=False))
else:
    print("6) ✅ No ghosted participants — baseline selection did NOT drop sex values.")
print()

# 7. Unknown raw SEX codes that are not in sex_map
invalid_codes = set(
    baseline["sex_code_effective"].dropna().unique()
) - set(sex_map.keys())
print("7) Unknown/invalid raw SEX codes (not in sex_map):", invalid_codes)
print()

print("============================")
print(" END OF SEX DEBUG SUMMARY ")
print("============================\n")

print("\n==============================")
print(" SEX MISSINGNESS — ROOT CAUSE ANALYSIS")
print("==============================\n")

# ------------------------------------------------------------
# Prep
# ------------------------------------------------------------

# Use the actual REDCap raw field ("sex" or "demo_sex"), NOT the derived field.
SEX_RAW = YOUTH_SEX_COL or PARENT_SEX_COL

baseline_sex_na = baseline["sex_lbl"].isna() | baseline["sex_lbl"].eq("NA")

# Build the baseline missing-sex frame safely
if SEX_RAW is None:
    print("\n[Root-cause] ⚠️ No raw REDCap sex column found in export. "
          "Skipping raw-level cross-checks.\n")
    na_baseline_df = baseline.loc[baseline_sex_na, ["record_id", "site"]].copy()
else:
    na_baseline_df = baseline.loc[
        baseline_sex_na, ["record_id", "site", SEX_RAW]
    ].copy()

na_ids = set(na_baseline_df["record_id"])

# ------------------------------------------------------------
# All T1 rows (raw REDCap)
# ------------------------------------------------------------
_t1_cols = ["record_id", "site", "redcap_repeat_instrument"]
if SEX_RAW is not None:
    _t1_cols.append(SEX_RAW)

t1_all = df_main[df_main["redcap_event_name"] == "t1_arm_1"][_t1_cols].copy()

# ------------------------------------------------------------
# Repeating-instrument rows
# ------------------------------------------------------------
_rep_cols = ["record_id", "site", "redcap_repeat_instrument"]
if SEX_RAW is not None:
    _rep_cols.append(SEX_RAW)

repeat_rows = df_main[df_main["redcap_repeat_instrument"].notna()][_rep_cols].copy()

# ------------------------------------------------------------
# 1. TRUE MISSING — no sex in ANY T1 row
# ------------------------------------------------------------
any_sex_in_t1 = (
    t1_all.groupby("record_id")[SEX_RAW]
    .apply(lambda x: x.notna().any())
    .reindex(na_ids, fill_value=False)
)

true_missing = {rid for rid, has in any_sex_in_t1.items() if not has}

# ------------------------------------------------------------
# 2. GHOSTED — sex exists in another T1 row, but baseline row missed it
# ------------------------------------------------------------
ghosted = {rid for rid, has in any_sex_in_t1.items() if has}

# ------------------------------------------------------------
# 3. INVALID CODES — raw code exists but does not map to sex_map
# ------------------------------------------------------------
valid_codes = set(sex_map.keys())
invalid_codes = {}

for _, row in na_baseline_df.iterrows():
    raw = row[SEX_RAW]
    rid = row["record_id"]
    if pd.notna(raw) and raw not in valid_codes:
        invalid_codes.setdefault(raw, []).append(rid)

invalid_ids = {rid for lst in invalid_codes.values() for rid in lst}

# ------------------------------------------------------------
# 4. REPEATING-ONLY — sex answered only in repeating instruments
# ------------------------------------------------------------
repeat_sex = repeat_rows[repeat_rows[SEX_RAW].notna()].copy()
repeat_only = set(
    repeat_sex["record_id"]
).intersection(na_ids) - true_missing - ghosted - invalid_ids

# Final classification table
import pandas as pd

rows = []
for rid in sorted(na_ids):
    if rid in invalid_ids:
        label = "INVALID_CODE"
    elif rid in repeat_only:
        label = "REPEATING_ONLY"
    elif rid in ghosted:
        label = "GHOSTED"
    else:
        label = "TRUE_MISSING"
    site = na_baseline_df.loc[na_baseline_df["record_id"] == rid, "site"].iloc[0]
    raw = na_baseline_df.loc[na_baseline_df["record_id"] == rid, SEX_RAW].iloc[0]
    rows.append({"record_id": rid, "site": site, "raw": raw, "category": label})

miss_df = pd.DataFrame(rows).sort_values(["category", "site", "record_id"])

print("\n===== SEX Missingness Classification =====")
print(miss_df.to_string(index=False))

print("\nSummary:")
print(miss_df.groupby("category")["record_id"].nunique().to_string())
print("\nInvalid raw codes detected:", invalid_codes)

print("\n──────── Showing 3 detailed examples of SEX-missing participants ────────")

na_ids = list(na_baseline_df["record_id"].unique())

if len(na_ids) == 0:
    print("✅ No missing-sex participants to inspect.")
else:
    example_ids = sorted(na_ids)[:3]

    for rid in example_ids:
        print(f"\n=== Participant {rid} ===")

        # Raw REDCap rows for this participant
        sub = df_main[df_main["record_id"] == rid].copy()

        # Merge in sex_code_effective from baseline (exists ONLY there)
        sub = sub.merge(
            baseline[["record_id", "sex_code_effective"]],
            on="record_id",
            how="left"
        )

        # Keep columns that actually exist
        cols = [
            "record_id",
            "site",
            "redcap_event_name",
            "redcap_repeat_instrument",
            "sex_code_effective",
        ]
        cols = [c for c in cols if c in sub.columns]

        # Sort and print
        print(
            sub[cols]
            .sort_values(["redcap_event_name", "redcap_repeat_instrument"])
            .to_string(index=False)
        )


print("\n==============================")
print(" CHECKING: Did participants fill other demographics but NOT sex?")
print("==============================\n")

# Demographic columns available in your dataset
AGE_COL = "age_years"                # derived age value
INCOME_COL = "demo_cg_income"        # merged income code
ETHNICITY_COLS = ETH_COLS            # list of checkbox vars

other_demo_cols = [AGE_COL, GENDER_COL, INCOME_COL] + ETHNICITY_COLS

# Participants missing sex at baseline
sex_na_ids = set(baseline[baseline["sex_lbl"].isna()]["record_id"])

rows = []

for rid in sorted(sex_na_ids):
    # all T1 rows for this participant
    sub = df_main[
        (df_main["record_id"] == rid) &
        (df_main["redcap_event_name"] == "t1_arm_1")
    ]

    if sub.empty:
        continue

    # Check if any OTHER demo fields are filled
    demo_filled = {
        col: sub[col].notna().any()
        for col in other_demo_cols
        if col in sub.columns
    }

    # Count of non-sex demo fields filled
    num_filled = sum(demo_filled.values())

    rows.append({
        "record_id": rid,
        "site": sub["site"].iloc[0],
        "sex_missing": True,
        "n_other_fields_filled": num_filled,
        **demo_filled
    })

demo_missing_df = pd.DataFrame(rows)

print("\n=== Summary of participants with sex missing but other fields filled ===")
print(
    demo_missing_df[demo_missing_df["n_other_fields_filled"] > 0]
    .to_string(index=False)
)

print("\nTotal participants missing sex but with at least one other field filled:",
      demo_missing_df["n_other_fields_filled"].gt(0).sum())

# Show up to 3 concrete cases
examples = demo_missing_df[
    demo_missing_df["n_other_fields_filled"] > 0
].head(3)["record_id"]

for rid in examples:
    print(f"\n--- Detailed rows for participant {rid} ---")
    sub = df_main[df_main["record_id"] == rid].copy()
    cols = ["record_id","site","redcap_event_name","redcap_repeat_instrument","sex_code_effective"] \
           + [AGE_COL, GENDER_COL, INCOME_COL] + ETHNICITY_COLS
    cols = [c for c in cols if c in sub.columns]  # keep only existing columns
    print(sub[cols]
          .sort_values(["redcap_event_name","redcap_repeat_instrument"])
          .to_string(index=False))


print("\n──────── Sex Missingness by Age Bin ────────")

# Use the derived age column created earlier in your script
AGE_COL = "age_years"

df = baseline.copy()
df["sex_missing"] = df["sex_lbl"].isna()

# Age bins (adjust if wanted)
age_bins = [8, 10, 12, 14, 16, 18]
df["age_bin"] = pd.cut(df[AGE_COL], bins=age_bins, right=False)

summary = (
    df.groupby(["site", "age_bin"])
      .agg(
          total=("record_id", "nunique"),
          sex_missing=("sex_missing", "sum"),
          sex_present=("sex_missing", lambda x: (~x).sum()),
      )
      .reset_index()
)

summary["percent_missing"] = (
    100 * summary["sex_missing"] / summary["total"]
).round(1)

print(summary.to_string(index=False))

# ----------------------------------------
# Completion-aware missingness for sex
# ----------------------------------------

# Form-level completion variables from REDCap
YOUTH_COMPLETE = "youth_demographics_baseline_complete"
PARENT_COMPLETE = "caregiver_demographics_baseline_about_child_complete"

# Normalize completion flags to booleans (2 = complete in REDCap)
def _is_complete(x):
    return x == 2

baseline["youth_form_complete"] = baseline[YOUTH_COMPLETE].apply(_is_complete)
baseline["parent_form_complete"] = baseline[PARENT_COMPLETE].apply(_is_complete)

# Sex variable comes from "sex_code_effective"; missing is only meaningful if the form was truly completed
baseline["sex_missing_effective"] = (
    baseline["sex_lbl"].isna() &    # missing sex
    (
        baseline["youth_form_complete"] | 
        baseline["parent_form_complete"]
    )   # but only count as missing if one of these forms was completed
)

# For comparison, also compute sex_present_effective
baseline["sex_present_effective"] = (
    (~baseline["sex_lbl"].isna()) &
    (baseline["youth_form_complete"] | baseline["parent_form_complete"])
)

print("\n──────── VALIDATION: Sex Missingness vs Form Completion ────────")

cols_to_show = [
    "record_id", "site", "age_years",
    "sex_code_effective", "sex_lbl",
    "youth_form_complete", "parent_form_complete",
    "sex_missing_effective", "sex_present_effective"
]

validation_df = baseline[cols_to_show].copy()

# Sort for easier reading
validation_df = validation_df.sort_values(["site", "record_id"])

print(validation_df.head(50).to_string(index=False))

# Optional: show ONLY cases counted as missing
print("\n──────── Cases Counted as TRUE Missing (effective) ────────")
missing_cases = validation_df[validation_df["sex_missing_effective"] == True]
print(missing_cases.head(50).to_string(index=False))

# Optional: summary counts for sanity
print("\n──────── Summary ────────")
print("Total participants:", len(baseline))
print("Form-complete participants:", 
      (baseline["youth_form_complete"] | baseline["parent_form_complete"]).sum())
print("True sex missing (effective):", baseline["sex_missing_effective"].sum())
print("True sex present (effective):", baseline["sex_present_effective"].sum())


# ----------------------------------------
# Completion-aware + age-aware missingness for sex
# ----------------------------------------

AGE_THRESHOLD = 11  # younger participants do not answer sex

baseline["age_valid_for_sex"] = baseline["age_years"] > AGE_THRESHOLD

baseline["sex_missing_effective"] = (
    baseline["sex_lbl"].isna()
    &
    (baseline["youth_form_complete"] | baseline["parent_form_complete"])  # form(s) completed
    &
    (baseline["age_valid_for_sex"])  # age > 11 years
)

baseline["sex_present_effective"] = (
    (~baseline["sex_lbl"].isna())
    &
    (baseline["youth_form_complete"] | baseline["parent_form_complete"])
    &
    (baseline["age_valid_for_sex"])
)

# Sex comes from caregiver demographics
FORM_FOR_SEX = "caregiver_demographics_baseline_about_child_complete"

baseline["parent_form_complete"] = baseline[FORM_FOR_SEX].eq(2)

baseline["sex_missing_effective"] = (
    baseline["sex_lbl"].isna()
    &
    baseline["parent_form_complete"]  # only check caregiver form
)

baseline["sex_present_effective"] = (
    (~baseline["sex_lbl"].isna())
    &
    baseline["parent_form_complete"]
)

print("\n──────── TRUE Missing Sex: Montreal Participants ────────")

# Filter for Montreal and true missing
mtl_missing = baseline[
    (baseline["site"] == "Montreal")
    & (baseline["sex_missing_effective"] == True)
]

cols_to_show = [
    "record_id",
    "site",
    "age_years",
    "sex_code_effective",
    "sex_lbl",
    "parent_form_complete",
    "youth_form_complete",
]

print(mtl_missing[cols_to_show].sort_values("record_id").to_string(index=False))

print("\nTotal Montreal participants with TRUE missing sex:", len(mtl_missing))

print("\n──────── Montreal: TRUE Missing Sex (record_id only) ────────")

mtl_missing_ids = baseline[
    (baseline["site"] == "Montreal")
    & (baseline["sex_missing_effective"] == True)
]["record_id"].sort_values()

print(mtl_missing_ids.to_list())
print("\nTotal Montreal TRUE missing sex:", len(mtl_missing_ids))


# ────────── Montreal: find the single TRUE-missing-sex participant ──────────
print("\n──────── Montreal TRUE-missing candidate search (age≥11, forms complete, T1 passed, no sex anywhere) ────────")

# Safety: define boolean completions if not already present
for _col in ["youth_form_complete", "parent_form_complete"]:
    if _col not in baseline.columns:
        baseline[_col] = False  # if you truly don't have these, everything will be treated as not complete

# T1 latest date per participant (we computed t1_dates earlier)
_t1 = df_main.loc[df_main["redcap_event_name"] == "t1_arm_1", ["record_id", "site", "mri_date"]].copy()
_t1["mri_date"] = pd.to_datetime(_t1["mri_date"], errors="coerce")
_t1 = _t1.dropna(subset=["mri_date"]).sort_values(["record_id", "mri_date"])
_t1_latest = _t1.drop_duplicates(subset="record_id", keep="last")
_today = pd.Timestamp.now().normalize()

# Any sex in ANY T1 row?  (raw REDCap → merge effective)
tmp = (
    df_main.loc[df_main["redcap_event_name"] == "t1_arm_1", ["record_id", SEX_RAW]]
    .merge(
        baseline[["record_id", "sex_code_effective"]],
        on="record_id",
        how="left"
    )
)

# Effective sex exists anywhere?
any_sex_in_t1 = (
    tmp.groupby("record_id")["sex_code_effective"]
       .apply(lambda s: s.notna().any())
       .rename("any_sex_in_t1")
)


# Assemble Montreal baseline with flags
mtl = (baseline[["record_id","site","age_years","sex_lbl","youth_form_complete","parent_form_complete"]]
       .merge(t1_latest[["record_id","mri_date"]], on="record_id", how="left")
       .merge(any_sex_in_t1, on="record_id", how="left"))

mtl["forms_complete"] = mtl["youth_form_complete"].fillna(False) | mtl["parent_form_complete"].fillna(False)
mtl["t1_passed"] = mtl["mri_date"].le(_today)
mtl["sex_missing_baseline"] = mtl["sex_lbl"].isna() | (mtl["sex_lbl"] == "NA")
mtl["age_ok"] = mtl["age_years"].ge(11)

# TRUE-missing definition (apply all filters)
mtl_true_missing = mtl.loc[
    (mtl["site"] == "Montreal")
    & mtl["sex_missing_baseline"]
    & mtl["forms_complete"]
    & mtl["age_ok"]
    & mtl["t1_passed"]
    & (~mtl["any_sex_in_t1"].fillna(False))
].copy()

ids = sorted(mtl_true_missing["record_id"].unique())

print(f"Total Montreal participants with TRUE missing sex (after all filters): {len(ids)}")
print("IDs:", ids)

# If exactly one, print it cleanly for copy/paste
if len(ids) == 1:
    print("\n👉 Montreal record_id with TRUE missing sex:", ids[0])
elif len(ids) == 0:
    print("\n✅ No Montreal participants meet the TRUE-missing criteria.")
else:
    # Show a concise table to help diagnose why multiples remain
    cols_show = ["record_id","age_years","sex_lbl","youth_form_complete","parent_form_complete","mri_date","any_sex_in_t1"]
    print("\n⚠️ More than one candidate matched. First few rows for debugging:")
    print(mtl_true_missing[cols_show].sort_values("record_id").head(15).to_string(index=False))


# ────────── TRUE MISSING AGE ANALYSIS ──────────
print("\n──────── TRUE Missing Age Analysis ────────")

# True missing = participant has no age info (both youth_age_y and youth_age_m missing)
age_missing = baseline[baseline["age_years"].isna()].copy()

# Count per site
age_missing_summary = (
    age_missing.groupby("site")["record_id"]
    .nunique()
    .reindex(SITE_ORDER, fill_value=0)
    .rename("TrueMissingAgeCount")
)

print("\nCounts of participants with true missing age per site:")
print(age_missing_summary.to_string())

# Optional: show examples
print("\nExamples of participants with true missing age:")
cols = ["record_id", "site", "youth_age_y", "youth_age_m", "age_years"]
print(age_missing[cols].sort_values(["site", "record_id"]).head(20).to_string(index=False))

# Also, proportion of missing age per site
total_per_site = baseline.groupby("site")["record_id"].nunique()
prop_missing = (100 * age_missing_summary / total_per_site).round(1).rename("PercentMissing")
summary_df = pd.concat([age_missing_summary, total_per_site.rename("TotalParticipants"), prop_missing], axis=1)
print("\nSummary (count and %):")
print(summary_df.to_string(index=True))
